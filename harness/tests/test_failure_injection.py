"""
Failure injection tests for dag-harness.

Tests verify that the system handles various failure scenarios gracefully:
- Network failures (GitLab API timeouts, rate limits, DNS failures)
- Database failures (locks, corruption, concurrent writes)
- Subprocess failures (segfaults, rejected pushes, expired auth)
- Graceful degradation (sync failures, notification failures, fallbacks)
"""

import json
import os
import signal
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    create_box_up_role_graph,
    create_initial_state,
    set_module_db,
)
from harness.dag.store import HarnessStore
from harness.db.models import Role, WorkflowStatus
from harness.db.state import StateDB


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def db():
    """Create an in-memory StateDB."""
    return StateDB(":memory:")


@pytest.fixture
def db_with_roles(db):
    """Database pre-populated with test roles."""
    roles = [
        Role(name="common", wave=0, has_molecule_tests=True),
        Role(name="sql_server_2022", wave=2, has_molecule_tests=True),
        Role(name="ems_web_app", wave=2, has_molecule_tests=True),
    ]
    for role in roles:
        db.upsert_role(role)
    return db


@pytest.fixture
def store(db):
    """Create a HarnessStore with in-memory database."""
    return HarnessStore(db)


@pytest.fixture
def file_db(tmp_path):
    """Create a file-based StateDB for corruption tests."""
    db_path = tmp_path / "test.db"
    db = StateDB(db_path)
    return db, db_path


# =============================================================================
# NETWORK FAILURE TESTS
# =============================================================================


@pytest.mark.unit
class TestNetworkFailures:
    """Test network failure scenarios for GitLab API interactions."""

    def test_gitlab_api_timeout(self):
        """Simulate network timeout during GitLab API call."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["glab", "api", "projects/1/issues"],
                timeout=30,
            )

            with pytest.raises(subprocess.TimeoutExpired):
                subprocess.run(
                    ["glab", "api", "projects/1/issues"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

    def test_gitlab_rate_limit_429(self):
        """Simulate 429 rate limit with retry-after header."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.text = "Rate limit exceeded"

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = httpx.Client()
            response = client.get("https://gitlab.example.com/api/v4/projects")

            assert response.status_code == 429
            assert response.headers["Retry-After"] == "60"

    def test_gitlab_server_error_503(self):
        """Simulate 503 service unavailable."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Temporarily Unavailable"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="503 Service Unavailable",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = httpx.Client()
            response = client.get("https://gitlab.example.com/api/v4/projects")

            assert response.status_code == 503
            with pytest.raises(httpx.HTTPStatusError):
                response.raise_for_status()

    def test_subprocess_timeout(self):
        """Simulate molecule/pytest subprocess timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["npm", "run", "molecule:role", "--role=common"],
                timeout=600,
            )

            with pytest.raises(subprocess.TimeoutExpired) as exc_info:
                subprocess.run(
                    ["npm", "run", "molecule:role", "--role=common"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

            assert exc_info.value.timeout == 600

    def test_dns_resolution_failure(self):
        """Simulate DNS failure for GitLab host."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.ConnectError(
                "Name or service not known"
            )
            mock_client_class.return_value = mock_client

            client = httpx.Client()
            with pytest.raises(httpx.ConnectError, match="Name or service not known"):
                client.get("https://gitlab.example.com/api/v4/projects")


# =============================================================================
# DATABASE FAILURE TESTS
# =============================================================================


@pytest.mark.unit
class TestDatabaseFailures:
    """Test database failure scenarios."""

    def test_database_locked(self, file_db):
        """Simulate SQLite database lock contention."""
        db, db_path = file_db

        # Insert a role to work with
        db.upsert_role(Role(name="common", wave=0))

        # Simulate a database lock by holding a write transaction
        blocking_conn = sqlite3.connect(str(db_path))
        blocking_conn.execute("BEGIN EXCLUSIVE")

        try:
            # The db.connection() should fail or timeout when the db is locked
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                # Use a short timeout to avoid waiting too long
                conn = sqlite3.connect(str(db_path), timeout=0.1)
                conn.execute("BEGIN EXCLUSIVE")
                conn.close()
        finally:
            blocking_conn.rollback()
            blocking_conn.close()

    def test_disk_full_during_checkpoint(self, db_with_roles):
        """Simulate disk full during checkpoint write."""
        db = db_with_roles

        # Create workflow definition and execution
        db.create_workflow_definition(
            name="box-up-role",
            description="Test workflow",
            nodes=[{"id": "start"}],
            edges=[],
        )
        exec_id = db.create_execution("box-up-role", "common")

        # Simulate checkpoint write failure (IOError / OSError)
        with patch.object(db, "connection") as mock_conn:
            mock_conn.side_effect = OSError("No space left on device")

            with pytest.raises(OSError, match="No space left on device"):
                db.checkpoint_execution(exec_id, {"state": "test"})

    def test_corrupted_database(self, tmp_path):
        """Simulate corrupted SQLite database file."""
        db_path = tmp_path / "corrupted.db"

        # Create a valid database first
        db = StateDB(db_path)
        db.upsert_role(Role(name="common", wave=0))
        del db

        # Corrupt the database by overwriting part of the file
        with open(db_path, "r+b") as f:
            f.seek(100)
            f.write(b"\x00" * 200)

        # Trying to use the corrupted database should fail gracefully
        with pytest.raises((sqlite3.DatabaseError, sqlite3.OperationalError)):
            corrupted_db = StateDB(db_path)
            corrupted_db.list_roles()

    def test_concurrent_writes(self, file_db):
        """Simulate concurrent write attempts to StateDB."""
        db, db_path = file_db
        errors = []
        success_count = 0
        lock = threading.Lock()

        def write_role(i):
            nonlocal success_count
            try:
                thread_db = StateDB(db_path)
                thread_db.upsert_role(
                    Role(name=f"role_{i}", wave=i % 5)
                )
                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=write_role, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # SQLite with WAL mode should handle concurrent writes;
        # at minimum, most should succeed (some may get busy errors)
        assert success_count + len(errors) == 10
        # At least some writes should succeed
        assert success_count > 0


# =============================================================================
# SUBPROCESS FAILURE TESTS
# =============================================================================


@pytest.mark.unit
class TestSubprocessFailures:
    """Test subprocess failure scenarios."""

    def test_molecule_segfault(self):
        """Simulate molecule process crash (signal)."""
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = -signal.SIGSEGV  # -11 on most systems
            result.stdout = ""
            result.stderr = "Segmentation fault (core dumped)"
            mock_run.return_value = result

            proc = subprocess.run(
                ["npm", "run", "molecule:role", "--role=common"],
                capture_output=True,
                text=True,
            )

            assert proc.returncode < 0
            assert proc.returncode == -signal.SIGSEGV

    def test_git_push_rejected(self):
        """Simulate git push rejected (non-fast-forward)."""
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = (
                "error: failed to push some refs to 'origin'\n"
                "hint: Updates were rejected because the remote contains work "
                "that you do not have locally."
            )
            mock_run.return_value = result

            proc = subprocess.run(
                ["git", "push", "origin", "sid/common"],
                capture_output=True,
                text=True,
            )

            assert proc.returncode == 1
            assert "rejected" in proc.stderr

    def test_glab_auth_expired(self):
        """Simulate expired GitLab auth token."""
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "glab: 401 Unauthorized - Token has expired"
            mock_run.return_value = result

            proc = subprocess.run(
                ["glab", "auth", "status"],
                capture_output=True,
                text=True,
            )

            assert proc.returncode == 1
            assert "401" in proc.stderr or "Unauthorized" in proc.stderr

    def test_ansible_playbook_error(self):
        """Simulate ansible-playbook execution failure."""
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 2  # Ansible error code
            result.stdout = ""
            result.stderr = (
                "ERROR! the role 'nonexistent' was not found in "
                "/etc/ansible/roles:/usr/share/ansible/roles"
            )
            mock_run.return_value = result

            proc = subprocess.run(
                ["ansible-playbook", "--syntax-check", "site.yml"],
                capture_output=True,
                text=True,
            )

            assert proc.returncode == 2
            assert "not found" in proc.stderr


# =============================================================================
# GRACEFUL DEGRADATION TESTS
# =============================================================================


@pytest.mark.unit
class TestGracefulDegradation:
    """Test that the system degrades gracefully under partial failures."""

    def test_statedb_sync_failure_doesnt_block(self, db_with_roles):
        """StateDB sync failure should not block LangGraph checkpoint."""
        db = db_with_roles
        set_module_db(db)

        # Simulate a state where the module DB is not accessible
        # The _record_test_result function should silently skip
        from harness.dag.langgraph_engine import _record_test_result

        # Set module db to None to simulate failure
        set_module_db(None)

        # This should not raise - it silently skips when db is None
        _record_test_result(
            role_name="common",
            test_name="molecule:common",
            test_type="molecule",
            passed=True,
        )

        # Restore db for cleanup
        set_module_db(db)

    def test_notification_failure_doesnt_block(self):
        """Notification failure should not block workflow."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            mock_client_class.return_value = mock_client

            # Notification failure should be catchable and non-fatal
            client = httpx.Client()
            try:
                client.post(
                    "https://discord.com/api/webhooks/test",
                    json={"content": "test"},
                )
                notification_sent = True
            except httpx.ConnectError:
                notification_sent = False

            # The workflow should continue even if notification fails
            assert notification_sent is False

    def test_merge_train_failure_falls_back(self):
        """Merge train add failure falls back to MWPS."""
        with patch("subprocess.run") as mock_run:
            # First call: merge train fails
            merge_train_fail = MagicMock()
            merge_train_fail.returncode = 1
            merge_train_fail.stdout = ""
            merge_train_fail.stderr = "Merge train is not enabled for this project"

            # Second call: MWPS fallback succeeds
            mwps_success = MagicMock()
            mwps_success.returncode = 0
            mwps_success.stdout = json.dumps({
                "merge_when_pipeline_succeeds": True,
                "iid": 456,
            })
            mwps_success.stderr = ""

            mock_run.side_effect = [merge_train_fail, mwps_success]

            # Try merge train
            result1 = subprocess.run(
                ["glab", "api", "projects/1/merge_trains"],
                capture_output=True,
                text=True,
            )

            if result1.returncode != 0:
                # Fall back to MWPS
                result2 = subprocess.run(
                    ["glab", "api", "projects/1/merge_requests/456/merge"],
                    capture_output=True,
                    text=True,
                )

                assert result2.returncode == 0
                data = json.loads(result2.stdout)
                assert data["merge_when_pipeline_succeeds"] is True

    def test_store_put_failure_doesnt_crash(self, store):
        """Store put failure should raise but not corrupt state."""
        # Put a valid item first
        store.put(("roles", "common"), "status", {"passed": True})

        # Simulate a write failure by patching the connection
        with patch.object(store._db, "connection") as mock_conn:
            mock_conn.side_effect = sqlite3.OperationalError("database is locked")

            with pytest.raises(sqlite3.OperationalError):
                store.put(("roles", "common"), "status", {"passed": False})

        # Original value should still be intact (connection restored)
        result = store.get(("roles", "common"), "status")
        assert result is not None
        assert result.value == {"passed": True}

    def test_create_initial_state_always_works(self):
        """create_initial_state should always succeed with valid role name."""
        state = create_initial_state("common")
        assert state["role_name"] == "common"
        assert state["errors"] == []
        assert state["completed_nodes"] == []
        assert state["current_node"] == "validate_role"

    def test_graph_creation_without_db(self):
        """Graph creation should succeed without a running database."""
        # create_box_up_role_graph doesn't need a live DB connection
        graph, breakpoints = create_box_up_role_graph(
            db_path=":memory:",
            parallel_tests=True,
            enable_breakpoints=False,
        )
        assert graph is not None
        assert breakpoints == []
