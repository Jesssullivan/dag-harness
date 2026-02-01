"""Pytest fixtures for harness tests."""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

from harness.db.state import StateDB
from harness.db.models import (
    Role, RoleDependency, DependencyType, Credential,
    Worktree, WorktreeStatus, TestType
)
from harness.config import HarnessConfig


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_harness.db"


@pytest.fixture
def db(temp_db_path: Path) -> Generator[StateDB, None, None]:
    """Create a fresh database for each test."""
    database = StateDB(temp_db_path)
    yield database
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def in_memory_db() -> Generator[StateDB, None, None]:
    """Create an in-memory database for fast tests."""
    database = StateDB(":memory:")
    yield database


@pytest.fixture
def db_with_roles(db: StateDB) -> StateDB:
    """Database pre-populated with test roles."""
    roles = [
        Role(name="common", wave=0, has_molecule_tests=True),
        Role(name="sql_server_2022", wave=2, has_molecule_tests=True),
        Role(name="sql_management_studio", wave=2, has_molecule_tests=False),
        Role(name="ems_web_app", wave=2, has_molecule_tests=True),
        Role(name="ems_platform_services", wave=3, has_molecule_tests=True),
    ]
    for role in roles:
        db.upsert_role(role)

    # Add dependencies: sql_server_2022 -> common
    common = db.get_role("common")
    sql_server = db.get_role("sql_server_2022")
    sql_mgmt = db.get_role("sql_management_studio")
    ems_web = db.get_role("ems_web_app")
    ems_platform = db.get_role("ems_platform_services")

    # sql_server_2022 depends on common
    db.add_dependency(RoleDependency(
        role_id=sql_server.id,
        depends_on_id=common.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    # sql_management_studio depends on common and sql_server_2022
    db.add_dependency(RoleDependency(
        role_id=sql_mgmt.id,
        depends_on_id=common.id,
        dependency_type=DependencyType.EXPLICIT
    ))
    db.add_dependency(RoleDependency(
        role_id=sql_mgmt.id,
        depends_on_id=sql_server.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    # ems_web_app depends on common
    db.add_dependency(RoleDependency(
        role_id=ems_web.id,
        depends_on_id=common.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    # ems_platform_services depends on ems_web_app
    db.add_dependency(RoleDependency(
        role_id=ems_platform.id,
        depends_on_id=ems_web.id,
        dependency_type=DependencyType.EXPLICIT
    ))

    return db


@pytest.fixture
def db_with_credentials(db_with_roles: StateDB) -> StateDB:
    """Database with roles and credentials."""
    common = db_with_roles.get_role("common")
    sql_server = db_with_roles.get_role("sql_server_2022")

    db_with_roles.add_credential(Credential(
        role_id=common.id,
        entry_name="ansible-self",
        purpose="WinRM authentication",
        is_base58=True
    ))

    db_with_roles.add_credential(Credential(
        role_id=sql_server.id,
        entry_name="dev-sql-sa",
        purpose="SQL Server SA password",
        is_base58=False
    ))

    db_with_roles.add_credential(Credential(
        role_id=sql_server.id,
        entry_name="test-windows-admin",
        purpose="Admin access",
        is_base58=False
    ))

    return db_with_roles


@pytest.fixture
def db_with_worktrees(db_with_roles: StateDB) -> StateDB:
    """Database with roles and worktrees."""
    common = db_with_roles.get_role("common")
    sql_server = db_with_roles.get_role("sql_server_2022")

    db_with_roles.upsert_worktree(Worktree(
        role_id=common.id,
        path="../.worktrees/sid-common",
        branch="sid/common",
        commits_ahead=2,
        commits_behind=0,
        uncommitted_changes=0,
        status=WorktreeStatus.ACTIVE
    ))

    db_with_roles.upsert_worktree(Worktree(
        role_id=sql_server.id,
        path="../.worktrees/sid-sql_server_2022",
        branch="sid/sql_server_2022",
        commits_ahead=0,
        commits_behind=5,
        uncommitted_changes=3,
        status=WorktreeStatus.DIRTY
    ))

    return db_with_roles


@pytest.fixture
def config(temp_db_path: Path) -> HarnessConfig:
    """Create test configuration."""
    return HarnessConfig(
        db_path=str(temp_db_path),
    )


# ============================================================================
# SAMPLE ROLE FIXTURES
# ============================================================================

@pytest.fixture
def sample_role_path() -> Path:
    """Path to the sample role in fixtures."""
    return Path(__file__).parent / "fixtures" / "sample_role"


@pytest.fixture
def temp_role_dir(tmp_path: Path) -> Path:
    """Create a temporary role directory structure."""
    role_dir = tmp_path / "test_role"
    role_dir.mkdir()

    # Create meta/main.yml
    meta_dir = role_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "main.yml").write_text("""---
galaxy_info:
  description: Test role for unit tests
dependencies:
  - common
  - role: other_role
""")

    # Create tasks/main.yml
    tasks_dir = role_dir / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "main.yml").write_text("""---
- name: Test task
  debug:
    msg: "Hello from test role"
""")

    # Create defaults/main.yml
    defaults_dir = role_dir / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "main.yml").write_text("""---
test_variable: "default_value"
""")

    return role_dir


@pytest.fixture
def temp_role_with_credentials(tmp_path: Path) -> Path:
    """Create a temporary role with credential references."""
    role_dir = tmp_path / "cred_role"
    role_dir.mkdir()

    # Create meta
    meta_dir = role_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "main.yml").write_text("""---
galaxy_info:
  description: Role with credentials
dependencies: []
""")

    # Create tasks with credentials
    tasks_dir = role_dir / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "main.yml").write_text("""---
- name: Get password
  set_fact:
    password: "{{ op_read('test-cred', 'password') }}"

- name: Get base58 token
  set_fact:
    token: "{{ op_read('api-token', 'credential') | decode_base58 }}"

- name: Get with lookup
  set_fact:
    secret: "{{ lookup('op', 'another-cred') }}"
""")

    return role_dir


# ============================================================================
# MOCK GITLAB FIXTURES
# ============================================================================

@pytest.fixture
def mock_gitlab_responses() -> dict:
    """Mock responses for GitLab API calls."""
    return {
        "iterations": [
            {
                "id": 12345,
                "title": "Sprint 42",
                "state": "opened",
                "start_date": "2024-01-01",
                "due_date": "2024-01-14",
                "group": {"id": 100}
            }
        ],
        "issue_created": {
            "id": 67890,
            "iid": 123,
            "title": "Box up `common` role",
            "state": "opened",
            "web_url": "https://gitlab.example.com/project/-/issues/123",
            "labels": ["role", "ansible"],
            "assignees": [{"username": "testuser"}],
            "weight": 3
        },
        "mr_created": {
            "id": 11111,
            "iid": 456,
            "source_branch": "sid/common",
            "target_branch": "main",
            "title": "Box up common role",
            "state": "opened",
            "web_url": "https://gitlab.example.com/project/-/merge_requests/456",
            "merge_status": "can_be_merged",
            "squash_on_merge": True,
            "force_remove_source_branch": True
        },
        "merge_train": [
            {
                "id": 1,
                "merge_request": {"iid": 456},
                "pipeline": {"id": 999, "status": "running"},
                "status": "merging"
            }
        ]
    }


@pytest.fixture
def mock_glab_run():
    """Mock subprocess.run for glab commands."""
    def _mock_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        cmd_str = " ".join(cmd)

        if "issue create" in cmd_str:
            result.stdout = "https://gitlab.example.com/project/-/issues/123"
        elif "mr create" in cmd_str:
            result.stdout = "https://gitlab.example.com/project/-/merge_requests/456"
        elif "auth status" in cmd_str:
            result.stdout = "Token: glpat-testtoken123"

        return result

    return _mock_run


@pytest.fixture
def mock_gitlab_api(mock_gitlab_responses):
    """Fixture that mocks GitLab API calls via glab."""
    with patch('subprocess.run') as mock_run:
        def run_side_effect(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""

            cmd_str = " ".join(cmd)

            if "glab api" in cmd_str:
                if "iterations" in cmd_str:
                    result.stdout = json.dumps(mock_gitlab_responses["iterations"])
                elif "/issues/" in cmd_str and "POST" not in cmd_str:
                    result.stdout = json.dumps(mock_gitlab_responses["issue_created"])
                elif "/merge_requests/" in cmd_str:
                    result.stdout = json.dumps(mock_gitlab_responses["mr_created"])
                elif "merge_trains" in cmd_str:
                    result.stdout = json.dumps(mock_gitlab_responses["merge_train"])
                else:
                    result.stdout = "[]"
            elif "issue create" in cmd_str:
                result.stdout = "https://gitlab.example.com/project/-/issues/123"
            elif "mr create" in cmd_str:
                result.stdout = "https://gitlab.example.com/project/-/merge_requests/456"
            elif "auth status" in cmd_str:
                result.stdout = "Token: glpat-testtoken123"
            else:
                result.stdout = ""

            return result

        mock_run.side_effect = run_side_effect
        yield mock_run


# ============================================================================
# MOCK NOTIFICATION FIXTURES
# ============================================================================

@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for notification tests."""
    with patch('httpx.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        yield mock_client


@pytest.fixture
def mock_async_httpx_client():
    """Mock async httpx client for notification tests."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        # Make post return a coroutine
        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mock_client.aclose = MagicMock(return_value=None)

        # Make aclose async
        async def mock_aclose():
            pass
        mock_client.aclose = mock_aclose

        mock_client_class.return_value = mock_client

        yield mock_client


@pytest.fixture
def notification_config():
    """Configuration for notification service tests."""
    from harness.hotl.notifications import NotificationConfig
    return NotificationConfig(
        discord_webhook_url="https://discord.com/api/webhooks/test/hook",
        email_smtp_host="smtp.example.com",
        email_smtp_port=587,
        email_from="harness@example.com",
        email_to="alerts@example.com",
    )


# ============================================================================
# WORKFLOW FIXTURES
# ============================================================================

@pytest.fixture
def workflow_db(db_with_roles: StateDB) -> StateDB:
    """Database with a workflow definition."""
    nodes = [
        {"id": "start", "type": "start"},
        {"id": "analyze", "type": "task"},
        {"id": "test", "type": "task"},
        {"id": "end", "type": "end"}
    ]
    edges = [
        {"from": "start", "to": "analyze"},
        {"from": "analyze", "to": "test"},
        {"from": "test", "to": "end"}
    ]

    db_with_roles.create_workflow_definition(
        name="box-up-role",
        description="Standard box-up workflow",
        nodes=nodes,
        edges=edges
    )

    return db_with_roles


# ============================================================================
# HOTL FIXTURES
# ============================================================================

@pytest.fixture
def hotl_state():
    """Create a sample HOTL state for testing."""
    from harness.hotl.state import HOTLState, HOTLPhase, create_initial_state
    return create_initial_state(
        max_iterations=10,
        notification_interval=60,
        config={"test_mode": True}
    )


@pytest.fixture
def mock_langgraph_checkpointer(tmp_path):
    """Mock LangGraph checkpointer for testing."""
    with patch('langgraph.checkpoint.sqlite.SqliteSaver') as mock_saver:
        mock_instance = MagicMock()
        mock_saver.from_conn_string.return_value = mock_instance
        yield mock_instance


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "pbt: mark test as property-based test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
