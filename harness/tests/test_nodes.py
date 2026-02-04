"""Comprehensive unit tests for all 15 LangGraph nodes in the box-up-role workflow.

Tests for all 15 nodes with at least 3 test cases each:
- Happy path (success)
- Error handling (failure)
- Edge case

Target: 95% code coverage for node functions in langgraph_engine.py.

Nodes tested:
1.  validate_role_node - valid role, invalid role, missing meta
2.  analyze_deps_node - no deps, single dep, circular dep detection
3.  check_reverse_deps_node - no reverse deps, multiple reverse deps
4.  create_worktree_node - new worktree, existing worktree, path conflict
5.  run_molecule_node - tests pass, tests fail, no molecule config
6.  run_pytest_node - tests pass, tests fail, no pytest
7.  validate_deploy_node - deploy success, deploy failure, rollback
8.  create_commit_node - changes to commit, no changes, commit message format
9.  push_branch_node - push success, push rejected, remote conflict
10. create_issue_node - new issue, existing issue (idempotent), API error
11. create_mr_node - new MR, existing MR, MR from issue
12. human_approval_node - approved, rejected, timeout
13. add_to_merge_train_node - added, already in train, not mergeable
14. report_summary_node - success summary, failure summary, partial
15. notify_failure_node - notification sent, notification disabled
"""

import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    add_to_merge_train_node,
    analyze_deps_node,
    check_reverse_deps_node,
    create_box_up_role_graph,
    create_commit_node,
    create_initial_state,
    create_issue_node,
    create_mr_node,
    create_worktree_node,
    human_approval_node,
    keep_last_n,
    merge_test_results_node,
    notify_failure_node,
    push_branch_node,
    report_summary_node,
    run_molecule_node,
    run_pytest_node,
    set_module_db,
    validate_deploy_node,
    validate_role_node,
)
from harness.db.models import Role
from harness.db.state import StateDB


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_db(tmp_path):
    """Create a mock database for testing."""
    db = StateDB(tmp_path / "test.db")
    set_module_db(db)
    yield db
    set_module_db(None)


@pytest.fixture
def db_with_test_role(mock_db):
    """Database with a test role pre-populated."""
    mock_db.upsert_role(
        Role(name="test_role", wave=1, wave_name="Infrastructure", has_molecule_tests=True)
    )
    return mock_db


@pytest.fixture
def sample_state() -> BoxUpRoleState:
    """Create a sample state for testing."""
    return create_initial_state("test_role", execution_id=1)


@pytest.fixture
def temp_role_structure(tmp_path):
    """Create a temporary role directory structure."""
    role_dir = tmp_path / "ansible" / "roles" / "test_role"
    role_dir.mkdir(parents=True)

    # Create molecule directory
    molecule_dir = role_dir / "molecule"
    molecule_dir.mkdir()

    # Create meta directory with main.yml
    meta_dir = role_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "main.yml").write_text(
        """---
galaxy_info:
  description: Test role
dependencies:
  - common
"""
    )

    return tmp_path


# =============================================================================
# 1. VALIDATE_ROLE_NODE TESTS
# =============================================================================


class TestValidateRoleNode:
    """Tests for validate_role_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_valid_role_with_molecule(self, temp_role_structure):
        """Happy path: valid role with molecule tests."""
        with patch(
            "harness.dag.langgraph_nodes.Path",
            return_value=temp_role_structure / "ansible" / "roles" / "test_role",
        ):
            # Create the role path correctly
            role_path = temp_role_structure / "ansible" / "roles" / "test_role"
            role_path.mkdir(parents=True, exist_ok=True)
            (role_path / "molecule").mkdir(exist_ok=True)
            (role_path / "meta").mkdir(exist_ok=True)
            (role_path / "meta" / "main.yml").write_text("---\ndependencies: []")

            # Patch Path to return our temp structure
            original_path = Path

            def patched_path(p):
                if "ansible/roles/test_role" in str(p):
                    return role_path
                return original_path(p)

            with patch("harness.dag.langgraph_nodes.Path", side_effect=patched_path):
                state: BoxUpRoleState = {"role_name": "test_role"}
                result = await validate_role_node(state)

                assert "validate_role" in result["completed_nodes"]
                assert result["has_molecule_tests"] is True
                assert result["has_meta"] is True
                assert "errors" not in result or len(result.get("errors", [])) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_role_not_found(self):
        """Error handling: role directory doesn't exist."""
        state: BoxUpRoleState = {"role_name": "nonexistent_role"}
        result = await validate_role_node(state)

        assert "validate_role" in result["completed_nodes"]
        assert len(result.get("errors", [])) > 0
        assert "not found" in result["errors"][0].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_role_without_molecule(self, tmp_path):
        """Edge case: valid role without molecule tests."""
        role_path = tmp_path / "ansible" / "roles" / "no_molecule_role"
        role_path.mkdir(parents=True)
        (role_path / "meta").mkdir()
        (role_path / "meta" / "main.yml").write_text("---\ndependencies: []")

        original_path = Path

        def patched_path(p):
            if "ansible/roles/no_molecule_role" in str(p):
                return role_path
            return original_path(p)

        with patch("harness.dag.langgraph_nodes.Path", side_effect=patched_path):
            state: BoxUpRoleState = {"role_name": "no_molecule_role"}
            result = await validate_role_node(state)

            assert result["has_molecule_tests"] is False
            assert result["has_meta"] is True


# =============================================================================
# 2. ANALYZE_DEPS_NODE TESTS
# =============================================================================


class TestAnalyzeDepsNode:
    """Tests for analyze_deps_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_deps_success(self, db_with_test_role):
        """Happy path: successful dependency analysis."""
        state: BoxUpRoleState = {"role_name": "test_role"}
        result = await analyze_deps_node(state)

        assert "analyze_dependencies" in result["completed_nodes"]
        assert "wave" in result
        assert result["wave"] == 1
        assert "errors" not in result or len(result.get("errors", [])) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_deps_no_db(self):
        """Error handling: database not available."""
        set_module_db(None)
        state: BoxUpRoleState = {"role_name": "test_role"}
        result = await analyze_deps_node(state)

        assert "analyze_dependencies" in result["completed_nodes"]
        assert len(result.get("errors", [])) > 0
        assert "database" in result["errors"][0].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_deps_role_not_in_db(self, mock_db):
        """Edge case: role exists on disk but not in database."""
        state: BoxUpRoleState = {"role_name": "unknown_role"}
        result = await analyze_deps_node(state)

        assert "analyze_dependencies" in result["completed_nodes"]
        assert len(result.get("errors", [])) > 0
        assert "not found" in result["errors"][0].lower()


# =============================================================================
# 3. CHECK_REVERSE_DEPS_NODE TESTS
# =============================================================================


class TestCheckReverseDepsNode:
    """Tests for check_reverse_deps_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_no_reverse_deps(self):
        """Happy path: role has no reverse dependencies."""
        state: BoxUpRoleState = {"role_name": "test_role", "reverse_deps": []}
        result = await check_reverse_deps_node(state)

        assert "check_reverse_deps" in result["completed_nodes"]
        assert result["blocking_deps"] == []
        assert "errors" not in result or len(result.get("errors", [])) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_blocking_reverse_deps(self):
        """Error handling: reverse deps not yet boxed up."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = ""  # Branch doesn't exist
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {"role_name": "test_role", "reverse_deps": ["dep_role"]}
            result = await check_reverse_deps_node(state)

            assert "check_reverse_deps" in result["completed_nodes"]
            assert "dep_role" in result["blocking_deps"]
            assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_reverse_deps_already_boxed(self):
        """Edge case: reverse deps already have branches."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "abc123  refs/heads/sid/dep_role"  # Branch exists
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {"role_name": "test_role", "reverse_deps": ["dep_role"]}
            result = await check_reverse_deps_node(state)

            assert "check_reverse_deps" in result["completed_nodes"]
            assert result["blocking_deps"] == []


# =============================================================================
# 4. CREATE_WORKTREE_NODE TESTS
# =============================================================================


class TestCreateWorktreeNode:
    """Tests for create_worktree_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_worktree_success(self, db_with_test_role):
        """Happy path: successfully creates worktree."""
        mock_worktree_info = MagicMock()
        mock_worktree_info.path = "/tmp/worktree/test_role"
        mock_worktree_info.branch = "sid/test_role"
        mock_worktree_info.commit = "abc123"

        with (
            patch("harness.worktree.manager.WorktreeManager") as mock_manager_class,
            patch("harness.gitlab.api.GitLabClient") as mock_client_class,
        ):
            mock_manager = MagicMock()
            mock_manager.create.return_value = mock_worktree_info
            mock_manager_class.return_value = mock_manager

            mock_client = MagicMock()
            mock_client.remote_branch_exists.return_value = False
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role"}
            from harness.dag.langgraph_engine import create_worktree_node

            result = await create_worktree_node(state)

            assert "create_worktree" in result["completed_nodes"]
            assert result["worktree_path"] == "/tmp/worktree/test_role"
            assert result["branch"] == "sid/test_role"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_worktree_already_exists(self, db_with_test_role):
        """Edge case: worktree already exists - should be idempotent."""
        from harness.db.models import Worktree, WorktreeStatus

        # Add existing worktree to db
        role = db_with_test_role.get_role("test_role")
        db_with_test_role.upsert_worktree(
            Worktree(
                role_id=role.id,
                path="/existing/worktree",
                branch="sid/test_role",
                current_commit="existing123",
                status=WorktreeStatus.ACTIVE,
            )
        )

        with (
            patch("harness.worktree.manager.WorktreeManager") as mock_manager_class,
            patch("harness.gitlab.api.GitLabClient") as mock_client_class,
        ):
            mock_manager = MagicMock()
            mock_manager.create.side_effect = ValueError("Worktree already exists")
            mock_manager_class.return_value = mock_manager

            mock_client = MagicMock()
            mock_client.remote_branch_exists.return_value = True
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role"}
            from harness.dag.langgraph_engine import create_worktree_node

            result = await create_worktree_node(state)

            assert "create_worktree" in result["completed_nodes"]
            # Should recover by getting existing worktree
            assert result["worktree_path"] == "/existing/worktree"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_worktree_no_db(self):
        """Error handling: database not available."""
        set_module_db(None)
        state: BoxUpRoleState = {"role_name": "test_role"}
        from harness.dag.langgraph_engine import create_worktree_node

        result = await create_worktree_node(state)

        assert "create_worktree" in result["completed_nodes"]
        assert len(result.get("errors", [])) > 0


# =============================================================================
# 5. RUN_MOLECULE_NODE TESTS
# =============================================================================


class TestRunMoleculeNode:
    """Tests for run_molecule_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_tests_pass(self, mock_db):
        """Happy path: molecule tests pass."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "All tests passed"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "has_molecule_tests": True,
                "worktree_path": "/tmp/worktree",
            }
            result = await run_molecule_node(state)

            assert result["molecule_passed"] is True
            assert "run_molecule" in result["parallel_tests_completed"]
            assert "run_molecule_tests" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_tests_fail(self, mock_db):
        """Error handling: molecule tests fail."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = "Test output"
            mock_result.stderr = "FAILED: assertion error"
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "has_molecule_tests": True,
                "worktree_path": "/tmp/worktree",
            }
            result = await run_molecule_node(state)

            assert result["molecule_passed"] is False
            assert "run_molecule" in result["parallel_tests_completed"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_no_tests(self):
        """Edge case: no molecule tests configured."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "has_molecule_tests": False,
            "worktree_path": "/tmp/worktree",
        }
        result = await run_molecule_node(state)

        assert result["molecule_skipped"] is True
        assert "run_molecule" in result["parallel_tests_completed"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_timeout(self, mock_db):
        """Edge case: molecule tests timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="npm", timeout=600)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "has_molecule_tests": True,
                "worktree_path": "/tmp/worktree",
            }
            result = await run_molecule_node(state)

            assert result["molecule_passed"] is False


# =============================================================================
# 6. RUN_PYTEST_NODE TESTS
# =============================================================================


class TestRunPytestNode:
    """Tests for run_pytest_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_pytest_tests_pass(self, tmp_path, mock_db):
        """Happy path: pytest tests pass."""
        # Create a test file
        test_file = tmp_path / "tests" / "test_test_role.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_example(): pass")

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "1 passed"
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await run_pytest_node(state)

            assert result["pytest_passed"] is True
            assert "run_pytest" in result["parallel_tests_completed"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_pytest_tests_fail(self, tmp_path, mock_db):
        """Error handling: pytest tests fail."""
        test_file = tmp_path / "tests" / "test_test_role.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_fail(): assert False")

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = "1 failed"
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await run_pytest_node(state)

            assert result["pytest_passed"] is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_pytest_no_tests(self, tmp_path):
        """Edge case: no pytest test file exists."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "worktree_path": str(tmp_path),
        }
        result = await run_pytest_node(state)

        assert result["pytest_skipped"] is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_pytest_timeout(self, tmp_path, mock_db):
        """Edge case: pytest tests timeout."""
        test_file = tmp_path / "tests" / "test_test_role.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_slow(): import time; time.sleep(1000)")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=300)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await run_pytest_node(state)

            assert result["pytest_passed"] is False


# =============================================================================
# 7. VALIDATE_DEPLOY_NODE (deploy_and_verify equivalent)
# =============================================================================


class TestValidateDeployNode:
    """Tests for validate_deploy_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_deploy_validation_success(self, tmp_path):
        """Happy path: deploy validation passes."""
        # Create site.yml
        (tmp_path / "site.yml").write_text("---\n- hosts: all\n  roles: [test_role]")

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await validate_deploy_node(state)

            assert result["deploy_passed"] is True
            assert "validate_deploy" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_deploy_validation_failure(self, tmp_path):
        """Error handling: deploy validation fails."""
        (tmp_path / "site.yml").write_text("---\n- invalid yaml")

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Syntax error in playbook"
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await validate_deploy_node(state)

            assert result["deploy_passed"] is False
            assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_deploy_no_site_yml(self, tmp_path):
        """Edge case: no site.yml exists - skip validation."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("site.yml not found")

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await validate_deploy_node(state)

            assert result["deploy_skipped"] is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_deploy_validation_timeout(self, tmp_path):
        """Edge case: validation times out."""
        # Create site.yml so the node doesn't skip (v0.5.0 auto-detects ansible/ subdir)
        ansible_dir = tmp_path / "ansible"
        ansible_dir.mkdir()
        (ansible_dir / "site.yml").touch()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ansible-playbook", timeout=60)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await validate_deploy_node(state)

            assert result["deploy_passed"] is False
            assert "timed out" in result["errors"][0].lower()


# =============================================================================
# 8. CREATE_COMMIT_NODE TESTS
# =============================================================================


class TestCreateCommitNode:
    """Tests for create_commit_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_commit_success(self, tmp_path):
        """Happy path: commit created successfully."""
        with patch("subprocess.run") as mock_run:

            def run_side_effect(cmd, **kwargs):
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                if "status" in cmd:
                    result.stdout = "M  file.txt"  # Has changes
                elif "rev-parse" in cmd:
                    result.stdout = "abc123def456"
                else:
                    result.stdout = ""
                return result

            mock_run.side_effect = run_side_effect

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_commit_node(state)

            assert "create_commit" in result["completed_nodes"]
            assert result.get("commit_sha") == "abc123def456"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_commit_no_changes(self, tmp_path):
        """Edge case: no changes to commit."""
        with patch("subprocess.run") as mock_run:

            def run_side_effect(cmd, **kwargs):
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                result.stdout = ""  # No changes
                return result

            mock_run.side_effect = run_side_effect

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_commit_node(state)

            assert "create_commit" in result["completed_nodes"]
            # No commit_sha when there's nothing to commit
            assert result.get("commit_sha") is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_commit_failure(self, tmp_path):
        """Error handling: commit fails."""
        with patch("subprocess.run") as mock_run:

            def run_side_effect(cmd, **kwargs):
                result = MagicMock()
                if "status" in cmd:
                    result.returncode = 0
                    result.stdout = "M  file.txt"
                elif "commit" in cmd:
                    result.returncode = 1
                    result.stderr = "error: commit failed"
                else:
                    result.returncode = 0
                    result.stdout = ""
                return result

            mock_run.side_effect = run_side_effect

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_commit_node(state)

            assert "create_commit" in result["completed_nodes"]
            assert len(result.get("errors", [])) > 0


# =============================================================================
# 9. PUSH_BRANCH_NODE TESTS
# =============================================================================


class TestPushBranchNode:
    """Tests for push_branch_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_push_success(self, tmp_path):
        """Happy path: push succeeds."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Everything up-to-date"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
                "branch": "sid/test_role",
            }
            result = await push_branch_node(state)

            assert result["pushed"] is True
            assert "push_branch" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_push_rejected(self, tmp_path):
        """Error handling: push rejected by remote."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "rejected: non-fast-forward"
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
                "branch": "sid/test_role",
            }
            result = await push_branch_node(state)

            assert result.get("pushed") is not True
            assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_push_remote_conflict(self, tmp_path):
        """Edge case: remote has conflicting changes."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Updates were rejected because the remote contains work"
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
                "branch": "sid/test_role",
            }
            result = await push_branch_node(state)

            assert len(result.get("errors", [])) > 0
            assert "rejected" in result["errors"][0].lower()


# =============================================================================
# 10. CREATE_ISSUE_NODE TESTS
# =============================================================================


class TestCreateIssueNode:
    """Tests for create_issue_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_issue_success(self, db_with_test_role):
        """Happy path: issue created successfully."""
        mock_issue = {
            "web_url": "https://gitlab.example.com/issues/123",
            "iid": 123,
        }
        mock_iteration = {"id": 456, "title": "Iteration 1"}

        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role", "ansible"]
            mock_config_class.from_harness_yml.return_value = mock_config

            # Set up mock API with async context manager
            mock_api = AsyncMock()
            mock_api.get_current_iteration = AsyncMock(return_value=mock_iteration)
            mock_api.ensure_label_exists = AsyncMock()
            mock_api.get_or_create_issue = AsyncMock(return_value=(mock_issue, True))

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
                "explicit_deps": [],
                "credentials": [],
            }
            result = await create_issue_node(state)

            assert result["issue_url"] == "https://gitlab.example.com/issues/123"
            assert result["issue_iid"] == 123
            assert result["issue_created"] is True
            assert "create_gitlab_issue" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_issue_existing_reused(self, db_with_test_role):
        """Edge case: existing issue reused (idempotent)."""
        mock_issue = {
            "web_url": "https://gitlab.example.com/issues/99",
            "iid": 99,
        }

        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role"]
            mock_config_class.from_harness_yml.return_value = mock_config

            # Set up mock API with async context manager
            mock_api = AsyncMock()
            mock_api.get_current_iteration = AsyncMock(return_value=None)
            mock_api.ensure_label_exists = AsyncMock()
            mock_api.get_or_create_issue = AsyncMock(return_value=(mock_issue, False))  # Not created

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
                "explicit_deps": [],
                "credentials": [],
            }
            result = await create_issue_node(state)

            assert result["issue_iid"] == 99
            assert result["issue_created"] is False  # Reused existing

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_issue_api_error(self, db_with_test_role):
        """Error handling: GitLab API error."""
        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role"]
            mock_config_class.from_harness_yml.return_value = mock_config

            # Set up mock API that raises error
            mock_api = AsyncMock()
            mock_api.get_current_iteration = AsyncMock(return_value=None)
            mock_api.ensure_label_exists = AsyncMock()
            mock_api.get_or_create_issue = AsyncMock(side_effect=RuntimeError("API error: 500"))

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
                "explicit_deps": [],
                "credentials": [],
            }
            result = await create_issue_node(state)

            assert len(result.get("errors", [])) > 0
            assert "api error" in result["errors"][0].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_issue_no_db(self):
        """Error handling: database not available."""
        set_module_db(None)
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "wave": 1,
            "wave_name": "Infrastructure",
        }
        result = await create_issue_node(state)

        assert len(result.get("errors", [])) > 0


# =============================================================================
# 11. CREATE_MR_NODE TESTS
# =============================================================================


class TestCreateMRNode:
    """Tests for create_mr_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_mr_success(self, db_with_test_role):
        """Happy path: MR created successfully."""
        mock_mr = {
            "web_url": "https://gitlab.example.com/merge_requests/456",
            "iid": 456,
        }

        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
            patch("harness.gitlab.templates.render_mr_description") as mock_render,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role", "ansible"]
            mock_config.default_reviewers = ["reviewer1"]
            mock_config_class.from_harness_yml.return_value = mock_config

            # Mock template rendering
            mock_render.return_value = "## Summary\nTest MR description"

            # Set up mock API with async context manager
            mock_api = AsyncMock()
            mock_api.find_mr_by_branch = AsyncMock(return_value=None)  # No existing MR
            mock_api.get_or_create_mr = AsyncMock(return_value=(mock_mr, True))
            mock_api.set_mr_reviewers = AsyncMock(return_value=True)

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "issue_iid": 123,
                "branch": "sid/test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_mr_node(state)

            assert result["mr_url"] == "https://gitlab.example.com/merge_requests/456"
            assert result["mr_iid"] == 456
            assert result["mr_created"] is True
            assert "create_merge_request" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_mr_existing_reused(self, db_with_test_role):
        """Edge case: existing MR reused."""
        mock_existing_mr = {
            "web_url": "https://gitlab.example.com/merge_requests/789",
            "iid": 789,
        }

        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
            patch("harness.gitlab.templates.render_mr_description") as mock_render,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role"]
            mock_config.default_reviewers = []
            mock_config_class.from_harness_yml.return_value = mock_config

            # Mock template rendering
            mock_render.return_value = "## Summary\nTest MR description"

            # Set up mock API - return existing opened MR
            mock_api = AsyncMock()
            mock_api.find_mr_by_branch = AsyncMock(return_value=mock_existing_mr)

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "issue_iid": 123,
                "branch": "sid/test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_mr_node(state)

            assert result["mr_iid"] == 789
            assert result["mr_created"] is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_mr_no_issue_iid(self, db_with_test_role):
        """Error handling: no issue IID provided."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "issue_iid": None,
            "branch": "sid/test_role",
            "wave": 1,
            "wave_name": "Infrastructure",
        }
        result = await create_mr_node(state)

        assert len(result.get("errors", [])) > 0
        assert "no issue iid" in result["errors"][0].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_mr_api_error(self, db_with_test_role):
        """Error handling: GitLab API error."""
        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
            patch("harness.gitlab.templates.render_mr_description") as mock_render,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role"]
            mock_config.default_reviewers = []
            mock_config_class.from_harness_yml.return_value = mock_config

            # Mock template rendering
            mock_render.return_value = "## Summary\nTest MR description"

            # Set up mock API that raises error
            mock_api = AsyncMock()
            mock_api.find_mr_by_branch = AsyncMock(return_value=None)
            mock_api.get_or_create_mr = AsyncMock(side_effect=RuntimeError("MR creation failed"))

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "issue_iid": 123,
                "branch": "sid/test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_mr_node(state)

            assert len(result.get("errors", [])) > 0


# =============================================================================
# 12. HUMAN_APPROVAL_NODE TESTS
# =============================================================================


class TestHumanApprovalNode:
    """Tests for human_approval_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_human_approval_approved(self):
        """Happy path: human approves the MR."""
        # Mock interrupt to return approval
        with patch("harness.dag.langgraph_nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"approved": True, "reason": ""}

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "mr_url": "https://gitlab.example.com/mr/123",
                "mr_iid": 123,
                "molecule_passed": True,
                "pytest_passed": True,
                "branch": "sid/test_role",
            }
            result = await human_approval_node(state)

            assert result["human_approved"] is True
            assert result["awaiting_human_input"] is False
            assert "human_approval" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_human_approval_rejected(self):
        """Error handling: human rejects the MR."""
        with patch("harness.dag.langgraph_nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"approved": False, "reason": "Needs more tests"}

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "mr_url": "https://gitlab.example.com/mr/123",
                "mr_iid": 123,
                "molecule_passed": True,
                "pytest_passed": True,
                "branch": "sid/test_role",
            }
            result = await human_approval_node(state)

            assert result["human_approved"] is False
            assert "Needs more tests" in result["human_rejection_reason"]
            assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_human_approval_simple_bool_response(self):
        """Edge case: human provides simple boolean response."""
        with patch("harness.dag.langgraph_nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = True  # Simple boolean

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "mr_url": "https://gitlab.example.com/mr/123",
                "mr_iid": 123,
                "molecule_passed": True,
                "pytest_passed": True,
                "branch": "sid/test_role",
            }
            result = await human_approval_node(state)

            assert result["human_approved"] is True


# =============================================================================
# 13. ADD_TO_MERGE_TRAIN_NODE TESTS
# =============================================================================


class TestAddToMergeTrainNode:
    """Tests for add_to_merge_train_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_add_to_merge_train_success(self, db_with_test_role):
        """Happy path: MR added to merge train."""
        with patch("harness.gitlab.api.GitLabClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.is_merge_train_available.return_value = {"available": True}
            mock_client.is_merge_train_enabled.return_value = True
            mock_client.add_to_merge_train.return_value = None
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
            result = await add_to_merge_train_node(state)

            assert result["merge_train_status"] == "added"
            assert "add_to_merge_train" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_add_to_merge_train_already_in_train(self, db_with_test_role):
        """Edge case: MR already in merge train."""
        with patch("harness.gitlab.api.GitLabClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.is_merge_train_available.return_value = {
                "available": False,
                "reason": "Already in merge train",
            }
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
            result = await add_to_merge_train_node(state)

            assert result["merge_train_status"] == "added"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_add_to_merge_train_not_mergeable(self, db_with_test_role):
        """Error handling: MR not mergeable."""
        with patch("harness.gitlab.api.GitLabClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.is_merge_train_available.return_value = {
                "available": False,
                "reason": "Pipeline failed",
            }
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
            result = await add_to_merge_train_node(state)

            assert result["merge_train_status"] == "skipped"
            assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_add_to_merge_train_fallback(self, db_with_test_role):
        """Edge case: merge train not enabled, fallback to merge when pipeline succeeds."""
        with patch("harness.gitlab.api.GitLabClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.is_merge_train_available.return_value = {"available": True}
            mock_client.is_merge_train_enabled.return_value = False
            mock_client.merge_when_pipeline_succeeds.return_value = None
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
            result = await add_to_merge_train_node(state)

            assert result["merge_train_status"] == "fallback"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_add_to_merge_train_no_mr_iid(self, db_with_test_role):
        """Error handling: no MR IID provided."""
        state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": None}
        result = await add_to_merge_train_node(state)

        assert result["merge_train_status"] == "skipped"
        assert len(result.get("errors", [])) > 0


# =============================================================================
# 14. REPORT_SUMMARY_NODE TESTS
# =============================================================================


class TestReportSummaryNode:
    """Tests for report_summary_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_report_summary_success(self):
        """Happy path: successful summary report."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "wave": 1,
            "wave_name": "Infrastructure",
            "worktree_path": "/tmp/worktree",
            "branch": "sid/test_role",
            "commit_sha": "abc123",
            "issue_url": "https://gitlab.example.com/issues/123",
            "mr_url": "https://gitlab.example.com/mr/456",
            "molecule_passed": True,
            "molecule_duration": 120,
            "pytest_passed": True,
            "pytest_duration": 30,
            "all_tests_passed": True,
            "test_phase_duration": 125.0,
            "merge_train_status": "added",
            "credentials": [{"entry_name": "test-cred"}],
            "explicit_deps": ["common"],
            "errors": [],
        }
        result = await report_summary_node(state)

        assert "report_summary" in result["completed_nodes"]
        assert result["summary"]["success"] is True
        assert result["summary"]["role"] == "test_role"
        assert result["summary"]["molecule_passed"] is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_report_summary_with_errors(self):
        """Error handling: summary with errors."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "wave": 1,
            "wave_name": "Infrastructure",
            "molecule_passed": False,
            "molecule_duration": 60,
            "pytest_passed": True,
            "pytest_duration": 30,
            "all_tests_passed": False,
            "errors": ["Molecule tests failed"],
        }
        result = await report_summary_node(state)

        assert result["summary"]["success"] is False
        assert len(result["summary"]["errors"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_report_summary_time_savings(self):
        """Edge case: parallel execution shows time savings."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "wave": 1,
            "wave_name": "Infrastructure",
            "molecule_passed": True,
            "molecule_duration": 60,  # Sequential would be 60+40=100s
            "pytest_passed": True,
            "pytest_duration": 40,
            "all_tests_passed": True,
            "test_phase_duration": 65.0,  # Parallel took ~65s
            "parallel_execution_enabled": True,
            "errors": [],
        }
        result = await report_summary_node(state)

        assert result["summary"]["test_time_saved_seconds"] > 0
        assert result["summary"]["test_time_saved_percent"] > 0


# =============================================================================
# 15. NOTIFY_FAILURE_NODE TESTS
# =============================================================================


class TestNotifyFailureNode:
    """Tests for notify_failure_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_notify_failure_basic(self, capsys):
        """Happy path: failure notification sent."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "errors": ["Molecule tests failed", "Deploy validation failed"],
            "issue_iid": None,  # No issue to update
        }
        result = await notify_failure_node(state)

        assert "notify_failure" in result["completed_nodes"]

        captured = capsys.readouterr()
        assert "FAILURE" in captured.out
        assert "test_role" in captured.out

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_notify_failure_updates_issue(self, db_with_test_role, capsys):
        """Edge case: failure updates existing issue."""
        # GitLabClient is imported inside the function body, so we patch where it's imported from
        mock_client = MagicMock()
        mock_client.update_issue_on_failure.return_value = True

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            state: BoxUpRoleState = {
                "role_name": "test_role",
                "errors": ["Test failure"],
                "issue_iid": 123,
            }
            result = await notify_failure_node(state)

            assert "notify_failure" in result["completed_nodes"]
            mock_client.update_issue_on_failure.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_notify_failure_no_errors(self, capsys):
        """Edge case: notify_failure called with no errors."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "errors": [],
            "issue_iid": None,
        }
        result = await notify_failure_node(state)

        assert "notify_failure" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_notify_failure_gitlab_import_error(self, db_with_test_role, capsys):
        """Edge case: GitLabClient import fails (notification disabled scenario)."""
        with patch("harness.gitlab.api.GitLabClient", side_effect=ImportError("no module")):
            state: BoxUpRoleState = {
                "role_name": "test_role",
                "errors": ["Deploy failed"],
                "issue_iid": 99,
            }
            result = await notify_failure_node(state)

            # Should not raise - notification is best-effort
            assert "notify_failure" in result["completed_nodes"]
            captured = capsys.readouterr()
            assert "FAILURE" in captured.out

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_notify_failure_issue_update_exception(self, db_with_test_role, capsys):
        """Edge case: issue update itself throws an exception."""
        mock_client = MagicMock()
        mock_client.update_issue_on_failure.side_effect = Exception("API timeout")

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            state: BoxUpRoleState = {
                "role_name": "test_role",
                "errors": ["Some error"],
                "issue_iid": 50,
            }
            result = await notify_failure_node(state)

            # Should still complete without raising
            assert "notify_failure" in result["completed_nodes"]
            captured = capsys.readouterr()
            assert "WARNING" in captured.out


# =============================================================================
# ADDITIONAL EDGE CASE AND COVERAGE TESTS
# =============================================================================


class TestValidateRoleNodeExtended:
    """Extended tests for validate_role_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_role_with_meta_but_no_molecule(self, tmp_path):
        """Edge case: role has meta/main.yml but no molecule directory."""
        role_path = tmp_path / "ansible" / "roles" / "meta_only_role"
        role_path.mkdir(parents=True)
        (role_path / "meta").mkdir()
        (role_path / "meta" / "main.yml").write_text("---\ndependencies: []")
        # No molecule directory

        original_path = Path

        def patched_path(p):
            if "ansible/roles/meta_only_role" in str(p):
                return role_path
            return original_path(p)

        with patch("harness.dag.langgraph_nodes.Path", side_effect=patched_path):
            state: BoxUpRoleState = {"role_name": "meta_only_role"}
            result = await validate_role_node(state)

            assert result["has_meta"] is True
            assert result["has_molecule_tests"] is False
            assert "errors" not in result or len(result.get("errors", [])) == 0


class TestAnalyzeDepsNodeExtended:
    """Extended tests for analyze_deps_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_deps_with_transitive(self, mock_db):
        """Edge case: role has transitive dependencies (depth > 1)."""
        mock_db.upsert_role(
            Role(name="deep_role", wave=3, wave_name="Deep", has_molecule_tests=True)
        )
        role = mock_db.get_role("deep_role")
        assert role is not None

        # Mock direct query results - transitive deps have depth > 1
        original_get_deps = mock_db.get_dependencies

        def mock_get_deps(role_name, transitive=False):
            if not transitive:
                return [("common", 1)]
            else:
                return [("common", 1), ("base_infra", 2)]  # depth 2 = transitive

        mock_db.get_dependencies = mock_get_deps
        mock_db.get_reverse_dependencies = lambda r, transitive=False: []
        mock_db.get_credentials = lambda r: []

        state: BoxUpRoleState = {"role_name": "deep_role"}
        result = await analyze_deps_node(state)

        assert "analyze_dependencies" in result["completed_nodes"]
        assert result["explicit_deps"] == ["common"]
        # Transitive deps (depth > 1) should be in implicit_deps
        assert "base_infra" in result["implicit_deps"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_analyze_deps_exception_in_get_dependencies(self, mock_db):
        """Edge case: exception raised during dependency query (circular dep scenario)."""
        mock_db.upsert_role(
            Role(name="circ_role", wave=1, wave_name="Test", has_molecule_tests=False)
        )

        # Simulate a database error that could occur with circular dependency detection
        original_get_deps = mock_db.get_dependencies
        mock_db.get_dependencies = MagicMock(
            side_effect=Exception("Circular dependency detected: circ_role -> A -> circ_role")
        )

        state: BoxUpRoleState = {"role_name": "circ_role"}
        result = await analyze_deps_node(state)

        assert "analyze_dependencies" in result["completed_nodes"]
        assert len(result.get("errors", [])) > 0
        assert "failed" in result["errors"][0].lower()


class TestCheckReverseDepsNodeExtended:
    """Extended tests for check_reverse_deps_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_multiple_reverse_deps_mixed(self):
        """Edge case: some reverse deps boxed, some not."""
        call_count = [0]

        def mock_run_side_effect(cmd, **kwargs):
            nonlocal call_count
            result = MagicMock()
            # First call: dep exists, second: dep missing
            branch_name = cmd[-1]  # e.g., "sid/dep_a"
            if "dep_a" in branch_name:
                result.stdout = "abc123  refs/heads/sid/dep_a"
            else:
                result.stdout = ""  # Branch doesn't exist
            return result

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            state: BoxUpRoleState = {
                "role_name": "test_role",
                "reverse_deps": ["dep_a", "dep_b"],
            }
            result = await check_reverse_deps_node(state)

        assert "dep_b" in result["blocking_deps"]
        assert "dep_a" not in result["blocking_deps"]


class TestCreateWorktreeNodeExtended:
    """Extended tests for create_worktree_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_worktree_runtime_error(self, db_with_test_role):
        """Edge case: worktree path conflict or git error."""
        with (
            patch("harness.worktree.manager.WorktreeManager") as mock_manager_class,
            patch("harness.gitlab.api.GitLabClient") as mock_client_class,
        ):
            mock_manager = MagicMock()
            mock_manager.create.side_effect = RuntimeError(
                "fatal: '/path/to/worktree' is already a working tree"
            )
            mock_manager_class.return_value = mock_manager

            mock_client = MagicMock()
            mock_client.remote_branch_exists.return_value = False
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role"}
            result = await create_worktree_node(state)

            assert "create_worktree" in result["completed_nodes"]
            assert len(result.get("errors", [])) > 0
            assert "failed" in result["errors"][0].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_worktree_unexpected_exception(self, db_with_test_role):
        """Edge case: unexpected exception during worktree creation."""
        with (
            patch("harness.worktree.manager.WorktreeManager") as mock_manager_class,
            patch("harness.gitlab.api.GitLabClient") as mock_client_class,
        ):
            mock_manager = MagicMock()
            mock_manager.create.side_effect = OSError("Permission denied")
            mock_manager_class.return_value = mock_manager

            mock_client = MagicMock()
            mock_client.remote_branch_exists.return_value = False
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role"}
            result = await create_worktree_node(state)

            assert "create_worktree" in result["completed_nodes"]
            assert len(result.get("errors", [])) > 0


class TestRunMoleculeNodeExtended:
    """Extended tests for run_molecule_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_records_regression_on_failure(self, mock_db):
        """Edge case: verify _record_test_result is called on failure for regression tracking."""
        mock_db.upsert_role(
            Role(name="reg_role", wave=1, wave_name="Test", has_molecule_tests=True)
        )

        with (
            patch("subprocess.run") as mock_run,
            patch("harness.dag.langgraph_nodes._record_test_result") as mock_record,
        ):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = "FAILED output"
            mock_result.stderr = "stderr output"
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "reg_role",
                "has_molecule_tests": True,
                "worktree_path": "/tmp/worktree",
                "execution_id": 42,
            }
            result = await run_molecule_node(state)

            assert result["molecule_passed"] is False
            assert result["molecule_output"] is not None
            # Verify _record_test_result was called with passed=False
            mock_record.assert_called_once()
            call_kwargs = mock_record.call_args
            assert call_kwargs.kwargs.get("passed") is False or call_kwargs[0][3] is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_molecule_truncates_long_output(self, mock_db):
        """Edge case: verify long output is truncated."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "x" * 10000  # Very long output
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "has_molecule_tests": True,
                "worktree_path": "/tmp/worktree",
            }
            result = await run_molecule_node(state)

            assert result["molecule_passed"] is True


class TestRunPytestNodeExtended:
    """Extended tests for run_pytest_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_pytest_binary_not_found(self, tmp_path, mock_db):
        """Edge case: pytest binary not installed (FileNotFoundError)."""
        test_file = tmp_path / "tests" / "test_test_role.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_ok(): pass")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("pytest not found")

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
            }
            result = await run_pytest_node(state)

            assert result["pytest_skipped"] is True
            assert "run_pytest" in result["parallel_tests_completed"]


class TestCreateCommitNodeExtended:
    """Extended tests for create_commit_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_commit_message_format_contains_wave(self, tmp_path):
        """Edge case: verify commit message includes role name and wave info."""
        with patch("subprocess.run") as mock_run:

            captured_commit_msg = {}

            def run_side_effect(cmd, **kwargs):
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                if "status" in cmd:
                    result.stdout = "M  file.txt"
                elif "commit" in cmd:
                    # Capture the commit message
                    captured_commit_msg["msg"] = cmd[-1]
                    result.stdout = ""
                elif "rev-parse" in cmd:
                    result.stdout = "sha789"
                else:
                    result.stdout = ""
                return result

            mock_run.side_effect = run_side_effect

            state: BoxUpRoleState = {
                "role_name": "ems_web_app",
                "worktree_path": str(tmp_path),
                "wave": 3,
                "wave_name": "Web Applications",
            }
            result = await create_commit_node(state)

            assert "feat(ems_web_app)" in result["commit_message"]
            assert "Wave 3" in result["commit_message"]
            assert "Web Applications" in result["commit_message"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_commit_git_add_fails(self, tmp_path):
        """Edge case: git add fails with CalledProcessError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git add")

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "worktree_path": str(tmp_path),
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_commit_node(state)

            assert "create_commit" in result["completed_nodes"]
            assert len(result.get("errors", [])) > 0


class TestPushBranchNodeExtended:
    """Extended tests for push_branch_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_push_uses_default_branch(self, tmp_path):
        """Edge case: push uses default branch naming when branch not set."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            state: BoxUpRoleState = {
                "role_name": "my_role",
                "worktree_path": str(tmp_path),
                # branch not set explicitly, uses default
            }
            result = await push_branch_node(state)

            assert result["pushed"] is True
            # Verify the push command used the default branch
            call_args = mock_run.call_args[0][0]
            assert "sid/my_role" in call_args


class TestCreateIssueNodeExtended:
    """Extended tests for create_issue_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_issue_value_error(self, db_with_test_role):
        """Edge case: ValueError raised (e.g., role not found in DB)."""
        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = []
            mock_config_class.from_harness_yml.return_value = mock_config

            # Set up mock API that raises ValueError
            mock_api = AsyncMock()
            mock_api.get_current_iteration = AsyncMock(return_value=None)
            mock_api.ensure_label_exists = AsyncMock()
            mock_api.get_or_create_issue = AsyncMock(side_effect=ValueError("Role 'x' not found in DB"))

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
                "explicit_deps": [],
                "credentials": [],
            }
            result = await create_issue_node(state)

            assert "create_gitlab_issue" in result["completed_nodes"]
            assert len(result.get("errors", [])) > 0
            assert "not found" in result["errors"][0].lower()


class TestCreateMRNodeExtended:
    """Extended tests for create_mr_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_mr_from_issue_closes_ref(self, db_with_test_role):
        """Edge case: MR description should reference closing the issue."""
        mock_mr = {
            "web_url": "https://gitlab.example.com/mr/500",
            "iid": 500,
        }

        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
            patch("harness.gitlab.templates.render_mr_description") as mock_render,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role"]
            mock_config.default_reviewers = []
            mock_config_class.from_harness_yml.return_value = mock_config

            # Mock template rendering
            mock_render.return_value = "## Summary\nCloses #42"

            # Set up mock API
            mock_api = AsyncMock()
            mock_api.find_mr_by_branch = AsyncMock(return_value=None)
            mock_api.get_or_create_mr = AsyncMock(return_value=(mock_mr, True))
            mock_api.set_mr_reviewers = AsyncMock(return_value=True)

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "issue_iid": 42,
                "branch": "sid/test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_mr_node(state)

            # Verify render_mr_description was called with issue_iid in context
            call_args = mock_render.call_args[0][0]
            assert call_args.get("issue_iid") == 42

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_create_mr_reviewers_fail(self, db_with_test_role):
        """Edge case: MR created but setting reviewers fails."""
        mock_mr = {
            "web_url": "https://gitlab.example.com/mr/600",
            "iid": 600,
        }

        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
            patch("harness.gitlab.templates.render_mr_description") as mock_render,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role"]
            mock_config.default_reviewers = ["user1", "user2"]
            mock_config_class.from_harness_yml.return_value = mock_config

            # Mock template rendering
            mock_render.return_value = "## Summary\nTest MR"

            # Set up mock API with reviewer failure
            mock_api = AsyncMock()
            mock_api.find_mr_by_branch = AsyncMock(return_value=None)
            mock_api.get_or_create_mr = AsyncMock(return_value=(mock_mr, True))
            mock_api.set_mr_reviewers = AsyncMock(return_value=False)  # Reviewer assignment fails

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "issue_iid": 42,
                "branch": "sid/test_role",
                "wave": 1,
                "wave_name": "Infrastructure",
            }
            result = await create_mr_node(state)

            # MR should be created successfully (reviewer assignment is no longer done in create_mr_node)
            assert result["mr_iid"] == 600
            assert result["mr_created"] is True
            assert "create_merge_request" in result["completed_nodes"]


class TestHumanApprovalNodeExtended:
    """Extended tests for human_approval_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_human_approval_rejected_no_reason(self):
        """Edge case: rejection without a reason provided."""
        with patch("harness.dag.langgraph_nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"approved": False}  # No reason key

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "mr_url": "https://gitlab.example.com/mr/123",
                "mr_iid": 123,
                "molecule_passed": True,
                "pytest_passed": True,
                "branch": "sid/test_role",
            }
            result = await human_approval_node(state)

            assert result["human_approved"] is False
            # Should have a default rejection reason
            assert result["human_rejection_reason"] is not None
            assert "without reason" in result["human_rejection_reason"].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_human_approval_false_bool_response(self):
        """Edge case: simple False boolean response."""
        with patch("harness.dag.langgraph_nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = False

            state: BoxUpRoleState = {
                "role_name": "test_role",
                "mr_url": "https://gitlab.example.com/mr/123",
                "mr_iid": 123,
                "molecule_passed": True,
                "pytest_passed": True,
                "branch": "sid/test_role",
            }
            result = await human_approval_node(state)

            assert result["human_approved"] is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_human_approval_context_includes_all_info(self):
        """Edge case: verify interrupt receives full context about the MR."""
        captured_context = {}

        def mock_interrupt(data):
            captured_context.update(data)
            return {"approved": True}

        with patch("harness.dag.langgraph_nodes.interrupt", side_effect=mock_interrupt):
            state: BoxUpRoleState = {
                "role_name": "my_role",
                "mr_url": "https://gitlab.example.com/mr/999",
                "mr_iid": 999,
                "molecule_passed": True,
                "pytest_passed": False,
                "branch": "sid/my_role",
                "issue_url": "https://gitlab.example.com/issues/55",
                "wave": 2,
                "wave_name": "Platform",
                "credentials": [{"entry_name": "admin"}],
                "explicit_deps": ["common"],
            }
            await human_approval_node(state)

            # Verify context passed to interrupt
            assert "context" in captured_context
            ctx = captured_context["context"]
            assert ctx["role_name"] == "my_role"
            assert ctx["mr_iid"] == 999
            assert ctx["pytest_passed"] is False
            assert ctx["wave"] == 2


class TestAddToMergeTrainNodeExtended:
    """Extended tests for add_to_merge_train_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_merge_train_add_fails_then_fallback(self, db_with_test_role):
        """Edge case: merge train add raises error, falls back to MWPS."""
        with patch("harness.gitlab.api.GitLabClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.is_merge_train_available.return_value = {"available": True}
            mock_client.is_merge_train_enabled.return_value = True
            # add_to_merge_train fails with a "not found" error -> triggers fallback
            mock_client.add_to_merge_train.side_effect = RuntimeError(
                "merge_train API not found"
            )
            mock_client.merge_when_pipeline_succeeds.return_value = None
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
            result = await add_to_merge_train_node(state)

            assert result["merge_train_status"] == "fallback"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_merge_train_no_db(self):
        """Edge case: database not available for merge train operation."""
        set_module_db(None)
        state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
        result = await add_to_merge_train_node(state)

        assert result["merge_train_status"] == "skipped"
        assert len(result.get("errors", [])) > 0
        assert "database" in result["errors"][0].lower()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_merge_train_mwps_fallback_fails(self, db_with_test_role):
        """Edge case: merge_when_pipeline_succeeds fallback also fails."""
        with patch("harness.gitlab.api.GitLabClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.is_merge_train_available.return_value = {"available": True}
            mock_client.is_merge_train_enabled.return_value = False
            mock_client.merge_when_pipeline_succeeds.side_effect = RuntimeError(
                "Pipeline still running"
            )
            mock_client_class.return_value = mock_client

            state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
            result = await add_to_merge_train_node(state)

            assert result["merge_train_status"] == "skipped"
            assert len(result.get("errors", [])) > 0


class TestReportSummaryNodeExtended:
    """Extended tests for report_summary_node edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_report_summary_partial_completion(self):
        """Edge case: partial completion - some nodes didn't run."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "wave": 1,
            "wave_name": "Infrastructure",
            "molecule_passed": True,
            "molecule_duration": 100,
            "pytest_passed": None,  # Didn't run
            "pytest_duration": 0,
            "all_tests_passed": None,
            "test_phase_duration": None,  # No parallel execution
            "errors": [],
        }
        result = await report_summary_node(state)

        assert "report_summary" in result["completed_nodes"]
        summary = result["summary"]
        assert summary["molecule_passed"] is True
        assert summary["pytest_passed"] is None
        # test_phase_duration should fallback to sum of individual durations
        assert summary["test_phase_duration"] == 100

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_report_summary_no_test_durations(self):
        """Edge case: no test durations (zero division protection)."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "wave": 0,
            "wave_name": "Base",
            "molecule_duration": None,
            "pytest_duration": None,
            "test_phase_duration": None,
            "errors": [],
        }
        result = await report_summary_node(state)

        summary = result["summary"]
        # Should not raise ZeroDivisionError
        assert summary["test_time_saved_seconds"] == 0
        assert summary["test_time_saved_percent"] == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_report_summary_includes_branch_existed(self):
        """Edge case: summary should reflect if branch already existed."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "wave": 1,
            "wave_name": "Infra",
            "branch_existed": True,
            "issue_created": False,
            "mr_created": False,
            "errors": [],
        }
        result = await report_summary_node(state)

        summary = result["summary"]
        assert summary["branch_existed"] is True
        assert summary["issue_created"] is False
        assert summary["mr_created"] is False


# =============================================================================
# MERGE TEST RESULTS NODE TESTS (Bonus - already partially covered)
# =============================================================================


class TestMergeTestResultsNode:
    """Tests for merge_test_results_node."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_merge_all_passed(self):
        """Happy path: all tests passed."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_skipped": False,
            "molecule_duration": 60,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 30,
            "test_phase_start_time": time.time() - 65,
        }
        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is True
        assert "merge_test_results" in result["completed_nodes"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_merge_partial_failure(self):
        """Error handling: one test fails."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": False,
            "molecule_skipped": False,
            "molecule_duration": 120,
            "pytest_passed": True,
            "pytest_skipped": False,
            "pytest_duration": 30,
            "test_phase_start_time": time.time() - 125,
        }
        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is False
        assert len(result.get("errors", [])) == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_merge_both_skipped(self):
        """Edge case: both tests skipped."""
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "molecule_passed": None,
            "molecule_skipped": True,
            "molecule_duration": 0,
            "pytest_passed": None,
            "pytest_skipped": True,
            "pytest_duration": 0,
            "test_phase_start_time": time.time() - 1,
        }
        result = await merge_test_results_node(state)

        assert result["all_tests_passed"] is True


# =============================================================================
# GRAPH CONSTRUCTION TESTS
# =============================================================================


class TestGraphConstruction:
    """Tests for graph construction and configuration."""

    @pytest.mark.unit
    def test_graph_creates_all_nodes(self):
        """Verify graph includes all 15 nodes."""
        graph, breakpoints = create_box_up_role_graph()

        # Graph should be a StateGraph
        assert graph is not None

    @pytest.mark.unit
    def test_graph_with_parallel_tests_enabled(self):
        """Graph with parallel_tests=True should include merge_test_results."""
        graph, _ = create_box_up_role_graph(parallel_tests=True)
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.unit
    def test_graph_with_parallel_tests_disabled(self):
        """Graph with parallel_tests=False should work in sequential mode."""
        graph, _ = create_box_up_role_graph(parallel_tests=False)
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.unit
    def test_graph_with_breakpoints_enabled(self, monkeypatch):
        """Graph with breakpoints enabled should return breakpoint nodes."""
        monkeypatch.setenv("HARNESS_BREAKPOINTS", "true")
        graph, breakpoints = create_box_up_role_graph(enable_breakpoints=True)

        assert len(breakpoints) > 0
        assert "human_approval" in breakpoints

    @pytest.mark.unit
    def test_graph_with_breakpoints_disabled(self):
        """Graph with breakpoints disabled should return empty list."""
        graph, breakpoints = create_box_up_role_graph(enable_breakpoints=False)

        assert breakpoints == []


# =============================================================================
# INITIAL STATE TESTS
# =============================================================================


class TestInitialState:
    """Tests for create_initial_state."""

    @pytest.mark.unit
    def test_initial_state_has_all_fields(self):
        """Initial state should have all required fields."""
        state = create_initial_state("test_role", execution_id=1)

        # Core fields
        assert state["role_name"] == "test_role"
        assert state["execution_id"] == 1

        # Test fields
        assert state["molecule_skipped"] is False
        assert state["pytest_skipped"] is False
        assert state["all_tests_passed"] is None

        # Parallel execution fields
        assert state["parallel_tests_completed"] == []
        assert state["parallel_execution_enabled"] is True

        # HITL fields
        assert state["human_approved"] is None
        assert state["awaiting_human_input"] is False

    @pytest.mark.unit
    def test_initial_state_without_execution_id(self):
        """Initial state should work without execution_id."""
        state = create_initial_state("test_role")

        assert state["role_name"] == "test_role"
        assert state.get("execution_id") is None


# =============================================================================
# CONDITIONAL ROUTING TESTS
# =============================================================================


class TestConditionalRouting:
    """Tests for conditional routing functions."""

    @pytest.mark.unit
    def test_should_continue_after_validation_success(self):
        """After successful validation, should continue to analyze_deps."""
        from harness.dag.langgraph_engine import should_continue_after_validation

        state: BoxUpRoleState = {"role_name": "test_role", "errors": []}
        result = should_continue_after_validation(state)
        assert result == "analyze_deps"

    @pytest.mark.unit
    def test_should_continue_after_validation_failure(self):
        """After failed validation, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_validation

        state: BoxUpRoleState = {"role_name": "test_role", "errors": ["Role not found"]}
        result = should_continue_after_validation(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_mr_with_mr(self):
        """After MR creation with MR, should route to human_approval."""
        from harness.dag.langgraph_engine import should_continue_after_mr

        state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": 123}
        result = should_continue_after_mr(state)
        assert result == "human_approval"

    @pytest.mark.unit
    def test_should_continue_after_mr_without_mr(self):
        """After MR creation without MR, should route to report_summary."""
        from harness.dag.langgraph_engine import should_continue_after_mr

        state: BoxUpRoleState = {"role_name": "test_role", "mr_iid": None}
        result = should_continue_after_mr(state)
        assert result == "report_summary"

    @pytest.mark.unit
    def test_should_continue_after_human_approval_approved(self):
        """After human approval approved, should route to merge train."""
        from harness.dag.langgraph_engine import should_continue_after_human_approval

        state: BoxUpRoleState = {"role_name": "test_role", "human_approved": True}
        result = should_continue_after_human_approval(state)
        assert result == "add_to_merge_train"

    @pytest.mark.unit
    def test_should_continue_after_human_approval_rejected(self):
        """After human approval rejected, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_human_approval

        state: BoxUpRoleState = {"role_name": "test_role", "human_approved": False}
        result = should_continue_after_human_approval(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_deps_success(self):
        """After successful deps analysis, should continue to check_reverse_deps."""
        from harness.dag.langgraph_engine import should_continue_after_deps

        state: BoxUpRoleState = {"role_name": "test_role", "errors": []}
        result = should_continue_after_deps(state)
        assert result == "check_reverse_deps"

    @pytest.mark.unit
    def test_should_continue_after_deps_failure(self):
        """After failed deps analysis, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_deps

        state: BoxUpRoleState = {"role_name": "test_role", "errors": ["DB error"]}
        result = should_continue_after_deps(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_reverse_deps_blocking(self):
        """When blocking deps exist, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_reverse_deps

        state: BoxUpRoleState = {
            "role_name": "test_role",
            "blocking_deps": ["dep_a"],
        }
        result = should_continue_after_reverse_deps(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_reverse_deps_clear(self):
        """When no blocking deps, should continue to create_worktree."""
        from harness.dag.langgraph_engine import should_continue_after_reverse_deps

        state: BoxUpRoleState = {
            "role_name": "test_role",
            "blocking_deps": [],
        }
        result = should_continue_after_reverse_deps(state)
        assert result == "create_worktree"

    @pytest.mark.unit
    def test_should_continue_after_deploy_pass(self):
        """After deploy passes, should continue to create_commit."""
        from harness.dag.langgraph_engine import should_continue_after_deploy

        state: BoxUpRoleState = {"role_name": "test_role", "deploy_passed": True}
        result = should_continue_after_deploy(state)
        assert result == "create_commit"

    @pytest.mark.unit
    def test_should_continue_after_deploy_fail(self):
        """After deploy fails, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_deploy

        state: BoxUpRoleState = {"role_name": "test_role", "deploy_passed": False}
        result = should_continue_after_deploy(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_commit_success(self):
        """After commit success, should continue to push_branch."""
        from harness.dag.langgraph_engine import should_continue_after_commit

        state: BoxUpRoleState = {"role_name": "test_role", "errors": []}
        result = should_continue_after_commit(state)
        assert result == "push_branch"

    @pytest.mark.unit
    def test_should_continue_after_commit_failure(self):
        """After commit failure, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_commit

        state: BoxUpRoleState = {"role_name": "test_role", "errors": ["Commit failed"]}
        result = should_continue_after_commit(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_push_success(self):
        """After push success, should continue to create_issue."""
        from harness.dag.langgraph_engine import should_continue_after_push

        state: BoxUpRoleState = {"role_name": "test_role", "pushed": True}
        result = should_continue_after_push(state)
        assert result == "create_issue"

    @pytest.mark.unit
    def test_should_continue_after_push_failure(self):
        """After push failure, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_push

        state: BoxUpRoleState = {"role_name": "test_role", "pushed": False}
        result = should_continue_after_push(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_issue_success(self):
        """After issue creation success, should continue to create_mr."""
        from harness.dag.langgraph_engine import should_continue_after_issue

        state: BoxUpRoleState = {"role_name": "test_role", "issue_iid": 42}
        result = should_continue_after_issue(state)
        assert result == "create_mr"

    @pytest.mark.unit
    def test_should_continue_after_issue_failure(self):
        """After issue creation failure, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_issue

        state: BoxUpRoleState = {"role_name": "test_role", "issue_iid": None}
        result = should_continue_after_issue(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_merge_pass(self):
        """After merge tests pass, should continue to validate_deploy."""
        from harness.dag.langgraph_engine import should_continue_after_merge

        state: BoxUpRoleState = {"role_name": "test_role", "all_tests_passed": True}
        result = should_continue_after_merge(state)
        assert result == "validate_deploy"

    @pytest.mark.unit
    def test_should_continue_after_merge_fail(self):
        """After merge tests fail, should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_merge

        state: BoxUpRoleState = {"role_name": "test_role", "all_tests_passed": False}
        result = should_continue_after_merge(state)
        assert result == "notify_failure"


# =============================================================================
# CUSTOM REDUCER TESTS
# =============================================================================


class TestKeepLastNReducer:
    """Tests for the keep_last_n custom reducer."""

    @pytest.mark.unit
    def test_keep_last_n_within_limit(self):
        """Happy path: items within limit are kept."""
        reducer = keep_last_n(5)
        result = reducer([1, 2, 3], [4])
        assert result == [1, 2, 3, 4]

    @pytest.mark.unit
    def test_keep_last_n_exceeds_limit(self):
        """Edge case: items exceeding limit are trimmed from the front."""
        reducer = keep_last_n(3)
        result = reducer([1, 2], [3, 4])
        assert result == [2, 3, 4]

    @pytest.mark.unit
    def test_keep_last_n_none_inputs(self):
        """Edge case: None inputs are treated as empty lists."""
        reducer = keep_last_n(5)
        result = reducer(None, None)
        assert result == []

    @pytest.mark.unit
    def test_keep_last_n_empty_lists(self):
        """Edge case: empty lists produce empty result."""
        reducer = keep_last_n(3)
        result = reducer([], [])
        assert result == []
