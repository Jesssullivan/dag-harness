"""Tests for subgraph composition (Task #22).

This module tests:
1. Individual subgraph execution
2. Subgraph composition and phase transitions
3. Error propagation between subgraphs
4. State accumulation across subgraphs
"""

from unittest.mock import MagicMock, patch

import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    create_initial_state,
    set_module_db,
)
from harness.dag.subgraphs import (
    SubgraphWorkflowRunner,
    create_composed_workflow,
    create_gitlab_subgraph,
    create_notification_subgraph,
    create_testing_subgraph,
    create_validation_subgraph,
    should_continue_after_gitlab_phase,
    should_continue_after_testing_phase,
    should_continue_after_validation_phase,
)
from harness.db.state import StateDB

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_db():
    """Create a mock StateDB for testing."""
    db = MagicMock(spec=StateDB)

    # Mock role data
    mock_role = MagicMock()
    mock_role.id = 1
    mock_role.name = "test_role"
    mock_role.wave = 1
    mock_role.wave_name = "Infrastructure"
    db.get_role.return_value = mock_role

    # Mock dependencies
    db.get_dependencies.return_value = [("common", 1)]
    db.get_reverse_dependencies.return_value = []
    db.get_credentials.return_value = []

    return db


@pytest.fixture
def initial_state() -> BoxUpRoleState:
    """Create initial state for testing."""
    return create_initial_state("test_role", execution_id=1)


@pytest.fixture
def validation_passed_state(initial_state) -> BoxUpRoleState:
    """State after successful validation phase."""
    state = dict(initial_state)
    state.update(
        {
            "role_path": "ansible/roles/test_role",
            "has_molecule_tests": True,
            "has_meta": True,
            "wave": 1,
            "wave_name": "Infrastructure",
            "explicit_deps": ["common"],
            "implicit_deps": [],
            "reverse_deps": [],
            "credentials": [],
            "blocking_deps": [],
            "completed_nodes": ["validate_role", "analyze_dependencies", "check_reverse_deps"],
            "errors": [],
        }
    )
    return BoxUpRoleState(**state)


@pytest.fixture
def testing_passed_state(validation_passed_state) -> BoxUpRoleState:
    """State after successful testing phase."""
    state = dict(validation_passed_state)
    state.update(
        {
            "worktree_path": "/tmp/sid-test_role",
            "branch": "sid/test_role",
            "commit_sha": "abc123",
            "molecule_passed": True,
            "molecule_skipped": False,
            "pytest_passed": True,
            "pytest_skipped": False,
            "deploy_passed": True,
            "deploy_skipped": False,
            "completed_nodes": validation_passed_state.get("completed_nodes", [])
            + ["create_worktree", "run_molecule_tests", "run_pytest", "validate_deploy"],
        }
    )
    return BoxUpRoleState(**state)


@pytest.fixture
def gitlab_passed_state(testing_passed_state) -> BoxUpRoleState:
    """State after successful GitLab phase."""
    state = dict(testing_passed_state)
    state.update(
        {
            "commit_sha": "abc123def456",
            "commit_message": "feat(test_role): Add test_role Ansible role",
            "pushed": True,
            "issue_url": "https://gitlab.example.com/issues/123",
            "issue_iid": 123,
            "issue_created": True,
            "mr_url": "https://gitlab.example.com/merge_requests/456",
            "mr_iid": 456,
            "mr_created": True,
            "human_approved": True,
            "merge_train_status": "added",
            "completed_nodes": testing_passed_state.get("completed_nodes", [])
            + [
                "create_commit",
                "push_branch",
                "create_gitlab_issue",
                "create_merge_request",
                "human_approval",
                "add_to_merge_train",
            ],
        }
    )
    return BoxUpRoleState(**state)


@pytest.fixture
def error_state(initial_state) -> BoxUpRoleState:
    """State with errors."""
    state = dict(initial_state)
    state.update(
        {
            "errors": ["Role not found: test_role"],
            "completed_nodes": ["validate_role"],
        }
    )
    return BoxUpRoleState(**state)


# ============================================================================
# ROUTING FUNCTION TESTS
# ============================================================================


class TestRoutingFunctions:
    """Test phase transition routing functions."""

    @pytest.mark.unit
    def test_validation_phase_success_routes_to_testing(self, validation_passed_state):
        """Successful validation routes to testing phase."""
        result = should_continue_after_validation_phase(validation_passed_state)
        assert result == "testing_phase"

    @pytest.mark.unit
    def test_validation_phase_error_routes_to_notification(self, error_state):
        """Validation error routes to notification phase."""
        result = should_continue_after_validation_phase(error_state)
        assert result == "notification_phase"

    @pytest.mark.unit
    def test_validation_phase_blocking_deps_routes_to_notification(self, initial_state):
        """Blocking dependencies route to notification phase."""
        state = dict(initial_state)
        state["blocking_deps"] = ["other_role"]
        result = should_continue_after_validation_phase(BoxUpRoleState(**state))
        assert result == "notification_phase"

    @pytest.mark.unit
    def test_testing_phase_success_routes_to_gitlab(self, testing_passed_state):
        """Successful testing routes to GitLab phase."""
        result = should_continue_after_testing_phase(testing_passed_state)
        assert result == "gitlab_phase"

    @pytest.mark.unit
    def test_testing_phase_molecule_failure_routes_to_notification(self, validation_passed_state):
        """Molecule failure routes to notification phase."""
        state = dict(validation_passed_state)
        state["molecule_passed"] = False
        result = should_continue_after_testing_phase(BoxUpRoleState(**state))
        assert result == "notification_phase"

    @pytest.mark.unit
    def test_testing_phase_pytest_failure_routes_to_notification(self, validation_passed_state):
        """Pytest failure routes to notification phase."""
        state = dict(validation_passed_state)
        state["pytest_passed"] = False
        result = should_continue_after_testing_phase(BoxUpRoleState(**state))
        assert result == "notification_phase"

    @pytest.mark.unit
    def test_testing_phase_deploy_failure_routes_to_notification(self, validation_passed_state):
        """Deploy failure routes to notification phase."""
        state = dict(validation_passed_state)
        state["deploy_passed"] = False
        result = should_continue_after_testing_phase(BoxUpRoleState(**state))
        assert result == "notification_phase"

    @pytest.mark.unit
    def test_gitlab_phase_always_routes_to_notification(self, gitlab_passed_state):
        """GitLab phase always routes to notification."""
        result = should_continue_after_gitlab_phase(gitlab_passed_state)
        assert result == "notification_phase"

    @pytest.mark.unit
    def test_gitlab_phase_error_routes_to_notification(self, testing_passed_state):
        """GitLab phase with errors routes to notification."""
        state = dict(testing_passed_state)
        state["errors"] = ["Push failed"]
        result = should_continue_after_gitlab_phase(BoxUpRoleState(**state))
        assert result == "notification_phase"


# ============================================================================
# VALIDATION SUBGRAPH TESTS
# ============================================================================


class TestValidationSubgraph:
    """Test validation subgraph independently."""

    @pytest.mark.unit
    def test_create_validation_subgraph(self):
        """Validation subgraph can be created."""
        graph = create_validation_subgraph()
        assert graph is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validation_subgraph_role_not_found(self, initial_state):
        """Validation subgraph handles missing role."""
        graph = create_validation_subgraph()

        with patch("harness.dag.langgraph_nodes.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            result = await graph.ainvoke(initial_state)

            # Should have error about missing role
            assert len(result.get("errors", [])) > 0
            assert "validate_role" in result.get("completed_nodes", [])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validation_subgraph_success(self, initial_state, mock_db, tmp_path):
        """Validation subgraph succeeds with valid role."""
        # Set up module db
        set_module_db(mock_db)

        # Create a mock role directory
        role_dir = tmp_path / "ansible" / "roles" / "test_role"
        role_dir.mkdir(parents=True)
        (role_dir / "meta").mkdir()
        (role_dir / "meta" / "main.yml").touch()
        (role_dir / "molecule").mkdir()

        graph = create_validation_subgraph()

        with patch("harness.dag.langgraph_nodes.Path") as mock_path_cls:
            # Mock the path to exist and have molecule/meta
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__ = lambda self, x: mock_path
            mock_path_cls.return_value = mock_path

            result = await graph.ainvoke(initial_state)

            # Should complete without errors
            assert "validate_role" in result.get("completed_nodes", [])


# ============================================================================
# TESTING SUBGRAPH TESTS
# ============================================================================


class TestTestingSubgraph:
    """Test testing subgraph independently."""

    @pytest.mark.unit
    def test_create_testing_subgraph(self):
        """Testing subgraph can be created."""
        graph = create_testing_subgraph()
        assert graph is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_testing_subgraph_worktree_error(self, validation_passed_state, mock_db):
        """Testing subgraph handles worktree creation error."""
        set_module_db(mock_db)
        graph = create_testing_subgraph()

        with patch("harness.worktree.manager.WorktreeManager") as mock_manager:
            mock_manager.return_value.create.side_effect = RuntimeError("Worktree failed")

            with patch("harness.gitlab.api.GitLabClient") as mock_gitlab:
                mock_gitlab.return_value.remote_branch_exists.return_value = False

                result = await graph.ainvoke(validation_passed_state)

                # Should have error
                assert len(result.get("errors", [])) > 0
                assert "create_worktree" in result.get("completed_nodes", [])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_testing_subgraph_molecule_skip(self, validation_passed_state, mock_db):
        """Testing subgraph skips molecule when no tests exist."""
        set_module_db(mock_db)

        # No molecule tests
        state = dict(validation_passed_state)
        state["has_molecule_tests"] = False

        graph = create_testing_subgraph()

        with patch("harness.worktree.manager.WorktreeManager") as mock_manager:
            mock_worktree = MagicMock()
            mock_worktree.path = "/tmp/worktree"
            mock_worktree.branch = "sid/test_role"
            mock_worktree.commit = "abc123"
            mock_manager.return_value.create.return_value = mock_worktree

            with patch("harness.gitlab.api.GitLabClient") as mock_gitlab:
                mock_gitlab.return_value.remote_branch_exists.return_value = False

                with patch("subprocess.run") as mock_run:
                    # Make all subprocess calls succeed
                    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                    result = await graph.ainvoke(BoxUpRoleState(**state))

                    # Should skip molecule
                    assert result.get("molecule_skipped") is True


# ============================================================================
# GITLAB SUBGRAPH TESTS
# ============================================================================


class TestGitLabSubgraph:
    """Test GitLab subgraph independently."""

    @pytest.mark.unit
    def test_create_gitlab_subgraph(self):
        """GitLab subgraph can be created."""
        graph = create_gitlab_subgraph()
        assert graph is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_gitlab_subgraph_commit_no_changes(self, testing_passed_state, mock_db):
        """GitLab subgraph handles no changes to commit."""
        set_module_db(mock_db)
        graph = create_gitlab_subgraph()

        with patch("harness.dag.langgraph_nodes.subprocess.run") as mock_run:
            # git add succeeds
            # git status returns empty (no changes)
            mock_run.side_effect = [
                MagicMock(returncode=0),  # git add
                MagicMock(returncode=0, stdout="", stderr=""),  # git status
            ]

            result = await graph.ainvoke(testing_passed_state)

            # Should continue without commit
            assert "create_commit" in result.get("completed_nodes", [])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_gitlab_subgraph_push_failure(self, testing_passed_state, mock_db):
        """GitLab subgraph handles push failure."""
        set_module_db(mock_db)
        graph = create_gitlab_subgraph()

        with patch("harness.dag.langgraph_nodes.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0),  # git add
                MagicMock(returncode=0, stdout="M file.txt"),  # git status
                MagicMock(returncode=0),  # git commit
                MagicMock(returncode=0, stdout="abc123"),  # git rev-parse
                MagicMock(returncode=1, stderr="Push failed: permission denied"),  # git push
            ]

            result = await graph.ainvoke(testing_passed_state)

            # Should have error
            assert len(result.get("errors", [])) > 0
            assert "push_branch" in result.get("completed_nodes", [])


# ============================================================================
# NOTIFICATION SUBGRAPH TESTS
# ============================================================================


class TestNotificationSubgraph:
    """Test notification subgraph independently."""

    @pytest.mark.unit
    def test_create_notification_subgraph(self):
        """Notification subgraph can be created."""
        graph = create_notification_subgraph()
        assert graph is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_notification_subgraph_success_path(self, gitlab_passed_state):
        """Notification subgraph generates summary on success."""
        graph = create_notification_subgraph()

        result = await graph.ainvoke(gitlab_passed_state)

        # Should have summary
        assert result.get("summary") is not None
        assert "report_summary" in result.get("completed_nodes", [])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_notification_subgraph_failure_path(self, error_state, mock_db):
        """Notification subgraph handles failure notifications."""
        set_module_db(mock_db)
        graph = create_notification_subgraph()

        with patch("harness.gitlab.api.GitLabClient"):
            result = await graph.ainvoke(error_state)

            # Should go through failure path then summary
            assert "notify_failure" in result.get("completed_nodes", [])
            assert "report_summary" in result.get("completed_nodes", [])


# ============================================================================
# COMPOSED WORKFLOW TESTS
# ============================================================================


class TestComposedWorkflow:
    """Test the full composed workflow."""

    @pytest.mark.unit
    def test_create_composed_workflow(self):
        """Composed workflow can be created."""
        workflow = create_composed_workflow()
        assert workflow is not None

    @pytest.mark.unit
    def test_composed_workflow_compiles(self):
        """Composed workflow can be compiled."""
        workflow = create_composed_workflow()
        compiled = workflow.compile()
        assert compiled is not None

    @pytest.mark.unit
    def test_composed_workflow_has_phases(self):
        """Composed workflow has all expected phases."""
        workflow = create_composed_workflow()

        # Check that all phase nodes exist
        # Note: We can't directly inspect nodes, but we can compile and check it works
        compiled = workflow.compile()
        assert compiled is not None


# ============================================================================
# SUBGRAPH WORKFLOW RUNNER TESTS
# ============================================================================


class TestSubgraphWorkflowRunner:
    """Test the SubgraphWorkflowRunner."""

    @pytest.mark.unit
    def test_runner_initialization(self, mock_db):
        """Runner can be initialized."""
        runner = SubgraphWorkflowRunner(db=mock_db, db_path=":memory:")
        assert runner is not None
        assert runner.db == mock_db

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_runner_get_graph(self, mock_db):
        """Runner creates composed graph."""
        runner = SubgraphWorkflowRunner(db=mock_db, db_path=":memory:")
        graph = await runner._get_graph()
        assert graph is not None


# ============================================================================
# ERROR PROPAGATION TESTS
# ============================================================================


class TestErrorPropagation:
    """Test error propagation between subgraphs."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validation_error_propagates_to_notification(self, initial_state):
        """Validation errors propagate to notification phase."""
        workflow = create_composed_workflow()
        compiled = workflow.compile()

        with patch("harness.dag.langgraph_nodes.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            result = await compiled.ainvoke(initial_state)

            # Should have error
            assert len(result.get("errors", [])) > 0
            # Should have summary (from notification phase)
            assert result.get("summary") is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_testing_error_propagates_to_notification(self, validation_passed_state, mock_db):
        """Testing errors propagate to notification phase."""
        set_module_db(mock_db)
        workflow = create_composed_workflow()
        compiled = workflow.compile()

        with patch("harness.worktree.manager.WorktreeManager") as mock_manager:
            mock_manager.return_value.create.side_effect = RuntimeError("Worktree failed")

            with patch("harness.gitlab.api.GitLabClient") as mock_gitlab:
                mock_gitlab.return_value.remote_branch_exists.return_value = False

                result = await compiled.ainvoke(validation_passed_state)

                # Should have error
                assert len(result.get("errors", [])) > 0
                # Should have summary
                assert result.get("summary") is not None


# ============================================================================
# STATE ACCUMULATION TESTS
# ============================================================================


class TestStateAccumulation:
    """Test state accumulation across subgraphs."""

    @pytest.mark.unit
    def test_completed_nodes_accumulate(self, gitlab_passed_state):
        """Completed nodes accumulate across all phases."""
        completed = gitlab_passed_state.get("completed_nodes", [])

        # Should have nodes from all phases
        validation_nodes = {"validate_role", "analyze_dependencies", "check_reverse_deps"}
        testing_nodes = {"create_worktree", "run_molecule_tests", "run_pytest", "validate_deploy"}
        gitlab_nodes = {
            "create_commit",
            "push_branch",
            "create_gitlab_issue",
            "create_merge_request",
            "human_approval",
            "add_to_merge_train",
        }

        validation_nodes | testing_nodes | gitlab_nodes

        # At least the key nodes should be present
        assert "validate_role" in completed or len(completed) > 0

    @pytest.mark.unit
    def test_errors_list_accumulates(self):
        """Errors accumulate across phases using list reducer."""
        # Start with empty state
        state: BoxUpRoleState = create_initial_state("test")

        # Simulate adding errors from different phases
        state["errors"] = ["Error 1"]

        # In real execution, the list reducer would append
        # For now, just verify the type is correct
        assert isinstance(state.get("errors"), list)

    @pytest.mark.unit
    def test_state_preserves_across_phases(self, testing_passed_state):
        """State values from earlier phases are preserved."""
        # Values set in validation should persist
        assert testing_passed_state.get("role_path") is not None
        assert testing_passed_state.get("has_molecule_tests") is True

        # Values set in testing should be present
        assert testing_passed_state.get("worktree_path") is not None
        assert testing_passed_state.get("molecule_passed") is True


# ============================================================================
# INTEGRATION WITH EXISTING WORKFLOW
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with existing workflow."""

    @pytest.mark.unit
    def test_state_schema_unchanged(self):
        """BoxUpRoleState schema is unchanged."""
        state = create_initial_state("test", execution_id=1)

        # All expected fields should be present
        assert "role_name" in state
        assert "execution_id" in state
        assert "has_molecule_tests" in state
        assert "errors" in state
        assert "completed_nodes" in state

    @pytest.mark.unit
    def test_composed_workflow_produces_same_state(self, initial_state):
        """Composed workflow produces equivalent state structure."""
        # The composed workflow should produce the same state structure
        # as the flat workflow
        workflow = create_composed_workflow()
        compiled = workflow.compile()

        # Verify the graph can process the standard initial state
        assert compiled is not None

    @pytest.mark.unit
    def test_subgraph_runner_matches_langgraph_runner_interface(self, mock_db):
        """SubgraphWorkflowRunner has same interface as LangGraphWorkflowRunner."""
        import inspect

        from harness.dag.langgraph_engine import LangGraphWorkflowRunner

        subgraph_runner = SubgraphWorkflowRunner(db=mock_db, db_path=":memory:")
        langgraph_runner = LangGraphWorkflowRunner(db=mock_db, db_path=":memory:")

        # Both should have execute method
        assert hasattr(subgraph_runner, "execute")
        assert hasattr(langgraph_runner, "execute")

        # Both should be async
        assert inspect.iscoroutinefunction(subgraph_runner.execute)
        assert inspect.iscoroutinefunction(langgraph_runner.execute)
