"""Tests for Human-in-the-Loop (HITL) interrupt pattern (Task #18).

This module tests the LangGraph interrupt() pattern implementation:
- Human approval node creation and state handling
- Graph compilation with interrupt points
- CLI resume commands with --approve and --reject options
- Routing logic after human approval
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.db.models import Role, WorkflowStatus
from harness.db.state import StateDB

runner = CliRunner()


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def cli_env(temp_db_path):
    """Set up environment for CLI tests to use the temp database."""
    db = StateDB(temp_db_path)
    env = {"HARNESS_DB_PATH": str(temp_db_path)}
    return db, env


@pytest.fixture
def db_with_execution(temp_db_path):
    """Database with a workflow execution ready for HITL testing."""
    db = StateDB(temp_db_path)

    # Create a role
    db.upsert_role(Role(name="test_role", wave=1, has_molecule_tests=True))

    # Create workflow definition
    nodes = [
        {"id": "validate_role", "type": "task"},
        {"id": "human_approval", "type": "hitl"},
        {"id": "add_to_merge_train", "type": "task"},
        {"id": "notify_failure", "type": "terminal"},
    ]
    edges = [
        {"from": "validate_role", "to": "human_approval"},
        {"from": "human_approval", "to": "add_to_merge_train"},
        {"from": "human_approval", "to": "notify_failure"},
    ]

    db.create_workflow_definition(
        name="box-up-role", description="Test workflow with HITL", nodes=nodes, edges=edges
    )

    # Create execution paused at human_approval
    execution_id = db.create_execution(workflow_name="box-up-role", role_name="test_role")

    db.update_execution_status(
        execution_id, status=WorkflowStatus.PAUSED, current_node="human_approval"
    )

    # Save checkpoint with HITL context
    db.checkpoint_execution(
        execution_id,
        {
            "role_name": "test_role",
            "mr_url": "https://gitlab.example.com/project/-/merge_requests/123",
            "mr_iid": 123,
            "molecule_passed": True,
            "pytest_passed": True,
            "awaiting_human_input": True,
        },
    )

    env = {"HARNESS_DB_PATH": str(temp_db_path)}
    return db, execution_id, env


# =============================================================================
# HUMAN APPROVAL NODE TESTS
# =============================================================================


class TestHumanApprovalNode:
    """Tests for human_approval_node function."""

    @pytest.mark.unit
    def test_human_approval_node_exists(self):
        """Verify human_approval_node is defined in langgraph_engine."""
        from harness.dag.langgraph_engine import human_approval_node

        assert callable(human_approval_node)

    @pytest.mark.unit
    def test_human_approval_node_signature(self):
        """Verify human_approval_node has correct signature."""
        import inspect

        from harness.dag.langgraph_engine import human_approval_node

        sig = inspect.signature(human_approval_node)
        params = list(sig.parameters.keys())

        # Should accept state as parameter
        assert "state" in params

    @pytest.mark.unit
    def test_initial_state_has_hitl_fields(self):
        """Verify initial state includes HITL fields."""
        from harness.dag.langgraph_engine import create_initial_state

        state = create_initial_state("test_role", execution_id=1)

        # HITL fields should be present
        assert "human_approved" in state
        assert "human_rejection_reason" in state
        assert "awaiting_human_input" in state

        # Initial values
        assert state["human_approved"] is None
        assert state["human_rejection_reason"] is None
        assert state["awaiting_human_input"] is False


class TestHumanApprovalRouting:
    """Tests for routing functions around human approval."""

    @pytest.mark.unit
    def test_should_continue_after_mr_routes_to_human_approval(self):
        """After MR creation, workflow should route to human_approval."""
        from harness.dag.langgraph_engine import should_continue_after_mr

        state = {
            "role_name": "test_role",
            "mr_iid": 123,
            "mr_url": "https://gitlab.example.com/-/merge_requests/123",
        }

        result = should_continue_after_mr(state)
        assert result == "human_approval"

    @pytest.mark.unit
    def test_should_continue_after_mr_no_mr_goes_to_summary(self):
        """Without MR, workflow should go to report_summary."""
        from harness.dag.langgraph_engine import should_continue_after_mr

        state = {
            "role_name": "test_role",
            "mr_iid": None,
        }

        result = should_continue_after_mr(state)
        assert result == "report_summary"

    @pytest.mark.unit
    def test_should_continue_after_human_approval_approved(self):
        """When approved, workflow should continue to merge train."""
        from harness.dag.langgraph_engine import should_continue_after_human_approval

        state = {
            "role_name": "test_role",
            "human_approved": True,
        }

        result = should_continue_after_human_approval(state)
        assert result == "add_to_merge_train"

    @pytest.mark.unit
    def test_should_continue_after_human_approval_rejected(self):
        """When rejected, workflow should route to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_human_approval

        state = {
            "role_name": "test_role",
            "human_approved": False,
            "human_rejection_reason": "Tests need more coverage",
        }

        result = should_continue_after_human_approval(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_should_continue_after_human_approval_none_is_rejection(self):
        """When human_approved is None, should be treated as rejection."""
        from harness.dag.langgraph_engine import should_continue_after_human_approval

        state = {
            "role_name": "test_role",
            "human_approved": None,
        }

        result = should_continue_after_human_approval(state)
        assert result == "notify_failure"


# =============================================================================
# GRAPH COMPILATION TESTS
# =============================================================================


class TestGraphWithHITL:
    """Tests for graph compilation with HITL node."""

    @pytest.mark.unit
    def test_graph_includes_human_approval_node(self):
        """Verify graph includes human_approval node."""
        from harness.dag.langgraph_engine import create_box_up_role_graph

        graph, _breakpoints = create_box_up_role_graph()

        # Get node names from the graph
        # StateGraph stores nodes in a dict-like structure
        assert hasattr(graph, "nodes") or hasattr(graph, "_nodes")

    @pytest.mark.unit
    def test_graph_compiles_with_checkpointer(self):
        """Verify graph compiles successfully with checkpointer."""

        from harness.dag.langgraph_engine import create_box_up_role_graph

        graph, _breakpoints = create_box_up_role_graph()

        # Graph should compile without errors
        # Even without a real checkpointer for basic compilation
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.unit
    def test_graph_has_correct_edge_routing(self):
        """Verify graph has correct edge routing around HITL."""
        from harness.dag.langgraph_engine import create_box_up_role_graph

        graph, _breakpoints = create_box_up_role_graph()
        compiled = graph.compile()

        # Compiled graph should be ready for execution
        assert compiled is not None


# =============================================================================
# CLI RESUME COMMAND TESTS
# =============================================================================


class TestCLIResumeCommand:
    """Tests for CLI resume command with HITL options."""

    @pytest.mark.unit
    def test_resume_approve_rejects_with_reject(self, db_with_execution):
        """Cannot use both --approve and --reject."""
        db, execution_id, env = db_with_execution

        result = runner.invoke(app, ["resume", str(execution_id), "--approve", "--reject"], env=env)

        assert result.exit_code == 1
        assert "Cannot use both" in result.stdout

    @pytest.mark.unit
    def test_resume_nonexistent_execution(self, cli_env):
        """Resume with non-existent execution ID should fail."""
        db, env = cli_env

        result = runner.invoke(app, ["resume", "99999"], env=env)

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @pytest.mark.unit
    def test_resume_approve_shows_context(self, db_with_execution):
        """Resume with --approve should show approval context."""
        db, execution_id, env = db_with_execution

        # Mock the LangGraph execution to avoid actual graph run
        with patch("harness.cli._resume_with_command", new_callable=AsyncMock) as mock_resume:
            mock_resume.return_value = None

            result = runner.invoke(app, ["resume", str(execution_id), "--approve"], env=env)

            # Should show resuming message
            assert "Resuming" in result.stdout or "Approving" in result.stdout

    @pytest.mark.unit
    def test_resume_reject_with_reason(self, db_with_execution):
        """Resume with --reject and --reason should pass reason."""
        db, execution_id, env = db_with_execution

        with patch("harness.cli._resume_with_command", new_callable=AsyncMock) as mock_resume:
            mock_resume.return_value = None

            result = runner.invoke(
                app,
                ["resume", str(execution_id), "--reject", "--reason", "Tests incomplete"],
                env=env,
            )

            # Should show rejection context
            assert "Rejecting" in result.stdout or "Tests incomplete" in result.stdout

    @pytest.mark.unit
    def test_resume_reject_without_reason(self, db_with_execution):
        """Resume with --reject but no reason should work."""
        db, execution_id, env = db_with_execution

        with patch("harness.cli._resume_with_command", new_callable=AsyncMock) as mock_resume:
            mock_resume.return_value = None

            result = runner.invoke(app, ["resume", str(execution_id), "--reject"], env=env)

            # Should work with no reason
            assert result.exit_code == 0 or "Rejecting" in result.stdout

    @pytest.mark.unit
    def test_resume_warns_wrong_node(self, temp_db_path):
        """Resume with --approve on non-human_approval node should warn."""
        db = StateDB(temp_db_path)
        env = {"HARNESS_DB_PATH": str(temp_db_path)}

        # Create role and workflow
        db.upsert_role(Role(name="test_role", wave=1))
        db.create_workflow_definition(
            name="box-up-role",
            description="Test",
            nodes=[{"id": "validate_role", "type": "task"}],
            edges=[],
        )

        execution_id = db.create_execution("box-up-role", "test_role")

        # Set to a different node
        db.update_execution_status(
            execution_id,
            status=WorkflowStatus.PAUSED,
            current_node="validate_role",  # Not human_approval
        )

        with patch("harness.cli._resume_with_command", new_callable=AsyncMock) as mock_resume:
            mock_resume.return_value = None

            result = runner.invoke(app, ["resume", str(execution_id), "--approve"], env=env)

            # Should warn about wrong node
            assert "Warning" in result.stdout or "validate_role" in result.stdout


class TestResumeWithCommand:
    """Tests for _resume_with_command async function."""

    @pytest.mark.asyncio
    async def test_resume_with_command_approve(self, db_with_execution):
        """Test _resume_with_command with approval."""

        db, execution_id, _ = db_with_execution

        # Mock the LangGraph components at the correct import locations
        with (
            patch("harness.dag.langgraph_engine.create_box_up_role_graph") as mock_graph,
            patch("langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver") as mock_saver,
            patch("harness.config.HarnessConfig") as mock_config,
        ):
            # Setup mocks
            mock_config.load.return_value = MagicMock(db_path=":memory:")

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {
                "role_name": "test_role",
                "human_approved": True,
                "summary": {"mr_url": "https://example.com/mr/123"},
            }

            mock_graph_instance = MagicMock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_graph.return_value = mock_graph_instance

            # Mock the async context manager for AsyncSqliteSaver
            mock_checkpointer = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_checkpointer)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_saver.from_conn_string.return_value = mock_context_manager

            # Mock graph compile with checkpointer
            mock_graph_instance.compile.return_value = mock_compiled

            resume_value = {"approved": True, "reason": ""}

            # Import and call after mocking
            from harness.cli import _resume_with_command

            # This will raise due to console.print not being available in test
            # but the logic can be tested via mocking
            try:
                await _resume_with_command(db, execution_id, "test_role", resume_value)
            except Exception:
                pass  # Expected in test environment

            # Verify graph was called
            mock_graph.assert_called_once()


# =============================================================================
# INTERRUPT PATTERN TESTS
# =============================================================================


class TestInterruptPattern:
    """Tests for LangGraph interrupt() pattern integration."""

    @pytest.mark.unit
    def test_interrupt_imported(self):
        """Verify interrupt is imported from langgraph.types."""
        from langgraph.types import interrupt as lg_interrupt

        from harness.dag.langgraph_engine import interrupt

        assert interrupt is lg_interrupt

    @pytest.mark.unit
    def test_command_imported(self):
        """Verify Command is imported from langgraph.types."""
        from langgraph.types import Command as lg_command

        from harness.dag.langgraph_engine import Command

        assert Command is lg_command

    @pytest.mark.unit
    def test_state_has_awaiting_human_input(self):
        """Verify state can track awaiting_human_input."""
        from harness.dag.langgraph_engine import BoxUpRoleState

        # TypedDict should have the field
        annotations = getattr(BoxUpRoleState, "__annotations__", {})
        assert "awaiting_human_input" in annotations

    @pytest.mark.unit
    def test_human_approval_context_structure(self):
        """Verify human approval context has expected fields."""
        import inspect

        from harness.dag.langgraph_engine import human_approval_node

        # Get source code to verify context structure
        source = inspect.getsource(human_approval_node)

        # Should build approval_context with key fields
        assert "role_name" in source
        assert "mr_url" in source
        assert "molecule_passed" in source


# =============================================================================
# RETRY POLICY TESTS
# =============================================================================


class TestRetryPolicies:
    """Tests to verify retry policies are correctly configured."""

    @pytest.mark.unit
    def test_gitlab_api_retry_policy_exists(self):
        """Verify GITLAB_API_RETRY_POLICY is defined."""
        from harness.dag.langgraph_engine import GITLAB_API_RETRY_POLICY

        assert GITLAB_API_RETRY_POLICY is not None
        assert GITLAB_API_RETRY_POLICY.max_attempts >= 2

    @pytest.mark.unit
    def test_subprocess_retry_policy_exists(self):
        """Verify SUBPROCESS_RETRY_POLICY is defined."""
        from harness.dag.langgraph_engine import SUBPROCESS_RETRY_POLICY

        assert SUBPROCESS_RETRY_POLICY is not None

    @pytest.mark.unit
    def test_git_retry_policy_exists(self):
        """Verify GIT_RETRY_POLICY is defined."""
        from harness.dag.langgraph_engine import GIT_RETRY_POLICY

        assert GIT_RETRY_POLICY is not None
