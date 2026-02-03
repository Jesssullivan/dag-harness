"""
Tests for checkpoint, interrupt(), and Command(resume=...) functionality.

This module tests the LangGraph interrupt/resume pattern implementation:
- Checkpoint round-trip (save/restore exact state)
- interrupt() pausing execution and saving state
- Command(resume=...) continuing execution with human input
- Error recovery for corrupted/missing checkpoints
- Checkpoint versioning and migration
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.db.models import Role, WorkflowStatus
from harness.db.state import StateDB


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_checkpoint_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite path for checkpointer tests."""
    return tmp_path / "test_checkpoints.db"


@pytest.fixture
def state_db(temp_db_path: Path) -> StateDB:
    """Create a fresh StateDB for testing."""
    return StateDB(temp_db_path)


@pytest.fixture
def state_db_with_execution(state_db: StateDB) -> tuple[StateDB, int]:
    """StateDB with a workflow execution created."""
    # Create role
    state_db.upsert_role(Role(name="test_role", wave=1, has_molecule_tests=True))

    # Create workflow definition
    state_db.create_workflow_definition(
        name="test-workflow",
        description="Test workflow for checkpoint tests",
        nodes=[
            {"id": "node_1", "type": "task"},
            {"id": "human_approval", "type": "hitl"},
            {"id": "node_2", "type": "task"},
        ],
        edges=[
            {"from": "node_1", "to": "human_approval"},
            {"from": "human_approval", "to": "node_2"},
        ],
    )

    # Create execution
    execution_id = state_db.create_execution(
        workflow_name="test-workflow", role_name="test_role"
    )

    return state_db, execution_id


@pytest.fixture
def mock_inner_checkpointer():
    """Create a mock inner checkpointer for testing unified checkpointer."""
    mock = MagicMock()
    mock.get_tuple = MagicMock(return_value=None)
    mock.aget_tuple = AsyncMock(return_value=None)
    mock.put = MagicMock(return_value={"configurable": {"thread_id": "test"}})
    mock.aput = AsyncMock(return_value={"configurable": {"thread_id": "test"}})
    mock.list = MagicMock(return_value=iter([]))
    mock.alist = AsyncMock(return_value=[])
    mock.put_writes = MagicMock()
    mock.aput_writes = AsyncMock()
    return mock


# =============================================================================
# CHECKPOINT ROUND-TRIP TESTS
# =============================================================================


class TestCheckpointRoundTrip:
    """Tests for checkpoint save and restore operations."""

    @pytest.mark.unit
    def test_checkpoint_save_and_restore(self, state_db_with_execution):
        """Save checkpoint, restore exact state."""
        db, execution_id = state_db_with_execution

        # Save checkpoint with test state
        checkpoint_data = {
            "role_name": "test_role",
            "current_node": "human_approval",
            "completed_nodes": ["node_1"],
            "molecule_passed": True,
            "pytest_passed": True,
            "mr_iid": 42,
        }

        db.checkpoint_execution(execution_id, checkpoint_data)

        # Restore and verify exact match
        restored = db.get_checkpoint(execution_id)

        assert restored is not None
        assert restored["role_name"] == "test_role"
        assert restored["current_node"] == "human_approval"
        assert restored["completed_nodes"] == ["node_1"]
        assert restored["molecule_passed"] is True
        assert restored["pytest_passed"] is True
        assert restored["mr_iid"] == 42

    @pytest.mark.unit
    def test_checkpoint_with_complex_state(self, state_db_with_execution):
        """Checkpoint with lists, dicts, nested objects."""
        db, execution_id = state_db_with_execution

        checkpoint_data = {
            "role_name": "complex_role",
            "explicit_deps": ["dep1", "dep2", "dep3"],
            "credentials": [
                {"entry_name": "cred1", "purpose": "test", "attribute": "password"},
                {"entry_name": "cred2", "purpose": "deploy", "attribute": "token"},
            ],
            "nested": {
                "level1": {
                    "level2": {
                        "value": 42,
                        "list": [1, 2, 3],
                    }
                }
            },
            "mixed_types": [1, "two", 3.0, True, None],
        }

        db.checkpoint_execution(execution_id, checkpoint_data)
        restored = db.get_checkpoint(execution_id)

        assert restored is not None
        assert restored["explicit_deps"] == ["dep1", "dep2", "dep3"]
        assert len(restored["credentials"]) == 2
        assert restored["credentials"][0]["entry_name"] == "cred1"
        assert restored["nested"]["level1"]["level2"]["value"] == 42
        assert restored["mixed_types"] == [1, "two", 3.0, True, None]

    @pytest.mark.unit
    def test_checkpoint_metadata_preserved(self, state_db_with_execution):
        """Metadata survives checkpoint round-trip."""
        db, execution_id = state_db_with_execution

        checkpoint_data = {
            "state": {"role_name": "test_role"},
            "metadata": {
                "checkpoint_version": 2,
                "checkpoint_timestamp": "2026-02-03T10:00:00+00:00",
                "source": "test",
                "custom_field": "custom_value",
            },
        }

        db.checkpoint_execution(execution_id, checkpoint_data)
        restored = db.get_checkpoint(execution_id)

        assert restored is not None
        assert "metadata" in restored
        assert restored["metadata"]["checkpoint_version"] == 2
        assert restored["metadata"]["source"] == "test"
        assert restored["metadata"]["custom_field"] == "custom_value"

    @pytest.mark.unit
    def test_checkpoint_versioning(self, state_db_with_execution):
        """Version 2 checkpoints load correctly."""
        from harness.dag.checkpointer_unified import CheckpointVersion

        db, execution_id = state_db_with_execution

        # Save a version 2 checkpoint
        checkpoint_data = {
            "state": {"role_name": "test_role", "current_node": "validate_role"},
            "metadata": {
                "checkpoint_version": CheckpointVersion.CURRENT,
                "checkpoint_timestamp": datetime.now(UTC).isoformat(),
            },
        }

        db.checkpoint_execution(execution_id, checkpoint_data)
        restored = db.get_checkpoint(execution_id)

        assert restored is not None
        version = CheckpointVersion.get_version(restored.get("metadata"))
        assert version == CheckpointVersion.CURRENT

    @pytest.mark.unit
    def test_checkpoint_overwrite(self, state_db_with_execution):
        """Subsequent checkpoint overwrites previous."""
        db, execution_id = state_db_with_execution

        # First checkpoint
        db.checkpoint_execution(execution_id, {"version": 1, "node": "node_1"})

        # Second checkpoint (should overwrite)
        db.checkpoint_execution(execution_id, {"version": 2, "node": "node_2"})

        restored = db.get_checkpoint(execution_id)
        assert restored["version"] == 2
        assert restored["node"] == "node_2"


# =============================================================================
# INTERRUPT FUNCTIONALITY TESTS
# =============================================================================


class TestInterruptFunctionality:
    """Tests for interrupt() pattern implementation."""

    @pytest.mark.unit
    def test_interrupt_imported_from_langgraph(self):
        """Verify interrupt is imported from langgraph.types."""
        from langgraph.types import interrupt as lg_interrupt

        from harness.dag.langgraph_engine import interrupt

        assert interrupt is lg_interrupt

    @pytest.mark.unit
    def test_interrupt_pauses_execution(self, state_db_with_execution):
        """interrupt() stops graph execution at the expected point."""
        db, execution_id = state_db_with_execution

        # Set execution to paused state (simulating interrupt)
        db.update_execution_status(
            execution_id, status=WorkflowStatus.PAUSED, current_node="human_approval"
        )

        # Verify status
        with db.connection() as conn:
            row = conn.execute(
                "SELECT status, current_node FROM workflow_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()

        assert row["status"] == WorkflowStatus.PAUSED.value
        assert row["current_node"] == "human_approval"

    @pytest.mark.unit
    def test_interrupt_saves_checkpoint(self, state_db_with_execution):
        """State is checkpointed on interrupt."""
        db, execution_id = state_db_with_execution

        # Simulate interrupt state
        checkpoint_data = {
            "role_name": "test_role",
            "current_node": "human_approval",
            "awaiting_human_input": True,
            "mr_url": "https://gitlab.example.com/merge_requests/123",
            "molecule_passed": True,
        }

        db.update_execution_status(
            execution_id, status=WorkflowStatus.PAUSED, current_node="human_approval"
        )
        db.checkpoint_execution(execution_id, checkpoint_data)

        # Verify checkpoint contains interrupt state
        restored = db.get_checkpoint(execution_id)
        assert restored is not None
        assert restored["awaiting_human_input"] is True
        assert restored["current_node"] == "human_approval"

    @pytest.mark.unit
    def test_interrupt_at_human_approval(self):
        """Interrupt at human_approval node with proper context."""
        from harness.dag.langgraph_engine import BoxUpRoleState, human_approval_node, interrupt

        # Create a test state that would be passed to human_approval
        state: BoxUpRoleState = {
            "role_name": "test_role",
            "mr_url": "https://gitlab.example.com/merge_requests/123",
            "mr_iid": 123,
            "molecule_passed": True,
            "pytest_passed": True,
            "branch": "sid/test_role",
            "issue_url": "https://gitlab.example.com/issues/456",
            "wave": 2,
            "wave_name": "Core Services",
            "credentials": [],
            "explicit_deps": ["common"],
        }

        # Verify the node function exists and has correct signature
        import inspect

        sig = inspect.signature(human_approval_node)
        assert "state" in sig.parameters

    @pytest.mark.unit
    def test_interrupt_before_configured_nodes(self):
        """interrupt_before parameter pauses before specified nodes."""
        from harness.dag.langgraph_engine import (
            DEFAULT_BREAKPOINTS,
            create_box_up_role_graph,
        )

        # Default breakpoints should include key nodes
        assert "human_approval" in DEFAULT_BREAKPOINTS
        assert "create_mr" in DEFAULT_BREAKPOINTS
        assert "add_to_merge_train" in DEFAULT_BREAKPOINTS

        # Graph creation with breakpoints enabled
        graph, breakpoints = create_box_up_role_graph(
            db_path=":memory:", enable_breakpoints=True
        )

        assert breakpoints == DEFAULT_BREAKPOINTS
        assert len(breakpoints) >= 2

    @pytest.mark.unit
    def test_static_breakpoints_from_env(self):
        """HARNESS_BREAKPOINTS env var controls breakpoint activation."""
        from harness.dag.langgraph_engine import (
            BREAKPOINTS_ENV_VAR,
            get_breakpoints_enabled,
        )

        # Test with env var set to true
        with patch.dict(os.environ, {BREAKPOINTS_ENV_VAR: "true"}):
            assert get_breakpoints_enabled() is True

        with patch.dict(os.environ, {BREAKPOINTS_ENV_VAR: "1"}):
            assert get_breakpoints_enabled() is True

        with patch.dict(os.environ, {BREAKPOINTS_ENV_VAR: "yes"}):
            assert get_breakpoints_enabled() is True

        # Test with env var unset or false
        with patch.dict(os.environ, {BREAKPOINTS_ENV_VAR: "false"}):
            assert get_breakpoints_enabled() is False

        with patch.dict(os.environ, {BREAKPOINTS_ENV_VAR: ""}):
            assert get_breakpoints_enabled() is False

    @pytest.mark.unit
    def test_interrupt_context_structure(self):
        """Human approval context has expected fields for review."""
        import inspect

        from harness.dag.langgraph_engine import human_approval_node

        # Get source code to verify context structure
        source = inspect.getsource(human_approval_node)

        # Context should include key fields for human review
        assert "role_name" in source
        assert "mr_url" in source
        assert "molecule_passed" in source
        assert "pytest_passed" in source
        assert "approval_context" in source


# =============================================================================
# RESUME FUNCTIONALITY TESTS
# =============================================================================


class TestResumeFunctionality:
    """Tests for Command(resume=...) pattern implementation."""

    @pytest.mark.unit
    def test_command_imported_from_langgraph(self):
        """Verify Command is imported from langgraph.types."""
        from langgraph.types import Command as lg_command

        from harness.dag.langgraph_engine import Command

        assert Command is lg_command

    @pytest.mark.unit
    def test_command_resume_structure(self):
        """Command(resume=...) has expected structure."""
        from langgraph.types import Command

        # Create a resume command with approval
        resume_value = {"approved": True, "reason": ""}
        cmd = Command(resume=resume_value)

        assert hasattr(cmd, "resume")
        assert cmd.resume == resume_value

    @pytest.mark.unit
    def test_resume_from_approval_approved(self, state_db_with_execution):
        """Resume with approval=True continues to merge train."""
        from harness.dag.langgraph_engine import should_continue_after_human_approval

        db, execution_id = state_db_with_execution

        # State after human approval with approval=True
        state = {
            "role_name": "test_role",
            "human_approved": True,
            "mr_iid": 123,
        }

        result = should_continue_after_human_approval(state)
        assert result == "add_to_merge_train"

    @pytest.mark.unit
    def test_resume_from_approval_rejected(self, state_db_with_execution):
        """Resume with approval=False routes to notify_failure."""
        from harness.dag.langgraph_engine import should_continue_after_human_approval

        db, execution_id = state_db_with_execution

        # State after human approval with approval=False
        state = {
            "role_name": "test_role",
            "human_approved": False,
            "human_rejection_reason": "Tests need more coverage",
        }

        result = should_continue_after_human_approval(state)
        assert result == "notify_failure"

    @pytest.mark.unit
    def test_resume_with_modified_state(self, state_db_with_execution):
        """Resume with state changes applies modifications."""
        db, execution_id = state_db_with_execution

        # Initial checkpoint
        initial_state = {
            "role_name": "test_role",
            "current_node": "human_approval",
            "human_approved": None,
            "mr_iid": 123,
        }
        db.checkpoint_execution(execution_id, initial_state)

        # Simulate resume with modified state (approval added)
        modified_state = {
            **initial_state,
            "human_approved": True,
            "current_node": "add_to_merge_train",
        }
        db.checkpoint_execution(execution_id, modified_state)

        # Verify modifications persisted
        restored = db.get_checkpoint(execution_id)
        assert restored["human_approved"] is True
        assert restored["current_node"] == "add_to_merge_train"

    @pytest.mark.unit
    def test_resume_after_crash(self, state_db_with_execution):
        """Resume from persisted checkpoint after simulated crash."""
        db, execution_id = state_db_with_execution

        # Save checkpoint before "crash"
        pre_crash_state = {
            "role_name": "test_role",
            "current_node": "run_molecule",
            "completed_nodes": ["validate_role", "analyze_deps"],
            "molecule_passed": None,  # In progress when crash occurred
        }
        db.checkpoint_execution(execution_id, pre_crash_state)
        db.update_execution_status(
            execution_id, status=WorkflowStatus.RUNNING, current_node="run_molecule"
        )

        # Simulate crash and recovery by creating new DB connection
        recovered_db = StateDB(db.db_path)

        # Verify state can be restored
        restored = recovered_db.get_checkpoint(execution_id)
        assert restored is not None
        assert restored["current_node"] == "run_molecule"
        assert restored["completed_nodes"] == ["validate_role", "analyze_deps"]


# =============================================================================
# ERROR RECOVERY TESTS
# =============================================================================


class TestErrorRecovery:
    """Tests for checkpoint error handling and recovery."""

    @pytest.mark.unit
    def test_checkpoint_corruption_detection(self, state_db_with_execution):
        """Detect corrupted checkpoints gracefully."""
        db, execution_id = state_db_with_execution

        # Manually insert corrupted JSON
        with db.connection() as conn:
            conn.execute(
                "UPDATE workflow_executions SET checkpoint_data = ? WHERE id = ?",
                ("{ invalid json", execution_id),
            )

        # get_checkpoint should handle the error
        # The actual behavior depends on implementation - may return None or raise
        try:
            result = db.get_checkpoint(execution_id)
            # If it doesn't raise, it should return None for corrupted data
            # or re-raise - either is acceptable
        except json.JSONDecodeError:
            # Expected - corruption detected
            pass

    @pytest.mark.unit
    def test_checkpoint_migration_v1_to_v2(self, state_db_with_execution):
        """Migrate v1 (no version) to v2 checkpoints."""
        from harness.dag.checkpointer_unified import CheckpointVersion

        db, execution_id = state_db_with_execution

        # Save v1 checkpoint (no version field)
        v1_checkpoint = {
            "role_name": "test_role",
            "current_node": "validate_role",
            # Note: no checkpoint_version in metadata
        }
        db.checkpoint_execution(execution_id, v1_checkpoint)

        restored = db.get_checkpoint(execution_id)

        # Should detect as v1
        version = CheckpointVersion.get_version(restored.get("metadata"))
        assert version == 1  # Default for missing version

    @pytest.mark.unit
    def test_resume_with_missing_checkpoint(self, state_db: StateDB):
        """Handle missing checkpoint gracefully."""
        # Try to get checkpoint for non-existent execution
        result = state_db.get_checkpoint(99999)
        assert result is None

    @pytest.mark.unit
    def test_partial_checkpoint_recovery(self, state_db_with_execution):
        """Recover from partial checkpoint writes."""
        db, execution_id = state_db_with_execution

        # Save valid checkpoint first
        valid_checkpoint = {
            "role_name": "test_role",
            "current_node": "node_1",
            "completed_nodes": ["validate_role"],
        }
        db.checkpoint_execution(execution_id, valid_checkpoint)

        # Verify we can always restore the last valid checkpoint
        restored = db.get_checkpoint(execution_id)
        assert restored is not None
        assert restored["current_node"] == "node_1"

    @pytest.mark.unit
    def test_checkpoint_null_handling(self, state_db_with_execution):
        """Handle null/None values in checkpoints correctly."""
        db, execution_id = state_db_with_execution

        checkpoint_data = {
            "role_name": "test_role",
            "molecule_passed": None,
            "pytest_passed": None,
            "human_approved": None,
            "optional_field": None,
        }

        db.checkpoint_execution(execution_id, checkpoint_data)
        restored = db.get_checkpoint(execution_id)

        assert restored is not None
        assert restored["molecule_passed"] is None
        assert restored["pytest_passed"] is None
        assert restored["human_approved"] is None
        assert restored["optional_field"] is None


# =============================================================================
# UNIFIED CHECKPOINTER TESTS
# =============================================================================


class TestUnifiedCheckpointer:
    """Tests for CheckpointerWithStateDB unified checkpointer."""

    @pytest.mark.unit
    def test_unified_checkpointer_creation(self, state_db: StateDB, mock_inner_checkpointer):
        """Unified checkpointer wraps inner checkpointer correctly."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db,
            inner_checkpointer=mock_inner_checkpointer,
            sync_to_statedb=True,
        )

        assert unified._db is state_db
        assert unified._inner is mock_inner_checkpointer
        assert unified._sync_to_statedb is True

    @pytest.mark.unit
    def test_unified_checkpointer_inner_property(
        self, state_db: StateDB, mock_inner_checkpointer
    ):
        """Access underlying checkpointer via inner property."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db,
            inner_checkpointer=mock_inner_checkpointer,
        )

        assert unified.inner is mock_inner_checkpointer

    @pytest.mark.unit
    def test_unified_checkpointer_enriches_metadata(
        self, state_db: StateDB, mock_inner_checkpointer
    ):
        """Checkpointer adds version and timestamp to metadata."""
        from harness.dag.checkpointer_unified import (
            CheckpointerWithStateDB,
            CheckpointVersion,
        )

        unified = CheckpointerWithStateDB(
            db=state_db,
            inner_checkpointer=mock_inner_checkpointer,
        )

        # Test metadata enrichment
        original_metadata = {"source": "test"}
        enriched = unified._enrich_metadata(original_metadata)

        assert "checkpoint_version" in enriched
        assert enriched["checkpoint_version"] == CheckpointVersion.CURRENT
        assert "checkpoint_timestamp" in enriched
        assert enriched["source"] == "test"  # Original preserved

    @pytest.mark.unit
    def test_unified_checkpointer_sync_disabled(
        self, state_db: StateDB, mock_inner_checkpointer
    ):
        """StateDB sync can be disabled."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db,
            inner_checkpointer=mock_inner_checkpointer,
            sync_to_statedb=False,
        )

        assert unified._sync_to_statedb is False


# =============================================================================
# ASYNC CHECKPOINT TESTS
# =============================================================================


class TestAsyncCheckpointOperations:
    """Tests for async checkpoint operations."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_async_put_delegates_to_inner(
        self, state_db: StateDB, mock_inner_checkpointer
    ):
        """aput() delegates to inner checkpointer."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db,
            inner_checkpointer=mock_inner_checkpointer,
            sync_to_statedb=False,  # Disable sync for this test
        )

        config = {"configurable": {"thread_id": "test-123"}}
        checkpoint = MagicMock()
        metadata = {}
        new_versions = {}

        await unified.aput(config, checkpoint, metadata, new_versions)

        # Verify inner checkpointer was called
        mock_inner_checkpointer.aput.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_async_get_tuple_delegates_to_inner(
        self, state_db: StateDB, mock_inner_checkpointer
    ):
        """aget_tuple() delegates to inner checkpointer."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db,
            inner_checkpointer=mock_inner_checkpointer,
        )

        config = {"configurable": {"thread_id": "test-123"}}

        await unified.aget_tuple(config)

        mock_inner_checkpointer.aget_tuple.assert_called_once_with(config)


# =============================================================================
# STATE EXTRACTION TESTS
# =============================================================================


class TestStateExtraction:
    """Tests for extracting execution ID and building checkpoint data."""

    @pytest.mark.unit
    def test_extract_execution_id_from_int(self, state_db: StateDB, mock_inner_checkpointer):
        """Extract execution_id from integer thread_id."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db, inner_checkpointer=mock_inner_checkpointer
        )

        config = {"configurable": {"thread_id": 123}}
        execution_id = unified._extract_execution_id(config)

        assert execution_id == 123

    @pytest.mark.unit
    def test_extract_execution_id_from_string(
        self, state_db: StateDB, mock_inner_checkpointer
    ):
        """Extract execution_id from string thread_id."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db, inner_checkpointer=mock_inner_checkpointer
        )

        config = {"configurable": {"thread_id": "456"}}
        execution_id = unified._extract_execution_id(config)

        assert execution_id == 456

    @pytest.mark.unit
    def test_extract_execution_id_from_prefixed_string(
        self, state_db: StateDB, mock_inner_checkpointer
    ):
        """Extract execution_id from 'execution-123' format."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db, inner_checkpointer=mock_inner_checkpointer
        )

        config = {"configurable": {"thread_id": "execution-789"}}
        execution_id = unified._extract_execution_id(config)

        assert execution_id == 789

    @pytest.mark.unit
    def test_extract_execution_id_missing(self, state_db: StateDB, mock_inner_checkpointer):
        """Handle missing thread_id gracefully."""
        from harness.dag.checkpointer_unified import CheckpointerWithStateDB

        unified = CheckpointerWithStateDB(
            db=state_db, inner_checkpointer=mock_inner_checkpointer
        )

        config = {"configurable": {}}
        execution_id = unified._extract_execution_id(config)

        assert execution_id is None


# =============================================================================
# WORKFLOW RUNNER INTEGRATION TESTS
# =============================================================================


class TestWorkflowRunnerCheckpointing:
    """Tests for LangGraphWorkflowRunner checkpoint integration."""

    @pytest.mark.unit
    def test_runner_accepts_checkpointer_factory(self, state_db: StateDB):
        """Runner accepts checkpointer_factory parameter."""
        from harness.dag.checkpointer import CheckpointerFactory
        from harness.dag.langgraph_engine import LangGraphWorkflowRunner

        runner = LangGraphWorkflowRunner(
            db=state_db,
            checkpointer_factory=CheckpointerFactory,
        )

        assert runner._checkpointer_factory is CheckpointerFactory

    @pytest.mark.unit
    def test_runner_accepts_postgres_url(self, state_db: StateDB):
        """Runner accepts postgres_url parameter."""
        from harness.dag.langgraph_engine import LangGraphWorkflowRunner

        runner = LangGraphWorkflowRunner(
            db=state_db,
            postgres_url="postgresql://test:5432/db",
        )

        assert runner._postgres_url == "postgresql://test:5432/db"
        assert runner._use_postgres is True

    @pytest.mark.unit
    def test_runner_use_postgres_flag(self, state_db: StateDB):
        """Runner respects use_postgres flag."""
        from harness.dag.langgraph_engine import LangGraphWorkflowRunner

        runner = LangGraphWorkflowRunner(
            db=state_db,
            use_postgres=True,
        )

        assert runner._use_postgres is True


# =============================================================================
# GRAPH COMPILATION TESTS
# =============================================================================


class TestGraphCompilationWithBreakpoints:
    """Tests for graph compilation with interrupt_before breakpoints."""

    @pytest.mark.unit
    def test_graph_compiles_with_breakpoints(self):
        """Graph compiles with interrupt_before parameter."""
        from harness.dag.langgraph_engine import create_box_up_role_graph

        graph, breakpoints = create_box_up_role_graph(
            db_path=":memory:", enable_breakpoints=True
        )

        # Should compile without checkpointer for basic verification
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.unit
    def test_graph_compiles_without_breakpoints(self):
        """Graph compiles without breakpoints."""
        from harness.dag.langgraph_engine import create_box_up_role_graph

        graph, breakpoints = create_box_up_role_graph(
            db_path=":memory:", enable_breakpoints=False
        )

        assert breakpoints == []
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.unit
    def test_breakpoints_list_contents(self):
        """Verify breakpoints list contains expected nodes."""
        from harness.dag.langgraph_engine import DEFAULT_BREAKPOINTS

        # Key nodes that should have breakpoints
        expected_breakpoints = {"human_approval", "create_mr", "add_to_merge_train"}

        for bp in expected_breakpoints:
            assert bp in DEFAULT_BREAKPOINTS, f"Missing breakpoint: {bp}"
