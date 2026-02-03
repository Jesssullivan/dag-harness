"""
Tests for Unified Checkpointer (CheckpointerWithStateDB).

Tests cover:
- Wrapper correctly delegates to inner checkpointer
- Checkpoint puts sync to StateDB
- Metadata enrichment (version, timestamp)
- Factory function and context manager
- Graceful degradation when StateDB sync fails
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.dag.checkpointer_unified import (
    CheckpointerWithStateDB,
    CheckpointVersion,
    UnifiedCheckpointerContext,
    create_unified_checkpointer,
)
from harness.db.state import StateDB


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_statedb():
    """Create a mock StateDB."""
    db = MagicMock(spec=StateDB)
    db.checkpoint_execution = MagicMock()
    db.get_checkpoint = MagicMock(return_value=None)
    return db


@pytest.fixture
def mock_inner_checkpointer():
    """Create a mock inner checkpointer."""
    checkpointer = MagicMock()
    checkpointer.get_tuple = MagicMock(return_value=None)
    checkpointer.aget_tuple = AsyncMock(return_value=None)
    checkpointer.put = MagicMock(return_value={"configurable": {"thread_id": "123"}})
    checkpointer.aput = AsyncMock(return_value={"configurable": {"thread_id": "123"}})
    checkpointer.list = MagicMock(return_value=iter([]))
    checkpointer.alist = AsyncMock(return_value=[])
    checkpointer.put_writes = MagicMock()
    checkpointer.aput_writes = AsyncMock()
    return checkpointer


@pytest.fixture
def unified_checkpointer(mock_statedb, mock_inner_checkpointer):
    """Create a unified checkpointer with mocks."""
    return CheckpointerWithStateDB(
        db=mock_statedb,
        inner_checkpointer=mock_inner_checkpointer,
        sync_to_statedb=True,
    )


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint dict."""
    return {
        "v": 1,
        "ts": datetime.now(UTC).isoformat(),
        "id": "checkpoint-123",
        "channel_values": {
            "role_name": "common",
            "current_node": "validate_role",
            "completed_nodes": ["validate_role"],
            "errors": [],
        },
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


@pytest.fixture
def sample_config():
    """Create a sample LangGraph config."""
    return {"configurable": {"thread_id": "123"}}


@pytest.fixture
def sample_metadata():
    """Create sample checkpoint metadata."""
    return {"step": 1, "source": "input"}


# =============================================================================
# CHECKPOINT VERSION TESTS
# =============================================================================


class TestCheckpointVersion:
    """Test CheckpointVersion utility class."""

    @pytest.mark.unit
    def test_current_version(self):
        """Current version should be 2."""
        assert CheckpointVersion.CURRENT == 2

    @pytest.mark.unit
    def test_get_version_from_metadata(self):
        """Should extract version from metadata."""
        metadata = {"checkpoint_version": 2, "other": "data"}
        assert CheckpointVersion.get_version(metadata) == 2

    @pytest.mark.unit
    def test_get_version_default(self):
        """Should return 1 if no version in metadata."""
        assert CheckpointVersion.get_version(None) == 1
        assert CheckpointVersion.get_version({}) == 1
        assert CheckpointVersion.get_version({"other": "data"}) == 1


# =============================================================================
# CHECKPOINTER WITH STATEDB TESTS
# =============================================================================


class TestCheckpointerWithStateDB:
    """Test CheckpointerWithStateDB wrapper."""

    @pytest.mark.unit
    def test_init(self, mock_statedb, mock_inner_checkpointer):
        """Should initialize with db, inner checkpointer, and sync flag."""
        wrapper = CheckpointerWithStateDB(
            db=mock_statedb,
            inner_checkpointer=mock_inner_checkpointer,
            sync_to_statedb=True,
        )

        assert wrapper._db is mock_statedb
        assert wrapper._inner is mock_inner_checkpointer
        assert wrapper._sync_to_statedb is True

    @pytest.mark.unit
    def test_inner_property(self, unified_checkpointer, mock_inner_checkpointer):
        """Should expose inner checkpointer via property."""
        assert unified_checkpointer.inner is mock_inner_checkpointer

    @pytest.mark.unit
    def test_get_tuple_delegates(
        self, unified_checkpointer, mock_inner_checkpointer, sample_config
    ):
        """get_tuple should delegate to inner checkpointer."""
        unified_checkpointer.get_tuple(sample_config)
        mock_inner_checkpointer.get_tuple.assert_called_once_with(sample_config)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_aget_tuple_delegates(
        self, unified_checkpointer, mock_inner_checkpointer, sample_config
    ):
        """aget_tuple should delegate to inner checkpointer."""
        await unified_checkpointer.aget_tuple(sample_config)
        mock_inner_checkpointer.aget_tuple.assert_called_once_with(sample_config)

    @pytest.mark.unit
    def test_list_delegates(self, unified_checkpointer, mock_inner_checkpointer, sample_config):
        """list should delegate to inner checkpointer."""
        list(unified_checkpointer.list(sample_config))
        mock_inner_checkpointer.list.assert_called_once()

    @pytest.mark.unit
    def test_put_delegates_and_syncs(
        self,
        unified_checkpointer,
        mock_inner_checkpointer,
        mock_statedb,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """put should delegate to inner and sync to StateDB."""
        unified_checkpointer.put(
            sample_config, sample_checkpoint, sample_metadata, {}
        )

        # Should call inner checkpointer
        mock_inner_checkpointer.put.assert_called_once()

        # Should sync to StateDB
        mock_statedb.checkpoint_execution.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_aput_delegates_and_syncs(
        self,
        unified_checkpointer,
        mock_inner_checkpointer,
        mock_statedb,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """aput should delegate to inner and sync to StateDB."""
        await unified_checkpointer.aput(
            sample_config, sample_checkpoint, sample_metadata, {}
        )

        # Should call inner checkpointer
        mock_inner_checkpointer.aput.assert_called_once()

        # Should sync to StateDB
        mock_statedb.checkpoint_execution.assert_called_once()

    @pytest.mark.unit
    def test_put_without_sync(
        self,
        mock_statedb,
        mock_inner_checkpointer,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """put should not sync when sync_to_statedb=False."""
        wrapper = CheckpointerWithStateDB(
            db=mock_statedb,
            inner_checkpointer=mock_inner_checkpointer,
            sync_to_statedb=False,
        )

        wrapper.put(sample_config, sample_checkpoint, sample_metadata, {})

        # Should call inner checkpointer
        mock_inner_checkpointer.put.assert_called_once()

        # Should NOT sync to StateDB
        mock_statedb.checkpoint_execution.assert_not_called()

    @pytest.mark.unit
    def test_put_writes_delegates(
        self, unified_checkpointer, mock_inner_checkpointer, sample_config
    ):
        """put_writes should delegate to inner checkpointer."""
        writes = [("channel", "value")]
        unified_checkpointer.put_writes(sample_config, writes, "task-1")
        mock_inner_checkpointer.put_writes.assert_called_once_with(
            sample_config, writes, "task-1"
        )


# =============================================================================
# METADATA ENRICHMENT TESTS
# =============================================================================


class TestMetadataEnrichment:
    """Test metadata enrichment."""

    @pytest.mark.unit
    def test_enrich_metadata_adds_version(self, unified_checkpointer):
        """Should add checkpoint_version to metadata."""
        original = {"step": 1}
        enriched = unified_checkpointer._enrich_metadata(original)

        assert enriched["checkpoint_version"] == CheckpointVersion.CURRENT
        assert enriched["step"] == 1

    @pytest.mark.unit
    def test_enrich_metadata_adds_timestamp(self, unified_checkpointer):
        """Should add checkpoint_timestamp to metadata."""
        enriched = unified_checkpointer._enrich_metadata({})

        assert "checkpoint_timestamp" in enriched
        # Should be a valid ISO format timestamp
        timestamp = enriched["checkpoint_timestamp"]
        assert "T" in timestamp

    @pytest.mark.unit
    def test_enrich_metadata_handles_none(self, unified_checkpointer):
        """Should handle None metadata."""
        enriched = unified_checkpointer._enrich_metadata(None)

        assert enriched["checkpoint_version"] == CheckpointVersion.CURRENT
        assert "checkpoint_timestamp" in enriched


# =============================================================================
# EXECUTION ID EXTRACTION TESTS
# =============================================================================


class TestExecutionIdExtraction:
    """Test execution ID extraction from config."""

    @pytest.mark.unit
    def test_extract_int_thread_id(self, unified_checkpointer):
        """Should extract integer thread_id directly."""
        config = {"configurable": {"thread_id": 123}}
        assert unified_checkpointer._extract_execution_id(config) == 123

    @pytest.mark.unit
    def test_extract_string_thread_id(self, unified_checkpointer):
        """Should parse string thread_id as integer."""
        config = {"configurable": {"thread_id": "456"}}
        assert unified_checkpointer._extract_execution_id(config) == 456

    @pytest.mark.unit
    def test_extract_execution_format(self, unified_checkpointer):
        """Should extract from execution-123 format."""
        config = {"configurable": {"thread_id": "execution-789"}}
        assert unified_checkpointer._extract_execution_id(config) == 789

    @pytest.mark.unit
    def test_extract_returns_none_for_invalid(self, unified_checkpointer):
        """Should return None for non-parseable thread_id."""
        config = {"configurable": {"thread_id": "role-common-workflow"}}
        assert unified_checkpointer._extract_execution_id(config) is None

    @pytest.mark.unit
    def test_extract_returns_none_for_missing(self, unified_checkpointer):
        """Should return None if thread_id missing."""
        assert unified_checkpointer._extract_execution_id({}) is None
        assert unified_checkpointer._extract_execution_id({"configurable": {}}) is None


# =============================================================================
# STATE SUMMARY TESTS
# =============================================================================


class TestStateSummary:
    """Test state summarization for observability."""

    @pytest.mark.unit
    def test_summarize_state_extracts_observable_fields(self, unified_checkpointer):
        """Should extract key observable fields."""
        values = {
            "role_name": "common",
            "current_node": "validate_role",
            "completed_nodes": ["validate_role", "analyze_deps"],
            "errors": [],
            "molecule_passed": True,
            "other_field": "ignored",
        }

        summary = unified_checkpointer._summarize_state(values)

        assert summary["role_name"] == "common"
        assert summary["current_node"] == "validate_role"
        assert summary["completed_nodes"] == ["validate_role", "analyze_deps"]
        assert summary["molecule_passed"] is True
        assert "other_field" not in summary

    @pytest.mark.unit
    def test_summarize_state_truncates_long_lists(self, unified_checkpointer):
        """Should truncate lists longer than 10 items."""
        values = {
            "completed_nodes": [f"node_{i}" for i in range(20)],
        }

        summary = unified_checkpointer._summarize_state(values)

        assert len(summary["completed_nodes"]) == 11  # 10 items + "... (10 more)"
        assert "... (10 more)" in summary["completed_nodes"][-1]

    @pytest.mark.unit
    def test_summarize_empty_values(self, unified_checkpointer):
        """Should handle empty values dict."""
        summary = unified_checkpointer._summarize_state({})
        assert summary == {}


# =============================================================================
# STATEDB SYNC TESTS
# =============================================================================


class TestStateDBSync:
    """Test StateDB synchronization."""

    @pytest.mark.unit
    def test_sync_builds_checkpoint_data(
        self, unified_checkpointer, mock_statedb, sample_checkpoint, sample_metadata
    ):
        """Sync should build proper checkpoint data structure."""
        config = {"configurable": {"thread_id": "123"}}

        unified_checkpointer._sync_to_statedb_impl(
            config, sample_checkpoint, sample_metadata
        )

        # Verify checkpoint_execution was called with correct structure
        mock_statedb.checkpoint_execution.assert_called_once()
        call_args = mock_statedb.checkpoint_execution.call_args

        execution_id = call_args[0][0]
        checkpoint_data = call_args[0][1]

        assert execution_id == 123
        assert "version" in checkpoint_data
        assert "timestamp" in checkpoint_data
        assert "metadata" in checkpoint_data
        assert "state_summary" in checkpoint_data

    @pytest.mark.unit
    def test_sync_skips_if_no_execution_id(
        self, unified_checkpointer, mock_statedb, sample_checkpoint, sample_metadata
    ):
        """Sync should skip if execution_id cannot be extracted."""
        config = {"configurable": {"thread_id": "non-numeric"}}

        unified_checkpointer._sync_to_statedb_impl(
            config, sample_checkpoint, sample_metadata
        )

        # Should not call checkpoint_execution
        mock_statedb.checkpoint_execution.assert_not_called()

    @pytest.mark.unit
    def test_sync_handles_statedb_error(
        self, unified_checkpointer, mock_statedb, sample_checkpoint, sample_metadata
    ):
        """Sync should log error but not raise if StateDB fails."""
        config = {"configurable": {"thread_id": "123"}}
        mock_statedb.checkpoint_execution.side_effect = Exception("DB error")

        # Should not raise
        unified_checkpointer._sync_to_statedb_impl(
            config, sample_checkpoint, sample_metadata
        )


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestUnifiedCheckpointerContext:
    """Test UnifiedCheckpointerContext context manager."""

    @pytest.mark.unit
    def test_context_init(self, mock_statedb, tmp_path):
        """Should initialize with proper parameters."""
        sqlite_path = str(tmp_path / "test.db")

        context = UnifiedCheckpointerContext(
            db=mock_statedb,
            sqlite_path=sqlite_path,
            postgres_url=None,
            sync_to_statedb=True,
        )

        assert context._db is mock_statedb
        assert context._sqlite_path == sqlite_path
        assert context._sync_to_statedb is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_creates_checkpointer(self, tmp_path):
        """Context manager should create CheckpointerWithStateDB."""
        db = StateDB(":memory:")
        sqlite_path = str(tmp_path / "test.db")

        async with UnifiedCheckpointerContext(
            db=db,
            sqlite_path=sqlite_path,
            sync_to_statedb=True,
        ) as checkpointer:
            assert isinstance(checkpointer, CheckpointerWithStateDB)
            assert checkpointer._sync_to_statedb is True


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateUnifiedCheckpointer:
    """Test create_unified_checkpointer factory function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_creates_checkpointer(self, tmp_path):
        """Factory should create CheckpointerWithStateDB."""
        db = StateDB(":memory:")
        sqlite_path = str(tmp_path / "test.db")

        # Patch at the source module where the import happens
        with patch("harness.dag.checkpointer.CheckpointerFactory") as mock_factory:
            mock_inner = MagicMock()
            mock_factory.create_async = AsyncMock(return_value=mock_inner)

            checkpointer = await create_unified_checkpointer(
                db=db,
                sqlite_path=sqlite_path,
                sync_to_statedb=True,
            )

            assert isinstance(checkpointer, CheckpointerWithStateDB)
            assert checkpointer._db is db
            assert checkpointer._inner is mock_inner
            assert checkpointer._sync_to_statedb is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_factory_passes_postgres_url(self, tmp_path):
        """Factory should pass postgres_url to inner factory."""
        db = StateDB(":memory:")
        sqlite_path = str(tmp_path / "test.db")
        postgres_url = "postgresql://test:5432/db"

        # Patch at the source module where the import happens
        with patch("harness.dag.checkpointer.CheckpointerFactory") as mock_factory:
            mock_inner = MagicMock()
            mock_factory.create_async = AsyncMock(return_value=mock_inner)

            await create_unified_checkpointer(
                db=db,
                sqlite_path=sqlite_path,
                postgres_url=postgres_url,
            )

            mock_factory.create_async.assert_called_once_with(
                postgres_url=postgres_url,
                sqlite_path=sqlite_path,
                fallback_to_sqlite=True,
            )
