"""
Unified Checkpointer that coordinates LangGraph checkpointing with StateDB.

This module bridges the dual persistence gap by wrapping LangGraph's native
checkpointer and synchronizing state with the application's StateDB.

Design Philosophy:
- LangGraph checkpointer remains the source of truth for graph state
- StateDB stores a copy in workflow_executions.checkpoint_data for observability
- Both systems are updated atomically where possible
- Graceful degradation if StateDB sync fails (LangGraph checkpoint still works)

Usage:
    async with CheckpointerWithStateDB(db, sqlite_path="harness.db") as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        await compiled.ainvoke(initial_state, config)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

if TYPE_CHECKING:
    from harness.db.state import StateDB

logger = logging.getLogger(__name__)


class CheckpointVersion:
    """Version tracking for checkpoint schema evolution."""

    CURRENT = 2

    @staticmethod
    def get_version(metadata: dict | None) -> int:
        """Extract version from checkpoint metadata."""
        if not metadata:
            return 1
        return metadata.get("checkpoint_version", 1)


class CheckpointerWithStateDB(BaseCheckpointSaver):
    """
    Unified checkpointer that wraps LangGraph checkpointer and syncs with StateDB.

    This class:
    1. Delegates all checkpoint operations to the underlying LangGraph checkpointer
    2. On put(), also updates StateDB.workflow_executions.checkpoint_data
    3. Adds version tracking to checkpoint metadata
    4. Provides observability via StateDB for external monitoring

    Thread Safety:
    - The underlying checkpointer handles its own thread safety
    - StateDB operations are connection-per-call, so thread-safe
    - Sync failures are logged but don't block checkpoint operations
    """

    def __init__(
        self,
        db: StateDB,
        inner_checkpointer: BaseCheckpointSaver,
        sync_to_statedb: bool = True,
    ):
        """
        Initialize unified checkpointer.

        Args:
            db: StateDB instance for application state persistence.
            inner_checkpointer: LangGraph checkpointer to wrap.
            sync_to_statedb: If True, sync checkpoints to StateDB (default True).
        """
        super().__init__()
        self._db = db
        self._inner = inner_checkpointer
        self._sync_to_statedb = sync_to_statedb

    @property
    def inner(self) -> BaseCheckpointSaver:
        """Access the underlying LangGraph checkpointer."""
        return self._inner

    # =========================================================================
    # REQUIRED ABSTRACT METHODS FROM BaseCheckpointSaver
    # =========================================================================

    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        """Get checkpoint tuple (sync version)."""
        return self._inner.get_tuple(config)

    async def aget_tuple(self, config: dict) -> CheckpointTuple | None:
        """Get checkpoint tuple (async version)."""
        if hasattr(self._inner, "aget_tuple"):
            return await self._inner.aget_tuple(config)
        return self._inner.get_tuple(config)

    def list(
        self,
        config: dict | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints (sync version)."""
        return self._inner.list(config, filter=filter, before=before, limit=limit)

    async def alist(
        self,
        config: dict | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> list[CheckpointTuple]:
        """List checkpoints (async version)."""
        if hasattr(self._inner, "alist"):
            result = []
            async for item in self._inner.alist(config, filter=filter, before=before, limit=limit):
                result.append(item)
            return result
        return list(self._inner.list(config, filter=filter, before=before, limit=limit))

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        """
        Store checkpoint (sync version).

        Also syncs to StateDB if enabled.
        """
        # Add version to metadata
        enriched_metadata = self._enrich_metadata(metadata)

        # Delegate to inner checkpointer
        result = self._inner.put(config, checkpoint, enriched_metadata, new_versions)

        # Sync to StateDB
        if self._sync_to_statedb:
            self._sync_to_statedb_impl(config, checkpoint, enriched_metadata)

        return result

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        """
        Store checkpoint (async version).

        Also syncs to StateDB if enabled.
        """
        # Add version to metadata
        enriched_metadata = self._enrich_metadata(metadata)

        # Delegate to inner checkpointer
        if hasattr(self._inner, "aput"):
            result = await self._inner.aput(config, checkpoint, enriched_metadata, new_versions)
        else:
            result = self._inner.put(config, checkpoint, enriched_metadata, new_versions)

        # Sync to StateDB
        if self._sync_to_statedb:
            await self._async_sync_to_statedb(config, checkpoint, enriched_metadata)

        return result

    def put_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes (sync version)."""
        if hasattr(self._inner, "put_writes"):
            self._inner.put_writes(config, writes, task_id)

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes (async version)."""
        if hasattr(self._inner, "aput_writes"):
            await self._inner.aput_writes(config, writes, task_id)
        elif hasattr(self._inner, "put_writes"):
            self._inner.put_writes(config, writes, task_id)

    # =========================================================================
    # STATEDB SYNC IMPLEMENTATION
    # =========================================================================

    def _enrich_metadata(self, metadata: CheckpointMetadata) -> CheckpointMetadata:
        """Add version and timestamp to checkpoint metadata."""
        enriched = dict(metadata) if metadata else {}
        enriched["checkpoint_version"] = CheckpointVersion.CURRENT
        enriched["checkpoint_timestamp"] = datetime.now(UTC).isoformat()
        return enriched

    def _sync_to_statedb_impl(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> None:
        """Sync checkpoint to StateDB (sync version)."""
        try:
            execution_id = self._extract_execution_id(config)
            if execution_id is None:
                return

            # Build checkpoint data for StateDB
            checkpoint_data = self._build_checkpoint_data(checkpoint, metadata)

            # Update StateDB
            self._db.checkpoint_execution(execution_id, checkpoint_data)

            logger.debug(f"Synced checkpoint to StateDB for execution {execution_id}")

        except Exception as e:
            # Log but don't fail - LangGraph checkpoint is the source of truth
            logger.warning(f"Failed to sync checkpoint to StateDB: {e}")

    async def _async_sync_to_statedb(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> None:
        """Sync checkpoint to StateDB (async version)."""
        # StateDB is sync-only, so we just call the sync method
        # In a production system, this could use run_in_executor
        self._sync_to_statedb_impl(config, checkpoint, metadata)

    def _extract_execution_id(self, config: dict) -> int | None:
        """Extract execution_id from LangGraph config."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")

        if thread_id is None:
            return None

        # thread_id can be "123" or "role-common-123" format
        if isinstance(thread_id, int):
            return thread_id

        if isinstance(thread_id, str):
            # Try direct integer parse
            try:
                return int(thread_id)
            except ValueError:
                pass

            # Try extracting from "execution-123" format
            if thread_id.startswith("execution-"):
                try:
                    return int(thread_id.split("-")[1])
                except (IndexError, ValueError):
                    pass

        return None

    def _build_checkpoint_data(
        self,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> dict:
        """Build checkpoint data dict for StateDB storage."""
        # Extract state values from checkpoint
        values = {}
        if hasattr(checkpoint, "channel_values"):
            values = checkpoint.channel_values
        elif isinstance(checkpoint, dict):
            values = checkpoint.get("channel_values", checkpoint)

        return {
            "version": CheckpointVersion.CURRENT,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": dict(metadata) if metadata else {},
            "state_summary": self._summarize_state(values),
        }

    def _summarize_state(self, values: dict) -> dict:
        """Create a summary of state for observability (not full state)."""
        summary = {}

        # Extract key fields for observability
        observable_fields = [
            "role_name",
            "current_node",
            "completed_nodes",
            "errors",
            "molecule_passed",
            "pytest_passed",
            "issue_iid",
            "mr_iid",
            "human_approved",
        ]

        for field in observable_fields:
            if field in values:
                value = values[field]
                # Truncate long lists for storage efficiency
                if isinstance(value, list) and len(value) > 10:
                    summary[field] = value[:10] + [f"... ({len(value) - 10} more)"]
                else:
                    summary[field] = value

        return summary


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


async def create_unified_checkpointer(
    db: StateDB,
    sqlite_path: str = "harness.db",
    postgres_url: str | None = None,
    sync_to_statedb: bool = True,
) -> CheckpointerWithStateDB:
    """
    Create a unified checkpointer that coordinates LangGraph and StateDB.

    Args:
        db: StateDB instance.
        sqlite_path: Path for SQLite database.
        postgres_url: Optional PostgreSQL URL (preferred over SQLite).
        sync_to_statedb: Whether to sync checkpoints to StateDB.

    Returns:
        CheckpointerWithStateDB instance.
    """
    from harness.dag.checkpointer import CheckpointerFactory

    # Create the underlying LangGraph checkpointer
    inner = await CheckpointerFactory.create_async(
        postgres_url=postgres_url,
        sqlite_path=sqlite_path,
        fallback_to_sqlite=True,
    )

    return CheckpointerWithStateDB(
        db=db,
        inner_checkpointer=inner,
        sync_to_statedb=sync_to_statedb,
    )


# =============================================================================
# CONTEXT MANAGER
# =============================================================================


class UnifiedCheckpointerContext:
    """
    Context manager for unified checkpointer lifecycle.

    Usage:
        async with UnifiedCheckpointerContext(db) as checkpointer:
            compiled = graph.compile(checkpointer=checkpointer)
            result = await compiled.ainvoke(state, config)
    """

    def __init__(
        self,
        db: StateDB,
        sqlite_path: str = "harness.db",
        postgres_url: str | None = None,
        sync_to_statedb: bool = True,
    ):
        self._db = db
        self._sqlite_path = sqlite_path
        self._postgres_url = postgres_url
        self._sync_to_statedb = sync_to_statedb
        self._checkpointer: CheckpointerWithStateDB | None = None
        self._inner_context: Any = None

    async def __aenter__(self) -> CheckpointerWithStateDB:
        # Create inner checkpointer via context manager
        self._inner_context = AsyncSqliteSaver.from_conn_string(self._sqlite_path)
        inner = await self._inner_context.__aenter__()

        self._checkpointer = CheckpointerWithStateDB(
            db=self._db,
            inner_checkpointer=inner,
            sync_to_statedb=self._sync_to_statedb,
        )

        return self._checkpointer

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._inner_context:
            await self._inner_context.__aexit__(exc_type, exc_val, exc_tb)
