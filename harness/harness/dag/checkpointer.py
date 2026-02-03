"""
PostgreSQL Checkpointer Factory with SQLite Fallback.

Provides production-ready checkpointing for LangGraph workflows with:
- PostgreSQL preference for production environments
- SQLite fallback for development/testing
- Connection pooling for Postgres
- Health checks and monitoring
- Migration utilities
- Checkpoint cleanup

Environment Variables:
- POSTGRES_URL: PostgreSQL connection string (preferred)
- DATABASE_URL: Alternative connection string (fallback)

Example usage:
    # Async factory (preferred)
    checkpointer = await CheckpointerFactory.create_async()
    compiled = graph.compile(checkpointer=checkpointer)

    # Sync factory (for simpler use cases)
    checkpointer = CheckpointerFactory.create_sync()
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)


class PostgresHealthResult(TypedDict):
    """Result of PostgreSQL health check."""

    connected: bool
    latency_ms: float
    pool_size: int
    error: str | None
    version: str | None


class MigrationResult(TypedDict):
    """Result of checkpoint migration."""

    success: bool
    checkpoints_migrated: int
    writes_migrated: int
    errors: list[str]
    duration_seconds: float


class CleanupResult(TypedDict):
    """Result of checkpoint cleanup."""

    deleted_count: int
    errors: list[str]


# =============================================================================
# POSTGRES AVAILABILITY CHECK
# =============================================================================

_POSTGRES_AVAILABLE: bool | None = None


def is_postgres_available() -> bool:
    """
    Check if PostgreSQL checkpointer dependencies are available.

    Returns:
        True if langgraph-checkpoint-postgres is installed.
    """
    global _POSTGRES_AVAILABLE

    if _POSTGRES_AVAILABLE is None:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: F401

            _POSTGRES_AVAILABLE = True
        except ImportError:
            _POSTGRES_AVAILABLE = False

    return _POSTGRES_AVAILABLE


def get_postgres_url() -> str | None:
    """
    Get PostgreSQL connection URL from environment.

    Checks in order:
    1. POSTGRES_URL
    2. DATABASE_URL

    Returns:
        Connection string or None if not configured.
    """
    return os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")


# =============================================================================
# CHECKPOINTER FACTORY
# =============================================================================


class CheckpointerFactory:
    """
    Factory for creating checkpointers with PostgreSQL preference and SQLite fallback.

    Design Philosophy:
    - Production: Use PostgreSQL for durability and scalability
    - Development: SQLite for simplicity and zero-config
    - Graceful fallback: If Postgres unavailable, use SQLite with warning

    Thread Safety:
    - Async methods create new connections per call (connection pooling in Postgres)
    - Sync methods create new connections per call
    """

    @staticmethod
    async def create_async(
        postgres_url: str | None = None,
        sqlite_path: str = "harness.db",
        fallback_to_sqlite: bool = True,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> BaseCheckpointSaver:
        """
        Create async checkpointer with Postgres preference and SQLite fallback.

        Args:
            postgres_url: PostgreSQL connection string. If None, uses environment.
            sqlite_path: Path for SQLite database (used as fallback).
            fallback_to_sqlite: If True, fall back to SQLite when Postgres unavailable.
            pool_size: Base connection pool size for Postgres.
            max_overflow: Maximum overflow connections for Postgres.

        Returns:
            Configured checkpointer (AsyncPostgresSaver or AsyncSqliteSaver).

        Raises:
            RuntimeError: If Postgres unavailable and fallback disabled.
        """
        # Try to get Postgres URL from parameter or environment
        url = postgres_url or get_postgres_url()

        if url and is_postgres_available():
            try:
                checkpointer = await create_postgres_checkpointer(
                    url=url,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                )
                logger.info("Created PostgreSQL checkpointer with connection pooling")
                return checkpointer
            except Exception as e:
                logger.warning(f"Failed to create PostgreSQL checkpointer: {e}")
                if not fallback_to_sqlite:
                    raise RuntimeError(f"PostgreSQL checkpointer creation failed: {e}")
        elif url and not is_postgres_available():
            logger.warning(
                "PostgreSQL URL configured but langgraph-checkpoint-postgres not installed. "
                "Install with: pip install dag-harness[postgres]"
            )
            if not fallback_to_sqlite:
                raise RuntimeError(
                    "PostgreSQL configured but dependencies not installed. "
                    "Install with: pip install dag-harness[postgres]"
                )

        # Fall back to SQLite
        if not fallback_to_sqlite:
            raise RuntimeError(
                "PostgreSQL not available and SQLite fallback disabled. "
                "Configure POSTGRES_URL or enable fallback_to_sqlite."
            )

        logger.info(f"Using SQLite checkpointer at {sqlite_path}")
        checkpointer = AsyncSqliteSaver.from_conn_string(sqlite_path)

        # AsyncSqliteSaver.from_conn_string returns a context manager
        # We need to enter it to get the actual saver
        return await checkpointer.__aenter__()

    @staticmethod
    def create_sync(
        postgres_url: str | None = None,
        sqlite_path: str = "harness.db",
        fallback_to_sqlite: bool = True,
    ) -> BaseCheckpointSaver:
        """
        Create sync checkpointer with Postgres preference and SQLite fallback.

        Note: For Postgres, prefer create_async() for better performance.
        This method uses sync Postgres driver which may block.

        Args:
            postgres_url: PostgreSQL connection string. If None, uses environment.
            sqlite_path: Path for SQLite database (used as fallback).
            fallback_to_sqlite: If True, fall back to SQLite when Postgres unavailable.

        Returns:
            Configured checkpointer (PostgresSaver or SqliteSaver).

        Raises:
            RuntimeError: If Postgres unavailable and fallback disabled.
        """
        url = postgres_url or get_postgres_url()

        if url and is_postgres_available():
            try:
                checkpointer = create_postgres_checkpointer_sync(url)
                logger.info("Created sync PostgreSQL checkpointer")
                return checkpointer
            except Exception as e:
                logger.warning(f"Failed to create sync PostgreSQL checkpointer: {e}")
                if not fallback_to_sqlite:
                    raise RuntimeError(f"PostgreSQL checkpointer creation failed: {e}")
        elif url and not is_postgres_available():
            logger.warning("PostgreSQL URL configured but dependencies not installed.")
            if not fallback_to_sqlite:
                raise RuntimeError("PostgreSQL dependencies not installed.")

        if not fallback_to_sqlite:
            raise RuntimeError("PostgreSQL not available and SQLite fallback disabled.")

        logger.info(f"Using sync SQLite checkpointer at {sqlite_path}")
        return SqliteSaver.from_conn_string(sqlite_path)


# =============================================================================
# POSTGRES CHECKPOINTER CREATION
# =============================================================================


async def create_postgres_checkpointer(
    url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> BaseCheckpointSaver:
    """
    Create PostgreSQL checkpointer with connection pooling.

    Connection Pool Configuration:
    - pool_size: Minimum connections maintained (default: 5)
    - max_overflow: Additional connections allowed above pool_size (default: 10)
    - Total max connections: pool_size + max_overflow = 15

    Args:
        url: PostgreSQL connection string (postgresql://user:pass@host:port/db)
        pool_size: Base pool size.
        max_overflow: Maximum overflow connections.

    Returns:
        Configured AsyncPostgresSaver with connection pooling.

    Raises:
        ImportError: If langgraph-checkpoint-postgres not installed.
        Exception: If connection fails.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        raise ImportError(
            "langgraph-checkpoint-postgres not installed. "
            "Install with: pip install dag-harness[postgres]"
        )

    # Create the checkpointer with connection pooling
    # Note: The actual API may vary by version - this follows LangGraph 2.0 patterns
    try:
        # Try the newer from_conn_string API with pool configuration
        checkpointer = await AsyncPostgresSaver.from_conn_string(
            url,
            pool_size=pool_size,
            max_overflow=max_overflow,
        )
    except TypeError:
        # Fall back to simpler API if pool params not supported
        checkpointer = await AsyncPostgresSaver.from_conn_string(url)
        logger.warning("Connection pool parameters not supported in this version")

    # Setup tables if needed
    try:
        await checkpointer.setup()
    except Exception as e:
        logger.warning(f"Checkpointer setup warning (may already exist): {e}")

    return checkpointer


def create_postgres_checkpointer_sync(url: str) -> BaseCheckpointSaver:
    """
    Create synchronous PostgreSQL checkpointer.

    Note: Prefer async version for production use.

    Args:
        url: PostgreSQL connection string.

    Returns:
        Configured PostgresSaver.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        raise ImportError(
            "langgraph-checkpoint-postgres not installed. "
            "Install with: pip install dag-harness[postgres]"
        )

    checkpointer = PostgresSaver.from_conn_string(url)

    # Setup tables if needed
    try:
        checkpointer.setup()
    except Exception as e:
        logger.warning(f"Checkpointer setup warning: {e}")

    return checkpointer


# =============================================================================
# HEALTH CHECK
# =============================================================================


async def check_postgres_health(url: str) -> PostgresHealthResult:
    """
    Check PostgreSQL connection health.

    Performs:
    1. Connection test
    2. Query latency measurement
    3. Pool status check (if available)
    4. Version check

    Args:
        url: PostgreSQL connection string.

    Returns:
        Health check results with connection status and metrics.
    """
    result: PostgresHealthResult = {
        "connected": False,
        "latency_ms": 0.0,
        "pool_size": 0,
        "error": None,
        "version": None,
    }

    if not is_postgres_available():
        result["error"] = "PostgreSQL dependencies not installed"
        return result

    try:
        import asyncpg
    except ImportError:
        result["error"] = "asyncpg not installed"
        return result

    start_time = time.perf_counter()

    try:
        # Create a direct connection for health check
        conn = await asyncpg.connect(url)

        try:
            # Measure query latency
            query_start = time.perf_counter()
            version = await conn.fetchval("SELECT version()")
            query_time = time.perf_counter() - query_start

            result["connected"] = True
            result["latency_ms"] = query_time * 1000
            result["version"] = version

            # Check if checkpointer tables exist
            tables_exist = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'checkpoints'
                )
            """)

            if tables_exist:
                # Get checkpoint count as a proxy for usage
                checkpoint_count = await conn.fetchval("SELECT COUNT(*) FROM checkpoints")
                result["pool_size"] = checkpoint_count or 0

        finally:
            await conn.close()

    except Exception as e:
        result["error"] = str(e)
        result["latency_ms"] = (time.perf_counter() - start_time) * 1000

    return result


# =============================================================================
# CHECKPOINT CLEANUP
# =============================================================================


async def cleanup_old_checkpoints(
    checkpointer: BaseCheckpointSaver,
    max_age_days: int = 30,
) -> CleanupResult:
    """
    Clean up checkpoints older than max_age_days.

    This function removes old checkpoint data to prevent unbounded growth.
    It preserves the most recent checkpoint per thread for resumability.

    Args:
        checkpointer: The checkpointer instance (Postgres or SQLite).
        max_age_days: Maximum age in days for checkpoints to keep.

    Returns:
        CleanupResult with count of deleted checkpoints and any errors.
    """
    result: CleanupResult = {
        "deleted_count": 0,
        "errors": [],
    }

    cutoff_date = datetime.now(UTC) - timedelta(days=max_age_days)
    cutoff_ts = cutoff_date.isoformat()

    # Handle different checkpointer types
    if hasattr(checkpointer, "conn") and hasattr(checkpointer.conn, "execute"):
        # SQLite checkpointer
        try:
            async with checkpointer.conn.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_ts < ?
                AND thread_id NOT IN (
                    SELECT thread_id
                    FROM checkpoints
                    GROUP BY thread_id
                    HAVING MAX(thread_ts) = thread_ts
                )
                """,
                (cutoff_ts,),
            ) as cursor:
                result["deleted_count"] = cursor.rowcount or 0
            await checkpointer.conn.commit()
        except Exception as e:
            result["errors"].append(f"SQLite cleanup failed: {e}")

    elif is_postgres_available():
        # Postgres checkpointer - use raw SQL
        try:
            # Get connection from checkpointer if possible
            if hasattr(checkpointer, "pool"):
                async with checkpointer.pool.acquire() as conn:
                    deleted = await conn.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_ts < $1
                        AND (thread_id, thread_ts) NOT IN (
                            SELECT thread_id, MAX(thread_ts)
                            FROM checkpoints
                            GROUP BY thread_id
                        )
                        """,
                        cutoff_date,
                    )
                    # Parse "DELETE N" response
                    if deleted:
                        parts = deleted.split()
                        if len(parts) >= 2:
                            result["deleted_count"] = int(parts[1])
            else:
                result["errors"].append("Cannot access Postgres pool for cleanup")

        except Exception as e:
            result["errors"].append(f"Postgres cleanup failed: {e}")
    else:
        result["errors"].append("Unknown checkpointer type for cleanup")

    return result


async def cleanup_old_checkpoints_by_url(
    postgres_url: str,
    max_age_days: int = 30,
) -> CleanupResult:
    """
    Clean up old checkpoints using a direct database connection.

    This is useful when you don't have a checkpointer instance handy.

    Args:
        postgres_url: PostgreSQL connection string.
        max_age_days: Maximum age in days for checkpoints to keep.

    Returns:
        CleanupResult with deletion count and errors.
    """
    result: CleanupResult = {
        "deleted_count": 0,
        "errors": [],
    }

    cutoff_date = datetime.now(UTC) - timedelta(days=max_age_days)

    try:
        import asyncpg

        conn = await asyncpg.connect(postgres_url)
        try:
            # Delete old checkpoints but keep the latest per thread
            deleted = await conn.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_ts < $1
                AND (thread_id, thread_ts) NOT IN (
                    SELECT thread_id, MAX(thread_ts)
                    FROM checkpoints
                    GROUP BY thread_id
                )
                """,
                cutoff_date,
            )

            # Also clean up old checkpoint writes
            await conn.execute(
                """
                DELETE FROM checkpoint_writes
                WHERE thread_ts < $1
                AND (thread_id, thread_ts, checkpoint_id) NOT IN (
                    SELECT DISTINCT thread_id, thread_ts, checkpoint_id
                    FROM checkpoints
                )
                """,
                cutoff_date,
            )

            if deleted:
                parts = deleted.split()
                if len(parts) >= 2:
                    result["deleted_count"] = int(parts[1])

        finally:
            await conn.close()

    except ImportError:
        result["errors"].append("asyncpg not installed")
    except Exception as e:
        result["errors"].append(str(e))

    return result


# =============================================================================
# MIGRATION
# =============================================================================


async def migrate_sqlite_to_postgres(
    sqlite_path: str,
    postgres_url: str,
    batch_size: int = 100,
) -> MigrationResult:
    """
    Migrate checkpoints from SQLite to PostgreSQL.

    This function:
    1. Reads all checkpoints from SQLite
    2. Creates corresponding entries in PostgreSQL
    3. Preserves thread IDs and timestamps
    4. Handles checkpoint writes if they exist

    Note: This is a one-way migration. Always backup before migrating.

    Args:
        sqlite_path: Path to SQLite database.
        postgres_url: PostgreSQL connection string.
        batch_size: Number of records to migrate per batch.

    Returns:
        MigrationResult with counts and any errors.
    """
    result: MigrationResult = {
        "success": False,
        "checkpoints_migrated": 0,
        "writes_migrated": 0,
        "errors": [],
        "duration_seconds": 0.0,
    }

    start_time = time.perf_counter()

    if not is_postgres_available():
        result["errors"].append("PostgreSQL dependencies not installed")
        result["duration_seconds"] = time.perf_counter() - start_time
        return result

    if not Path(sqlite_path).exists():
        result["errors"].append(f"SQLite database not found: {sqlite_path}")
        result["duration_seconds"] = time.perf_counter() - start_time
        return result

    try:
        import aiosqlite
        import asyncpg

        # Connect to both databases
        sqlite_conn = await aiosqlite.connect(sqlite_path)
        sqlite_conn.row_factory = aiosqlite.Row

        pg_conn = await asyncpg.connect(postgres_url)

        try:
            # Ensure Postgres tables exist
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            pg_saver = await AsyncPostgresSaver.from_conn_string(postgres_url)
            await pg_saver.setup()

            # Migrate checkpoints table
            async with sqlite_conn.execute(
                "SELECT * FROM checkpoints ORDER BY thread_ts"
            ) as cursor:
                batch = []
                async for row in cursor:
                    batch.append(dict(row))

                    if len(batch) >= batch_size:
                        await _migrate_checkpoint_batch(pg_conn, batch)
                        result["checkpoints_migrated"] += len(batch)
                        batch = []

                # Migrate remaining
                if batch:
                    await _migrate_checkpoint_batch(pg_conn, batch)
                    result["checkpoints_migrated"] += len(batch)

            # Migrate checkpoint_writes table if it exists
            try:
                async with sqlite_conn.execute(
                    "SELECT * FROM checkpoint_writes ORDER BY thread_ts"
                ) as cursor:
                    batch = []
                    async for row in cursor:
                        batch.append(dict(row))

                        if len(batch) >= batch_size:
                            await _migrate_writes_batch(pg_conn, batch)
                            result["writes_migrated"] += len(batch)
                            batch = []

                    if batch:
                        await _migrate_writes_batch(pg_conn, batch)
                        result["writes_migrated"] += len(batch)
            except Exception:
                # checkpoint_writes may not exist in older schemas
                pass

            result["success"] = True

        finally:
            await sqlite_conn.close()
            await pg_conn.close()

    except Exception as e:
        result["errors"].append(str(e))

    result["duration_seconds"] = time.perf_counter() - start_time
    return result


async def _migrate_checkpoint_batch(
    pg_conn: Any,
    batch: list[dict],
) -> None:
    """Migrate a batch of checkpoints to PostgreSQL."""
    for row in batch:
        await pg_conn.execute(
            """
            INSERT INTO checkpoints (
                thread_id, checkpoint_ns, checkpoint_id,
                parent_checkpoint_id, type, checkpoint, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO NOTHING
            """,
            row.get("thread_id"),
            row.get("checkpoint_ns", ""),
            row.get("checkpoint_id"),
            row.get("parent_checkpoint_id"),
            row.get("type", ""),
            row.get("checkpoint"),
            row.get("metadata"),
        )


async def _migrate_writes_batch(
    pg_conn: Any,
    batch: list[dict],
) -> None:
    """Migrate a batch of checkpoint writes to PostgreSQL."""
    for row in batch:
        await pg_conn.execute(
            """
            INSERT INTO checkpoint_writes (
                thread_id, checkpoint_ns, checkpoint_id,
                task_id, idx, channel, type, blob
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT DO NOTHING
            """,
            row.get("thread_id"),
            row.get("checkpoint_ns", ""),
            row.get("checkpoint_id"),
            row.get("task_id"),
            row.get("idx"),
            row.get("channel"),
            row.get("type"),
            row.get("blob"),
        )


# =============================================================================
# CONTEXT MANAGER HELPERS
# =============================================================================


class CheckpointerContext:
    """
    Context manager for checkpointer lifecycle management.

    Handles proper setup and teardown of checkpointers,
    including connection pool cleanup.

    Usage:
        async with CheckpointerContext() as checkpointer:
            compiled = graph.compile(checkpointer=checkpointer)
            await compiled.ainvoke(...)

    For unified checkpointing with StateDB sync, use UnifiedCheckpointerContext instead:
        from harness.dag.checkpointer_unified import UnifiedCheckpointerContext
        async with UnifiedCheckpointerContext(db) as checkpointer:
            ...
    """

    def __init__(
        self,
        postgres_url: str | None = None,
        sqlite_path: str = "harness.db",
        fallback_to_sqlite: bool = True,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        self.postgres_url = postgres_url
        self.sqlite_path = sqlite_path
        self.fallback_to_sqlite = fallback_to_sqlite
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._checkpointer: BaseCheckpointSaver | None = None
        self._is_postgres = False

    async def __aenter__(self) -> BaseCheckpointSaver:
        url = self.postgres_url or get_postgres_url()

        if url and is_postgres_available():
            try:
                self._checkpointer = await create_postgres_checkpointer(
                    url=url,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                )
                self._is_postgres = True
                return self._checkpointer
            except Exception as e:
                if not self.fallback_to_sqlite:
                    raise
                logger.warning(f"Postgres unavailable, falling back to SQLite: {e}")

        # SQLite fallback
        self._checkpointer = await AsyncSqliteSaver.from_conn_string(self.sqlite_path).__aenter__()
        return self._checkpointer

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._checkpointer is None:
            return

        try:
            if self._is_postgres and hasattr(self._checkpointer, "pool"):
                await self._checkpointer.pool.close()
            elif hasattr(self._checkpointer, "__aexit__"):
                await self._checkpointer.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.warning(f"Error closing checkpointer: {e}")
