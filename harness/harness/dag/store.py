"""
LangGraph Store Interface Implementation for dag-harness.

This module provides a BaseStore implementation backed by StateDB for
cross-thread memory persistence. It enables sharing state between different
workflow executions and threads.

Namespace Structure:
- ("roles", "common"): Role-specific memory
- ("waves", "0"): Wave-level coordination
- ("users", "jsullivan2"): User preferences and session data
- ("workflows", "box-up-role"): Workflow execution memory

Usage:
    from harness.dag.store import HarnessStore
    from harness.db.state import StateDB

    db = StateDB("harness.db")
    store = HarnessStore(db)

    # Store data
    store.put(("roles", "common"), "status", {"last_run": "2025-02-03", "passed": True})

    # Retrieve data
    item = store.get(("roles", "common"), "status")
    print(item.value)  # {"last_run": "2025-02-03", "passed": True}

    # Search within namespace
    results = store.search(("roles",), filter={"passed": True})
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

if TYPE_CHECKING:
    from harness.db.state import StateDB

logger = logging.getLogger(__name__)

# =============================================================================
# TABLE SCHEMA
# =============================================================================

STORE_ITEMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,           -- JSON-encoded tuple e.g. '["roles", "common"]'
    key TEXT NOT NULL,
    value TEXT NOT NULL,               -- JSON-encoded dict
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(namespace, key)
);

CREATE INDEX IF NOT EXISTS idx_store_items_namespace ON store_items(namespace);
CREATE INDEX IF NOT EXISTS idx_store_items_namespace_key ON store_items(namespace, key);
"""


# =============================================================================
# HARNESS STORE IMPLEMENTATION
# =============================================================================


class HarnessStore(BaseStore):
    """
    LangGraph Store implementation backed by StateDB SQLite.

    This store enables cross-thread memory persistence for the harness,
    allowing different workflow executions to share state.

    Thread Safety:
    - SQLite with connection-per-call pattern is thread-safe
    - Operations are atomic at the individual item level
    - For multi-item atomicity, use batch() method

    Limitations:
    - No semantic search support (index parameter ignored)
    - No TTL support (items persist indefinitely)
    - Simple key-value filter matching only
    """

    supports_ttl: bool = False

    def __init__(self, db: StateDB):
        """
        Initialize HarnessStore with StateDB backend.

        Args:
            db: StateDB instance for persistence.
        """
        super().__init__()
        self._db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure store_items table exists."""
        with self._db.connection() as conn:
            conn.executescript(STORE_ITEMS_SCHEMA)

    def _namespace_to_str(self, namespace: tuple[str, ...]) -> str:
        """Convert namespace tuple to JSON string for storage."""
        return json.dumps(list(namespace))

    def _str_to_namespace(self, s: str) -> tuple[str, ...]:
        """Convert JSON string back to namespace tuple."""
        return tuple(json.loads(s))

    # =========================================================================
    # REQUIRED ABSTRACT METHODS
    # =========================================================================

    def batch(self, ops: Iterable[GetOp | PutOp | SearchOp | ListNamespacesOp]) -> list[Result]:
        """
        Execute multiple operations synchronously in a single batch.

        Args:
            ops: Iterable of operations (GetOp, PutOp, SearchOp, ListNamespacesOp)

        Returns:
            List of results corresponding to each operation.
        """
        results: list[Result] = []
        ops_list = list(ops)

        with self._db.connection() as conn:
            for op in ops_list:
                if isinstance(op, GetOp):
                    results.append(self._execute_get(conn, op))
                elif isinstance(op, PutOp):
                    results.append(self._execute_put(conn, op))
                elif isinstance(op, SearchOp):
                    results.append(self._execute_search(conn, op))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._execute_list_namespaces(conn, op))
                else:
                    raise TypeError(f"Unknown operation type: {type(op)}")

        return results

    async def abatch(
        self, ops: Iterable[GetOp | PutOp | SearchOp | ListNamespacesOp]
    ) -> list[Result]:
        """
        Execute multiple operations asynchronously in a single batch.

        Note: This is a thin wrapper around batch() since SQLite doesn't
        have true async support. For production use with high concurrency,
        consider using AsyncPostgresStore.

        Args:
            ops: Iterable of operations.

        Returns:
            List of results corresponding to each operation.
        """
        # SQLite is sync-only; delegate to sync implementation
        return self.batch(ops)

    # =========================================================================
    # OPERATION EXECUTION
    # =========================================================================

    def _execute_get(self, conn: sqlite3.Connection, op: GetOp) -> Item | None:
        """Execute a GetOp and return Item or None."""
        namespace_str = self._namespace_to_str(op.namespace)

        cursor = conn.execute(
            """
            SELECT key, value, created_at, updated_at
            FROM store_items
            WHERE namespace = ? AND key = ?
            """,
            (namespace_str, op.key),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        key, value_str, created_at_str, updated_at_str = row

        return Item(
            namespace=op.namespace,
            key=key,
            value=json.loads(value_str),
            created_at=self._parse_timestamp(created_at_str),
            updated_at=self._parse_timestamp(updated_at_str),
        )

    def _execute_put(self, conn: sqlite3.Connection, op: PutOp) -> None:
        """Execute a PutOp (insert, update, or delete)."""
        namespace_str = self._namespace_to_str(op.namespace)

        if op.value is None:
            # Delete operation
            conn.execute(
                "DELETE FROM store_items WHERE namespace = ? AND key = ?",
                (namespace_str, op.key),
            )
        else:
            # Insert or update
            value_str = json.dumps(op.value)
            now = datetime.now(UTC).isoformat()

            conn.execute(
                """
                INSERT INTO store_items (namespace, key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (namespace_str, op.key, value_str, now, now),
            )

        return None

    def _execute_search(self, conn: sqlite3.Connection, op: SearchOp) -> list[SearchItem]:
        """Execute a SearchOp and return matching items."""
        namespace_prefix_str = self._namespace_to_str(op.namespace_prefix)

        # Build query to find items with namespace starting with prefix
        # We use LIKE with escaped JSON prefix
        like_pattern = namespace_prefix_str.rstrip("]") + "%"

        cursor = conn.execute(
            """
            SELECT namespace, key, value, created_at, updated_at
            FROM store_items
            WHERE namespace LIKE ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (like_pattern, op.limit, op.offset),
        )

        results: list[SearchItem] = []
        for row in cursor:
            namespace_str, key, value_str, created_at_str, updated_at_str = row
            value = json.loads(value_str)

            # Apply filter if provided
            if op.filter:
                if not self._matches_filter(value, op.filter):
                    continue

            results.append(
                SearchItem(
                    namespace=self._str_to_namespace(namespace_str),
                    key=key,
                    value=value,
                    created_at=self._parse_timestamp(created_at_str),
                    updated_at=self._parse_timestamp(updated_at_str),
                    score=None,  # No semantic search support
                )
            )

        return results

    def _execute_list_namespaces(
        self, conn: sqlite3.Connection, op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        """Execute a ListNamespacesOp and return matching namespaces."""
        cursor = conn.execute(
            """
            SELECT DISTINCT namespace FROM store_items
            ORDER BY namespace
            LIMIT ? OFFSET ?
            """,
            (op.limit, op.offset),
        )

        namespaces: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()

        for (namespace_str,) in cursor:
            namespace = self._str_to_namespace(namespace_str)

            # Apply match conditions
            if op.match_conditions:
                if not self._matches_conditions(namespace, op.match_conditions):
                    continue

            # Apply max_depth truncation
            if op.max_depth is not None and len(namespace) > op.max_depth:
                namespace = namespace[: op.max_depth]

            # Deduplicate after truncation
            if namespace not in seen:
                seen.add(namespace)
                namespaces.append(namespace)

        return namespaces

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp string to datetime."""
        try:
            # Try ISO format first
            if "T" in ts_str:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            # SQLite default format
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        except ValueError:
            return datetime.now(UTC)

    def _matches_filter(self, value: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        """Check if value matches all filter criteria."""
        for key, expected in filter_dict.items():
            if key not in value:
                return False
            if value[key] != expected:
                return False
        return True

    def _matches_conditions(
        self, namespace: tuple[str, ...], conditions: tuple[MatchCondition, ...]
    ) -> bool:
        """Check if namespace matches all conditions."""
        for condition in conditions:
            if condition.match_type == "prefix":
                if not self._starts_with(namespace, condition.path):
                    return False
            elif condition.match_type == "suffix":
                if not self._ends_with(namespace, condition.path):
                    return False
        return True

    def _starts_with(
        self, namespace: tuple[str, ...], prefix: tuple[str, ...] | list[str]
    ) -> bool:
        """Check if namespace starts with prefix."""
        prefix_tuple = tuple(prefix) if isinstance(prefix, list) else prefix
        return namespace[: len(prefix_tuple)] == prefix_tuple

    def _ends_with(
        self, namespace: tuple[str, ...], suffix: tuple[str, ...] | list[str]
    ) -> bool:
        """Check if namespace ends with suffix."""
        suffix_tuple = tuple(suffix) if isinstance(suffix, list) else suffix
        return namespace[-len(suffix_tuple) :] == suffix_tuple


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_harness_store(db: StateDB) -> HarnessStore:
    """
    Create a HarnessStore instance.

    Args:
        db: StateDB instance.

    Returns:
        Configured HarnessStore.
    """
    return HarnessStore(db)
