"""
Database migration system for harness.

Provides versioned, ordered schema migrations with up/down support,
idempotent application, and status tracking via a schema_migrations table.

Usage:
    from harness.db.migrations import MigrationRunner

    runner = MigrationRunner("harness.db")
    runner.migrate()          # Apply all pending migrations
    runner.migrate(target=2)  # Migrate to version 2
    runner.rollback(target=1) # Roll back to version 1
    runner.status()           # Show current migration state
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# SCHEMA MIGRATIONS TABLE
# =============================================================================

SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# =============================================================================
# BASE MIGRATION CLASS
# =============================================================================


class Migration(ABC):
    """
    Base class for database migrations.

    Each migration must define:
    - version: Integer version number (unique, sequential)
    - description: Human-readable description of what this migration does
    - up(): Apply the migration
    - down(): Reverse the migration
    """

    version: int
    description: str

    @abstractmethod
    def up(self, conn: sqlite3.Connection) -> None:
        """Apply the migration."""
        ...

    @abstractmethod
    def down(self, conn: sqlite3.Connection) -> None:
        """Reverse the migration."""
        ...

    def __repr__(self) -> str:
        return f"Migration(version={self.version}, description={self.description!r})"


# =============================================================================
# MIGRATION REGISTRY
# =============================================================================


def _discover_migrations() -> list[Migration]:
    """
    Discover all migration modules in this package.

    Scans for modules matching the pattern NNN_*.py and instantiates
    their Migration subclass. Each module must define exactly one
    Migration subclass.

    Returns:
        List of Migration instances sorted by version.
    """
    migrations: list[Migration] = []
    package_path = Path(__file__).parent

    for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if ispkg or modname.startswith("_"):
            continue

        # Only load modules that start with a number (migration files)
        parts = modname.split("_", 1)
        if not parts[0].isdigit():
            continue

        try:
            module = importlib.import_module(f".{modname}", package=__name__)
        except ImportError as e:
            logger.warning("Failed to import migration module %s: %s", modname, e)
            continue

        # Find Migration subclasses in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Migration)
                and attr is not Migration
            ):
                instance = attr()
                migrations.append(instance)

    migrations.sort(key=lambda m: m.version)
    return migrations


# =============================================================================
# MIGRATION RUNNER
# =============================================================================


@dataclass
class MigrationRunner:
    """
    Runs migrations against a SQLite database.

    Tracks applied migrations in a schema_migrations table and supports
    forward migration, rollback, and status queries.
    """

    db_path: str
    _migrations: list[Migration] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._migrations = _discover_migrations()
        self._ensure_schema_migrations_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema_migrations_table(self) -> None:
        """Create the schema_migrations table if it doesn't exist."""
        conn = self._get_connection()
        try:
            conn.executescript(SCHEMA_MIGRATIONS_DDL)
            conn.commit()
        finally:
            conn.close()

    @property
    def all_migrations(self) -> list[Migration]:
        """Return all discovered migrations sorted by version."""
        return list(self._migrations)

    def get_current_version(self) -> int:
        """
        Get the current schema version.

        Returns:
            The highest applied migration version, or 0 if none applied.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT MAX(version) as version FROM schema_migrations"
            ).fetchone()
            return row["version"] if row and row["version"] is not None else 0
        finally:
            conn.close()

    def get_applied_versions(self) -> set[int]:
        """Get the set of all applied migration versions."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            return {row["version"] for row in rows}
        finally:
            conn.close()

    def get_pending(self) -> list[Migration]:
        """
        Get migrations that haven't been applied yet.

        Returns:
            List of pending Migration instances in version order.
        """
        applied = self.get_applied_versions()
        return [m for m in self._migrations if m.version not in applied]

    def migrate(self, target: int | None = None) -> list[str]:
        """
        Apply pending migrations up to an optional target version.

        Args:
            target: If specified, only apply migrations up to this version.
                    If None, apply all pending migrations.

        Returns:
            List of description strings for applied migrations.
        """
        pending = self.get_pending()
        if target is not None:
            pending = [m for m in pending if m.version <= target]

        if not pending:
            logger.info("No pending migrations to apply")
            return []

        applied: list[str] = []
        conn = self._get_connection()
        try:
            for migration in pending:
                logger.info(
                    "Applying migration %d: %s",
                    migration.version,
                    migration.description,
                )
                try:
                    migration.up(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                        (migration.version, migration.description),
                    )
                    conn.commit()
                    applied.append(
                        f"Applied migration {migration.version}: {migration.description}"
                    )
                except Exception as e:
                    conn.rollback()
                    logger.error(
                        "Migration %d failed: %s", migration.version, e
                    )
                    raise RuntimeError(
                        f"Migration {migration.version} failed: {e}"
                    ) from e
        finally:
            conn.close()

        return applied

    def rollback(self, target: int) -> list[str]:
        """
        Roll back migrations down to a target version.

        Rolls back all applied migrations with version > target,
        in reverse order.

        Args:
            target: The version to roll back to. Migrations with version
                    greater than this will be reversed.

        Returns:
            List of description strings for rolled-back migrations.
        """
        applied = self.get_applied_versions()
        to_rollback = sorted(
            [m for m in self._migrations if m.version in applied and m.version > target],
            key=lambda m: m.version,
            reverse=True,
        )

        if not to_rollback:
            logger.info("No migrations to roll back")
            return []

        rolled_back: list[str] = []
        conn = self._get_connection()
        try:
            for migration in to_rollback:
                logger.info(
                    "Rolling back migration %d: %s",
                    migration.version,
                    migration.description,
                )
                try:
                    migration.down(conn)
                    conn.execute(
                        "DELETE FROM schema_migrations WHERE version = ?",
                        (migration.version,),
                    )
                    conn.commit()
                    rolled_back.append(
                        f"Rolled back migration {migration.version}: {migration.description}"
                    )
                except Exception as e:
                    conn.rollback()
                    logger.error(
                        "Rollback of migration %d failed: %s",
                        migration.version,
                        e,
                    )
                    raise RuntimeError(
                        f"Rollback of migration {migration.version} failed: {e}"
                    ) from e
        finally:
            conn.close()

        return rolled_back

    def status(self) -> dict[str, Any]:
        """
        Get the current migration status.

        Returns:
            Dictionary with:
            - current_version: Highest applied version
            - total_migrations: Total number of available migrations
            - applied_count: Number of applied migrations
            - pending_count: Number of pending migrations
            - applied: List of applied migration info dicts
            - pending: List of pending migration info dicts
        """
        applied_versions = self.get_applied_versions()
        current_version = self.get_current_version()

        # Get applied migration details with timestamps
        conn = self._get_connection()
        try:
            applied_rows = conn.execute(
                "SELECT version, description, applied_at FROM schema_migrations ORDER BY version"
            ).fetchall()
        finally:
            conn.close()

        applied_info = [
            {
                "version": row["version"],
                "description": row["description"],
                "applied_at": row["applied_at"],
            }
            for row in applied_rows
        ]

        pending_info = [
            {
                "version": m.version,
                "description": m.description,
            }
            for m in self._migrations
            if m.version not in applied_versions
        ]

        return {
            "current_version": current_version,
            "total_migrations": len(self._migrations),
            "applied_count": len(applied_versions),
            "pending_count": len(pending_info),
            "applied": applied_info,
            "pending": pending_info,
        }
