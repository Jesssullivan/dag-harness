"""
Tests for the database migration system.

Covers:
- Initial migration application and table verification
- Rollback behavior
- Migration status reporting
- Idempotency (running twice doesn't fail)
- Targeted migration to specific versions
- Migration ordering guarantees
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from harness.db.migrations import Migration, MigrationRunner


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Create a temporary database path."""
    return str(tmp_path / "test_migrations.db")


@pytest.fixture
def runner(db_path: str) -> MigrationRunner:
    """Create a MigrationRunner with a fresh database."""
    return MigrationRunner(db_path)


def _get_tables(db_path: str) -> set[str]:
    """Helper to get all table names in a database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _get_views(db_path: str) -> set[str]:
    """Helper to get all view names in a database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _get_indexes(db_path: str) -> set[str]:
    """Helper to get all index names in a database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _get_triggers(db_path: str) -> set[str]:
    """Helper to get all trigger names in a database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _table_has_column(db_path: str, table: str, column: str) -> bool:
    """Check if a table has a specific column."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in rows)
    finally:
        conn.close()


# =============================================================================
# TEST: INITIAL MIGRATION
# =============================================================================


class TestInitialMigrationUp:
    """Test that migration 001 creates all expected tables."""

    def test_applies_initial_migration(self, runner: MigrationRunner, db_path: str):
        """001 migration creates all core tables."""
        results = runner.migrate(target=1)
        assert len(results) == 1
        assert "Initial schema" in results[0]

        tables = _get_tables(db_path)

        # Core tables
        assert "roles" in tables
        assert "role_dependencies" in tables
        assert "credentials" in tables

        # Git worktrees
        assert "worktrees" in tables

        # GitLab entities
        assert "iterations" in tables
        assert "issues" in tables
        assert "merge_requests" in tables

        # Workflow execution
        assert "workflow_definitions" in tables
        assert "workflow_executions" in tables
        assert "node_executions" in tables

        # Test results
        assert "test_runs" in tables
        assert "test_cases" in tables

        # Audit log
        assert "audit_log" in tables

        # Context control
        assert "execution_contexts" in tables
        assert "context_capabilities" in tables
        assert "tool_invocations" in tables

        # Regression tracking
        assert "test_regressions" in tables
        assert "merge_train_entries" in tables

        # Agent sessions
        assert "agent_sessions" in tables
        assert "agent_file_changes" in tables

    def test_creates_views(self, runner: MigrationRunner, db_path: str):
        """001 migration creates expected views."""
        runner.migrate(target=1)
        views = _get_views(db_path)

        assert "v_role_status" in views
        assert "v_dependency_graph" in views
        assert "v_active_regressions" in views
        assert "v_context_capabilities" in views
        assert "v_agent_sessions" in views

    def test_creates_indexes(self, runner: MigrationRunner, db_path: str):
        """001 migration creates expected indexes."""
        runner.migrate(target=1)
        indexes = _get_indexes(db_path)

        assert "idx_role_deps_role" in indexes
        assert "idx_worktrees_role" in indexes
        assert "idx_test_runs_role" in indexes
        assert "idx_agent_sessions_status" in indexes

    def test_creates_triggers(self, runner: MigrationRunner, db_path: str):
        """001 migration creates expected triggers."""
        runner.migrate(target=1)
        triggers = _get_triggers(db_path)

        assert "trg_roles_updated" in triggers
        assert "trg_worktrees_updated" in triggers
        assert "trg_workflow_exec_updated" in triggers

    def test_version_recorded(self, runner: MigrationRunner):
        """Migration version is recorded in schema_migrations."""
        runner.migrate(target=1)
        assert runner.get_current_version() == 1
        assert 1 in runner.get_applied_versions()


# =============================================================================
# TEST: MIGRATION ROLLBACK
# =============================================================================


class TestMigrationRollback:
    """Test that migrations can be rolled back."""

    def test_rollback_initial_migration(self, runner: MigrationRunner, db_path: str):
        """Rolling back 001 drops all core tables."""
        runner.migrate(target=1)
        assert "roles" in _get_tables(db_path)

        results = runner.rollback(target=0)
        assert len(results) == 1
        assert "Rolled back" in results[0]

        tables = _get_tables(db_path)
        # schema_migrations should remain (it's not part of migrations)
        assert "schema_migrations" in tables
        assert "roles" not in tables
        assert "worktrees" not in tables
        assert "issues" not in tables

    def test_rollback_token_usage(self, runner: MigrationRunner, db_path: str):
        """Rolling back 002 drops token_usage table and its views."""
        runner.migrate(target=2)
        assert "token_usage" in _get_tables(db_path)
        assert "v_daily_costs" in _get_views(db_path)

        runner.rollback(target=1)
        assert "token_usage" not in _get_tables(db_path)
        assert "v_daily_costs" not in _get_views(db_path)
        assert "v_session_costs" not in _get_views(db_path)

        # Core tables should still exist
        assert "roles" in _get_tables(db_path)

    def test_rollback_store(self, runner: MigrationRunner, db_path: str):
        """Rolling back 003 drops store_items table."""
        runner.migrate()
        assert "store_items" in _get_tables(db_path)

        runner.rollback(target=2)
        assert "store_items" not in _get_tables(db_path)
        # token_usage should still exist
        assert "token_usage" in _get_tables(db_path)

    def test_rollback_version_updated(self, runner: MigrationRunner):
        """Rollback updates the current version."""
        runner.migrate()
        assert runner.get_current_version() == 3

        runner.rollback(target=1)
        assert runner.get_current_version() == 1

    def test_rollback_multiple_at_once(self, runner: MigrationRunner, db_path: str):
        """Rolling back multiple migrations in one operation."""
        runner.migrate()
        assert runner.get_current_version() == 3

        results = runner.rollback(target=0)
        assert len(results) == 3
        assert runner.get_current_version() == 0

        # Only schema_migrations should remain
        tables = _get_tables(db_path)
        assert tables == {"schema_migrations"}


# =============================================================================
# TEST: MIGRATION STATUS
# =============================================================================


class TestMigrationStatus:
    """Test migration status reporting."""

    def test_status_no_migrations_applied(self, runner: MigrationRunner):
        """Status shows all migrations as pending when none applied."""
        status = runner.status()
        assert status["current_version"] == 0
        assert status["applied_count"] == 0
        assert status["pending_count"] == 3
        assert len(status["pending"]) == 3
        assert len(status["applied"]) == 0

    def test_status_all_applied(self, runner: MigrationRunner):
        """Status shows no pending when all migrations applied."""
        runner.migrate()
        status = runner.status()
        assert status["current_version"] == 3
        assert status["applied_count"] == 3
        assert status["pending_count"] == 0
        assert len(status["applied"]) == 3

    def test_status_partial(self, runner: MigrationRunner):
        """Status reflects partially applied migrations."""
        runner.migrate(target=1)
        status = runner.status()
        assert status["current_version"] == 1
        assert status["applied_count"] == 1
        assert status["pending_count"] == 2

    def test_status_includes_descriptions(self, runner: MigrationRunner):
        """Status includes migration descriptions."""
        runner.migrate()
        status = runner.status()

        descriptions = [m["description"] for m in status["applied"]]
        assert any("Initial schema" in d for d in descriptions)
        assert any("token_usage" in d for d in descriptions)
        assert any("store_items" in d for d in descriptions)

    def test_status_includes_applied_timestamps(self, runner: MigrationRunner):
        """Applied migrations have timestamps."""
        runner.migrate()
        status = runner.status()

        for m in status["applied"]:
            assert m["applied_at"] is not None

    def test_total_migrations_count(self, runner: MigrationRunner):
        """Total migrations count reflects all discovered migrations."""
        status = runner.status()
        assert status["total_migrations"] == 3


# =============================================================================
# TEST: IDEMPOTENCY
# =============================================================================


class TestMigrationsIdempotent:
    """Test that running migrations twice doesn't fail."""

    def test_migrate_twice_no_error(self, runner: MigrationRunner):
        """Running migrate() twice succeeds without error."""
        results1 = runner.migrate()
        assert len(results1) == 3

        results2 = runner.migrate()
        assert len(results2) == 0  # Nothing to apply

    def test_migrate_twice_same_version(self, runner: MigrationRunner):
        """Version stays the same after second run."""
        runner.migrate()
        version1 = runner.get_current_version()

        runner.migrate()
        version2 = runner.get_current_version()

        assert version1 == version2 == 3

    def test_migrate_after_rollback_and_reapply(self, runner: MigrationRunner, db_path: str):
        """Migrations can be rolled back and reapplied cleanly."""
        runner.migrate()
        runner.rollback(target=0)
        assert runner.get_current_version() == 0

        results = runner.migrate()
        assert len(results) == 3
        assert runner.get_current_version() == 3

        # Verify tables exist
        tables = _get_tables(db_path)
        assert "roles" in tables
        assert "token_usage" in tables
        assert "store_items" in tables

    def test_partial_migrate_then_complete(self, runner: MigrationRunner):
        """Partial migration followed by full migration works."""
        runner.migrate(target=1)
        assert runner.get_current_version() == 1

        results = runner.migrate()
        assert len(results) == 2  # migrations 2 and 3
        assert runner.get_current_version() == 3


# =============================================================================
# TEST: TARGETED MIGRATION
# =============================================================================


class TestMigrateToTarget:
    """Test migrating to a specific version."""

    def test_migrate_to_version_1(self, runner: MigrationRunner, db_path: str):
        """Migrating to version 1 applies only initial schema."""
        results = runner.migrate(target=1)
        assert len(results) == 1
        assert runner.get_current_version() == 1

        tables = _get_tables(db_path)
        assert "roles" in tables
        assert "token_usage" not in tables
        assert "store_items" not in tables

    def test_migrate_to_version_2(self, runner: MigrationRunner, db_path: str):
        """Migrating to version 2 applies initial and token_usage."""
        results = runner.migrate(target=2)
        assert len(results) == 2
        assert runner.get_current_version() == 2

        tables = _get_tables(db_path)
        assert "roles" in tables
        assert "token_usage" in tables
        assert "store_items" not in tables

    def test_migrate_to_version_3(self, runner: MigrationRunner, db_path: str):
        """Migrating to version 3 applies all migrations."""
        results = runner.migrate(target=3)
        assert len(results) == 3
        assert runner.get_current_version() == 3

        tables = _get_tables(db_path)
        assert "roles" in tables
        assert "token_usage" in tables
        assert "store_items" in tables

    def test_migrate_to_already_applied_version(self, runner: MigrationRunner):
        """Migrating to an already-applied version does nothing."""
        runner.migrate(target=2)
        results = runner.migrate(target=1)
        assert len(results) == 0
        assert runner.get_current_version() == 2

    def test_migrate_incrementally(self, runner: MigrationRunner):
        """Migrating step by step produces same result as all at once."""
        runner.migrate(target=1)
        assert runner.get_current_version() == 1

        runner.migrate(target=2)
        assert runner.get_current_version() == 2

        runner.migrate(target=3)
        assert runner.get_current_version() == 3


# =============================================================================
# TEST: MIGRATION ORDERING
# =============================================================================


class TestMigrationOrdering:
    """Test that migrations run in version order."""

    def test_migrations_discovered_in_order(self, runner: MigrationRunner):
        """All migrations are discovered in ascending version order."""
        migrations = runner.all_migrations
        versions = [m.version for m in migrations]
        assert versions == sorted(versions)
        assert versions == [1, 2, 3]

    def test_migrations_applied_in_order(self, runner: MigrationRunner):
        """Migrations are applied in version order."""
        results = runner.migrate()
        # Check that version 1 appears before version 2, etc.
        assert "migration 1:" in results[0].lower()
        assert "migration 2:" in results[1].lower()
        assert "migration 3:" in results[2].lower()

    def test_rollback_runs_in_reverse_order(self, runner: MigrationRunner):
        """Rollback runs migrations in reverse version order."""
        runner.migrate()
        results = runner.rollback(target=0)

        assert "migration 3:" in results[0].lower()
        assert "migration 2:" in results[1].lower()
        assert "migration 1:" in results[2].lower()

    def test_pending_returns_correct_order(self, runner: MigrationRunner):
        """Pending migrations are returned in version order."""
        pending = runner.get_pending()
        versions = [m.version for m in pending]
        assert versions == [1, 2, 3]

        runner.migrate(target=1)
        pending = runner.get_pending()
        versions = [m.version for m in pending]
        assert versions == [2, 3]

    def test_each_migration_has_unique_version(self, runner: MigrationRunner):
        """All migrations have unique version numbers."""
        migrations = runner.all_migrations
        versions = [m.version for m in migrations]
        assert len(versions) == len(set(versions))


# =============================================================================
# TEST: TOKEN USAGE TABLE STRUCTURE
# =============================================================================


class TestTokenUsageMigration:
    """Test migration 002 creates proper token_usage schema."""

    def test_token_usage_columns(self, runner: MigrationRunner, db_path: str):
        """token_usage table has all required columns."""
        runner.migrate(target=2)

        assert _table_has_column(db_path, "token_usage", "id")
        assert _table_has_column(db_path, "token_usage", "session_id")
        assert _table_has_column(db_path, "token_usage", "model")
        assert _table_has_column(db_path, "token_usage", "input_tokens")
        assert _table_has_column(db_path, "token_usage", "output_tokens")
        assert _table_has_column(db_path, "token_usage", "cost")
        assert _table_has_column(db_path, "token_usage", "context")
        assert _table_has_column(db_path, "token_usage", "timestamp")

    def test_token_usage_cost_views(self, runner: MigrationRunner, db_path: str):
        """Cost summary views are created."""
        runner.migrate(target=2)
        views = _get_views(db_path)
        assert "v_daily_costs" in views
        assert "v_session_costs" in views

    def test_can_insert_token_usage(self, runner: MigrationRunner, db_path: str):
        """Can insert data into token_usage table."""
        runner.migrate(target=2)

        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, cost)
                   VALUES ('test-session', 'claude-opus-4-5-20251101', 1000, 500, 0.025)"""
            )
            conn.commit()

            row = conn.execute("SELECT * FROM token_usage").fetchone()
            assert row is not None
        finally:
            conn.close()


# =============================================================================
# TEST: STORE ITEMS TABLE STRUCTURE
# =============================================================================


class TestStoreMigration:
    """Test migration 003 creates proper store_items schema."""

    def test_store_items_columns(self, runner: MigrationRunner, db_path: str):
        """store_items table has all required columns."""
        runner.migrate(target=3)

        assert _table_has_column(db_path, "store_items", "id")
        assert _table_has_column(db_path, "store_items", "namespace")
        assert _table_has_column(db_path, "store_items", "key")
        assert _table_has_column(db_path, "store_items", "value")
        assert _table_has_column(db_path, "store_items", "created_at")
        assert _table_has_column(db_path, "store_items", "updated_at")

    def test_store_items_indexes(self, runner: MigrationRunner, db_path: str):
        """store_items namespace indexes are created."""
        runner.migrate(target=3)
        indexes = _get_indexes(db_path)
        assert "idx_store_items_namespace" in indexes
        assert "idx_store_items_namespace_key" in indexes

    def test_store_items_trigger(self, runner: MigrationRunner, db_path: str):
        """store_items updated_at trigger is created."""
        runner.migrate(target=3)
        triggers = _get_triggers(db_path)
        assert "trg_store_items_updated" in triggers

    def test_can_insert_store_item(self, runner: MigrationRunner, db_path: str):
        """Can insert data into store_items table."""
        runner.migrate(target=3)

        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """INSERT INTO store_items (namespace, key, value)
                   VALUES ('["roles", "common"]', 'status', '{"passed": true}')"""
            )
            conn.commit()

            row = conn.execute("SELECT * FROM store_items").fetchone()
            assert row is not None
        finally:
            conn.close()

    def test_store_items_unique_constraint(self, runner: MigrationRunner, db_path: str):
        """store_items enforces unique (namespace, key) constraint."""
        runner.migrate(target=3)

        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """INSERT INTO store_items (namespace, key, value)
                   VALUES ('["test"]', 'key1', '{}')"""
            )
            conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO store_items (namespace, key, value)
                       VALUES ('["test"]', 'key1', '{"new": true}')"""
                )
        finally:
            conn.close()


# =============================================================================
# TEST: DATA PRESERVATION ACROSS MIGRATIONS
# =============================================================================


class TestDataPreservation:
    """Test that data is preserved when applying later migrations."""

    def test_data_preserved_after_migration_2(self, runner: MigrationRunner, db_path: str):
        """Data in roles table persists after applying migration 2."""
        runner.migrate(target=1)

        # Insert test data
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO roles (name, wave) VALUES ('test_role', 1)"
        )
        conn.commit()
        conn.close()

        # Apply next migration
        runner.migrate(target=2)

        # Verify data still exists
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT name FROM roles WHERE name='test_role'").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "test_role"

    def test_data_preserved_after_migration_3(self, runner: MigrationRunner, db_path: str):
        """Data in token_usage persists after applying migration 3."""
        runner.migrate(target=2)

        conn = sqlite3.connect(db_path)
        conn.execute(
            """INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, cost)
               VALUES ('preserve-test', 'claude-3-haiku', 100, 50, 0.001)"""
        )
        conn.commit()
        conn.close()

        runner.migrate(target=3)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT session_id FROM token_usage WHERE session_id='preserve-test'"
        ).fetchone()
        conn.close()
        assert row is not None


# =============================================================================
# TEST: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_schema_migrations_table_created_on_init(self, db_path: str):
        """schema_migrations table is created when MigrationRunner initializes."""
        runner = MigrationRunner(db_path)
        tables = _get_tables(db_path)
        assert "schema_migrations" in tables

    def test_rollback_at_version_0_noop(self, runner: MigrationRunner):
        """Rollback when at version 0 does nothing."""
        results = runner.rollback(target=0)
        assert len(results) == 0

    def test_rollback_above_current_noop(self, runner: MigrationRunner):
        """Rollback to version above current does nothing."""
        runner.migrate(target=1)
        results = runner.rollback(target=5)
        assert len(results) == 0
        assert runner.get_current_version() == 1

    def test_get_pending_after_all_applied(self, runner: MigrationRunner):
        """get_pending returns empty list when all applied."""
        runner.migrate()
        assert runner.get_pending() == []

    def test_multiple_runner_instances(self, db_path: str):
        """Multiple MigrationRunner instances see the same state."""
        runner1 = MigrationRunner(db_path)
        runner1.migrate(target=1)

        runner2 = MigrationRunner(db_path)
        assert runner2.get_current_version() == 1
        assert len(runner2.get_pending()) == 2
