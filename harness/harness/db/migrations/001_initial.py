"""
Migration 001: Initial schema.

Creates all base tables for the harness database:
- roles, role_dependencies, credentials
- worktrees
- iterations, issues, merge_requests
- workflow_definitions, workflow_executions, node_executions
- test_runs, test_cases
- audit_log
- execution_contexts, context_capabilities, tool_invocations
- test_regressions, merge_train_entries
- agent_sessions, agent_file_changes
- Views and triggers
"""

import sqlite3

from harness.db.migrations import Migration


class InitialSchema(Migration):
    version = 1
    description = "Initial schema with core tables, views, and triggers"

    def up(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            -- ================================================================
            -- CORE ENTITIES
            -- ================================================================

            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                wave INTEGER NOT NULL CHECK (wave BETWEEN 0 AND 10),
                wave_name TEXT,
                description TEXT,
                molecule_path TEXT,
                has_molecule_tests BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS role_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                depends_on_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                dependency_type TEXT NOT NULL CHECK (dependency_type IN ('explicit', 'implicit', 'credential')),
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(role_id, depends_on_id, dependency_type)
            );

            CREATE TABLE IF NOT EXISTS credentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                entry_name TEXT NOT NULL,
                purpose TEXT,
                is_base58 BOOLEAN DEFAULT 0,
                attribute TEXT,
                source_file TEXT,
                source_line INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(role_id, entry_name, attribute)
            );

            -- ================================================================
            -- GIT WORKTREES
            -- ================================================================

            CREATE TABLE IF NOT EXISTS worktrees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                branch TEXT NOT NULL,
                base_commit TEXT,
                current_commit TEXT,
                commits_ahead INTEGER DEFAULT 0,
                commits_behind INTEGER DEFAULT 0,
                uncommitted_changes INTEGER DEFAULT 0,
                status TEXT CHECK (status IN ('active', 'stale', 'dirty', 'merged', 'pruned')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(role_id)
            );

            -- ================================================================
            -- GITLAB ENTITIES
            -- ================================================================

            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY,
                title TEXT,
                state TEXT CHECK (state IN ('opened', 'closed')),
                start_date DATE,
                due_date DATE,
                group_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY,
                iid INTEGER NOT NULL,
                role_id INTEGER REFERENCES roles(id) ON DELETE SET NULL,
                iteration_id INTEGER REFERENCES iterations(id) ON DELETE SET NULL,
                title TEXT NOT NULL,
                state TEXT CHECK (state IN ('opened', 'closed')),
                web_url TEXT,
                labels TEXT,
                assignee TEXT,
                weight INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS merge_requests (
                id INTEGER PRIMARY KEY,
                iid INTEGER NOT NULL,
                role_id INTEGER REFERENCES roles(id) ON DELETE SET NULL,
                issue_id INTEGER REFERENCES issues(id) ON DELETE SET NULL,
                source_branch TEXT NOT NULL,
                target_branch TEXT DEFAULT 'main',
                title TEXT NOT NULL,
                state TEXT CHECK (state IN ('opened', 'merged', 'closed')),
                web_url TEXT,
                merge_status TEXT,
                squash_on_merge BOOLEAN DEFAULT 1,
                remove_source_branch BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- ================================================================
            -- WORKFLOW EXECUTION (DAG State)
            -- ================================================================

            CREATE TABLE IF NOT EXISTS workflow_definitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                nodes_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS workflow_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id INTEGER NOT NULL REFERENCES workflow_definitions(id),
                role_id INTEGER NOT NULL REFERENCES roles(id),
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'paused', 'completed', 'failed', 'cancelled')),
                current_node TEXT,
                checkpoint_data TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS node_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id INTEGER NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
                node_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                retry_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(execution_id, node_name)
            );

            -- ================================================================
            -- TEST RESULTS
            -- ================================================================

            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                worktree_id INTEGER REFERENCES worktrees(id) ON DELETE SET NULL,
                execution_id INTEGER REFERENCES workflow_executions(id) ON DELETE SET NULL,
                test_type TEXT NOT NULL CHECK (test_type IN ('molecule', 'pytest', 'deploy')),
                status TEXT NOT NULL CHECK (status IN ('running', 'passed', 'failed', 'skipped')),
                duration_seconds INTEGER,
                log_path TEXT,
                output_json TEXT,
                commit_sha TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS test_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_run_id INTEGER NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('passed', 'failed', 'skipped', 'error')),
                duration_ms INTEGER,
                error_message TEXT,
                failure_output TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- ================================================================
            -- AUDIT LOG
            -- ================================================================

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                actor TEXT DEFAULT 'harness',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- ================================================================
            -- SEE/ACP CONTEXT CONTROL
            -- ================================================================

            CREATE TABLE IF NOT EXISTS execution_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id TEXT,
                request_id TEXT,
                capabilities TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS context_capabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id INTEGER NOT NULL REFERENCES execution_contexts(id) ON DELETE CASCADE,
                capability TEXT NOT NULL,
                scope TEXT,
                granted_by TEXT DEFAULT 'system',
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                revoked_at TIMESTAMP,
                UNIQUE(context_id, capability, scope)
            );

            CREATE TABLE IF NOT EXISTS tool_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id INTEGER REFERENCES execution_contexts(id) ON DELETE SET NULL,
                tool_name TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                status TEXT CHECK (status IN ('pending', 'running', 'completed', 'failed', 'blocked')),
                duration_ms INTEGER,
                blocked_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- ================================================================
            -- TEST REGRESSION TRACKING
            -- ================================================================

            CREATE TABLE IF NOT EXISTS test_regressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                test_name TEXT NOT NULL,
                test_type TEXT CHECK (test_type IN ('molecule', 'pytest', 'deploy')),
                first_failure_run_id INTEGER REFERENCES test_runs(id) ON DELETE SET NULL,
                resolved_run_id INTEGER REFERENCES test_runs(id) ON DELETE SET NULL,
                failure_count INTEGER DEFAULT 1,
                consecutive_failures INTEGER DEFAULT 1,
                last_failure_at TIMESTAMP,
                last_error_message TEXT,
                status TEXT CHECK (status IN ('active', 'resolved', 'flaky', 'known_issue')) DEFAULT 'active',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(role_id, test_name, test_type)
            );

            CREATE TABLE IF NOT EXISTS merge_train_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mr_id INTEGER NOT NULL REFERENCES merge_requests(id) ON DELETE CASCADE,
                position INTEGER,
                target_branch TEXT DEFAULT 'main',
                status TEXT CHECK (status IN ('queued', 'merging', 'merged', 'failed', 'cancelled')),
                pipeline_id INTEGER,
                pipeline_status TEXT,
                queued_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                failure_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- ================================================================
            -- AGENT SESSIONS
            -- ================================================================

            CREATE TABLE IF NOT EXISTS agent_sessions (
                id TEXT PRIMARY KEY,
                execution_id INTEGER REFERENCES workflow_executions(id) ON DELETE SET NULL,
                task TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'needs_human', 'cancelled')),
                output TEXT,
                error_message TEXT,
                intervention_reason TEXT,
                context_json TEXT,
                progress_json TEXT,
                working_dir TEXT NOT NULL,
                pid INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS agent_file_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                change_type TEXT NOT NULL CHECK (change_type IN ('create', 'modify', 'delete', 'rename')),
                diff TEXT,
                old_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, file_path, change_type)
            );

            -- ================================================================
            -- INDEXES
            -- ================================================================

            CREATE INDEX IF NOT EXISTS idx_role_deps_role ON role_dependencies(role_id);
            CREATE INDEX IF NOT EXISTS idx_role_deps_depends ON role_dependencies(depends_on_id);
            CREATE INDEX IF NOT EXISTS idx_credentials_role ON credentials(role_id);
            CREATE INDEX IF NOT EXISTS idx_worktrees_role ON worktrees(role_id);
            CREATE INDEX IF NOT EXISTS idx_worktrees_status ON worktrees(status);
            CREATE INDEX IF NOT EXISTS idx_issues_role ON issues(role_id);
            CREATE INDEX IF NOT EXISTS idx_issues_iteration ON issues(iteration_id);
            CREATE INDEX IF NOT EXISTS idx_mrs_role ON merge_requests(role_id);
            CREATE INDEX IF NOT EXISTS idx_workflow_exec_status ON workflow_executions(status);
            CREATE INDEX IF NOT EXISTS idx_workflow_exec_role ON workflow_executions(role_id);
            CREATE INDEX IF NOT EXISTS idx_node_exec_status ON node_executions(status);
            CREATE INDEX IF NOT EXISTS idx_test_runs_role ON test_runs(role_id);
            CREATE INDEX IF NOT EXISTS idx_test_runs_status ON test_runs(status);
            CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);

            CREATE INDEX IF NOT EXISTS idx_exec_ctx_session ON execution_contexts(session_id);
            CREATE INDEX IF NOT EXISTS idx_exec_ctx_user ON execution_contexts(user_id);
            CREATE INDEX IF NOT EXISTS idx_ctx_caps_context ON context_capabilities(context_id);
            CREATE INDEX IF NOT EXISTS idx_ctx_caps_capability ON context_capabilities(capability);
            CREATE INDEX IF NOT EXISTS idx_tool_inv_context ON tool_invocations(context_id);
            CREATE INDEX IF NOT EXISTS idx_tool_inv_tool ON tool_invocations(tool_name);
            CREATE INDEX IF NOT EXISTS idx_tool_inv_status ON tool_invocations(status);

            CREATE INDEX IF NOT EXISTS idx_regressions_role ON test_regressions(role_id);
            CREATE INDEX IF NOT EXISTS idx_regressions_status ON test_regressions(status);
            CREATE INDEX IF NOT EXISTS idx_regressions_test ON test_regressions(test_name);
            CREATE INDEX IF NOT EXISTS idx_merge_train_mr ON merge_train_entries(mr_id);
            CREATE INDEX IF NOT EXISTS idx_merge_train_status ON merge_train_entries(status);

            CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status);
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_execution ON agent_sessions(execution_id);
            CREATE INDEX IF NOT EXISTS idx_agent_file_changes_session ON agent_file_changes(session_id);

            -- ================================================================
            -- VIEWS
            -- ================================================================

            CREATE VIEW IF NOT EXISTS v_role_status AS
            SELECT
                r.id,
                r.name,
                r.wave,
                r.wave_name,
                w.status as worktree_status,
                w.commits_ahead,
                w.commits_behind,
                i.state as issue_state,
                i.web_url as issue_url,
                mr.state as mr_state,
                mr.web_url as mr_url,
                (SELECT COUNT(*) FROM test_runs tr WHERE tr.role_id = r.id AND tr.status = 'passed') as passed_tests,
                (SELECT COUNT(*) FROM test_runs tr WHERE tr.role_id = r.id AND tr.status = 'failed') as failed_tests
            FROM roles r
            LEFT JOIN worktrees w ON w.role_id = r.id
            LEFT JOIN issues i ON i.role_id = r.id
            LEFT JOIN merge_requests mr ON mr.role_id = r.id;

            CREATE VIEW IF NOT EXISTS v_dependency_graph AS
            SELECT
                r1.name as from_role,
                r2.name as to_role,
                rd.dependency_type,
                r1.wave as from_wave,
                r2.wave as to_wave
            FROM role_dependencies rd
            JOIN roles r1 ON rd.role_id = r1.id
            JOIN roles r2 ON rd.depends_on_id = r2.id;

            CREATE VIEW IF NOT EXISTS v_active_regressions AS
            SELECT
                tr.id,
                r.name as role_name,
                r.wave,
                tr.test_name,
                tr.test_type,
                tr.failure_count,
                tr.consecutive_failures,
                tr.last_failure_at,
                tr.last_error_message,
                tr.status,
                tr.notes
            FROM test_regressions tr
            JOIN roles r ON tr.role_id = r.id
            WHERE tr.status IN ('active', 'flaky')
            ORDER BY tr.consecutive_failures DESC, tr.last_failure_at DESC;

            CREATE VIEW IF NOT EXISTS v_context_capabilities AS
            SELECT
                ec.session_id,
                ec.user_id,
                cc.capability,
                cc.scope,
                cc.granted_at,
                cc.expires_at,
                CASE
                    WHEN cc.revoked_at IS NOT NULL THEN 'revoked'
                    WHEN cc.expires_at IS NOT NULL AND cc.expires_at < CURRENT_TIMESTAMP THEN 'expired'
                    ELSE 'active'
                END as status
            FROM context_capabilities cc
            JOIN execution_contexts ec ON cc.context_id = ec.id;

            CREATE VIEW IF NOT EXISTS v_agent_sessions AS
            SELECT
                a.id,
                a.execution_id,
                a.task,
                a.status,
                a.working_dir,
                a.intervention_reason,
                a.created_at,
                a.started_at,
                a.completed_at,
                a.pid,
                (SELECT COUNT(*) FROM agent_file_changes fc WHERE fc.session_id = a.id) as file_change_count,
                CASE
                    WHEN a.completed_at IS NOT NULL AND a.started_at IS NOT NULL
                    THEN (julianday(a.completed_at) - julianday(a.started_at)) * 86400
                    ELSE NULL
                END as duration_seconds
            FROM agent_sessions a;

            -- ================================================================
            -- TRIGGERS FOR UPDATED_AT
            -- ================================================================

            CREATE TRIGGER IF NOT EXISTS trg_roles_updated
            AFTER UPDATE ON roles
            BEGIN
                UPDATE roles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_worktrees_updated
            AFTER UPDATE ON worktrees
            BEGIN
                UPDATE worktrees SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_workflow_exec_updated
            AFTER UPDATE ON workflow_executions
            BEGIN
                UPDATE workflow_executions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_exec_ctx_updated
            AFTER UPDATE ON execution_contexts
            BEGIN
                UPDATE execution_contexts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_regressions_updated
            AFTER UPDATE ON test_regressions
            BEGIN
                UPDATE test_regressions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_merge_train_updated
            AFTER UPDATE ON merge_train_entries
            BEGIN
                UPDATE merge_train_entries SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            """
        )

    def down(self, conn: sqlite3.Connection) -> None:
        # Drop views first (they depend on tables)
        conn.executescript(
            """
            DROP VIEW IF EXISTS v_agent_sessions;
            DROP VIEW IF EXISTS v_context_capabilities;
            DROP VIEW IF EXISTS v_active_regressions;
            DROP VIEW IF EXISTS v_dependency_graph;
            DROP VIEW IF EXISTS v_role_status;

            DROP TRIGGER IF EXISTS trg_merge_train_updated;
            DROP TRIGGER IF EXISTS trg_regressions_updated;
            DROP TRIGGER IF EXISTS trg_exec_ctx_updated;
            DROP TRIGGER IF EXISTS trg_workflow_exec_updated;
            DROP TRIGGER IF EXISTS trg_worktrees_updated;
            DROP TRIGGER IF EXISTS trg_roles_updated;

            DROP TABLE IF EXISTS agent_file_changes;
            DROP TABLE IF EXISTS agent_sessions;
            DROP TABLE IF EXISTS merge_train_entries;
            DROP TABLE IF EXISTS test_regressions;
            DROP TABLE IF EXISTS tool_invocations;
            DROP TABLE IF EXISTS context_capabilities;
            DROP TABLE IF EXISTS execution_contexts;
            DROP TABLE IF EXISTS audit_log;
            DROP TABLE IF EXISTS test_cases;
            DROP TABLE IF EXISTS test_runs;
            DROP TABLE IF EXISTS node_executions;
            DROP TABLE IF EXISTS workflow_executions;
            DROP TABLE IF EXISTS workflow_definitions;
            DROP TABLE IF EXISTS merge_requests;
            DROP TABLE IF EXISTS issues;
            DROP TABLE IF EXISTS iterations;
            DROP TABLE IF EXISTS worktrees;
            DROP TABLE IF EXISTS credentials;
            DROP TABLE IF EXISTS role_dependencies;
            DROP TABLE IF EXISTS roles;
            """
        )
