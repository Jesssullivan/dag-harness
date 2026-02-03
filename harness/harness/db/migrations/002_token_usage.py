"""
Migration 002: Token usage tracking.

Adds the token_usage table for LLM cost tracking, with indexes
and views for daily and session cost summaries.
"""

import sqlite3

from harness.db.migrations import Migration


class TokenUsageMigration(Migration):
    version = 2
    description = "Add token_usage table for LLM cost tracking"

    def up(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            -- Token usage / cost tracking
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost REAL NOT NULL,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for cost queries
            CREATE INDEX IF NOT EXISTS idx_token_usage_session ON token_usage(session_id);
            CREATE INDEX IF NOT EXISTS idx_token_usage_model ON token_usage(model);
            CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp ON token_usage(timestamp);

            -- View for daily cost summary
            CREATE VIEW IF NOT EXISTS v_daily_costs AS
            SELECT
                date(timestamp) as day,
                model,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cost) as total_cost,
                COUNT(*) as record_count
            FROM token_usage
            GROUP BY date(timestamp), model
            ORDER BY day DESC, model;

            -- View for session cost summary
            CREATE VIEW IF NOT EXISTS v_session_costs AS
            SELECT
                session_id,
                MIN(timestamp) as first_use,
                MAX(timestamp) as last_use,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cost) as total_cost,
                COUNT(*) as record_count,
                GROUP_CONCAT(DISTINCT model) as models_used
            FROM token_usage
            GROUP BY session_id
            ORDER BY total_cost DESC;
            """
        )

    def down(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP VIEW IF EXISTS v_session_costs;
            DROP VIEW IF EXISTS v_daily_costs;
            DROP INDEX IF EXISTS idx_token_usage_timestamp;
            DROP INDEX IF EXISTS idx_token_usage_model;
            DROP INDEX IF EXISTS idx_token_usage_session;
            DROP TABLE IF EXISTS token_usage;
            """
        )
