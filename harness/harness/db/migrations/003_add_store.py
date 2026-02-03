"""
Migration 003: LangGraph Store.

Creates the store_items table for cross-thread memory persistence
(LangGraph Store interface), with indexes for namespace queries
and an updated_at trigger.
"""

import sqlite3

from harness.db.migrations import Migration


class AddStoreMigration(Migration):
    version = 3
    description = "Add store_items table for LangGraph Store cross-thread memory"

    def up(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            -- LangGraph Store: cross-thread memory persistence
            CREATE TABLE IF NOT EXISTS store_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(namespace, key)
            );

            CREATE INDEX IF NOT EXISTS idx_store_items_namespace ON store_items(namespace);
            CREATE INDEX IF NOT EXISTS idx_store_items_namespace_key ON store_items(namespace, key);

            -- Trigger for updated_at
            CREATE TRIGGER IF NOT EXISTS trg_store_items_updated
            AFTER UPDATE ON store_items
            BEGIN
                UPDATE store_items SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
            """
        )

    def down(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS trg_store_items_updated;
            DROP INDEX IF EXISTS idx_store_items_namespace_key;
            DROP INDEX IF EXISTS idx_store_items_namespace;
            DROP TABLE IF EXISTS store_items;
            """
        )
