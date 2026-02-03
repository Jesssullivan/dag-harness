"""Log Claude Code tool invocations to harness.db for audit trail.

This module provides functions for logging tool invocations, checking
capabilities, and tracking file changes. It integrates with the existing
SQLite schema (tool_invocations and agent_file_changes tables) to provide
a persistent audit trail of all tool operations.

The functions here are designed to be called from Claude Code hook scripts
(pre_tool_use.py and post_tool_use.py) as well as from the harness hooks
framework directly.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Dangerous command patterns that require explicit capability grants
DANGEROUS_PATTERNS: list[str] = [
    "rm -rf",
    "git push --force",
    "git reset --hard",
    "DROP TABLE",
    "DROP DATABASE",
    "TRUNCATE TABLE",
    "DELETE FROM",
    "FORMAT",
    "mkfs",
    "dd if=",
    "> /dev/",
]

# Tool operations that modify files
FILE_TOOLS: dict[str, str] = {
    "Write": "create",
    "Edit": "edit",
    "NotebookEdit": "edit",
}


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection with proper settings.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        A configured sqlite3 Connection.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the tool_invocations table exists (idempotent).

    This is a fallback for cases where the full schema hasn't been applied,
    such as when the hook scripts run standalone with a fresh database.

    Args:
        conn: Active database connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_id INTEGER,
            tool_name TEXT NOT NULL,
            arguments TEXT,
            result TEXT,
            status TEXT CHECK (status IN ('pending', 'running', 'completed', 'failed', 'blocked')),
            duration_ms INTEGER,
            blocked_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            change_type TEXT NOT NULL CHECK (change_type IN ('create', 'edit', 'delete', 'rename')),
            tool_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def log_tool_invocation(
    tool_name: str,
    tool_input: dict[str, Any],
    phase: str = "pre",
    result: str | None = None,
    db_path: str = ".harness/harness.db",
) -> int | None:
    """Log a tool invocation to the tool_invocations table.

    For the "pre" phase, creates a new record with status "pending".
    For the "post" phase, updates the existing record with the result
    and sets status to "completed".

    Args:
        tool_name: Name of the tool being invoked (e.g., "Bash", "Write").
        tool_input: Dictionary of tool input parameters.
        phase: Either "pre" (before execution) or "post" (after execution).
        result: Tool result string (only used for "post" phase).
        db_path: Path to the harness SQLite database.

    Returns:
        The row ID of the inserted/updated record, or None on error.
    """
    try:
        db_file = Path(db_path)
        if not db_file.parent.exists():
            db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = _get_connection(str(db_file))
        _ensure_schema(conn)

        now = datetime.now(timezone.utc).isoformat()
        arguments_json = json.dumps(tool_input, default=str)

        if phase == "pre":
            cursor = conn.execute(
                """INSERT INTO tool_invocations
                   (tool_name, arguments, status, created_at)
                   VALUES (?, ?, 'pending', ?)""",
                (tool_name, arguments_json, now),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.debug("Logged pre-tool invocation: %s (id=%d)", tool_name, row_id)
            conn.close()
            return row_id

        elif phase == "post":
            # Try to find the most recent pending invocation for this tool
            cursor = conn.execute(
                """SELECT id FROM tool_invocations
                   WHERE tool_name = ? AND status = 'pending'
                   ORDER BY created_at DESC LIMIT 1""",
                (tool_name,),
            )
            row = cursor.fetchone()

            if row:
                result_text = result[:10000] if result else None
                conn.execute(
                    """UPDATE tool_invocations
                       SET status = 'completed', result = ?, completed_at = ?
                       WHERE id = ?""",
                    (result_text, now, row["id"]),
                )
                conn.commit()
                row_id = row["id"]
                logger.debug("Logged post-tool completion: %s (id=%d)", tool_name, row_id)
                conn.close()
                return row_id
            else:
                # No pending invocation found; create a completed record
                cursor = conn.execute(
                    """INSERT INTO tool_invocations
                       (tool_name, arguments, result, status, created_at, completed_at)
                       VALUES (?, ?, ?, 'completed', ?, ?)""",
                    (tool_name, arguments_json, result, now, now),
                )
                conn.commit()
                row_id = cursor.lastrowid
                conn.close()
                return row_id

    except Exception as e:
        logger.warning("Failed to log tool invocation: %s", e)
        return None


def log_blocked_invocation(
    tool_name: str,
    tool_input: dict[str, Any],
    reason: str,
    db_path: str = ".harness/harness.db",
) -> int | None:
    """Log a blocked tool invocation.

    Args:
        tool_name: Name of the tool that was blocked.
        tool_input: Dictionary of tool input parameters.
        reason: Reason the invocation was blocked.
        db_path: Path to the harness SQLite database.

    Returns:
        The row ID of the inserted record, or None on error.
    """
    try:
        db_file = Path(db_path)
        if not db_file.parent.exists():
            db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = _get_connection(str(db_file))
        _ensure_schema(conn)

        now = datetime.now(timezone.utc).isoformat()
        arguments_json = json.dumps(tool_input, default=str)

        cursor = conn.execute(
            """INSERT INTO tool_invocations
               (tool_name, arguments, status, blocked_reason, created_at)
               VALUES (?, ?, 'blocked', ?, ?)""",
            (tool_name, arguments_json, reason, now),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.info("Blocked tool invocation: %s reason=%s", tool_name, reason)
        conn.close()
        return row_id

    except Exception as e:
        logger.warning("Failed to log blocked invocation: %s", e)
        return None


def check_capability(
    tool_name: str,
    tool_input: dict[str, Any],
    capabilities: list[str] | None = None,
) -> bool:
    """Check if the current context allows this tool operation.

    Examines the tool name and input for dangerous patterns. If a dangerous
    pattern is detected, checks whether the provided capabilities include
    an explicit override.

    Args:
        tool_name: Name of the tool being invoked.
        tool_input: Dictionary of tool input parameters.
        capabilities: List of granted capabilities (e.g., ["destructive", "force-push"]).

    Returns:
        True if the operation is allowed, False if it should be blocked.
    """
    capabilities = capabilities or []

    # Non-Bash tools are generally safe
    if tool_name != "Bash":
        return True

    command = tool_input.get("command", "")
    if not command:
        return True

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            # Check if there's an explicit capability override
            if "destructive" in capabilities:
                logger.info(
                    "Allowing dangerous command (destructive capability granted): %s",
                    pattern,
                )
                return True
            # Check for specific pattern-based capabilities
            pattern_cap = pattern.replace(" ", "-").lower()
            if pattern_cap in capabilities:
                logger.info(
                    "Allowing dangerous command (specific capability granted): %s",
                    pattern,
                )
                return True

            logger.warning("Blocking dangerous command pattern: %s", pattern)
            return False

    return True


def track_file_change(
    file_path: str,
    change_type: str,
    tool_name: str | None = None,
    db_path: str = ".harness/harness.db",
) -> int | None:
    """Track a file change for audit trail.

    Records file creation, modification, or deletion events in the database
    for later review and audit.

    Args:
        file_path: Absolute or relative path to the changed file.
        change_type: Type of change - "create", "edit", or "delete".
        tool_name: Name of the tool that made the change.
        db_path: Path to the harness SQLite database.

    Returns:
        The row ID of the inserted record, or None on error.
    """
    valid_types = ("create", "edit", "delete", "rename")
    if change_type not in valid_types:
        logger.warning("Invalid change_type: %s (expected one of %s)", change_type, valid_types)
        return None

    try:
        db_file = Path(db_path)
        if not db_file.parent.exists():
            db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = _get_connection(str(db_file))
        _ensure_schema(conn)

        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """INSERT INTO file_changes
               (file_path, change_type, tool_name, created_at)
               VALUES (?, ?, ?, ?)""",
            (file_path, change_type, tool_name, now),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.debug("Tracked file change: %s %s (id=%d)", change_type, file_path, row_id)
        conn.close()
        return row_id

    except Exception as e:
        logger.warning("Failed to track file change: %s", e)
        return None


def get_dangerous_pattern(command: str) -> str | None:
    """Check if a command contains a dangerous pattern.

    Args:
        command: The shell command to check.

    Returns:
        The matched dangerous pattern, or None if safe.
    """
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return pattern
    return None
