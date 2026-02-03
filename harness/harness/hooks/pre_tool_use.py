#!/usr/bin/env python3
"""Pre-tool-use hook for Claude Code harness integration.

Logs tool invocations and checks capabilities before tool execution.
Follows the Claude Code hook protocol:
  - Reads JSON from stdin with tool_name and tool_input
  - Exit 0 to allow the tool execution
  - Exit 2 to block with a JSON message on stdout

This script can be referenced in .claude/settings.json:

    {
        "hooks": {
            "PreToolUse": [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": "python harness/hooks/pre_tool_use.py"
                }]
            }]
        }
    }
"""

import json
import os
import sys
from pathlib import Path


def _find_db_path() -> str:
    """Locate the harness database path.

    Searches for .harness/harness.db relative to the current directory
    and common parent directories.

    Returns:
        Path to the database file.
    """
    # Check environment variable first
    env_path = os.environ.get("HARNESS_DB_PATH")
    if env_path:
        return env_path

    # Search upward for .harness directory
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".harness" / "harness.db"
        if candidate.parent.exists():
            return str(candidate)
        # Also check for harness.db directly
        candidate = parent / "harness.db"
        if candidate.exists():
            return str(candidate)

    return ".harness/harness.db"


def main() -> None:
    """Main entry point for the pre-tool-use hook.

    Reads hook data from stdin, logs the invocation, and checks
    for dangerous command patterns.
    """
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.exit(0)

        hook_data = json.loads(raw_input)
    except (json.JSONDecodeError, OSError):
        # If we can't parse the input, allow execution
        sys.exit(0)

    tool_name = hook_data.get("tool_name", "")
    tool_input = hook_data.get("tool_input", {})

    db_path = _find_db_path()

    # Import tool_logger (handle case where harness package isn't installed)
    try:
        from harness.hooks.tool_logger import (
            check_capability,
            get_dangerous_pattern,
            log_blocked_invocation,
            log_tool_invocation,
        )

        # Log the invocation
        log_tool_invocation(tool_name, tool_input, phase="pre", db_path=db_path)

        # Check capabilities for dangerous operations
        capabilities = hook_data.get("capabilities", [])
        if not check_capability(tool_name, tool_input, capabilities):
            command = tool_input.get("command", "")
            pattern = get_dangerous_pattern(command)
            reason = f"Blocked dangerous pattern: {pattern}"
            log_blocked_invocation(tool_name, tool_input, reason, db_path=db_path)
            print(json.dumps({"blocked": True, "reason": reason}))
            sys.exit(2)

    except ImportError:
        # Fallback: check dangerous patterns without database logging
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            dangerous_patterns = [
                "rm -rf",
                "git push --force",
                "DROP TABLE",
                "git reset --hard",
            ]
            for pattern in dangerous_patterns:
                if pattern in command:
                    print(json.dumps({"blocked": True, "reason": f"Blocked: {pattern}"}))
                    sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
