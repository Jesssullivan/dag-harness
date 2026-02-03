#!/usr/bin/env python3
"""Post-tool-use hook for tracking results and file changes.

Follows the Claude Code hook protocol:
  - Reads JSON from stdin with tool_name, tool_input, and tool_output
  - Logs the tool completion and tracks any file changes
  - Always exits 0 (post hooks are informational only)

This script can be referenced in .claude/settings.json:

    {
        "hooks": {
            "PostToolUse": [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": "python harness/hooks/post_tool_use.py"
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
    env_path = os.environ.get("HARNESS_DB_PATH")
    if env_path:
        return env_path

    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".harness" / "harness.db"
        if candidate.parent.exists():
            return str(candidate)
        candidate = parent / "harness.db"
        if candidate.exists():
            return str(candidate)

    return ".harness/harness.db"


# Mapping of tool names to file change types
_FILE_CHANGE_TOOLS: dict[str, str] = {
    "Write": "create",
    "Edit": "edit",
    "NotebookEdit": "edit",
}


def main() -> None:
    """Main entry point for the post-tool-use hook.

    Reads hook data from stdin, logs the tool completion, and tracks
    any file changes made by file-modifying tools.
    """
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.exit(0)

        hook_data = json.loads(raw_input)
    except (json.JSONDecodeError, OSError):
        sys.exit(0)

    tool_name = hook_data.get("tool_name", "")
    tool_input = hook_data.get("tool_input", {})
    tool_output = hook_data.get("tool_output", "")

    db_path = _find_db_path()

    try:
        from harness.hooks.tool_logger import log_tool_invocation, track_file_change

        # Truncate result to avoid storing massive outputs
        result_str = str(tool_output)[:1000] if tool_output else None

        # Log the completion
        log_tool_invocation(
            tool_name, tool_input, phase="post", result=result_str, db_path=db_path
        )

        # Track file changes for file-modifying tools
        if tool_name in _FILE_CHANGE_TOOLS:
            file_path = tool_input.get("file_path", "")
            if file_path:
                change_type = _FILE_CHANGE_TOOLS[tool_name]
                track_file_change(
                    file_path, change_type, tool_name=tool_name, db_path=db_path
                )

        # Track Bash commands that might create/delete files
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            # Detect file creation via redirect
            if " > " in command or " >> " in command:
                # Extract target file from simple redirects
                parts = command.split(" > ")
                if len(parts) >= 2:
                    target = parts[-1].strip().split()[0] if parts[-1].strip() else ""
                    if target and not target.startswith("/dev/"):
                        track_file_change(
                            target, "create", tool_name="Bash", db_path=db_path
                        )

    except ImportError:
        # If harness package isn't installed, silently continue
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
