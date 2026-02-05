#!/usr/bin/env python3
"""Audit logger hook for harness operations.

This hook runs after all tool invocations to log:
- Tool name and input
- Execution result
- Timing information

Logs are written to ~/.claude/audit.jsonl by default.

Environment variables:
    HARNESS_AUDIT_LOG: Path to audit log file
    HARNESS_AUDIT_ENABLED: Set to "0" to disable logging
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def get_audit_log_path() -> Path:
    """Get path to audit log file."""
    default = Path.home() / ".claude" / "audit.jsonl"
    return Path(os.environ.get("HARNESS_AUDIT_LOG", str(default)))


def log_tool_usage():
    """Log tool usage to audit log."""
    if os.environ.get("HARNESS_AUDIT_ENABLED") == "0":
        return

    # Collect audit data
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": os.environ.get("TOOL_NAME", "unknown"),
        "tool_input": os.environ.get("TOOL_INPUT", "")[:1000],  # Truncate long inputs
        "exit_code": os.environ.get("EXIT_CODE", ""),
        "session_id": os.environ.get("SESSION_ID", ""),
        "working_dir": os.getcwd(),
    }

    # Write to audit log
    try:
        log_path = get_audit_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    except Exception as e:
        # Don't fail on audit log errors
        print(f"Audit log warning: {e}", file=sys.stderr)


def main():
    """Main entry point."""
    log_tool_usage()
    sys.exit(0)


if __name__ == "__main__":
    main()
