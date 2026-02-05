#!/usr/bin/env python3
"""Universal pre-tool hook for harness operations.

This hook runs before all tool invocations and can:
- Log tool usage for debugging
- Enforce project-specific policies
- Inject environment variables

Environment variables:
    HARNESS_HOOK_DEBUG: Set to "1" to enable debug logging
    HARNESS_HOOK_STRICT: Set to "1" to enforce strict policies
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def log_debug(message: str):
    """Log debug message if debug mode is enabled."""
    if os.environ.get("HARNESS_HOOK_DEBUG") == "1":
        timestamp = datetime.now().isoformat()
        print(f"[HOOK DEBUG {timestamp}] {message}", file=sys.stderr)


def get_tool_context() -> dict:
    """Get tool context from environment."""
    return {
        "tool_name": os.environ.get("TOOL_NAME", "unknown"),
        "tool_input": os.environ.get("TOOL_INPUT", ""),
        "session_id": os.environ.get("SESSION_ID", ""),
        "working_dir": os.getcwd(),
    }


def check_policies(context: dict) -> tuple[bool, str]:
    """Check if tool invocation passes policy checks.

    Returns:
        Tuple of (passed, message)
    """
    tool_name = context["tool_name"]
    tool_input = context["tool_input"]

    # Policy: Prevent destructive git commands unless explicitly allowed
    if tool_name == "Bash":
        dangerous_patterns = [
            "git push --force",
            "git reset --hard",
            "rm -rf /",
            "rm -rf ~",
        ]

        for pattern in dangerous_patterns:
            if pattern in tool_input:
                if os.environ.get("HARNESS_ALLOW_DANGEROUS") != "1":
                    return False, f"Blocked dangerous command: {pattern}"

    return True, ""


def main():
    """Main entry point."""
    context = get_tool_context()
    log_debug(f"Tool: {context['tool_name']}")

    # Check policies if strict mode is enabled
    if os.environ.get("HARNESS_HOOK_STRICT") == "1":
        passed, message = check_policies(context)
        if not passed:
            print(f"Policy violation: {message}", file=sys.stderr)
            sys.exit(1)

    # All checks passed
    sys.exit(0)


if __name__ == "__main__":
    main()
