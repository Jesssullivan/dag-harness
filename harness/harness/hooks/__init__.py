"""Enhanced hooks framework for DAG harness agent integration.

This module provides a comprehensive hooks system for intercepting and
modifying agent behavior at various lifecycle points:

- PreToolUse: Modify tool inputs before execution
- PostToolUse: Process tool outputs after execution
- SubagentStart: Notification when subagent is spawned
- SubagentStop: Audit hook for subagent completion

Hook Types:
    - PreToolUseHook: Can modify input before tool execution
    - PostToolUseHook: Can process/log output after tool execution
    - SubagentStartHook: Called when a subagent is spawned
    - SubagentStopHook: Called when a subagent completes

Specialized Hooks:
    - VerificationHook: For verification subagents
    - FileChangeTrackerHook: Tracks all file modifications
    - AuditHook: Logs all subagent activity
"""

from harness.hooks.audit import (
    AuditEntry,
    AuditHook,
    AuditLevel,
    AuditLogger,
)
from harness.hooks.base import (
    Hook,
    HookContext,
    HookManager,
    HookPriority,
    HookResult,
    PostToolUseHook,
    PreToolUseHook,
    SubagentStartHook,
    SubagentStopHook,
)
from harness.hooks.file_tracker import (
    FileChange,
    FileChangeTrackerHook,
    FileChangeType,
    TrackedFile,
)
from harness.hooks.tool_logger import (
    DANGEROUS_PATTERNS,
    check_capability,
    get_dangerous_pattern,
    log_blocked_invocation,
    log_tool_invocation,
    track_file_change,
)
from harness.hooks.verification import (
    VerificationHook,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    # Base classes
    "Hook",
    "HookContext",
    "HookManager",
    "HookPriority",
    "HookResult",
    "PreToolUseHook",
    "PostToolUseHook",
    "SubagentStartHook",
    "SubagentStopHook",
    # Audit hooks
    "AuditEntry",
    "AuditHook",
    "AuditLevel",
    "AuditLogger",
    # File tracker hooks
    "FileChange",
    "FileChangeTrackerHook",
    "FileChangeType",
    "TrackedFile",
    # Tool logger
    "DANGEROUS_PATTERNS",
    "check_capability",
    "get_dangerous_pattern",
    "log_blocked_invocation",
    "log_tool_invocation",
    "track_file_change",
    # Verification hooks
    "VerificationHook",
    "VerificationResult",
    "VerificationStatus",
]
