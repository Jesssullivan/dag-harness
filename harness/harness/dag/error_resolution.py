"""
Error classification and resolution strategies for LangGraph workflow.

This module implements the error resolution architecture that enables
self-correction loops for recoverable errors. Instead of binary
success/failure routing, nodes can:

1. Classify errors by type (transient, recoverable, user-fixable, unexpected)
2. Attempt resolution using context available in state
3. Loop back for retry with bounded attempt limits

Error Types:
- TRANSIENT: Network timeout, API rate limit - use RetryPolicy (existing)
- RECOVERABLE: Wrong path, missing config, worktree exists - self-correction loop
- USER_FIXABLE: Test failures, validation errors - agent or human fix
- UNEXPECTED: Crashes, bugs, unknown errors - fail fast, log details

Recovery Tiers (v0.6.0):
- Tier 0: LangGraph RetryPolicy (transient, existing)
- Tier 1: Inline fix via state update (v0.5.0 pattern)
- Tier 2: Recovery subgraph with tool execution
- Tier 3: Claude SDK agent spawned to fix code
- Tier 4: Fail fast, persist diagnostics, escalate (UNEXPECTED)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.dag.recovery_config import RecoveryConfig

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of errors for routing decisions."""

    TRANSIENT = "transient"  # Network, timeout - use RetryPolicy
    RECOVERABLE = "recoverable"  # Path issues, config - self-correction
    USER_FIXABLE = "user_fixable"  # Test failures - escalate with context
    UNEXPECTED = "unexpected"  # Bugs - fail fast


@dataclass
class ClassifiedError:
    """Classified error with resolution context."""

    error_type: ErrorType
    message: str
    resolution_hint: str | None = None
    context: dict[str, Any] | None = None


def classify_error(error_msg: str, node_name: str, state: dict) -> ClassifiedError:
    """
    Classify an error and suggest resolution strategy.

    Uses pattern matching on error messages and node context to determine
    the appropriate error type and resolution hint.

    Args:
        error_msg: The error message to classify
        node_name: The name of the node that failed
        state: Current workflow state for context

    Returns:
        ClassifiedError with type, message, and optional resolution hint
    """
    error_lower = error_msg.lower()

    # Path-related errors are recoverable
    if "could not be found" in error_lower or "no such file" in error_lower:
        if "playbook" in error_lower or "site.yml" in error_lower:
            return ClassifiedError(
                error_type=ErrorType.RECOVERABLE,
                message=error_msg,
                resolution_hint="check_ansible_subdirectory",
                context={"expected_path": "ansible/site.yml"},
            )
        if "role" in error_lower:
            return ClassifiedError(
                error_type=ErrorType.RECOVERABLE,
                message=error_msg,
                resolution_hint="check_role_path",
                context={"role_name": state.get("role_name")},
            )

    # Worktree exists is recoverable (clean and retry)
    if "worktree already exists" in error_lower or "already checked out" in error_lower:
        return ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message=error_msg,
            resolution_hint="remove_existing_worktree",
            context={"role_name": state.get("role_name")},
        )

    # Branch exists is recoverable
    if "branch" in error_lower and "already exists" in error_lower:
        return ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message=error_msg,
            resolution_hint="delete_existing_branch",
            context={"branch": f"sid/{state.get('role_name')}"},
        )

    # Fatal git error - branch checked out elsewhere
    if "is already checked out at" in error_lower:
        return ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message=error_msg,
            resolution_hint="remove_existing_worktree",
            context={"role_name": state.get("role_name")},
        )

    # Syntax check failure with specific patterns
    if node_name == "validate_deploy":
        if "syntax check failed" in error_lower:
            # Check if it's a path issue vs actual syntax error
            if "site.yml" in error_lower and (
                "could not be found" in error_lower or "not found" in error_lower
            ):
                return ClassifiedError(
                    error_type=ErrorType.RECOVERABLE,
                    message=error_msg,
                    resolution_hint="check_ansible_subdirectory",
                    context={"worktree_path": state.get("worktree_path")},
                )
            # Actual syntax errors are user-fixable
            return ClassifiedError(
                error_type=ErrorType.USER_FIXABLE,
                message=error_msg,
                resolution_hint="fix_ansible_syntax",
                context={"errors": error_msg[-2000:]},
            )

    # Test failures are user-fixable
    if node_name in ("run_molecule", "run_pytest") and "failed" in error_lower:
        return ClassifiedError(
            error_type=ErrorType.USER_FIXABLE,
            message=error_msg,
            resolution_hint="fix_failing_tests",
            context={"test_output": error_msg[-2000:]},
        )

    # Network/timeout errors are transient
    if any(
        term in error_lower
        for term in ["timeout", "connection refused", "network", "rate limit", "429", "503"]
    ):
        return ClassifiedError(
            error_type=ErrorType.TRANSIENT,
            message=error_msg,
            resolution_hint="retry_with_backoff",
        )

    # Permission errors may be recoverable
    if "permission denied" in error_lower:
        return ClassifiedError(
            error_type=ErrorType.USER_FIXABLE,
            message=error_msg,
            resolution_hint="check_permissions",
            context={"path": state.get("worktree_path")},
        )

    # Default to unexpected
    return ClassifiedError(error_type=ErrorType.UNEXPECTED, message=error_msg, resolution_hint=None)


def attempt_resolution(hint: str, context: dict | None, state: dict) -> dict | None:
    """
    Attempt to resolve an error using the provided hint and context.

    This function implements resolution strategies for recoverable errors.
    Each strategy returns state updates that, when applied, should allow
    the failing node to succeed on retry.

    Args:
        hint: Resolution hint from classify_error
        context: Additional context about the error
        state: Current workflow state

    Returns:
        State updates if resolution can be attempted, None otherwise
    """
    context = context or {}

    if hint == "check_ansible_subdirectory":
        # Fix: Use ansible/ subdirectory for playbook execution
        worktree_path = state.get("worktree_path", ".")
        ansible_path = Path(worktree_path) / "ansible"
        if ansible_path.exists() and (ansible_path / "site.yml").exists():
            return {"ansible_cwd": str(ansible_path)}
        # Also check if site.yml is at root
        root_site = Path(worktree_path) / "site.yml"
        if root_site.exists():
            return {"ansible_cwd": str(worktree_path)}
        return None

    if hint == "check_role_path":
        # Fix: Locate role in different possible paths
        role_name = context.get("role_name") or state.get("role_name")
        worktree_path = state.get("worktree_path", ".")
        possible_paths = [
            Path(worktree_path) / "ansible" / "roles" / role_name,
            Path(worktree_path) / "roles" / role_name,
            Path("ansible") / "roles" / role_name,
        ]
        for path in possible_paths:
            if path.exists():
                return {"role_path": str(path)}
        return None

    if hint == "remove_existing_worktree":
        # Signal that worktree should be force-recreated
        return {"worktree_force_recreate": True}

    if hint == "delete_existing_branch":
        # Signal that branch should be force-recreated
        return {"branch_force_recreate": True}

    # No resolution available for this hint
    return None


def get_recovery_attempt_count(state: dict, node_name: str) -> int:
    """
    Count how many recovery attempts have been made for a specific node.

    Args:
        state: Current workflow state
        node_name: Name of the node to check

    Returns:
        Number of recovery attempts for this node
    """
    recovery_attempts = state.get("recovery_attempts", [])
    return len([a for a in recovery_attempts if a.get("node") == node_name])


def should_attempt_recovery(state: dict, node_name: str, max_attempts: int = 2) -> bool:
    """
    Check if recovery should be attempted for a node.

    Args:
        state: Current workflow state
        node_name: Name of the node
        max_attempts: Maximum recovery attempts allowed

    Returns:
        True if recovery should be attempted
    """
    attempts = get_recovery_attempt_count(state, node_name)
    return attempts < max_attempts


def create_recovery_state_update(
    node_name: str,
    error: ClassifiedError,
    resolution: dict | None,
) -> dict:
    """
    Create state update for a recovery attempt.

    Args:
        node_name: Name of the failing node
        error: Classified error information
        resolution: Resolution state updates (or None)

    Returns:
        State update dict with recovery tracking
    """
    update: dict[str, Any] = {
        "recovery_context": {
            "node": node_name,
            "error": error.message,
            "error_type": error.error_type.value,
            "strategy": error.resolution_hint,
        },
        "recovery_attempts": [
            {
                "node": node_name,
                "hint": error.resolution_hint,
                "error_type": error.error_type.value,
            }
        ],
        "last_error_type": error.error_type.value,
        "last_error_message": error.message,
        "last_error_node": node_name,
    }

    if resolution:
        update.update(resolution)

    return update


def select_recovery_tier(error: ClassifiedError, node_name: str) -> int:
    """
    Select the appropriate recovery tier for a classified error.

    Uses error type and node configuration to determine the tier:
    - TRANSIENT -> Tier 0 (RetryPolicy handles it)
    - RECOVERABLE -> Tier 1 (simple inline fix) or Tier 2 (subgraph)
    - USER_FIXABLE -> Tier 3 (agent) if node allows, else Tier 2
    - UNEXPECTED -> Tier 4 (fail fast)

    The returned tier is capped by the node's max_tier from RecoveryConfig.

    Args:
        error: Classified error with type and hints.
        node_name: Name of the failing node.

    Returns:
        Recovery tier (0-4).
    """
    from harness.dag.recovery_config import get_max_tier

    max_tier = get_max_tier(node_name)

    if error.error_type == ErrorType.TRANSIENT:
        return 0
    if error.error_type == ErrorType.UNEXPECTED:
        return 4
    if error.error_type == ErrorType.RECOVERABLE:
        # Simple inline fixes are tier 1, complex ones tier 2
        simple_hints = {
            "remove_existing_worktree",
            "delete_existing_branch",
            "check_ansible_subdirectory",
            "check_role_path",
        }
        if error.resolution_hint in simple_hints:
            return min(1, max_tier)
        return min(2, max_tier)
    if error.error_type == ErrorType.USER_FIXABLE:
        # Test failures can be fixed by agent (tier 3) if allowed
        return min(3, max_tier)

    return min(1, max_tier)


def should_attempt_recovery_v2(
    state: dict,
    node_name: str,
    config: RecoveryConfig | None = None,
) -> bool:
    """
    Check if recovery should be attempted for a node (v0.6.0).

    Uses per-node RecoveryConfig for iteration limits instead of
    the global max_recovery_attempts.

    Args:
        state: Current workflow state.
        node_name: Name of the node.
        config: Optional RecoveryConfig override. If None, looked up from registry.

    Returns:
        True if recovery should be attempted.
    """
    if config is None:
        from harness.dag.recovery_config import get_recovery_config

        config = get_recovery_config(node_name)

    attempts = get_recovery_attempt_count(state, node_name)
    return attempts < config.max_iterations


def lookup_recovery_memory(
    store: Any,
    node_name: str,
    role_name: str,
    error_pattern: str,
) -> dict | None:
    """
    Look up prior recovery learnings from HarnessStore.

    Checks both role-specific and cross-role pattern namespaces
    for a matching fix.

    Args:
        store: HarnessStore instance (or None).
        node_name: Name of the failing node.
        role_name: Name of the role being processed.
        error_pattern: Key part of the error message to match.

    Returns:
        Dict with fix details if a match is found, None otherwise.
    """
    if store is None:
        return None

    try:
        from langgraph.store.base import GetOp

        # Check role-specific namespace first
        role_ns = ("recovery", node_name, role_name)
        results = store.batch([GetOp(namespace=role_ns, key=error_pattern)])
        if results and results[0] is not None:
            item = results[0]
            if hasattr(item, "value") and item.value.get("success"):
                logger.info(
                    f"Found recovery memory for {node_name}/{role_name}: {error_pattern}"
                )
                return item.value

        # Check cross-role patterns
        pattern_ns = ("recovery", node_name, "_patterns")
        results = store.batch([GetOp(namespace=pattern_ns, key=error_pattern)])
        if results and results[0] is not None:
            item = results[0]
            if hasattr(item, "value") and item.value.get("success"):
                logger.info(
                    f"Found cross-role recovery pattern for {node_name}: {error_pattern}"
                )
                return item.value

    except Exception as e:
        logger.debug(f"Recovery memory lookup failed: {e}")

    return None


def persist_recovery_memory(
    store: Any,
    node_name: str,
    role_name: str,
    error_pattern: str,
    fix_applied: str,
    success: bool,
    iterations: int,
) -> None:
    """
    Persist recovery outcome to HarnessStore for future runs.

    Saves to both role-specific and cross-role pattern namespaces.

    Args:
        store: HarnessStore instance (or None).
        node_name: Name of the failing node.
        role_name: Name of the role being processed.
        error_pattern: Key part of the error message.
        fix_applied: Description of the fix that was applied.
        success: Whether the fix resolved the error.
        iterations: Number of iterations it took.
    """
    if store is None:
        return

    try:
        import datetime

        from langgraph.store.base import PutOp

        entry = {
            "error_pattern": error_pattern,
            "fix_applied": fix_applied,
            "success": success,
            "iterations": iterations,
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }

        ops = [
            # Role-specific entry
            PutOp(
                namespace=("recovery", node_name, role_name),
                key=error_pattern,
                value=entry,
            ),
        ]

        # Only persist to cross-role patterns if successful
        if success:
            ops.append(
                PutOp(
                    namespace=("recovery", node_name, "_patterns"),
                    key=error_pattern,
                    value=entry,
                )
            )

        store.batch(ops)
        logger.info(
            f"Persisted recovery memory: {node_name}/{role_name} "
            f"pattern={error_pattern} success={success}"
        )

    except Exception as e:
        logger.debug(f"Failed to persist recovery memory: {e}")


__all__ = [
    "ErrorType",
    "ClassifiedError",
    "classify_error",
    "attempt_resolution",
    "get_recovery_attempt_count",
    "should_attempt_recovery",
    "create_recovery_state_update",
    # v0.6.0
    "select_recovery_tier",
    "should_attempt_recovery_v2",
    "lookup_recovery_memory",
    "persist_recovery_memory",
]
