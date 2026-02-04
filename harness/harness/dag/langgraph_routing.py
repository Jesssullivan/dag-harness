"""
Conditional routing functions for LangGraph workflow.

These functions determine the next node based on current state.
Used with StateGraph.add_conditional_edges() for failure routing.
"""

import time
from typing import Literal

from langgraph.types import Send

from harness.dag.langgraph_state import BoxUpRoleState


# ============================================================================
# PARALLEL TEST ROUTING (Task #21)
# ============================================================================


def route_to_parallel_tests(state: BoxUpRoleState) -> list[Send]:
    """
    Fan out to parallel test nodes based on what tests are available.

    Uses LangGraph's Send API to execute molecule and pytest tests in parallel.
    This can reduce test phase time by 30%+ when both test types are present.

    Returns:
        List of Send objects targeting test nodes, or validate_deploy if no tests.
    """
    sends = []

    # Record test phase start time for performance tracking
    test_phase_start = time.time()

    # Always try molecule if tests exist
    has_molecule = state.get("has_molecule_tests", True)
    if has_molecule:
        # Pass current state to molecule node
        sends.append(Send("run_molecule", {**state, "test_phase_start_time": test_phase_start}))

    # Always try pytest (it will skip if no tests exist)
    sends.append(Send("run_pytest", {**state, "test_phase_start_time": test_phase_start}))

    # If no tests to run (unlikely but handle edge case), skip to merge
    if not sends:
        sends.append(
            Send("merge_test_results", {**state, "test_phase_start_time": test_phase_start})
        )

    return sends


def should_continue_after_merge(
    state: BoxUpRoleState,
) -> Literal["validate_deploy", "notify_failure"]:
    """
    Route after test merge - continues if all tests passed or were skipped.

    Handles partial failures:
    - If both fail: route to failure
    - If one fails: route to failure (strict mode)
    - If both pass/skip: continue to deploy validation
    """
    all_passed = state.get("all_tests_passed", False)
    if all_passed:
        return "validate_deploy"
    return "notify_failure"


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================


def should_continue_after_validation(
    state: BoxUpRoleState,
) -> Literal["analyze_deps", "notify_failure"]:
    """Route after validation node."""
    if state.get("errors"):
        return "notify_failure"
    return "analyze_deps"


def should_continue_after_deps(
    state: BoxUpRoleState,
) -> Literal["check_reverse_deps", "notify_failure"]:
    """Route after dependency analysis."""
    if state.get("errors"):
        return "notify_failure"
    return "check_reverse_deps"


def should_continue_after_reverse_deps(
    state: BoxUpRoleState,
) -> Literal["create_worktree", "notify_failure"]:
    """Route after reverse deps check."""
    if state.get("blocking_deps"):
        return "notify_failure"
    return "create_worktree"


def should_continue_after_worktree(
    state: BoxUpRoleState,
) -> Literal["run_molecule", "create_worktree", "notify_failure"]:
    """
    Route after worktree creation with recovery support (sequential mode).

    If worktree_force_recreate is True, loop back for retry.
    For parallel test execution, use _route_after_worktree_with_recovery in builder.
    """
    # Check for recovery loop - worktree needs to be recreated
    if state.get("worktree_force_recreate") or state.get("branch_force_recreate"):
        # Check max retries to prevent infinite loops
        recovery_attempts = state.get("recovery_attempts", [])
        worktree_attempts = [a for a in recovery_attempts if a.get("node") == "create_worktree"]
        max_attempts = state.get("max_recovery_attempts", 2)
        if len(worktree_attempts) >= max_attempts:
            return "notify_failure"
        return "create_worktree"  # Loop back for retry

    if state.get("errors"):
        return "notify_failure"
    return "run_molecule"


# Note: should_continue_after_molecule and should_continue_after_pytest are deprecated
# in favor of parallel execution. Kept for backwards compatibility with sequential mode.


def should_continue_after_molecule(
    state: BoxUpRoleState,
) -> Literal["run_pytest", "notify_failure"]:
    """
    Route after molecule tests (sequential mode only).

    Deprecated: Use parallel execution with route_to_parallel_tests instead.
    """
    if state.get("molecule_passed") is False:
        return "notify_failure"
    return "run_pytest"


def should_continue_after_pytest(
    state: BoxUpRoleState,
) -> Literal["validate_deploy", "notify_failure"]:
    """
    Route after pytest (sequential mode only).

    Deprecated: Use parallel execution with merge_test_results instead.
    """
    if state.get("pytest_passed") is False:
        return "notify_failure"
    return "validate_deploy"


def should_continue_after_deploy(
    state: BoxUpRoleState,
) -> Literal["create_commit", "validate_deploy", "recovery", "notify_failure"]:
    """
    Route after deploy validation with recovery support.

    v0.6.0: Routes to "recovery" subgraph for complex recoverable errors.
    Simple path corrections still use the self-loop (tier 1).
    """
    from harness.dag.recovery_config import get_recovery_config

    # Check if recovery resolved the issue (v0.6.0)
    if state.get("recovery_result") == "resolved" and state.get("last_error_node") == "validate_deploy":
        return "create_commit"

    # Check for recovery loop - path was corrected (tier 1 inline)
    if state.get("ansible_cwd") and state.get("deploy_passed") is None:
        config = get_recovery_config("validate_deploy")
        recovery_attempts = state.get("recovery_attempts", [])
        deploy_attempts = [a for a in recovery_attempts if a.get("node") == "validate_deploy"]
        if len(deploy_attempts) >= config.max_iterations:
            return "notify_failure"
        return "validate_deploy"  # Loop back for retry

    # v0.6.0: Route to recovery subgraph for complex errors
    if (
        state.get("last_error_node") == "validate_deploy"
        and state.get("last_error_type") in ("recoverable", "user_fixable")
        and state.get("recovery_result") != "resolved"
        and state.get("deploy_passed") is None
        and not state.get("errors")
    ):
        return "recovery"

    if state.get("deploy_passed") is False:
        return "notify_failure"
    return "create_commit"


def should_continue_after_commit(state: BoxUpRoleState) -> Literal["push_branch", "notify_failure"]:
    """Route after commit."""
    if state.get("errors"):
        return "notify_failure"
    return "push_branch"


def should_continue_after_push(state: BoxUpRoleState) -> Literal["create_issue", "notify_failure"]:
    """Route after push."""
    if not state.get("pushed"):
        return "notify_failure"
    return "create_issue"


def should_continue_after_issue(state: BoxUpRoleState) -> Literal["create_mr", "notify_failure"]:
    """Route after issue creation."""
    if not state.get("issue_iid"):
        return "notify_failure"
    return "create_mr"


def should_continue_after_mr(state: BoxUpRoleState) -> Literal["human_approval", "report_summary"]:
    """Route after MR creation - now goes to human approval instead of merge train."""
    if state.get("mr_iid"):
        return "human_approval"
    return "report_summary"


def should_continue_after_human_approval(
    state: BoxUpRoleState,
) -> Literal["add_to_merge_train", "notify_failure"]:
    """Route after human approval - continues to merge train or fails."""
    if state.get("human_approved"):
        return "add_to_merge_train"
    return "notify_failure"


# ============================================================================
# RECOVERY SUBGRAPH ROUTING (v0.6.0)
# ============================================================================


def route_after_recovery(
    state: BoxUpRoleState,
) -> Literal[
    "create_worktree", "validate_deploy", "run_molecule", "run_pytest", "notify_failure"
]:
    """
    Route after recovery subgraph completes.

    Based on recovery_result and last_error_node, routes back to the
    node that originally failed, or to notify_failure if escalated.
    """
    result = state.get("recovery_result")
    failed_node = state.get("last_error_node")

    if result == "escalate" or result is None:
        return "notify_failure"

    if result == "resolved":
        # Route back to the node that failed so it can retry
        valid_targets = {
            "create_worktree", "validate_deploy", "run_molecule", "run_pytest",
        }
        if failed_node in valid_targets:
            return failed_node
        return "notify_failure"

    # "continue" shouldn't reach here (handled internally by subgraph),
    # but route to failure as safety net
    return "notify_failure"


__all__ = [
    # Parallel test routing
    "route_to_parallel_tests",
    "should_continue_after_merge",
    # Conditional routing
    "should_continue_after_validation",
    "should_continue_after_deps",
    "should_continue_after_reverse_deps",
    "should_continue_after_worktree",
    "should_continue_after_molecule",
    "should_continue_after_pytest",
    "should_continue_after_deploy",
    "should_continue_after_commit",
    "should_continue_after_push",
    "should_continue_after_issue",
    "should_continue_after_mr",
    "should_continue_after_human_approval",
    # Recovery routing (v0.6.0)
    "route_after_recovery",
]
