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
) -> Literal["parallel_tests", "notify_failure"]:
    """
    Route after worktree creation.

    Task #21: Now routes to parallel test execution via route_to_parallel_tests.
    """
    if state.get("errors"):
        return "notify_failure"
    return "parallel_tests"


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
) -> Literal["create_commit", "notify_failure"]:
    """Route after deploy validation."""
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
]
