"""
LangGraph state schema and module-level database access.

Defines:
- BoxUpRoleState TypedDict with Annotated reducers
- Module-level database access functions for nodes
- Custom reducers for list accumulation
- Initial state creation
"""

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from harness.db.models import TestType
from harness.db.state import StateDB


# =============================================================================
# CUSTOM REDUCERS
# =============================================================================


def keep_last_n(n: int) -> Callable[[list, list], list]:
    """
    Create a reducer that keeps only the last N items.

    Useful for state_snapshots and similar fields where we want a rolling
    history without unbounded growth.

    Args:
        n: Maximum number of items to keep.

    Returns:
        A reducer function compatible with LangGraph's Annotated pattern.

    Usage:
        state_snapshots: Annotated[list[dict], keep_last_n(10)]
    """

    def reducer(current: list | None, new: list | None) -> list:
        current = current or []
        new = new or []
        combined = current + new
        return combined[-n:] if len(combined) > n else combined

    return reducer


# =============================================================================
# STATIC BREAKPOINTS (Task #23 - Week 1 Day 5)
# =============================================================================

# Default breakpoints for the box-up-role workflow.
# These nodes will automatically pause execution when breakpoints are enabled,
# allowing operators to review state before critical operations.
DEFAULT_BREAKPOINTS: list[str] = [
    "human_approval",  # Always pause before human approval (redundant with interrupt())
    "create_mr",  # Pause before creating merge request
    "add_to_merge_train",  # Pause before adding to merge train
]

# Environment variable to enable static breakpoints
# Set HARNESS_BREAKPOINTS=true to enable automatic pausing at breakpoints
BREAKPOINTS_ENV_VAR = "HARNESS_BREAKPOINTS"


def get_breakpoints_enabled() -> bool:
    """Check if static breakpoints are enabled via environment variable."""
    import os

    return os.environ.get(BREAKPOINTS_ENV_VAR, "").lower() in ("true", "1", "yes")


# ============================================================================
# MODULE-LEVEL DATABASE ACCESS FOR NODES
# ============================================================================

# Module-level database instance for node access
# Set by create_box_up_role_graph() or LangGraphWorkflowRunner
_module_db: StateDB | None = None
_module_config: Any = None


def set_module_db(db: StateDB) -> None:
    """Set the module-level database instance for node access."""
    global _module_db
    _module_db = db


def set_module_config(config: Any) -> None:
    """Set the module-level harness config for node access."""
    global _module_config
    _module_config = config


def get_module_db() -> StateDB | None:
    """Get the module-level database instance."""
    return _module_db


def get_module_config() -> Any:
    """Get the module-level harness config."""
    return _module_config


def _record_test_result(
    role_name: str,
    test_name: str,
    test_type: TestType,
    passed: bool,
    error_message: str | None = None,
    execution_id: int | None = None,
) -> None:
    """Record a test result in the database for regression tracking.

    Args:
        role_name: Name of the role being tested
        test_name: Name of the test (e.g., "molecule:common")
        test_type: Type of test (MOLECULE, PYTEST, etc.)
        passed: Whether the test passed
        error_message: Error message if test failed
        execution_id: Workflow execution ID (used as test_run_id for tracking)
    """
    db = get_module_db()
    if db is None:
        return  # Silently skip if no db available (e.g., in testing)

    # Skip recording if no execution_id (required for test run tracking)
    if execution_id is None:
        return

    if passed:
        db.record_test_success(role_name, test_name, test_type, execution_id)
    else:
        db.record_test_failure(
            role_name, test_name, test_type, execution_id, error_message=error_message
        )


# ============================================================================
# STATE SCHEMA
# ============================================================================


class BoxUpRoleState(TypedDict, total=False):
    """
    TypedDict state schema for box-up-role workflow.

    Uses Annotated types with operator.add reducers for proper list accumulation.
    This ensures lists are appended to (not overwritten) across node executions.
    """

    # Role identification
    role_name: str
    execution_id: int | None

    # Role metadata (from validation and analysis)
    role_path: str
    has_molecule_tests: bool
    has_meta: bool
    wave: int
    wave_name: str

    # Dependency analysis - use reducers to accumulate across analysis passes
    explicit_deps: Annotated[list[str], operator.add]
    implicit_deps: Annotated[list[str], operator.add]
    reverse_deps: Annotated[list[str], operator.add]
    credentials: Annotated[list[dict], operator.add]
    tags: Annotated[list[str], operator.add]
    blocking_deps: Annotated[list[str], operator.add]

    # Worktree and git state
    worktree_path: str
    branch: str
    commit_sha: str | None
    commit_message: str | None
    pushed: bool

    # Test results
    molecule_passed: bool | None
    molecule_skipped: bool
    molecule_duration: int | None
    molecule_output: str | None
    pytest_passed: bool | None
    pytest_skipped: bool
    pytest_duration: int | None
    deploy_passed: bool | None
    deploy_skipped: bool

    # Parallel test execution state (Task #21)
    all_tests_passed: bool | None
    parallel_tests_completed: Annotated[list[str], operator.add]
    test_phase_start_time: float | None
    test_phase_duration: float | None
    parallel_execution_enabled: bool

    # GitLab integration
    issue_url: str | None
    issue_iid: int | None
    issue_created: bool  # True if issue was newly created, False if existing was reused
    mr_url: str | None
    mr_iid: int | None
    mr_created: bool  # True if MR was newly created, False if existing was reused
    reviewers_set: bool  # True if reviewers were successfully assigned to MR
    iteration_assigned: bool
    merge_train_status: str | None  # "added" | "fallback" | "skipped" | "error"
    branch_existed: bool  # True if branch already existed on remote before workflow

    # Workflow control - use reducers for accumulating nodes and errors
    current_node: str
    completed_nodes: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # Extended observability (Task #23 - Week 1 Day 4)
    # These reducers accumulate detailed operation logs for debugging and auditing
    test_results: Annotated[list[dict], operator.add]  # Individual test execution records
    git_operations: Annotated[list[str], operator.add]  # Git command history
    api_calls: Annotated[list[dict], operator.add]  # External API call records
    timing_metrics: Annotated[list[dict], operator.add]  # Performance timing data

    # State snapshots with bounded history (keep last 10 snapshots)
    state_snapshots: Annotated[list[dict], keep_last_n(10)]

    # Final summary
    summary: dict | None

    # Human-in-the-loop state (Task #18)
    human_approved: bool | None
    human_rejection_reason: str | None
    awaiting_human_input: bool


def create_initial_state(role_name: str, execution_id: int | None = None) -> BoxUpRoleState:
    """Create initial state for a workflow execution."""
    return BoxUpRoleState(
        role_name=role_name,
        execution_id=execution_id,
        has_molecule_tests=False,
        has_meta=False,
        wave=0,
        wave_name="",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=[],
        credentials=[],
        tags=[],
        blocking_deps=[],
        pushed=False,
        molecule_skipped=False,
        pytest_skipped=False,
        deploy_skipped=False,
        all_tests_passed=None,
        parallel_tests_completed=[],
        test_phase_start_time=None,
        test_phase_duration=None,
        parallel_execution_enabled=True,
        iteration_assigned=False,
        issue_created=False,
        mr_created=False,
        reviewers_set=False,
        branch_existed=False,
        current_node="validate_role",
        completed_nodes=[],
        errors=[],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


__all__ = [
    # Custom reducers
    "keep_last_n",
    # Breakpoints
    "DEFAULT_BREAKPOINTS",
    "BREAKPOINTS_ENV_VAR",
    "get_breakpoints_enabled",
    # Module DB access
    "set_module_db",
    "set_module_config",
    "get_module_db",
    "get_module_config",
    "_record_test_result",
    # State
    "BoxUpRoleState",
    "create_initial_state",
]
