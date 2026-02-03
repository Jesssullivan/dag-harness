"""
LangGraph-based workflow execution engine.

Implements proper LangGraph patterns:
- TypedDict-based state schema with Annotated reducers
- StateGraph with node and edge definitions
- SqliteSaver checkpointer for persistence
- Conditional edges for failure routing
- Parallel wave execution support
- RetryPolicy for external API and subprocess nodes (LangGraph 1.0.x)
- Human-in-the-loop via interrupt() pattern (Task #18)
- Parallel test execution via Send API (Task #21)

Migration Notes (LangGraph 0.2.x -> 1.0.x, Task #13):
- RetryPolicy imported from langgraph.types (new in 1.0.x)
- AsyncSqliteSaver path unchanged (langgraph.checkpoint.sqlite.aio)
- StateGraph, END imports unchanged from langgraph.graph
- add_node() now accepts 'retry_policy' parameter for RetryPolicy
- Default retry behavior: retries on 5xx HTTP errors, skips 4xx
- langgraph-checkpoint-sqlite upgraded to 3.0.0+ for compatibility

HITL Pattern (Task #18):
- interrupt() from langgraph.types pauses execution and stores state
- Command(resume=...) provides human input to resume execution
- Checkpointer required for interrupt to work (SqliteSaver)
- interrupt_before parameter in compile() for automatic breakpoints

Parallel Test Execution (Task #21):
- Send API from langgraph.types enables parallel node execution
- route_to_parallel_tests() fans out to run_molecule and run_pytest
- merge_test_results_node() aggregates results from parallel tests
- Performance tracking: logs time savings from parallel execution
- Partial failure handling: workflow tracks which tests failed
- Target: 30%+ time reduction for test phase when both test types exist
"""

import operator
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict


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

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import (  # noqa: F401 (Command re-exported)
    Command,
    RetryPolicy,
    Send,
    interrupt,
)

from harness.config import NotificationConfig
from harness.dag.store import HarnessStore
from harness.db.models import TestType, WorkflowStatus
from harness.db.state import StateDB
from harness.notifications import (
    NotificationService,
    notify_workflow_completed,
    notify_workflow_failed,
    notify_workflow_started,
)

# ============================================================================
# MODULE-LEVEL DATABASE ACCESS FOR NODES
# ============================================================================

# Module-level database instance for node access
# Set by create_box_up_role_graph() or LangGraphWorkflowRunner
_module_db: StateDB | None = None


def set_module_db(db: StateDB) -> None:
    """Set the module-level database instance for node access."""
    global _module_db
    _module_db = db


def get_module_db() -> StateDB | None:
    """Get the module-level database instance."""
    return _module_db


def _record_test_result(
    role_name: str,
    test_name: str,
    test_type: TestType,
    passed: bool,
    error_message: str | None = None,
    execution_id: int | None = None,
) -> None:
    """Record a test result in the database for regression tracking."""
    db = get_module_db()
    if db is None:
        return  # Silently skip if no db available (e.g., in testing)

    role = db.get_role(role_name)
    if not role or not role.id:
        return

    if passed:
        db.record_test_success(role.id, test_name, test_type)
    else:
        db.record_test_failure(role.id, test_name, test_type, error_message=error_message)


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


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


async def validate_role_node(state: BoxUpRoleState) -> dict:
    """Validate that the role exists and extract metadata."""
    role_name = state["role_name"]
    role_path = Path(f"ansible/roles/{role_name}")

    if not role_path.exists():
        return {
            "errors": [f"Role not found: {role_name}"],
            "completed_nodes": ["validate_role"],
        }

    has_molecule = (role_path / "molecule").exists()
    has_meta = (role_path / "meta" / "main.yml").exists()

    return {
        "role_path": str(role_path),
        "has_molecule_tests": has_molecule,
        "has_meta": has_meta,
        "completed_nodes": ["validate_role"],
    }


async def analyze_deps_node(state: BoxUpRoleState) -> dict:
    """Analyze role dependencies using StateDB directly."""
    role_name = state["role_name"]
    db = get_module_db()

    if db is None:
        return {
            "errors": ["Database not available for dependency analysis"],
            "completed_nodes": ["analyze_dependencies"],
        }

    try:
        role = db.get_role(role_name)
        if not role:
            return {
                "errors": [f"Role '{role_name}' not found in database"],
                "completed_nodes": ["analyze_dependencies"],
            }

        # Get dependencies from database
        explicit_deps = db.get_dependencies(role_name, transitive=False)
        implicit_deps = db.get_dependencies(role_name, transitive=True)
        reverse_deps = db.get_reverse_dependencies(role_name, transitive=False)

        # Get credentials
        credentials = db.get_credentials(role_name)
        cred_dicts = [
            {"entry_name": c.entry_name, "purpose": c.purpose, "attribute": c.attribute}
            for c in credentials
        ]

        return {
            "wave": role.wave or 0,
            "wave_name": role.wave_name or "",
            "explicit_deps": [dep[0] for dep in explicit_deps],
            "implicit_deps": [
                dep[0] for dep in implicit_deps if dep[1] > 1
            ],  # Only truly transitive
            "credentials": cred_dicts,
            "reverse_deps": [dep[0] for dep in reverse_deps],
            "tags": [],  # Tags not yet tracked in DB
            "completed_nodes": ["analyze_dependencies"],
        }

    except Exception as e:
        return {
            "errors": [f"Dependency analysis failed: {e}"],
            "completed_nodes": ["analyze_dependencies"],
        }


async def check_reverse_deps_node(state: BoxUpRoleState) -> dict:
    """Check if reverse dependencies are already boxed up."""
    reverse_deps = state.get("reverse_deps", [])

    if not reverse_deps:
        return {
            "blocking_deps": [],
            "completed_nodes": ["check_reverse_deps"],
        }

    blocking = []
    for dep in reverse_deps:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", f"sid/{dep}"], capture_output=True, text=True
        )
        if not result.stdout.strip():
            blocking.append(dep)

    if blocking:
        return {
            "blocking_deps": blocking,
            "errors": [f"Must box up first: {', '.join(blocking)}"],
            "completed_nodes": ["check_reverse_deps"],
        }

    return {
        "blocking_deps": [],
        "completed_nodes": ["check_reverse_deps"],
    }


async def create_worktree_node(state: BoxUpRoleState) -> dict:
    """Create git worktree for isolated development using WorktreeManager."""
    role_name = state["role_name"]
    db = get_module_db()

    if db is None:
        return {
            "errors": ["Database not available for worktree creation"],
            "completed_nodes": ["create_worktree"],
        }

    try:
        from harness.gitlab.api import GitLabClient
        from harness.worktree.manager import WorktreeManager

        # Check if branch already exists on remote before creating worktree
        branch_name = f"sid/{role_name}"
        client = GitLabClient(db)
        branch_existed = client.remote_branch_exists(branch_name)

        manager = WorktreeManager(db)
        worktree_info = manager.create(role_name, force=False)

        return {
            "worktree_path": worktree_info.path,
            "branch": worktree_info.branch,
            "commit_sha": worktree_info.commit,
            "branch_existed": branch_existed,
            "completed_nodes": ["create_worktree"],
        }

    except ValueError as e:
        # Worktree already exists - this is not necessarily an error for idempotency
        # Try to get existing worktree info
        try:
            worktree = db.get_worktree(role_name)
            if worktree:
                return {
                    "worktree_path": worktree.path,
                    "branch": worktree.branch,
                    "commit_sha": worktree.current_commit,
                    "branch_existed": True,  # If worktree exists, branch likely does too
                    "completed_nodes": ["create_worktree"],
                }
        except Exception:
            pass
        return {
            "errors": [str(e)],
            "completed_nodes": ["create_worktree"],
        }
    except RuntimeError as e:
        return {
            "errors": [f"Worktree creation failed: {e}"],
            "completed_nodes": ["create_worktree"],
        }
    except Exception as e:
        return {
            "errors": [str(e)],
            "completed_nodes": ["create_worktree"],
        }


async def run_molecule_node(state: BoxUpRoleState) -> dict:
    """
    Run molecule tests for the role.

    Supports parallel execution (Task #21) - can run concurrently with run_pytest.
    Tracks execution time for performance benchmarking.
    """
    role_name = state["role_name"]
    has_molecule = state.get("has_molecule_tests", False)
    worktree_path = state.get("worktree_path", ".")
    execution_id = state.get("execution_id")

    if not has_molecule:
        return {
            "molecule_skipped": True,
            "parallel_tests_completed": ["run_molecule"],
            "completed_nodes": ["run_molecule_tests"],
        }

    start_time = time.time()
    test_name = f"molecule:{role_name}"

    try:
        result = subprocess.run(
            ["npm", "run", "molecule:role", f"--role={role_name}"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
            cwd=worktree_path,
        )

        duration = int(time.time() - start_time)

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
            # Record failure for regression tracking
            _record_test_result(
                role_name,
                test_name,
                TestType.MOLECULE,
                passed=False,
                error_message=error_msg,
                execution_id=execution_id,
            )
            # Note: errors are tracked but don't prevent pytest from running in parallel mode
            return {
                "molecule_passed": False,
                "molecule_duration": duration,
                "molecule_output": result.stdout[-5000:],
                "parallel_tests_completed": ["run_molecule"],
                "completed_nodes": ["run_molecule_tests"],
            }

        # Record success for regression tracking
        _record_test_result(
            role_name, test_name, TestType.MOLECULE, passed=True, execution_id=execution_id
        )
        return {
            "molecule_passed": True,
            "molecule_duration": duration,
            "parallel_tests_completed": ["run_molecule"],
            "completed_nodes": ["run_molecule_tests"],
        }

    except subprocess.TimeoutExpired:
        # Record timeout as failure
        _record_test_result(
            role_name,
            test_name,
            TestType.MOLECULE,
            passed=False,
            error_message="Test timed out after 600 seconds",
            execution_id=execution_id,
        )
        return {
            "molecule_passed": False,
            "parallel_tests_completed": ["run_molecule"],
            "completed_nodes": ["run_molecule_tests"],
        }


async def run_pytest_node(state: BoxUpRoleState) -> dict:
    """
    Run pytest tests for the role.

    Supports parallel execution (Task #21) - can run concurrently with run_molecule.
    Tracks execution time for performance benchmarking.
    """
    role_name = state["role_name"]
    worktree_path = state.get("worktree_path", ".")
    execution_id = state.get("execution_id")

    # Check if pytest tests exist for this role
    test_path = Path(worktree_path) / "tests" / f"test_{role_name}.py"
    if not test_path.exists():
        return {
            "pytest_skipped": True,
            "parallel_tests_completed": ["run_pytest"],
            "completed_nodes": ["run_pytest"],
        }

    start_time = time.time()
    test_name = f"pytest:{role_name}"

    try:
        result = subprocess.run(
            ["pytest", str(test_path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            cwd=worktree_path,
        )

        duration = int(time.time() - start_time)

        if result.returncode != 0:
            error_msg = result.stdout[-2000:]
            # Record failure for regression tracking
            _record_test_result(
                role_name,
                test_name,
                TestType.PYTEST,
                passed=False,
                error_message=error_msg,
                execution_id=execution_id,
            )
            # Note: errors are tracked but don't prevent molecule from running in parallel mode
            return {
                "pytest_passed": False,
                "pytest_duration": duration,
                "parallel_tests_completed": ["run_pytest"],
                "completed_nodes": ["run_pytest"],
            }

        # Record success for regression tracking
        _record_test_result(
            role_name, test_name, TestType.PYTEST, passed=True, execution_id=execution_id
        )
        return {
            "pytest_passed": True,
            "pytest_duration": duration,
            "parallel_tests_completed": ["run_pytest"],
            "completed_nodes": ["run_pytest"],
        }

    except subprocess.TimeoutExpired:
        # Record timeout as failure
        _record_test_result(
            role_name,
            test_name,
            TestType.PYTEST,
            passed=False,
            error_message="Test timed out after 300 seconds",
            execution_id=execution_id,
        )
        return {
            "pytest_passed": False,
            "parallel_tests_completed": ["run_pytest"],
            "completed_nodes": ["run_pytest"],
        }
    except FileNotFoundError:
        return {
            "pytest_skipped": True,
            "parallel_tests_completed": ["run_pytest"],
            "completed_nodes": ["run_pytest"],
        }


async def validate_deploy_node(state: BoxUpRoleState) -> dict:
    """Validate deployment configuration."""
    # This node validates deployment readiness without actually deploying
    # Checks: syntax validation, variable completeness, etc.

    role_name = state["role_name"]
    worktree_path = state.get("worktree_path", ".")

    try:
        # Run ansible-playbook --syntax-check
        result = subprocess.run(
            ["ansible-playbook", "--syntax-check", "site.yml", "-e", f"target_role={role_name}"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=worktree_path,
        )

        if result.returncode != 0:
            return {
                "deploy_passed": False,
                "errors": [f"Syntax check failed: {result.stderr}"],
                "completed_nodes": ["validate_deploy"],
            }

        return {
            "deploy_passed": True,
            "completed_nodes": ["validate_deploy"],
        }

    except subprocess.TimeoutExpired:
        return {
            "deploy_passed": False,
            "errors": ["Deploy validation timed out"],
            "completed_nodes": ["validate_deploy"],
        }
    except FileNotFoundError:
        # No site.yml - skip validation
        return {
            "deploy_skipped": True,
            "completed_nodes": ["validate_deploy"],
        }


# ============================================================================
# PARALLEL TEST EXECUTION (Task #21)
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


async def merge_test_results_node(state: BoxUpRoleState) -> dict:
    """
    Merge parallel test results before continuing to deploy validation.

    This node:
    1. Aggregates results from parallel molecule and pytest execution
    2. Calculates overall test phase timing for performance comparison
    3. Determines if workflow should continue based on test outcomes
    4. Handles partial failures (one test type fails, other passes)

    Performance Tracking:
    - Compares parallel execution time vs estimated sequential time
    - Logs time savings for benchmarking

    Returns:
        Dict with merged test results and performance metrics.
    """
    # Both molecule and pytest results should be in state via reducers
    molecule_passed = state.get("molecule_passed")
    molecule_skipped = state.get("molecule_skipped", False)
    molecule_duration = state.get("molecule_duration", 0) or 0

    pytest_passed = state.get("pytest_passed")
    pytest_skipped = state.get("pytest_skipped", False)
    pytest_duration = state.get("pytest_duration", 0) or 0

    # Calculate test phase timing
    test_phase_start = state.get("test_phase_start_time")
    if test_phase_start:
        actual_parallel_duration = time.time() - test_phase_start
    else:
        # Fallback: use max of individual durations as approximation
        actual_parallel_duration = max(molecule_duration, pytest_duration)

    # Estimate sequential duration for comparison
    estimated_sequential_duration = molecule_duration + pytest_duration

    # Calculate time savings
    if estimated_sequential_duration > 0:
        time_saved = estimated_sequential_duration - actual_parallel_duration
        time_saved_percent = (time_saved / estimated_sequential_duration) * 100
    else:
        time_saved = 0
        time_saved_percent = 0

    # Determine overall test pass status with partial failure handling
    # Tests are considered passed if they pass OR are skipped
    molecule_ok = molecule_passed is True or molecule_skipped
    pytest_ok = pytest_passed is True or pytest_skipped

    all_passed = molecule_ok and pytest_ok

    # Collect any errors from failed tests (for partial failure reporting)
    errors = []
    if molecule_passed is False:
        errors.append(f"Molecule tests failed (duration: {molecule_duration}s)")
    if pytest_passed is False:
        errors.append(f"Pytest tests failed (duration: {pytest_duration}s)")

    # Log performance metrics
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"Parallel test execution completed: "
        f"actual={actual_parallel_duration:.1f}s, "
        f"estimated_sequential={estimated_sequential_duration:.1f}s, "
        f"saved={time_saved:.1f}s ({time_saved_percent:.1f}%)"
    )

    result = {
        "all_tests_passed": all_passed,
        "test_phase_duration": actual_parallel_duration,
        "completed_nodes": ["merge_test_results"],
    }

    if errors:
        result["errors"] = errors

    return result


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


async def create_commit_node(state: BoxUpRoleState) -> dict:
    """Create commit with semantic message."""
    role_name = state["role_name"]
    worktree_path = state.get("worktree_path", ".")
    wave = state.get("wave", 0)
    wave_name = state.get("wave_name", "")

    commit_msg = f"""feat({role_name}): Add {role_name} Ansible role

Wave {wave}: {wave_name}

- Molecule tests passing
- Ready for merge train
"""

    try:
        # Stage all changes
        subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True)

        # Check if there are changes to commit
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=worktree_path, capture_output=True, text=True
        )

        if not status.stdout.strip():
            return {
                "completed_nodes": ["create_commit"],
            }

        # Create commit
        result = subprocess.run(
            ["git", "commit", "--author=Jess Sullivan <jsullivan2@bates.edu>", "-m", commit_msg],
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {
                "errors": [f"Commit failed: {result.stderr}"],
                "completed_nodes": ["create_commit"],
            }

        # Get commit SHA
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=worktree_path, capture_output=True, text=True
        )

        return {
            "commit_sha": sha_result.stdout.strip(),
            "commit_message": commit_msg,
            "completed_nodes": ["create_commit"],
        }

    except subprocess.CalledProcessError as e:
        return {
            "errors": [str(e)],
            "completed_nodes": ["create_commit"],
        }


async def push_branch_node(state: BoxUpRoleState) -> dict:
    """Push branch to origin."""
    role_name = state["role_name"]
    worktree_path = state.get("worktree_path", ".")
    branch = state.get("branch", f"sid/{role_name}")

    try:
        result = subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {
                "errors": [f"Push failed: {result.stderr}"],
                "completed_nodes": ["push_branch"],
            }

        return {
            "pushed": True,
            "completed_nodes": ["push_branch"],
        }

    except Exception as e:
        return {
            "errors": [str(e)],
            "completed_nodes": ["push_branch"],
        }


async def create_issue_node(state: BoxUpRoleState) -> dict:
    """
    Create GitLab issue with iteration assignment using GitLabClient.

    Uses get_or_create_issue() for idempotency - if an issue already exists
    for this role, it will be reused instead of creating a duplicate.
    """
    role_name = state["role_name"]
    wave = state.get("wave", 0)
    wave_name = state.get("wave_name", "")
    db = get_module_db()

    if db is None:
        return {
            "errors": ["Database not available for issue creation"],
            "completed_nodes": ["create_gitlab_issue"],
        }

    try:
        from harness.gitlab.api import GitLabClient

        client = GitLabClient(db)

        # Build issue description
        description = f"""## Box up `{role_name}` Ansible role

Wave {wave}: {wave_name}

### Tasks
- [ ] Create molecule tests
- [ ] Verify all dependencies are satisfied
- [ ] Run molecule converge successfully
- [ ] Create MR and add to merge train

### Dependencies
{chr(10).join("- `" + dep + "`" for dep in state.get("explicit_deps", [])) or "None"}

### Credentials
{chr(10).join("- " + cred.get("entry_name", "unknown") for cred in state.get("credentials", [])) or "None required"}
"""

        # Get current iteration
        iteration = client.get_current_iteration()
        iteration_id = iteration.id if iteration else None

        # Prepare labels for the role - this ensures all labels exist with proper colors
        # before creating the issue, avoiding label creation failures
        labels = client.prepare_labels_for_role(role_name, wave)

        # Use get_or_create_issue for idempotency - avoids duplicate issues
        issue, created = client.get_or_create_issue(
            role_name=role_name,
            title=f"feat({role_name}): Box up `{role_name}` Ansible role",
            description=description,
            labels=labels,
            iteration_id=iteration_id,
            weight=2,
        )

        return {
            "issue_url": issue.web_url,
            "issue_iid": issue.iid,
            "issue_created": created,
            "iteration_assigned": iteration_id is not None,
            "completed_nodes": ["create_gitlab_issue"],
        }

    except ValueError as e:
        return {
            "errors": [str(e)],
            "completed_nodes": ["create_gitlab_issue"],
        }
    except RuntimeError as e:
        return {
            "errors": [f"Issue creation failed: {e}"],
            "completed_nodes": ["create_gitlab_issue"],
        }
    except Exception as e:
        return {
            "errors": [str(e)],
            "completed_nodes": ["create_gitlab_issue"],
        }


async def create_mr_node(state: BoxUpRoleState) -> dict:
    """
    Create GitLab merge request using GitLabClient.

    Uses get_or_create_mr() for idempotency - if an MR already exists
    for this branch, it will be reused instead of creating a duplicate.
    """
    role_name = state["role_name"]
    issue_iid = state.get("issue_iid")
    branch = state.get("branch", f"sid/{role_name}")
    wave = state.get("wave", 0)
    wave_name = state.get("wave_name", "")
    db = get_module_db()

    if db is None:
        return {
            "errors": ["Database not available for MR creation"],
            "completed_nodes": ["create_merge_request"],
        }

    if not issue_iid:
        return {
            "errors": ["No issue IID available for MR"],
            "completed_nodes": ["create_merge_request"],
        }

    try:
        from harness.gitlab.api import GitLabClient

        client = GitLabClient(db)

        # Build MR description
        description = f"""## Summary

Box up the `{role_name}` Ansible role with molecule tests.

Wave {wave}: {wave_name}

Closes #{issue_iid}

## Test Plan

- [ ] Molecule converge passes
- [ ] Molecule verify passes
- [ ] Molecule idempotence passes
- [ ] CI pipeline passes

## Checklist

- [x] Code follows project conventions
- [x] Tests added/updated
- [ ] Documentation updated (if needed)
"""

        # Use get_or_create_mr for idempotency - avoids duplicate MRs
        mr, created = client.get_or_create_mr(
            role_name=role_name,
            source_branch=branch,
            title=f"feat({role_name}): Add `{role_name}` Ansible role",
            description=description,
            issue_iid=issue_iid,
            draft=False,
        )

        # Set reviewers from config if available
        reviewers_set = False
        if client.config.default_reviewers:
            reviewers_set = client.set_mr_reviewers(mr.iid, client.config.default_reviewers)
            if not reviewers_set:
                # Log but don't fail - reviewers are optional
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to set reviewers for MR !{mr.iid}: {client.config.default_reviewers}"
                )

        return {
            "mr_url": mr.web_url,
            "mr_iid": mr.iid,
            "mr_created": created,
            "reviewers_set": reviewers_set,
            "completed_nodes": ["create_merge_request"],
        }

    except ValueError as e:
        return {
            "errors": [str(e)],
            "completed_nodes": ["create_merge_request"],
        }
    except RuntimeError as e:
        return {
            "errors": [f"MR creation failed: {e}"],
            "completed_nodes": ["create_merge_request"],
        }
    except Exception as e:
        return {
            "errors": [str(e)],
            "completed_nodes": ["create_merge_request"],
        }


async def add_to_merge_train_node(state: BoxUpRoleState) -> dict:
    """
    Add MR to merge train queue with automatic fallback.

    This node:
    1. Checks if merge trains are enabled for the project
    2. If enabled, adds the MR to the merge train
    3. If not enabled, falls back to merge_when_pipeline_succeeds
    4. Returns status indicating which path was taken

    Returns:
        Dict with merge_train_status: "added" | "fallback" | "skipped" | "error"
    """
    mr_iid = state.get("mr_iid")
    db = get_module_db()

    if not mr_iid:
        return {
            "merge_train_status": "skipped",
            "errors": ["No MR IID available for merge train"],
            "completed_nodes": ["add_to_merge_train"],
        }

    if db is None:
        return {
            "merge_train_status": "skipped",
            "errors": ["Database not available for merge train operation"],
            "completed_nodes": ["add_to_merge_train"],
        }

    try:
        from harness.gitlab.api import GitLabClient

        client = GitLabClient(db)

        # Check if MR can be added to merge train
        availability = client.is_merge_train_available(mr_iid)
        if not availability.get("available"):
            reason = availability.get("reason", "Unknown reason")
            # If already in merge train, that's fine
            if "already in merge train" in reason.lower():
                return {
                    "merge_train_status": "added",
                    "completed_nodes": ["add_to_merge_train"],
                }
            # If there's a blocking issue, report it but don't fail
            return {
                "merge_train_status": "skipped",
                "errors": [f"Cannot add to merge train: {reason}"],
                "completed_nodes": ["add_to_merge_train"],
            }

        # Check if merge trains are enabled for the project
        if client.is_merge_train_enabled():
            # Add to merge train
            try:
                client.add_to_merge_train(mr_iid, when_pipeline_succeeds=True, squash=True)
                return {
                    "merge_train_status": "added",
                    "completed_nodes": ["add_to_merge_train"],
                }
            except RuntimeError as e:
                # If merge train add fails, try fallback
                if "not found" in str(e).lower() or "merge_train" in str(e).lower():
                    # Fall through to fallback
                    pass
                else:
                    return {
                        "merge_train_status": "error",
                        "errors": [f"Failed to add to merge train: {e}"],
                        "completed_nodes": ["add_to_merge_train"],
                    }

        # Fallback: merge when pipeline succeeds (for projects without merge trains)
        try:
            client.merge_when_pipeline_succeeds(mr_iid)
            return {
                "merge_train_status": "fallback",
                "completed_nodes": ["add_to_merge_train"],
            }
        except RuntimeError as e:
            # Pipeline might still be running or MR not ready - non-blocking
            return {
                "merge_train_status": "skipped",
                "errors": [f"Could not set merge when pipeline succeeds: {e}"],
                "completed_nodes": ["add_to_merge_train"],
            }

    except ImportError:
        return {
            "merge_train_status": "error",
            "errors": ["GitLabClient not available"],
            "completed_nodes": ["add_to_merge_train"],
        }
    except Exception as e:
        # Non-blocking error - workflow can continue
        return {
            "merge_train_status": "error",
            "errors": [f"Merge train operation failed: {e}"],
            "completed_nodes": ["add_to_merge_train"],
        }


async def human_approval_node(state: BoxUpRoleState) -> dict:
    """
    Node that requires human approval before continuing to merge train.

    Uses LangGraph's interrupt() pattern to pause execution and wait for
    human input. The interrupt stores context about the MR and test results
    to help the human make an informed decision.

    When resumed with Command(resume={"approved": True}), continues to merge train.
    When resumed with Command(resume={"approved": False, "reason": "..."}), stops workflow.

    Returns:
        Dict with human_approved status and completion marker.
    """
    role_name = state["role_name"]
    mr_url = state.get("mr_url")
    mr_iid = state.get("mr_iid")
    molecule_passed = state.get("molecule_passed")
    pytest_passed = state.get("pytest_passed")
    branch = state.get("branch", f"sid/{role_name}")

    # Build context for human reviewer
    approval_context = {
        "role_name": role_name,
        "mr_url": mr_url,
        "mr_iid": mr_iid,
        "branch": branch,
        "molecule_passed": molecule_passed,
        "pytest_passed": pytest_passed,
        "issue_url": state.get("issue_url"),
        "wave": state.get("wave"),
        "wave_name": state.get("wave_name"),
        "credentials": state.get("credentials", []),
        "dependencies": state.get("explicit_deps", []),
    }

    # Use interrupt() to pause execution and wait for human input
    # The interrupt stores the context in the persistence layer
    # Execution resumes when Command(resume=...) is called
    approval = interrupt(
        {
            "question": f"Approve merge train for role '{role_name}'?",
            "context": approval_context,
            "instructions": (
                f"Review the MR at {mr_url} and approve or reject.\n"
                f"Resume with: harness resume <execution_id> --approve\n"
                f"Or reject:   harness resume <execution_id> --reject --reason '...'"
            ),
        }
    )

    # After resume, approval contains the human's response
    # Expected format: {"approved": True/False, "reason": "optional reason"}
    if isinstance(approval, dict):
        approved = approval.get("approved", False)
        reason = approval.get("reason", "")
    else:
        # Handle simple boolean response
        approved = bool(approval)
        reason = ""

    if approved:
        return {
            "human_approved": True,
            "awaiting_human_input": False,
            "completed_nodes": ["human_approval"],
        }
    else:
        return {
            "human_approved": False,
            "human_rejection_reason": reason or "Human rejected without reason",
            "awaiting_human_input": False,
            "errors": [f"Human rejected: {reason or 'No reason given'}"],
            "completed_nodes": ["human_approval"],
        }


async def report_summary_node(state: BoxUpRoleState) -> dict:
    """Generate and report final summary."""
    # Calculate test timing metrics for performance tracking
    molecule_duration = state.get("molecule_duration", 0) or 0
    pytest_duration = state.get("pytest_duration", 0) or 0
    test_phase_duration = state.get("test_phase_duration")

    # If parallel execution was used, we have the actual duration
    # Otherwise, estimate from individual test durations
    if test_phase_duration is None:
        test_phase_duration = molecule_duration + pytest_duration

    # Calculate estimated time savings from parallel execution
    estimated_sequential = molecule_duration + pytest_duration
    if estimated_sequential > 0 and test_phase_duration is not None:
        time_saved = estimated_sequential - test_phase_duration
        time_saved_percent = (time_saved / estimated_sequential) * 100
    else:
        time_saved = 0
        time_saved_percent = 0

    summary = {
        "role": state["role_name"],
        "wave": state.get("wave"),
        "wave_name": state.get("wave_name"),
        "worktree_path": state.get("worktree_path"),
        "branch": state.get("branch"),
        "branch_existed": state.get("branch_existed", False),
        "commit_sha": state.get("commit_sha"),
        "issue_url": state.get("issue_url"),
        "issue_created": state.get("issue_created", False),
        "mr_url": state.get("mr_url"),
        "mr_created": state.get("mr_created", False),
        "merge_train_status": state.get("merge_train_status"),
        "molecule_passed": state.get("molecule_passed"),
        "molecule_duration": molecule_duration,
        "pytest_passed": state.get("pytest_passed"),
        "pytest_duration": pytest_duration,
        "all_tests_passed": state.get("all_tests_passed"),
        "test_phase_duration": test_phase_duration,
        "parallel_execution_enabled": state.get("parallel_execution_enabled", True),
        "test_time_saved_seconds": round(time_saved, 2),
        "test_time_saved_percent": round(time_saved_percent, 1),
        "credentials": state.get("credentials", []),
        "dependencies": state.get("explicit_deps", []),
        "errors": state.get("errors", []),
        "success": len(state.get("errors", [])) == 0,
    }

    return {
        "summary": summary,
        "completed_nodes": ["report_summary"],
    }


async def notify_failure_node(state: BoxUpRoleState) -> dict:
    """
    Handle workflow failure and send notifications.

    This node:
    1. Logs the failure details
    2. If an issue exists, updates it with failure info (adds status::blocked label, adds comment)
    3. Returns completion status
    """
    errors = state.get("errors", [])
    role_name = state["role_name"]
    issue_iid = state.get("issue_iid")

    # Log the failure
    print(f"[FAILURE] Workflow failed for role {role_name}")
    for error in errors:
        print(f"  - {error}")

    # Update issue with failure information if an issue exists
    if issue_iid:
        db = get_module_db()
        if db is not None:
            try:
                from harness.gitlab.api import GitLabClient

                client = GitLabClient(db)
                error_msg = "; ".join(errors) if errors else "Unknown error"

                # Update issue with failure info (adds status::blocked label and comment)
                success = client.update_issue_on_failure(issue_iid, error_msg)
                if success:
                    print(f"[INFO] Updated issue #{issue_iid} with failure information")
                else:
                    print(f"[WARNING] Failed to update issue #{issue_iid} with failure information")

            except ImportError:
                print("[WARNING] GitLabClient not available for issue update")
            except Exception as e:
                print(f"[WARNING] Failed to update issue on failure: {e}")

    return {
        "completed_nodes": ["notify_failure"],
    }


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


# ============================================================================
# RETRY POLICIES (LangGraph 1.0.x)
# ============================================================================

# Retry policy for GitLab API calls (create_issue, create_mr, add_to_merge_train)
# Retries on network errors, timeouts, and 5xx server errors
# Does NOT retry on 4xx client errors (bad request, auth, not found)
GITLAB_API_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,  # Start with 1 second delay
    backoff_factor=2.0,  # Exponential backoff: 1s, 2s, 4s
    max_interval=10.0,  # Cap at 10 seconds between retries
    jitter=True,  # Add randomness to prevent thundering herd
    retry_on=(
        httpx.RequestError,  # Network errors (connection, DNS, etc.)
        httpx.TimeoutException,  # Request timeouts
        RuntimeError,  # Generic runtime errors from API client
    ),
)

# Retry policy for subprocess execution (molecule, pytest)
# Retries on timeout only - test failures are NOT retried
# This handles transient infrastructure issues (VM spin-up, network)
SUBPROCESS_RETRY_POLICY = RetryPolicy(
    max_attempts=2,  # Only retry once for subprocess timeouts
    initial_interval=5.0,  # Wait 5 seconds before retry
    backoff_factor=1.0,  # No backoff for subprocess retries
    max_interval=5.0,
    jitter=False,
    retry_on=(
        subprocess.TimeoutExpired,  # Only retry on timeout
    ),
)

# Retry policy for git operations (push_branch)
# Retries on network errors during push
GIT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=15.0,
    jitter=True,
    retry_on=(
        subprocess.CalledProcessError,  # Git command failures
        RuntimeError,
    ),
)


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_box_up_role_graph(
    db_path: str = "harness.db",
    parallel_tests: bool = True,
    enable_breakpoints: bool | None = None,
) -> tuple[StateGraph, list[str]]:
    """
    Create the LangGraph-based box-up-role workflow.

    This implements the full workflow with:
    - TypedDict state schema
    - Conditional routing on failures
    - Checkpointing via SqliteSaver
    - All gates (molecule, pytest, deploy)
    - RetryPolicy for external API and subprocess nodes (LangGraph 1.0.x)
    - Parallel test execution via Send API (Task #21)
    - Static breakpoints via interrupt_before (Task #23)

    Args:
        db_path: Path to the SQLite database for checkpointing
        parallel_tests: If True (default), run molecule and pytest in parallel.
                       If False, run tests sequentially for debugging.
        enable_breakpoints: If True, return breakpoint nodes for interrupt_before.
                           If None, check HARNESS_BREAKPOINTS env var.
                           If False, return empty breakpoints list.

    Returns:
        Tuple of (StateGraph, breakpoint_nodes) where breakpoint_nodes is
        a list of node names to pass to compile(interrupt_before=...).

    Graph Structure (parallel_tests=True):
        validate_role -> analyze_deps -> check_reverse_deps -> create_worktree
        -> [parallel: run_molecule, run_pytest] -> merge_test_results
        -> validate_deploy -> create_commit -> push_branch -> create_issue
        -> create_mr -> human_approval -> add_to_merge_train -> report_summary

    Graph Structure (parallel_tests=False, legacy):
        validate_role -> analyze_deps -> check_reverse_deps -> create_worktree
        -> run_molecule -> run_pytest -> validate_deploy -> ...

    Retry Policies Applied:
    - GitLab API nodes: 3 attempts with exponential backoff (1s, 2s, 4s)
    - Subprocess nodes: 2 attempts for timeout recovery
    - Git operations: 3 attempts with backoff for push reliability
    """
    # Create the graph
    graph = StateGraph(BoxUpRoleState)

    # Add nodes without retry policies (local operations)
    graph.add_node("validate_role", validate_role_node)
    graph.add_node("analyze_deps", analyze_deps_node)
    graph.add_node("check_reverse_deps", check_reverse_deps_node)
    graph.add_node("create_worktree", create_worktree_node)

    # Add subprocess nodes with retry policy for timeout handling
    graph.add_node("run_molecule", run_molecule_node, retry_policy=SUBPROCESS_RETRY_POLICY)
    graph.add_node("run_pytest", run_pytest_node, retry_policy=SUBPROCESS_RETRY_POLICY)

    # Add test results merger for parallel execution (Task #21)
    graph.add_node("merge_test_results", merge_test_results_node)

    graph.add_node("validate_deploy", validate_deploy_node)

    # Add git nodes with retry policy for push reliability
    graph.add_node("create_commit", create_commit_node)
    graph.add_node("push_branch", push_branch_node, retry_policy=GIT_RETRY_POLICY)

    # Add GitLab API nodes with retry policy for network resilience
    graph.add_node("create_issue", create_issue_node, retry_policy=GITLAB_API_RETRY_POLICY)
    graph.add_node("create_mr", create_mr_node, retry_policy=GITLAB_API_RETRY_POLICY)

    # Human-in-the-loop node (uses interrupt() for approval)
    graph.add_node("human_approval", human_approval_node)

    # Add merge train node with retry policy
    graph.add_node(
        "add_to_merge_train", add_to_merge_train_node, retry_policy=GITLAB_API_RETRY_POLICY
    )

    # Add terminal nodes (no retry needed)
    graph.add_node("report_summary", report_summary_node)
    graph.add_node("notify_failure", notify_failure_node)

    # Set entry point
    graph.set_entry_point("validate_role")

    # Add edges with conditional routing
    graph.add_conditional_edges("validate_role", should_continue_after_validation)
    graph.add_conditional_edges("analyze_deps", should_continue_after_deps)
    graph.add_conditional_edges("check_reverse_deps", should_continue_after_reverse_deps)

    if parallel_tests:
        # Task #21: Parallel test execution using Send API
        # create_worktree -> [parallel: run_molecule, run_pytest] -> merge_test_results
        graph.add_conditional_edges(
            "create_worktree",
            route_to_parallel_tests,
            ["run_molecule", "run_pytest", "merge_test_results"],
        )

        # Both parallel test nodes converge at merge_test_results
        graph.add_edge("run_molecule", "merge_test_results")
        graph.add_edge("run_pytest", "merge_test_results")

        # After merge, route based on combined test results
        graph.add_conditional_edges("merge_test_results", should_continue_after_merge)
    else:
        # Legacy sequential test execution (for debugging or backwards compatibility)
        graph.add_conditional_edges(
            "create_worktree",
            lambda state: "run_molecule" if not state.get("errors") else "notify_failure",
        )
        graph.add_conditional_edges("run_molecule", should_continue_after_molecule)
        graph.add_conditional_edges("run_pytest", should_continue_after_pytest)

    graph.add_conditional_edges("validate_deploy", should_continue_after_deploy)
    graph.add_conditional_edges("create_commit", should_continue_after_commit)
    graph.add_conditional_edges("push_branch", should_continue_after_push)
    graph.add_conditional_edges("create_issue", should_continue_after_issue)
    graph.add_conditional_edges("create_mr", should_continue_after_mr)
    graph.add_conditional_edges("human_approval", should_continue_after_human_approval)

    # Terminal edges
    graph.add_edge("add_to_merge_train", "report_summary")
    graph.add_edge("report_summary", END)
    graph.add_edge("notify_failure", END)

    # Determine breakpoints
    if enable_breakpoints is None:
        enable_breakpoints = get_breakpoints_enabled()

    breakpoints = DEFAULT_BREAKPOINTS if enable_breakpoints else []

    return graph, breakpoints


async def create_compiled_graph(
    db_path: str = "harness.db",
    enable_breakpoints: bool | None = None,
):
    """
    Create and compile the graph with SqliteSaver checkpointer.

    Args:
        db_path: Path to SQLite database for checkpointing.
        enable_breakpoints: If True, enable static breakpoints (interrupt_before).
                           If None, check HARNESS_BREAKPOINTS env var.

    Returns a compiled graph that can be executed with .ainvoke().
    """
    graph, breakpoints = create_box_up_role_graph(
        db_path, enable_breakpoints=enable_breakpoints
    )

    # Create async SQLite checkpointer
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compile_kwargs = {"checkpointer": checkpointer}
        if breakpoints:
            compile_kwargs["interrupt_before"] = breakpoints
        compiled = graph.compile(**compile_kwargs)
        return compiled


class LangGraphWorkflowRunner:
    """
    Wrapper for LangGraph workflow execution with database integration.

    Provides:
    - Easy execution interface
    - State persistence to StateDB
    - Resume from checkpoint support
    - Event emission for observability
    - Regression tracking via module-level db
    - Configurable checkpointer (SQLite or PostgreSQL)

    Checkpointer Configuration (Task #23):
        The runner supports multiple checkpointer backends:
        - SQLite (default): Uses db_path for local development
        - PostgreSQL: Uses checkpointer_factory for production

        To use PostgreSQL checkpointing:
            from harness.dag.checkpointer import CheckpointerFactory

            runner = LangGraphWorkflowRunner(
                db=db,
                checkpointer_factory=CheckpointerFactory,
                postgres_url="postgresql://user:pass@host/db"
            )

        Or configure via environment:
            export POSTGRES_URL=postgresql://user:pass@host/db
            runner = LangGraphWorkflowRunner(db=db, use_postgres=True)
    """

    def __init__(
        self,
        db: StateDB,
        db_path: str = "harness.db",
        notification_config: NotificationConfig | None = None,
        checkpointer_factory: Any | None = None,
        postgres_url: str | None = None,
        use_postgres: bool = False,
        store: HarnessStore | None = None,
    ):
        """
        Initialize the workflow runner.

        Args:
            db: StateDB instance for workflow tracking
            db_path: Path to SQLite database (default checkpointer)
            notification_config: Optional notification configuration
            checkpointer_factory: Optional CheckpointerFactory class for custom checkpointer
            postgres_url: Optional PostgreSQL URL (overrides environment)
            use_postgres: If True, prefer PostgreSQL from environment variable
            store: Optional HarnessStore for cross-thread memory persistence
        """
        self.db = db
        self.db_path = db_path
        self._graph = None
        self._checkpointer = None
        self._checkpointer_factory = checkpointer_factory
        self._postgres_url = postgres_url
        self._use_postgres = use_postgres or postgres_url is not None
        self._store = store
        # Set module-level db for node access (regression tracking)
        set_module_db(db)
        # Initialize notification service
        self._notification_service = NotificationService(
            notification_config or NotificationConfig(enabled=False)
        )

    async def _get_graph(self):
        """Lazily create and cache the compiled graph with appropriate checkpointer."""
        if self._graph is None:
            graph, breakpoints = create_box_up_role_graph(self.db_path)

            # Determine which checkpointer to use
            checkpointer = None

            if self._checkpointer_factory is not None:
                # Use provided factory
                try:
                    checkpointer = await self._checkpointer_factory.create_async(
                        postgres_url=self._postgres_url,
                        sqlite_path=self.db_path,
                        fallback_to_sqlite=True,
                    )
                    self._checkpointer = checkpointer
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Failed to create checkpointer from factory: {e}. "
                        "Falling back to in-memory."
                    )
            elif self._use_postgres:
                # Try to use PostgreSQL from environment
                try:
                    from harness.dag.checkpointer import CheckpointerFactory

                    checkpointer = await CheckpointerFactory.create_async(
                        postgres_url=self._postgres_url,
                        sqlite_path=self.db_path,
                        fallback_to_sqlite=True,
                    )
                    self._checkpointer = checkpointer
                except ImportError:
                    pass  # checkpointer module not available
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Failed to create PostgreSQL checkpointer: {e}. "
                        "Compiling without checkpointer."
                    )

            # Build compile kwargs
            compile_kwargs = {}
            if checkpointer is not None:
                compile_kwargs["checkpointer"] = checkpointer
            if breakpoints:
                compile_kwargs["interrupt_before"] = breakpoints
            if self._store is not None:
                compile_kwargs["store"] = self._store

            # Compile with or without checkpointer/breakpoints/store
            self._graph = graph.compile(**compile_kwargs)

        return self._graph

    async def execute(
        self, role_name: str, resume_from: int | None = None, config: RunnableConfig | None = None
    ) -> dict:
        """
        Execute the box-up-role workflow.

        Args:
            role_name: Name of the role to process
            resume_from: Optional execution ID to resume from
            config: Optional LangGraph configuration

        Returns:
            Final state after execution
        """
        graph = await self._get_graph()

        # Create or restore initial state
        if resume_from:
            checkpoint = self.db.get_checkpoint(resume_from)
            if checkpoint:
                initial_state = checkpoint.get("state", {})
                initial_state["role_name"] = role_name
                execution_id = resume_from
            else:
                raise ValueError(f"No checkpoint found for execution {resume_from}")
        else:
            # Create new execution record
            self.db.create_workflow_definition(
                "box_up_role_langgraph",
                "LangGraph-based box-up-role workflow",
                nodes=[
                    {"name": n}
                    for n in [
                        "validate_role",
                        "analyze_deps",
                        "check_reverse_deps",
                        "create_worktree",
                        "run_molecule",
                        "run_pytest",
                        "validate_deploy",
                        "create_commit",
                        "push_branch",
                        "create_issue",
                        "create_mr",
                        "add_to_merge_train",
                        "report_summary",
                        "notify_failure",
                    ]
                ],
                edges=[],
            )
            execution_id = self.db.create_execution("box_up_role_langgraph", role_name)
            initial_state = create_initial_state(role_name, execution_id)

        # Update execution status
        self.db.update_execution_status(
            execution_id, WorkflowStatus.RUNNING, current_node="validate_role"
        )

        # Send workflow started notification
        await notify_workflow_started(self._notification_service, role_name, execution_id)

        try:
            # Execute the graph
            final_state = await graph.ainvoke(initial_state, config=config or {})

            # Save checkpoint
            self.db.checkpoint_execution(
                execution_id,
                {
                    "state": dict(final_state),
                    "completed_nodes": final_state.get("completed_nodes", []),
                },
            )

            # Update final status and send notifications
            if final_state.get("errors"):
                error_msg = "; ".join(final_state.get("errors", []))
                self.db.update_execution_status(
                    execution_id, WorkflowStatus.FAILED, error_message=error_msg
                )
                await notify_workflow_failed(
                    self._notification_service,
                    role_name,
                    execution_id,
                    error=error_msg,
                    failed_node=final_state.get("current_node"),
                )
            else:
                self.db.update_execution_status(execution_id, WorkflowStatus.COMPLETED)
                await notify_workflow_completed(
                    self._notification_service,
                    role_name,
                    execution_id,
                    summary=final_state.get("summary", {}),
                )

            return {
                "status": "completed" if not final_state.get("errors") else "failed",
                "execution_id": execution_id,
                "state": dict(final_state),
                "summary": final_state.get("summary"),
            }

        except Exception as e:
            self.db.update_execution_status(
                execution_id, WorkflowStatus.FAILED, error_message=str(e)
            )
            await notify_workflow_failed(
                self._notification_service,
                role_name,
                execution_id,
                error=str(e),
                failed_node="unknown",
            )

            # Try to update issue with failure info if an issue exists
            # This handles unexpected errors that bypass the normal failure path
            issue_iid = initial_state.get("issue_iid")
            if issue_iid:
                try:
                    from harness.gitlab.api import GitLabClient

                    client = GitLabClient(self.db)
                    client.update_issue_on_failure(issue_iid, str(e))
                except Exception:
                    pass  # Best effort - don't let this cause additional failures

            return {"status": "error", "execution_id": execution_id, "error": str(e)}

    # =========================================================================
    # DAG Modification Entrypoints
    # =========================================================================

    def modify_edge(self, from_node: str, condition: str, new_target: str) -> bool:
        """
        Store an edge modification to be applied on next graph build.

        Args:
            from_node: Source node name
            condition: Edge condition (e.g., "success", "failure", "default")
            new_target: New target node name

        Returns:
            True if modification was stored successfully
        """
        if not hasattr(self, "_edge_modifications"):
            self._edge_modifications: list[dict] = []

        self._edge_modifications.append(
            {
                "type": "modify",
                "from": from_node,
                "condition": condition,
                "target": new_target,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return True

    def insert_node_after(
        self,
        existing_node: str,
        new_node: str,
        node_func: Callable[[BoxUpRoleState], dict],
        description: str,
    ) -> bool:
        """
        Store a node insertion to be applied on next graph build.

        Args:
            existing_node: Node to insert after
            new_node: Name for the new node
            node_func: Function to execute for the new node
            description: Node description for documentation

        Returns:
            True if insertion was stored successfully
        """
        if not hasattr(self, "_node_insertions"):
            self._node_insertions: list[dict] = []

        self._node_insertions.append(
            {
                "after": existing_node,
                "name": new_node,
                "func": node_func,
                "description": description,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return True

    def remove_node(self, node_name: str) -> bool:
        """
        Store a node removal to be applied on next graph build.

        Args:
            node_name: Node to remove

        Returns:
            True if removal was stored successfully
        """
        if not hasattr(self, "_node_removals"):
            self._node_removals: list[str] = []

        if node_name not in self._node_removals:
            self._node_removals.append(node_name)
        return True

    def export_graph(self) -> dict:
        """
        Export the graph definition as a serializable dict.

        Returns:
            Dict containing graph structure and pending modifications
        """
        # Get node list from the default node order
        default_nodes = [
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
            "run_pytest",
            "validate_deploy",
            "create_commit",
            "push_branch",
            "create_issue",
            "create_mr",
            "add_to_merge_train",
            "report_summary",
            "notify_failure",
        ]

        return {
            "workflow_type": "box_up_role",
            "entry_point": "validate_role",
            "nodes": default_nodes,
            "pending_modifications": {
                "edge_modifications": getattr(self, "_edge_modifications", []),
                "node_insertions": [
                    {
                        "after": ins["after"],
                        "name": ins["name"],
                        "description": ins["description"],
                        "timestamp": ins["timestamp"],
                    }
                    for ins in getattr(self, "_node_insertions", [])
                ],
                "node_removals": getattr(self, "_node_removals", []),
            },
            "exported_at": datetime.utcnow().isoformat(),
        }

    def import_graph(self, definition: dict) -> None:
        """
        Import a graph definition from a serialized dict.

        Note: Node functions cannot be serialized, so insertions
        from imported definitions will need their functions re-attached.

        Args:
            definition: Previously exported graph definition
        """
        mods = definition.get("pending_modifications", {})

        self._edge_modifications = mods.get("edge_modifications", [])
        self._node_removals = mods.get("node_removals", [])

        # For insertions, we store the metadata but func must be re-attached
        self._node_insertions = [
            {
                "after": ins["after"],
                "name": ins["name"],
                "func": None,  # Must be re-attached via register_node_func
                "description": ins["description"],
                "timestamp": ins.get("timestamp", datetime.utcnow().isoformat()),
            }
            for ins in mods.get("node_insertions", [])
        ]

    def register_node_func(self, node_name: str, func: Callable[[BoxUpRoleState], dict]) -> bool:
        """
        Register a function for a node (used after import_graph).

        Args:
            node_name: Name of the node
            func: Function to execute for this node

        Returns:
            True if registration was successful
        """
        if not hasattr(self, "_node_insertions"):
            return False

        for insertion in self._node_insertions:
            if insertion["name"] == node_name:
                insertion["func"] = func
                return True
        return False

    def clear_modifications(self) -> None:
        """Clear all pending modifications."""
        self._edge_modifications = []
        self._node_insertions = []
        self._node_removals = []

    def get_pending_modifications_count(self) -> dict[str, int]:
        """Get count of pending modifications by type."""
        return {
            "edge_modifications": len(getattr(self, "_edge_modifications", [])),
            "node_insertions": len(getattr(self, "_node_insertions", [])),
            "node_removals": len(getattr(self, "_node_removals", [])),
        }
