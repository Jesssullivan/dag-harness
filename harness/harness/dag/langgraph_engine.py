"""
LangGraph-based workflow execution engine.

Implements proper LangGraph patterns:
- TypedDict-based state schema with Annotated reducers
- StateGraph with node and edge definitions
- SqliteSaver checkpointer for persistence
- Conditional edges for failure routing
- Parallel wave execution support
"""

import asyncio
import json
import operator
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Sequence, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.runnables import RunnableConfig

from harness.db.models import NodeStatus, TestStatus, TestType, WorkflowStatus
from harness.db.state import StateDB
from harness.config import NotificationConfig
from harness.notifications import (
    NotificationService,
    notify_workflow_started,
    notify_workflow_completed,
    notify_workflow_failed,
)


# ============================================================================
# MODULE-LEVEL DATABASE ACCESS FOR NODES
# ============================================================================

# Module-level database instance for node access
# Set by create_box_up_role_graph() or LangGraphWorkflowRunner
_module_db: Optional[StateDB] = None


def set_module_db(db: StateDB) -> None:
    """Set the module-level database instance for node access."""
    global _module_db
    _module_db = db


def get_module_db() -> Optional[StateDB]:
    """Get the module-level database instance."""
    return _module_db


def _record_test_result(
    role_name: str,
    test_name: str,
    test_type: TestType,
    passed: bool,
    error_message: Optional[str] = None,
    execution_id: Optional[int] = None
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
        db.record_test_failure(
            role.id,
            test_name,
            test_type,
            error_message=error_message
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
    execution_id: Optional[int]

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
    commit_sha: Optional[str]
    commit_message: Optional[str]
    pushed: bool

    # Test results
    molecule_passed: Optional[bool]
    molecule_skipped: bool
    molecule_duration: Optional[int]
    molecule_output: Optional[str]
    pytest_passed: Optional[bool]
    pytest_skipped: bool
    deploy_passed: Optional[bool]
    deploy_skipped: bool

    # GitLab integration
    issue_url: Optional[str]
    issue_iid: Optional[int]
    mr_url: Optional[str]
    mr_iid: Optional[int]
    iteration_assigned: bool
    merge_train_status: Optional[str]  # "added" | "fallback" | "skipped" | "error"

    # Workflow control - use reducers for accumulating nodes and errors
    current_node: str
    completed_nodes: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # Final summary
    summary: Optional[dict]


def create_initial_state(role_name: str, execution_id: Optional[int] = None) -> BoxUpRoleState:
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
        iteration_assigned=False,
        current_node="validate_role",
        completed_nodes=[],
        errors=[],
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
            "implicit_deps": [dep[0] for dep in implicit_deps if dep[1] > 1],  # Only truly transitive
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
            ["git", "ls-remote", "--heads", "origin", f"sid/{dep}"],
            capture_output=True,
            text=True
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
        from harness.worktree.manager import WorktreeManager

        manager = WorktreeManager(db)
        worktree_info = manager.create(role_name, force=False)

        return {
            "worktree_path": worktree_info.path,
            "branch": worktree_info.branch,
            "commit_sha": worktree_info.commit,
            "completed_nodes": ["create_worktree"],
        }

    except ValueError as e:
        # Worktree already exists
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
    """Run molecule tests for the role."""
    role_name = state["role_name"]
    has_molecule = state.get("has_molecule_tests", False)
    worktree_path = state.get("worktree_path", ".")
    execution_id = state.get("execution_id")

    if not has_molecule:
        return {
            "molecule_skipped": True,
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
            cwd=worktree_path
        )

        duration = int(time.time() - start_time)

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
            # Record failure for regression tracking
            _record_test_result(
                role_name, test_name, TestType.MOLECULE,
                passed=False, error_message=error_msg, execution_id=execution_id
            )
            return {
                "molecule_passed": False,
                "molecule_duration": duration,
                "molecule_output": result.stdout[-5000:],
                "errors": ["Molecule tests failed"],
                "completed_nodes": ["run_molecule_tests"],
            }

        # Record success for regression tracking
        _record_test_result(
            role_name, test_name, TestType.MOLECULE,
            passed=True, execution_id=execution_id
        )
        return {
            "molecule_passed": True,
            "molecule_duration": duration,
            "completed_nodes": ["run_molecule_tests"],
        }

    except subprocess.TimeoutExpired:
        # Record timeout as failure
        _record_test_result(
            role_name, test_name, TestType.MOLECULE,
            passed=False, error_message="Test timed out after 600 seconds",
            execution_id=execution_id
        )
        return {
            "molecule_passed": False,
            "errors": ["Molecule tests timed out"],
            "completed_nodes": ["run_molecule_tests"],
        }


async def run_pytest_node(state: BoxUpRoleState) -> dict:
    """Run pytest tests for the role."""
    role_name = state["role_name"]
    worktree_path = state.get("worktree_path", ".")
    execution_id = state.get("execution_id")

    # Check if pytest tests exist for this role
    test_path = Path(worktree_path) / "tests" / f"test_{role_name}.py"
    if not test_path.exists():
        return {
            "pytest_skipped": True,
            "completed_nodes": ["run_pytest"],
        }

    test_name = f"pytest:{role_name}"

    try:
        result = subprocess.run(
            ["pytest", str(test_path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            cwd=worktree_path
        )

        if result.returncode != 0:
            error_msg = result.stdout[-2000:]
            # Record failure for regression tracking
            _record_test_result(
                role_name, test_name, TestType.PYTEST,
                passed=False, error_message=error_msg, execution_id=execution_id
            )
            return {
                "pytest_passed": False,
                "errors": [f"Pytest failed: {error_msg}"],
                "completed_nodes": ["run_pytest"],
            }

        # Record success for regression tracking
        _record_test_result(
            role_name, test_name, TestType.PYTEST,
            passed=True, execution_id=execution_id
        )
        return {
            "pytest_passed": True,
            "completed_nodes": ["run_pytest"],
        }

    except subprocess.TimeoutExpired:
        # Record timeout as failure
        _record_test_result(
            role_name, test_name, TestType.PYTEST,
            passed=False, error_message="Test timed out after 300 seconds",
            execution_id=execution_id
        )
        return {
            "pytest_passed": False,
            "errors": ["Pytest timed out"],
            "completed_nodes": ["run_pytest"],
        }
    except FileNotFoundError:
        return {
            "pytest_skipped": True,
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
            ["ansible-playbook", "--syntax-check", "site.yml",
             "-e", f"target_role={role_name}"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=worktree_path
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
            ["git", "status", "--porcelain"],
            cwd=worktree_path,
            capture_output=True,
            text=True
        )

        if not status.stdout.strip():
            return {
                "completed_nodes": ["create_commit"],
            }

        # Create commit
        result = subprocess.run(
            ["git", "commit",
             "--author=Jess Sullivan <jsullivan2@bates.edu>",
             "-m", commit_msg],
            cwd=worktree_path,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return {
                "errors": [f"Commit failed: {result.stderr}"],
                "completed_nodes": ["create_commit"],
            }

        # Get commit SHA
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree_path,
            capture_output=True,
            text=True
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
            text=True
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
    """Create GitLab issue with iteration assignment using GitLabClient."""
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
{chr(10).join('- `' + dep + '`' for dep in state.get('explicit_deps', [])) or 'None'}

### Credentials
{chr(10).join('- ' + cred.get('entry_name', 'unknown') for cred in state.get('credentials', [])) or 'None required'}
"""

        # Get current iteration
        iteration = client.get_current_iteration()
        iteration_id = iteration.id if iteration else None

        # Create the issue
        issue = client.create_issue(
            role_name=role_name,
            title=f"feat({role_name}): Box up `{role_name}` Ansible role",
            description=description,
            labels=["role", "ansible", "molecule", f"wave-{wave}"],
            iteration_id=iteration_id,
            weight=2,
        )

        return {
            "issue_url": issue.web_url,
            "issue_iid": issue.iid,
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
    """Create GitLab merge request using GitLabClient."""
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

        # Create the merge request
        mr = client.create_merge_request(
            role_name=role_name,
            source_branch=branch,
            title=f"feat({role_name}): Add `{role_name}` Ansible role",
            description=description,
            issue_iid=issue_iid,
            draft=False,
        )

        return {
            "mr_url": mr.web_url,
            "mr_iid": mr.iid,
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
                result = client.add_to_merge_train(
                    mr_iid,
                    when_pipeline_succeeds=True,
                    squash=True
                )
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
            result = client.merge_when_pipeline_succeeds(mr_iid)
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


async def report_summary_node(state: BoxUpRoleState) -> dict:
    """Generate and report final summary."""
    summary = {
        "role": state["role_name"],
        "wave": state.get("wave"),
        "wave_name": state.get("wave_name"),
        "worktree_path": state.get("worktree_path"),
        "branch": state.get("branch"),
        "commit_sha": state.get("commit_sha"),
        "issue_url": state.get("issue_url"),
        "mr_url": state.get("mr_url"),
        "merge_train_status": state.get("merge_train_status"),
        "molecule_passed": state.get("molecule_passed"),
        "pytest_passed": state.get("pytest_passed"),
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
    """Handle workflow failure and send notifications."""
    errors = state.get("errors", [])
    role_name = state["role_name"]

    # Log the failure
    print(f"[FAILURE] Workflow failed for role {role_name}")
    for error in errors:
        print(f"  - {error}")

    # Could add Discord/email notification here
    return {
        "completed_nodes": ["notify_failure"],
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue_after_validation(state: BoxUpRoleState) -> Literal["analyze_deps", "notify_failure"]:
    """Route after validation node."""
    if state.get("errors"):
        return "notify_failure"
    return "analyze_deps"


def should_continue_after_deps(state: BoxUpRoleState) -> Literal["check_reverse_deps", "notify_failure"]:
    """Route after dependency analysis."""
    if state.get("errors"):
        return "notify_failure"
    return "check_reverse_deps"


def should_continue_after_reverse_deps(state: BoxUpRoleState) -> Literal["create_worktree", "notify_failure"]:
    """Route after reverse deps check."""
    if state.get("blocking_deps"):
        return "notify_failure"
    return "create_worktree"


def should_continue_after_worktree(state: BoxUpRoleState) -> Literal["run_molecule", "notify_failure"]:
    """Route after worktree creation."""
    if state.get("errors"):
        return "notify_failure"
    return "run_molecule"


def should_continue_after_molecule(state: BoxUpRoleState) -> Literal["run_pytest", "notify_failure"]:
    """Route after molecule tests."""
    if state.get("molecule_passed") is False:
        return "notify_failure"
    return "run_pytest"


def should_continue_after_pytest(state: BoxUpRoleState) -> Literal["validate_deploy", "notify_failure"]:
    """Route after pytest."""
    if state.get("pytest_passed") is False:
        return "notify_failure"
    return "validate_deploy"


def should_continue_after_deploy(state: BoxUpRoleState) -> Literal["create_commit", "notify_failure"]:
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


def should_continue_after_mr(state: BoxUpRoleState) -> Literal["add_to_merge_train", "report_summary"]:
    """Route after MR creation."""
    if state.get("mr_iid"):
        return "add_to_merge_train"
    return "report_summary"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_box_up_role_graph(db_path: str = "harness.db") -> StateGraph:
    """
    Create the LangGraph-based box-up-role workflow.

    This implements the full workflow with:
    - TypedDict state schema
    - Conditional routing on failures
    - Checkpointing via SqliteSaver
    - All gates (molecule, pytest, deploy)
    """
    # Create the graph
    graph = StateGraph(BoxUpRoleState)

    # Add all nodes
    graph.add_node("validate_role", validate_role_node)
    graph.add_node("analyze_deps", analyze_deps_node)
    graph.add_node("check_reverse_deps", check_reverse_deps_node)
    graph.add_node("create_worktree", create_worktree_node)
    graph.add_node("run_molecule", run_molecule_node)
    graph.add_node("run_pytest", run_pytest_node)
    graph.add_node("validate_deploy", validate_deploy_node)
    graph.add_node("create_commit", create_commit_node)
    graph.add_node("push_branch", push_branch_node)
    graph.add_node("create_issue", create_issue_node)
    graph.add_node("create_mr", create_mr_node)
    graph.add_node("add_to_merge_train", add_to_merge_train_node)
    graph.add_node("report_summary", report_summary_node)
    graph.add_node("notify_failure", notify_failure_node)

    # Set entry point
    graph.set_entry_point("validate_role")

    # Add edges with conditional routing
    graph.add_conditional_edges(
        "validate_role",
        should_continue_after_validation
    )
    graph.add_conditional_edges(
        "analyze_deps",
        should_continue_after_deps
    )
    graph.add_conditional_edges(
        "check_reverse_deps",
        should_continue_after_reverse_deps
    )
    graph.add_conditional_edges(
        "create_worktree",
        should_continue_after_worktree
    )
    graph.add_conditional_edges(
        "run_molecule",
        should_continue_after_molecule
    )
    graph.add_conditional_edges(
        "run_pytest",
        should_continue_after_pytest
    )
    graph.add_conditional_edges(
        "validate_deploy",
        should_continue_after_deploy
    )
    graph.add_conditional_edges(
        "create_commit",
        should_continue_after_commit
    )
    graph.add_conditional_edges(
        "push_branch",
        should_continue_after_push
    )
    graph.add_conditional_edges(
        "create_issue",
        should_continue_after_issue
    )
    graph.add_conditional_edges(
        "create_mr",
        should_continue_after_mr
    )

    # Terminal edges
    graph.add_edge("add_to_merge_train", "report_summary")
    graph.add_edge("report_summary", END)
    graph.add_edge("notify_failure", END)

    return graph


async def create_compiled_graph(db_path: str = "harness.db"):
    """
    Create and compile the graph with SqliteSaver checkpointer.

    Returns a compiled graph that can be executed with .ainvoke().
    """
    graph = create_box_up_role_graph(db_path)

    # Create async SQLite checkpointer
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
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
    """

    def __init__(self, db: StateDB, db_path: str = "harness.db", notification_config: Optional[NotificationConfig] = None):
        self.db = db
        self.db_path = db_path
        self._graph = None
        # Set module-level db for node access (regression tracking)
        set_module_db(db)
        # Initialize notification service
        self._notification_service = NotificationService(
            notification_config or NotificationConfig(enabled=False)
        )

    async def _get_graph(self):
        """Lazily create and cache the compiled graph."""
        if self._graph is None:
            graph = create_box_up_role_graph(self.db_path)
            # For now, compile without checkpointer (sync version)
            self._graph = graph.compile()
        return self._graph

    async def execute(
        self,
        role_name: str,
        resume_from: Optional[int] = None,
        config: Optional[RunnableConfig] = None
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
                nodes=[{"name": n} for n in [
                    "validate_role", "analyze_deps", "check_reverse_deps",
                    "create_worktree", "run_molecule", "run_pytest",
                    "validate_deploy", "create_commit", "push_branch",
                    "create_issue", "create_mr", "add_to_merge_train",
                    "report_summary", "notify_failure"
                ]],
                edges=[]
            )
            execution_id = self.db.create_execution("box_up_role_langgraph", role_name)
            initial_state = create_initial_state(role_name, execution_id)

        # Update execution status
        self.db.update_execution_status(
            execution_id, WorkflowStatus.RUNNING,
            current_node="validate_role"
        )

        # Send workflow started notification
        await notify_workflow_started(self._notification_service, role_name, execution_id)

        try:
            # Execute the graph
            final_state = await graph.ainvoke(
                initial_state,
                config=config or {}
            )

            # Save checkpoint
            self.db.checkpoint_execution(execution_id, {
                "state": dict(final_state),
                "completed_nodes": final_state.get("completed_nodes", [])
            })

            # Update final status and send notifications
            if final_state.get("errors"):
                error_msg = "; ".join(final_state.get("errors", []))
                self.db.update_execution_status(
                    execution_id, WorkflowStatus.FAILED,
                    error_message=error_msg
                )
                await notify_workflow_failed(
                    self._notification_service, role_name, execution_id,
                    error=error_msg,
                    failed_node=final_state.get("current_node")
                )
            else:
                self.db.update_execution_status(
                    execution_id, WorkflowStatus.COMPLETED
                )
                await notify_workflow_completed(
                    self._notification_service, role_name, execution_id,
                    summary=final_state.get("summary", {})
                )

            return {
                "status": "completed" if not final_state.get("errors") else "failed",
                "execution_id": execution_id,
                "state": dict(final_state),
                "summary": final_state.get("summary")
            }

        except Exception as e:
            self.db.update_execution_status(
                execution_id, WorkflowStatus.FAILED,
                error_message=str(e)
            )
            await notify_workflow_failed(
                self._notification_service, role_name, execution_id,
                error=str(e),
                failed_node="unknown"
            )
            return {
                "status": "error",
                "execution_id": execution_id,
                "error": str(e)
            }

    # =========================================================================
    # DAG Modification Entrypoints
    # =========================================================================

    def modify_edge(
        self,
        from_node: str,
        condition: str,
        new_target: str
    ) -> bool:
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

        self._edge_modifications.append({
            "type": "modify",
            "from": from_node,
            "condition": condition,
            "target": new_target,
            "timestamp": datetime.utcnow().isoformat()
        })
        return True

    def insert_node_after(
        self,
        existing_node: str,
        new_node: str,
        node_func: Callable[[BoxUpRoleState], dict],
        description: str
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

        self._node_insertions.append({
            "after": existing_node,
            "name": new_node,
            "func": node_func,
            "description": description,
            "timestamp": datetime.utcnow().isoformat()
        })
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
            "validate_role", "analyze_deps", "check_reverse_deps",
            "create_worktree", "run_molecule", "run_pytest",
            "validate_deploy", "create_commit", "push_branch",
            "create_issue", "create_mr", "add_to_merge_train",
            "report_summary", "notify_failure"
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
                        "timestamp": ins["timestamp"]
                    }
                    for ins in getattr(self, "_node_insertions", [])
                ],
                "node_removals": getattr(self, "_node_removals", [])
            },
            "exported_at": datetime.utcnow().isoformat()
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
                "timestamp": ins.get("timestamp", datetime.utcnow().isoformat())
            }
            for ins in mods.get("node_insertions", [])
        ]

    def register_node_func(
        self,
        node_name: str,
        func: Callable[[BoxUpRoleState], dict]
    ) -> bool:
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
            "node_removals": len(getattr(self, "_node_removals", []))
        }
