"""
LangGraph workflow node functions.

Each node function takes BoxUpRoleState and returns a dict of state updates.
Nodes are async for consistency with LangGraph patterns.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

from langgraph.types import interrupt

from harness.dag.langgraph_state import (
    BoxUpRoleState,
    _record_test_result,
    get_module_config,
    get_module_db,
)
from harness.db.models import TestType

logger = logging.getLogger(__name__)


def _load_env_file(env_path: Path) -> dict[str, str]:
    """Load environment variables from a .env file.

    Returns a dict that can be merged with os.environ for subprocess calls.
    """
    env = {}
    if not env_path.exists():
        return env

    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Handle export VAR=value and VAR=value formats
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env[key.strip()] = value
    except Exception as e:
        logger.warning(f"Failed to load .env file {env_path}: {e}")

    return env


def _get_subprocess_env(worktree_path: str) -> dict[str, str]:
    """Get environment variables for subprocess calls.

    Merges current environment with .env file from ansible directory.
    """
    env = os.environ.copy()

    # Look for .env file in ansible directory
    ansible_env = Path(worktree_path) / "ansible" / ".env"
    if ansible_env.exists():
        env.update(_load_env_file(ansible_env))
        logger.debug(f"Loaded environment from {ansible_env}")

    # Also check repo root .env
    root_env = Path(worktree_path) / ".env"
    if root_env.exists():
        env.update(_load_env_file(root_env))
        logger.debug(f"Loaded environment from {root_env}")

    return env


# ============================================================================
# VALIDATION AND ANALYSIS NODES
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


# ============================================================================
# WORKTREE NODE
# ============================================================================


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


# ============================================================================
# TEST NODES
# ============================================================================


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

    # Get timeout from config
    harness_config = get_module_config()
    timeout = 600  # default 10 minutes
    if harness_config and hasattr(harness_config, "testing"):
        timeout = harness_config.testing.molecule_timeout

    try:
        # Load environment variables from .env files
        env = _get_subprocess_env(worktree_path)

        result = subprocess.run(
            ["npm", "run", "molecule:role", f"--role={role_name}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=worktree_path,
            env=env,
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
            error_message=f"Test timed out after {timeout} seconds",
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
        # Load environment variables from .env files
        env = _get_subprocess_env(worktree_path)

        result = subprocess.run(
            ["pytest", str(test_path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            cwd=worktree_path,
            env=env,
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
# GIT NODES
# ============================================================================


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


# ============================================================================
# GITLAB NODES
# ============================================================================


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
        from harness.gitlab.api import GitLabClient, GitLabConfig as ApiGitLabConfig

        # Use config from harness.yml if available
        harness_config = get_module_config()
        api_config = None
        if harness_config and hasattr(harness_config, "gitlab"):
            gl = harness_config.gitlab
            api_config = ApiGitLabConfig(
                project_path=gl.project_path,
                group_path=gl.group_path,
                default_assignee=gl.default_assignee,
                default_labels=gl.default_labels,
            )
        client = GitLabClient(db, config=api_config)

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
        from harness.gitlab.api import GitLabClient, GitLabConfig as ApiGitLabConfig

        # Use config from harness.yml if available
        harness_config = get_module_config()
        api_config = None
        if harness_config and hasattr(harness_config, "gitlab"):
            gl = harness_config.gitlab
            api_config = ApiGitLabConfig(
                project_path=gl.project_path,
                group_path=gl.group_path,
                default_assignee=gl.default_assignee,
                default_labels=gl.default_labels,
            )
        client = GitLabClient(db, config=api_config)

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
                logger.warning(
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


# ============================================================================
# HUMAN-IN-THE-LOOP NODE
# ============================================================================


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


# ============================================================================
# SUMMARY AND FAILURE NODES
# ============================================================================


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
                from harness.gitlab.api import GitLabClient, GitLabConfig as ApiGitLabConfig

                harness_config = get_module_config()
                api_config = None
                if harness_config and hasattr(harness_config, "gitlab"):
                    gl = harness_config.gitlab
                    api_config = ApiGitLabConfig(
                        project_path=gl.project_path,
                        group_path=gl.group_path,
                        default_assignee=gl.default_assignee,
                        default_labels=gl.default_labels,
                    )
                client = GitLabClient(db, config=api_config)
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


__all__ = [
    # Validation and analysis
    "validate_role_node",
    "analyze_deps_node",
    "check_reverse_deps_node",
    # Worktree
    "create_worktree_node",
    # Tests
    "run_molecule_node",
    "run_pytest_node",
    "merge_test_results_node",
    "validate_deploy_node",
    # Git
    "create_commit_node",
    "push_branch_node",
    # GitLab
    "create_issue_node",
    "create_mr_node",
    "add_to_merge_train_node",
    # Human-in-the-loop
    "human_approval_node",
    # Summary and failure
    "report_summary_node",
    "notify_failure_node",
]
