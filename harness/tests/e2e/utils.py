"""E2E utility functions for DAG Harness tests.

All GitLab operations use the `glab` CLI via subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# GITLAB WAIT HELPERS
# ============================================================================


async def wait_for_pipeline(
    project_path: str,
    ref: str,
    timeout: int = 300,
) -> dict:
    """Wait for a pipeline to complete on the given ref.

    Polls the GitLab API via ``glab`` until the pipeline reaches a terminal
    state (``success``, ``failed``, ``canceled``, ``skipped``) or the timeout
    is exceeded.

    Args:
        project_path: GitLab project path (e.g. "tinyland/projects/dag-harness-tests").
        ref: Git ref (branch or tag) whose pipeline to monitor.
        timeout: Maximum seconds to wait (default 300).

    Returns:
        Pipeline status dict with at least ``id`` and ``status`` keys.

    Raises:
        TimeoutError: If no terminal status is reached within *timeout* seconds.
    """
    terminal_states = {"success", "failed", "canceled", "skipped"}
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "glab",
                "api",
                f"projects/{_encode_path(project_path)}/pipelines",
                "-X",
                "GET",
                "-f",
                f"ref={ref}",
                "-f",
                "per_page=1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            pipelines = json.loads(result.stdout)
            if pipelines and isinstance(pipelines, list):
                pipeline = pipelines[0]
                status = pipeline.get("status", "")
                logger.debug("Pipeline %s status: %s", pipeline.get("id"), status)
                if status in terminal_states:
                    return pipeline

        await asyncio.sleep(10)

    raise TimeoutError(
        f"Pipeline on ref '{ref}' in '{project_path}' did not complete within {timeout}s"
    )


async def wait_for_merge_train(
    project_path: str,
    mr_iid: int,
    timeout: int = 300,
) -> dict:
    """Wait for an MR to be processed by the merge train.

    Polls until the MR is merged, the merge train removes it, or the timeout
    is exceeded.

    Args:
        project_path: GitLab project path.
        mr_iid: Merge request IID.
        timeout: Maximum seconds to wait (default 300).

    Returns:
        Merge request dict with current state.

    Raises:
        TimeoutError: If the MR is not processed within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "glab",
                "api",
                f"projects/{_encode_path(project_path)}/merge_requests/{mr_iid}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and result.stdout.strip():
            mr = json.loads(result.stdout)
            state = mr.get("state", "")
            logger.debug("MR !%d state: %s", mr_iid, state)
            if state == "merged":
                return mr
            if state == "closed":
                return mr

        await asyncio.sleep(10)

    raise TimeoutError(
        f"MR !{mr_iid} in '{project_path}' was not processed by merge train within {timeout}s"
    )


# ============================================================================
# CLEANUP
# ============================================================================


def cleanup_test_artifacts(project_path: str, tracker: dict) -> None:
    """Clean up issues, MRs, and branches created during tests.

    Best-effort: logs warnings on failure but never raises.

    Args:
        project_path: GitLab project path.
        tracker: Dict with keys ``issues``, ``merge_requests``, ``branches``
                 each containing a list of IIDs / branch names.
    """
    encoded = _encode_path(project_path)

    # Close merge requests first (before deleting branches)
    for mr_iid in tracker.get("merge_requests", []):
        try:
            subprocess.run(
                [
                    "glab",
                    "api",
                    f"projects/{encoded}/merge_requests/{mr_iid}",
                    "-X",
                    "PUT",
                    "-f",
                    "state_event=close",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.debug("Closed MR !%d", mr_iid)
        except Exception:
            logger.warning("Failed to close MR !%d during cleanup", mr_iid, exc_info=True)

    # Close issues
    for issue_iid in tracker.get("issues", []):
        try:
            subprocess.run(
                [
                    "glab",
                    "api",
                    f"projects/{encoded}/issues/{issue_iid}",
                    "-X",
                    "PUT",
                    "-f",
                    "state_event=close",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.debug("Closed issue #%d", issue_iid)
        except Exception:
            logger.warning("Failed to close issue #%d during cleanup", issue_iid, exc_info=True)

    # Delete branches
    for branch in tracker.get("branches", []):
        try:
            subprocess.run(
                [
                    "glab",
                    "api",
                    f"projects/{encoded}/repository/branches/{branch}",
                    "-X",
                    "DELETE",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.debug("Deleted branch %s", branch)
        except Exception:
            logger.warning("Failed to delete branch '%s' during cleanup", branch, exc_info=True)


# ============================================================================
# ROLE STRUCTURE
# ============================================================================


def create_test_role_structure(base_path: Path, role_name: str) -> Path:
    """Create a minimal Ansible role directory structure for testing.

    Creates the standard ``tasks/``, ``defaults/``, and ``meta/`` directories
    with stub ``main.yml`` files.

    Args:
        base_path: Parent directory under which the role directory is created.
        role_name: Name of the role (used as directory name).

    Returns:
        Path to the created role directory.
    """
    role_dir = base_path / role_name
    role_dir.mkdir(parents=True, exist_ok=True)

    # tasks/main.yml
    tasks_dir = role_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    (tasks_dir / "main.yml").write_text(
        f"---\n- name: E2E test task for {role_name}\n  ansible.builtin.debug:\n"
        f'    msg: "Hello from {role_name}"\n'
    )

    # defaults/main.yml
    defaults_dir = role_dir / "defaults"
    defaults_dir.mkdir(exist_ok=True)
    (defaults_dir / "main.yml").write_text(f"---\n{role_name}_enabled: true\n")

    # meta/main.yml
    meta_dir = role_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    (meta_dir / "main.yml").write_text(
        f"---\ngalaxy_info:\n  description: E2E test role {role_name}\ndependencies: []\n"
    )

    return role_dir


# ============================================================================
# ASSERTIONS
# ============================================================================


def assert_gitlab_resource_exists(
    project_path: str,
    resource_type: str,
    iid: int,
) -> dict:
    """Assert a GitLab resource exists and return it.

    Args:
        project_path: GitLab project path.
        resource_type: One of ``issues`` or ``merge_requests``.
        iid: The IID of the resource.

    Returns:
        The resource dict from the GitLab API.

    Raises:
        AssertionError: If the resource does not exist or the API call fails.
    """
    encoded = _encode_path(project_path)
    result = subprocess.run(
        [
            "glab",
            "api",
            f"projects/{encoded}/{resource_type}/{iid}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"glab api call failed for {resource_type}/{iid}: {result.stderr}"
    )

    data = json.loads(result.stdout)
    assert "id" in data, f"Resource {resource_type}/{iid} not found in {project_path}"
    return data


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _encode_path(project_path: str) -> str:
    """URL-encode a GitLab project path for API calls.

    Replaces ``/`` with ``%2F`` so that paths like
    ``tinyland/projects/dag-harness`` become
    ``tinyland%2Fprojects%2Fdag-harness``.
    """
    return project_path.replace("/", "%2F")
