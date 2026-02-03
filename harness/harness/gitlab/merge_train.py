"""
Enhanced merge train functionality for GitLab.

Provides comprehensive merge readiness checks, merge train position monitoring,
and pre-flight validation for adding MRs to merge trains.

This module extends the basic merge train operations in advanced.py with:
- MergeReadinessResult: Comprehensive MR readiness assessment
- get_mr_merge_readiness: Check pipeline, conflicts, approvals
- wait_for_merge_train_position: Poll for MR position in train
- preflight_merge_train_check: Validate MR can be added to train
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

from harness.gitlab.advanced import (
    GitLabAsyncBase,
    GitLabAsyncConfig,
    encode_project_path,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MergeReadinessResult:
    """
    Comprehensive assessment of merge request readiness.

    Provides detailed information about whether an MR can be merged,
    including specific blockers and approval status.

    Attributes:
        mergeable: Whether the MR can be merged (no blockers)
        blockers: List of blocker identifiers explaining why MR cannot merge
        pipeline_status: Current pipeline status (success, failed, running, etc.)
        has_conflicts: Whether the MR has merge conflicts with target branch
        approvals_required: Number of approvals required by project rules
        approvals_given: Number of approvals already received

    Blocker identifiers:
        - "pipeline_failed": Pipeline status is 'failed' or 'canceled'
        - "pipeline_running": Pipeline is still running
        - "pipeline_pending": Pipeline has not started
        - "no_pipeline": No pipeline exists for this MR
        - "has_conflicts": MR has merge conflicts
        - "needs_rebase": MR needs to be rebased
        - "insufficient_approvals": Not enough approvals
        - "discussions_unresolved": Unresolved discussions/threads
        - "draft_status": MR is still in draft mode
        - "blocked_status": MR is blocked by another MR
        - "not_open": MR is not in 'opened' state
        - "already_in_train": MR is already in the merge train
    """

    mergeable: bool
    blockers: list[str] = field(default_factory=list)
    pipeline_status: str | None = None
    has_conflicts: bool = False
    approvals_required: int = 0
    approvals_given: int = 0

    @property
    def needs_pipeline(self) -> bool:
        """Check if pipeline needs to pass before merge."""
        return self.pipeline_status in (None, "pending", "running", "created")

    @property
    def pipeline_passed(self) -> bool:
        """Check if pipeline has passed."""
        return self.pipeline_status == "success"

    @property
    def pipeline_failed(self) -> bool:
        """Check if pipeline has failed."""
        return self.pipeline_status in ("failed", "canceled")

    @property
    def approvals_satisfied(self) -> bool:
        """Check if approval requirements are met."""
        return self.approvals_given >= self.approvals_required

    @property
    def blocker_summary(self) -> str:
        """Human-readable summary of blockers."""
        if not self.blockers:
            return "No blockers - MR is ready to merge"

        blocker_descriptions = {
            "pipeline_failed": "Pipeline has failed",
            "pipeline_running": "Pipeline is still running",
            "pipeline_pending": "Pipeline has not started",
            "no_pipeline": "No pipeline exists",
            "has_conflicts": "Has merge conflicts",
            "needs_rebase": "Needs to be rebased",
            "insufficient_approvals": f"Needs {self.approvals_required - self.approvals_given} more approval(s)",
            "discussions_unresolved": "Has unresolved discussions",
            "draft_status": "MR is marked as draft",
            "blocked_status": "Blocked by another MR",
            "not_open": "MR is not open",
            "already_in_train": "Already in merge train",
        }

        descriptions = [
            blocker_descriptions.get(b, b) for b in self.blockers
        ]
        return "; ".join(descriptions)


# =============================================================================
# MERGE TRAIN HELPER CLASS
# =============================================================================


class MergeTrainHelper(GitLabAsyncBase):
    """
    Enhanced merge train operations with readiness checks and monitoring.

    Extends GitLabAsyncBase to provide comprehensive merge train management
    including readiness assessment, position monitoring, and pre-flight checks.
    """

    def __init__(self, config: GitLabAsyncConfig | None = None):
        super().__init__(config)

    async def _get_mr_details(
        self, project_path: str, mr_iid: int
    ) -> dict:
        """
        Get detailed merge request information.

        Args:
            project_path: Full project path
            mr_iid: Merge request IID

        Returns:
            MR data from GitLab API
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(f"projects/{encoded}/merge_requests/{mr_iid}")

    async def _get_mr_approvals(
        self, project_path: str, mr_iid: int
    ) -> dict:
        """
        Get merge request approval status.

        Args:
            project_path: Full project path
            mr_iid: Merge request IID

        Returns:
            Approval data from GitLab API
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(f"projects/{encoded}/merge_requests/{mr_iid}/approvals")

    async def _get_mr_pipelines(
        self, project_path: str, mr_iid: int
    ) -> list:
        """
        Get pipelines for a merge request.

        Args:
            project_path: Full project path
            mr_iid: Merge request IID

        Returns:
            List of pipeline data from GitLab API
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(f"projects/{encoded}/merge_requests/{mr_iid}/pipelines")

    async def _get_merge_train(
        self, project_path: str, target_branch: str = "main"
    ) -> list:
        """
        Get the merge train queue for a target branch.

        Args:
            project_path: Full project path
            target_branch: Target branch name

        Returns:
            List of merge train entries
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(
            f"projects/{encoded}/merge_trains?target_branch={target_branch}"
        )

    async def _is_mr_in_train(
        self, project_path: str, mr_iid: int, target_branch: str = "main"
    ) -> bool:
        """
        Check if an MR is already in the merge train.

        Args:
            project_path: Full project path
            mr_iid: Merge request IID
            target_branch: Target branch name

        Returns:
            True if MR is in train, False otherwise
        """
        try:
            train = await self._get_merge_train(project_path, target_branch)
            for entry in train:
                if entry.get("merge_request", {}).get("iid") == mr_iid:
                    return True
            return False
        except Exception:
            return False


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================


async def get_mr_merge_readiness(
    project_path: str,
    mr_iid: int,
    config: GitLabAsyncConfig | None = None,
) -> MergeReadinessResult:
    """
    Get comprehensive merge readiness assessment for an MR.

    Checks:
    - Pipeline status (passed, failed, running, pending)
    - Merge conflicts
    - Approval requirements
    - MR state (draft, blocked, closed)
    - Whether already in merge train

    Args:
        project_path: Full project path (e.g., "bates-ils/projects/ems/ems-mono")
        mr_iid: Merge request IID (project-scoped)
        config: Optional GitLab configuration

    Returns:
        MergeReadinessResult with detailed assessment

    Example:
        >>> result = await get_mr_merge_readiness("group/project", 123)
        >>> if result.mergeable:
        ...     print("Ready to merge!")
        ... else:
        ...     print(f"Blocked: {result.blocker_summary}")
    """
    helper = MergeTrainHelper(config)
    blockers: list[str] = []

    try:
        # Get MR details, approvals, and pipelines in parallel
        mr_task = helper._get_mr_details(project_path, mr_iid)
        approvals_task = helper._get_mr_approvals(project_path, mr_iid)
        pipelines_task = helper._get_mr_pipelines(project_path, mr_iid)

        mr_data, approvals_data, pipelines = await asyncio.gather(
            mr_task, approvals_task, pipelines_task,
            return_exceptions=True
        )

        # Handle errors from parallel requests
        if isinstance(mr_data, Exception):
            raise mr_data
        if isinstance(approvals_data, Exception):
            # Approvals might not be available (not GitLab Premium)
            approvals_data = {}
        if isinstance(pipelines, Exception):
            pipelines = []

        # Check MR state
        mr_state = mr_data.get("state", "")
        if mr_state != "opened":
            blockers.append("not_open")

        # Check draft status
        if mr_data.get("draft", False) or mr_data.get("work_in_progress", False):
            blockers.append("draft_status")

        # Check merge status / conflicts
        merge_status = mr_data.get("merge_status", "")
        has_conflicts = merge_status in (
            "cannot_be_merged",
            "cannot_be_merged_recheck",
            "cannot_be_merged_rechecking",
        )

        if has_conflicts:
            blockers.append("has_conflicts")

        # Check if needs rebase
        if mr_data.get("should_be_rebased", False):
            blockers.append("needs_rebase")

        # Check unresolved discussions
        if mr_data.get("blocking_discussions_resolved") is False:
            blockers.append("discussions_unresolved")

        # Check pipeline status
        pipeline_status = None
        if pipelines and len(pipelines) > 0:
            pipeline_status = pipelines[0].get("status")

            if pipeline_status in ("failed", "canceled"):
                blockers.append("pipeline_failed")
            elif pipeline_status == "running":
                blockers.append("pipeline_running")
            elif pipeline_status in ("pending", "created", "waiting_for_resource"):
                blockers.append("pipeline_pending")
        else:
            # No pipeline - might be required
            if mr_data.get("only_allow_merge_if_pipeline_succeeds", False):
                blockers.append("no_pipeline")

        # Check approvals
        approvals_required = approvals_data.get("approvals_required", 0)
        approvals_given = len(approvals_data.get("approved_by", []))

        if approvals_required > 0 and approvals_given < approvals_required:
            blockers.append("insufficient_approvals")

        # Check if already in merge train
        try:
            target_branch = mr_data.get("target_branch", "main")
            in_train = await helper._is_mr_in_train(project_path, mr_iid, target_branch)
            if in_train:
                blockers.append("already_in_train")
        except Exception:
            pass  # Not critical if we can't check

        return MergeReadinessResult(
            mergeable=len(blockers) == 0,
            blockers=blockers,
            pipeline_status=pipeline_status,
            has_conflicts=has_conflicts,
            approvals_required=approvals_required,
            approvals_given=approvals_given,
        )

    except Exception as e:
        logger.error(f"Failed to get merge readiness for MR !{mr_iid}: {e}")
        return MergeReadinessResult(
            mergeable=False,
            blockers=[f"error: {str(e)}"],
            pipeline_status=None,
            has_conflicts=False,
            approvals_required=0,
            approvals_given=0,
        )


async def wait_for_merge_train_position(
    project_path: str,
    mr_iid: int,
    timeout_seconds: int = 300,
    poll_interval: int = 10,
    config: GitLabAsyncConfig | None = None,
) -> int | None:
    """
    Wait for an MR to appear in the merge train and return its position.

    Polls the merge train queue until the MR appears or timeout is reached.
    Useful for monitoring MRs that were just added to the train.

    Args:
        project_path: Full project path
        mr_iid: Merge request IID
        timeout_seconds: Maximum time to wait (default: 300 = 5 minutes)
        poll_interval: Time between polls in seconds (default: 10)
        config: Optional GitLab configuration

    Returns:
        Position (1-indexed) in merge train, or None if timeout reached

    Example:
        >>> position = await wait_for_merge_train_position("group/project", 123)
        >>> if position:
        ...     print(f"MR is at position {position} in the merge train")
        ... else:
        ...     print("MR did not appear in merge train within timeout")
    """
    helper = MergeTrainHelper(config)
    start_time = asyncio.get_event_loop().time()

    # First, get the target branch from the MR
    try:
        mr_data = await helper._get_mr_details(project_path, mr_iid)
        target_branch = mr_data.get("target_branch", "main")
    except Exception as e:
        logger.error(f"Failed to get MR details: {e}")
        return None

    while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
        try:
            train = await helper._get_merge_train(project_path, target_branch)

            for i, entry in enumerate(train, 1):
                if entry.get("merge_request", {}).get("iid") == mr_iid:
                    logger.info(f"MR !{mr_iid} found at position {i} in merge train")
                    return i

        except Exception as e:
            logger.warning(f"Error checking merge train: {e}")

        await asyncio.sleep(poll_interval)

    logger.warning(
        f"Timeout waiting for MR !{mr_iid} to appear in merge train "
        f"after {timeout_seconds} seconds"
    )
    return None


async def preflight_merge_train_check(
    project_path: str,
    mr_iid: int,
    config: GitLabAsyncConfig | None = None,
) -> tuple[bool, list[str]]:
    """
    Perform pre-flight checks before adding MR to merge train.

    This is a comprehensive validation that checks all requirements
    for adding an MR to the merge train:
    - Pipeline must be passing (or not required)
    - No merge conflicts
    - Not already in the merge train
    - MR is in 'opened' state
    - Not a draft MR

    Args:
        project_path: Full project path
        mr_iid: Merge request IID
        config: Optional GitLab configuration

    Returns:
        Tuple of (can_add: bool, blockers: list[str])
        - can_add: True if MR can be added to merge train
        - blockers: List of reasons preventing addition (empty if can_add)

    Example:
        >>> can_add, blockers = await preflight_merge_train_check("group/project", 123)
        >>> if can_add:
        ...     # Safe to add to merge train
        ...     await add_to_merge_train(...)
        ... else:
        ...     print(f"Cannot add to train: {blockers}")
    """
    # Get full readiness assessment
    readiness = await get_mr_merge_readiness(project_path, mr_iid, config)

    # Filter blockers to only those that prevent merge train addition
    # Some blockers like "pipeline_running" don't prevent addition if
    # using when_pipeline_succeeds=True
    critical_blockers = []

    for blocker in readiness.blockers:
        # These blockers prevent merge train addition
        if blocker in (
            "not_open",
            "draft_status",
            "has_conflicts",
            "already_in_train",
            "pipeline_failed",
            "needs_rebase",
        ):
            critical_blockers.append(blocker)
        # pipeline_running and pipeline_pending are OK if using when_pipeline_succeeds
        # insufficient_approvals might be OK depending on project settings
        # discussions_unresolved depends on project settings

    can_add = len(critical_blockers) == 0

    if can_add:
        logger.info(f"MR !{mr_iid} passes pre-flight checks for merge train")
    else:
        logger.warning(
            f"MR !{mr_iid} failed pre-flight checks: {critical_blockers}"
        )

    return (can_add, critical_blockers)


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================


def create_merge_train_helper(
    project_path: str | None = None,
) -> MergeTrainHelper:
    """
    Create a MergeTrainHelper with optional custom project path.

    Args:
        project_path: Override default project path

    Returns:
        Configured MergeTrainHelper instance
    """
    config = GitLabAsyncConfig()
    if project_path:
        config.project_path = project_path
    return MergeTrainHelper(config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MergeReadinessResult",
    "MergeTrainHelper",
    "get_mr_merge_readiness",
    "wait_for_merge_train_position",
    "preflight_merge_train_check",
    "create_merge_train_helper",
]
