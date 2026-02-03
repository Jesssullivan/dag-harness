"""
GitLab status reporting for DAG harness workflows.

Provides comprehensive status reporting for:
- Individual role GitLab status (issues, MRs, pipelines)
- Wave-level aggregation and progress tracking
- Merge train health monitoring
- Full project status overview

The status reporter aggregates data from the local StateDB and GitLab API
to provide a unified view of the workflow progress.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from harness.db.state import StateDB
from harness.gitlab.labels import WAVE_LABEL_DESCRIPTIONS, WAVE_LABELS

logger = logging.getLogger(__name__)


# =============================================================================
# STATUS DATACLASSES
# =============================================================================


@dataclass
class RoleGitLabStatus:
    """
    GitLab status for a single role.

    Tracks the issue, merge request, pipeline, and merge train status
    for a role in the DAG workflow.
    """

    role_name: str
    wave: int
    issue_iid: int | None = None
    issue_state: str | None = None  # "opened", "closed"
    mr_iid: int | None = None
    mr_state: str | None = None  # "opened", "merged", "closed"
    pipeline_status: str | None = None  # "success", "failed", "running", "pending"
    train_position: int | None = None
    iteration: str | None = None

    @property
    def has_issue(self) -> bool:
        """Check if role has an associated issue."""
        return self.issue_iid is not None

    @property
    def has_mr(self) -> bool:
        """Check if role has an associated merge request."""
        return self.mr_iid is not None

    @property
    def is_merged(self) -> bool:
        """Check if role's MR has been merged."""
        return self.mr_state == "merged"

    @property
    def in_merge_train(self) -> bool:
        """Check if role's MR is in the merge train."""
        return self.train_position is not None

    @property
    def pipeline_passing(self) -> bool:
        """Check if the pipeline is in a passing state."""
        return self.pipeline_status == "success"

    @property
    def progress_stage(self) -> str:
        """
        Get a human-readable progress stage for the role.

        Returns one of:
        - "not_started": No issue created
        - "issue_created": Issue exists but no MR
        - "mr_created": MR exists but not in train
        - "in_train": MR is in merge train
        - "merged": MR has been merged
        """
        if self.is_merged:
            return "merged"
        if self.in_merge_train:
            return "in_train"
        if self.has_mr:
            return "mr_created"
        if self.has_issue:
            return "issue_created"
        return "not_started"


@dataclass
class WaveStatus:
    """
    Aggregated status for a wave of roles.

    Provides counts and percentages for tracking wave-level progress.
    """

    wave: int
    wave_name: str
    total_roles: int
    roles_with_issues: int = 0
    roles_with_mrs: int = 0
    mrs_merged: int = 0
    mrs_in_train: int = 0
    roles: list[RoleGitLabStatus] = field(default_factory=list)

    @property
    def completion_percentage(self) -> float:
        """Calculate wave completion percentage based on merged MRs."""
        if self.total_roles == 0:
            return 0.0
        return (self.mrs_merged / self.total_roles) * 100

    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage (issues + MRs + merged)."""
        if self.total_roles == 0:
            return 0.0
        # Weight: issue=25%, MR=50%, merged=100%
        score = (
            self.roles_with_issues * 0.25
            + self.roles_with_mrs * 0.50
            + self.mrs_merged * 1.0
        )
        max_score = self.total_roles * 1.0
        return (score / max_score) * 100

    @property
    def remaining_roles(self) -> int:
        """Get count of roles not yet merged."""
        return self.total_roles - self.mrs_merged


@dataclass
class MergeTrainHealth:
    """
    Health metrics for the merge train.

    Tracks queue depth, timing estimates, and blocked MRs.
    """

    total_in_train: int = 0
    oldest_mr_age_hours: float | None = None
    estimated_completion_hours: float | None = None
    blocked_mrs: list[int] = field(default_factory=list)
    avg_pipeline_duration_minutes: float | None = None

    @property
    def is_healthy(self) -> bool:
        """
        Check if merge train is in a healthy state.

        Unhealthy indicators:
        - More than 10 MRs in train
        - Oldest MR > 24 hours old
        - Any blocked MRs
        """
        if self.total_in_train > 10:
            return False
        if self.oldest_mr_age_hours and self.oldest_mr_age_hours > 24:
            return False
        if self.blocked_mrs:
            return False
        return True

    @property
    def health_status(self) -> str:
        """Get a human-readable health status."""
        if self.total_in_train == 0:
            return "empty"
        if self.is_healthy:
            return "healthy"
        if self.blocked_mrs:
            return "blocked"
        if self.oldest_mr_age_hours and self.oldest_mr_age_hours > 24:
            return "stale"
        if self.total_in_train > 10:
            return "congested"
        return "degraded"


@dataclass
class FullStatus:
    """
    Complete GitLab status for all roles and waves.

    Provides a comprehensive view of the entire workflow progress.
    """

    waves: list[WaveStatus] = field(default_factory=list)
    merge_train: MergeTrainHealth = field(default_factory=MergeTrainHealth)
    total_issues: int = 0
    total_mrs: int = 0
    total_merged: int = 0
    total_roles: int = 0
    last_updated: datetime | None = None

    @property
    def overall_completion_percentage(self) -> float:
        """Calculate overall completion percentage across all roles."""
        if self.total_roles == 0:
            return 0.0
        return (self.total_merged / self.total_roles) * 100

    @property
    def roles_remaining(self) -> int:
        """Get count of roles not yet merged."""
        return self.total_roles - self.total_merged

    def get_wave(self, wave_number: int) -> WaveStatus | None:
        """Get status for a specific wave by number."""
        for wave in self.waves:
            if wave.wave == wave_number:
                return wave
        return None


# =============================================================================
# STATUS REPORTER CLASS
# =============================================================================


class GitLabStatusReporter:
    """
    Comprehensive GitLab status reporter for DAG harness workflows.

    Aggregates data from:
    - Local StateDB for role, issue, and MR metadata
    - GitLab API for real-time pipeline and merge train status

    Usage:
        >>> from harness.db.state import StateDB
        >>> db = StateDB("harness.db")
        >>> reporter = GitLabStatusReporter(db, "bates-ils/projects/ems/ems-mono")
        >>> status = await reporter.get_full_status()
        >>> print(reporter.format_status_table(status))
    """

    def __init__(
        self,
        db: StateDB,
        project_path: str = "bates-ils/projects/ems/ems-mono",
    ):
        """
        Initialize the status reporter.

        Args:
            db: StateDB instance for local data access
            project_path: GitLab project path for API calls
        """
        self.db = db
        self.project_path = project_path
        self._project_path_encoded = project_path.replace("/", "%2F")

    # =========================================================================
    # ROLE STATUS METHODS
    # =========================================================================

    async def get_role_status(self, role_name: str) -> RoleGitLabStatus:
        """
        Get comprehensive GitLab status for a single role.

        Args:
            role_name: Name of the role to get status for

        Returns:
            RoleGitLabStatus with issue, MR, pipeline, and train info

        Raises:
            ValueError: If role is not found in database
        """
        role = self.db.get_role(role_name)
        if not role:
            raise ValueError(f"Role '{role_name}' not found in database")

        # Get issue and MR from database
        issue = self.db.get_issue(role_name)
        mr = self.db.get_merge_request(role_name)

        status = RoleGitLabStatus(
            role_name=role_name,
            wave=role.wave,
            issue_iid=issue.iid if issue else None,
            issue_state=issue.state if issue else None,
            mr_iid=mr.iid if mr else None,
            mr_state=mr.state if mr else None,
        )

        # Get pipeline status if MR exists
        if mr and mr.iid:
            status.pipeline_status = await self._get_mr_pipeline_status(mr.iid)
            status.train_position = await self._get_mr_train_position(mr.iid)

        return status

    async def _get_mr_pipeline_status(self, mr_iid: int) -> str | None:
        """
        Get the latest pipeline status for an MR.

        Args:
            mr_iid: Merge request IID

        Returns:
            Pipeline status string or None if no pipeline
        """
        try:
            import json
            import subprocess

            result = subprocess.run(
                [
                    "glab",
                    "api",
                    f"projects/{self._project_path_encoded}/merge_requests/{mr_iid}/pipelines",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            pipelines = json.loads(result.stdout)
            if pipelines and len(pipelines) > 0:
                return pipelines[0].get("status")

            return None
        except Exception as e:
            logger.warning(f"Failed to get pipeline status for MR !{mr_iid}: {e}")
            return None

    async def _get_mr_train_position(self, mr_iid: int) -> int | None:
        """
        Get the merge train position for an MR.

        Args:
            mr_iid: Merge request IID

        Returns:
            1-indexed position in train, or None if not in train
        """
        try:
            import json
            import subprocess

            result = subprocess.run(
                [
                    "glab",
                    "api",
                    f"projects/{self._project_path_encoded}/merge_trains?target_branch=main",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            trains = json.loads(result.stdout)
            for i, entry in enumerate(trains, 1):
                if entry.get("merge_request", {}).get("iid") == mr_iid:
                    return i

            return None
        except Exception as e:
            logger.warning(f"Failed to get train position for MR !{mr_iid}: {e}")
            return None

    # =========================================================================
    # WAVE STATUS METHODS
    # =========================================================================

    async def get_wave_status(self, wave: int) -> WaveStatus:
        """
        Get aggregated status for a wave of roles.

        Args:
            wave: Wave number (0-8)

        Returns:
            WaveStatus with counts and individual role statuses
        """
        roles = self.db.list_roles(wave=wave)

        wave_name = WAVE_LABEL_DESCRIPTIONS.get(f"wave-{wave}", f"Wave {wave}")

        wave_status = WaveStatus(
            wave=wave,
            wave_name=wave_name,
            total_roles=len(roles),
        )

        for role in roles:
            role_status = await self.get_role_status(role.name)
            wave_status.roles.append(role_status)

            if role_status.has_issue:
                wave_status.roles_with_issues += 1
            if role_status.has_mr:
                wave_status.roles_with_mrs += 1
            if role_status.is_merged:
                wave_status.mrs_merged += 1
            if role_status.in_merge_train:
                wave_status.mrs_in_train += 1

        return wave_status

    # =========================================================================
    # MERGE TRAIN HEALTH METHODS
    # =========================================================================

    async def get_merge_train_health(self) -> MergeTrainHealth:
        """
        Get health metrics for the merge train.

        Returns:
            MergeTrainHealth with queue stats and timing estimates
        """
        health = MergeTrainHealth()

        try:
            import json
            import subprocess

            result = subprocess.run(
                [
                    "glab",
                    "api",
                    f"projects/{self._project_path_encoded}/merge_trains?target_branch=main",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return health

            trains = json.loads(result.stdout)
            health.total_in_train = len(trains)

            if not trains:
                return health

            # Calculate timing metrics
            durations = []
            oldest_age = None
            now = datetime.utcnow()

            for entry in trains:
                # Track duration
                if entry.get("duration"):
                    durations.append(entry["duration"])

                # Track oldest entry
                created_at = entry.get("created_at")
                if created_at:
                    try:
                        # Parse ISO format datetime
                        created = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                        age_hours = (
                            now - created.replace(tzinfo=None)
                        ).total_seconds() / 3600
                        if oldest_age is None or age_hours > oldest_age:
                            oldest_age = age_hours
                    except Exception:
                        pass

                # Check for blocked MRs (failed pipeline)
                mr = entry.get("merge_request", {})
                pipeline = mr.get("head_pipeline", {})
                if pipeline.get("status") == "failed":
                    health.blocked_mrs.append(mr.get("iid"))

            health.oldest_mr_age_hours = oldest_age

            # Calculate average pipeline duration
            if durations:
                avg_duration_seconds = sum(durations) / len(durations)
                health.avg_pipeline_duration_minutes = avg_duration_seconds / 60

                # Estimate completion time for entire train
                remaining_entries = health.total_in_train
                health.estimated_completion_hours = (
                    remaining_entries * avg_duration_seconds
                ) / 3600

        except Exception as e:
            logger.warning(f"Failed to get merge train health: {e}")

        return health

    # =========================================================================
    # FULL STATUS METHODS
    # =========================================================================

    async def get_full_status(self) -> FullStatus:
        """
        Get comprehensive status for all roles and waves.

        Returns:
            FullStatus with all wave statuses and merge train health
        """
        status = FullStatus(last_updated=datetime.utcnow())

        # Get all roles to determine wave range
        all_roles = self.db.list_roles()
        status.total_roles = len(all_roles)

        # Determine which waves have roles
        waves_with_roles = set(role.wave for role in all_roles)

        # Get status for each wave
        for wave_num in sorted(waves_with_roles):
            wave_status = await self.get_wave_status(wave_num)
            status.waves.append(wave_status)

            # Aggregate totals
            status.total_issues += wave_status.roles_with_issues
            status.total_mrs += wave_status.roles_with_mrs
            status.total_merged += wave_status.mrs_merged

        # Get merge train health
        status.merge_train = await self.get_merge_train_health()

        return status

    # =========================================================================
    # FORMATTING METHODS
    # =========================================================================

    def format_status_table(self, status: FullStatus) -> str:
        """
        Format the full status as an ASCII table.

        Args:
            status: FullStatus to format

        Returns:
            Formatted ASCII table string
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("GitLab Status Report")
        if status.last_updated:
            lines.append(f"Updated: {status.last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 80)
        lines.append("")

        # Overall summary
        lines.append("OVERALL PROGRESS")
        lines.append("-" * 40)
        lines.append(f"  Total Roles:     {status.total_roles}")
        lines.append(f"  Issues Created:  {status.total_issues}")
        lines.append(f"  MRs Created:     {status.total_mrs}")
        lines.append(f"  MRs Merged:      {status.total_merged}")
        lines.append(
            f"  Completion:      {status.overall_completion_percentage:.1f}%"
        )
        lines.append(f"  Remaining:       {status.roles_remaining}")
        lines.append("")

        # Merge train health
        lines.append("MERGE TRAIN")
        lines.append("-" * 40)
        train = status.merge_train
        lines.append(f"  Status:          {train.health_status.upper()}")
        lines.append(f"  In Queue:        {train.total_in_train}")
        if train.oldest_mr_age_hours:
            lines.append(f"  Oldest MR:       {train.oldest_mr_age_hours:.1f} hours")
        if train.estimated_completion_hours:
            lines.append(
                f"  Est. Completion: {train.estimated_completion_hours:.1f} hours"
            )
        if train.avg_pipeline_duration_minutes:
            lines.append(
                f"  Avg Pipeline:    {train.avg_pipeline_duration_minutes:.1f} min"
            )
        if train.blocked_mrs:
            lines.append(f"  Blocked MRs:     {', '.join(f'!{iid}' for iid in train.blocked_mrs)}")
        lines.append("")

        # Wave breakdown table
        lines.append("WAVE BREAKDOWN")
        lines.append("-" * 80)
        lines.append(
            f"{'Wave':<8} {'Name':<30} {'Total':<7} {'Issues':<8} {'MRs':<6} {'Merged':<8} {'%':<6}"
        )
        lines.append("-" * 80)

        for wave in status.waves:
            lines.append(
                f"{wave.wave:<8} "
                f"{wave.wave_name[:28]:<30} "
                f"{wave.total_roles:<7} "
                f"{wave.roles_with_issues:<8} "
                f"{wave.roles_with_mrs:<6} "
                f"{wave.mrs_merged:<8} "
                f"{wave.completion_percentage:>5.1f}%"
            )

        lines.append("-" * 80)
        lines.append("")

        # Detailed role status for each wave
        for wave in status.waves:
            if not wave.roles:
                continue

            lines.append(f"WAVE {wave.wave}: {wave.wave_name}")
            lines.append("-" * 60)
            lines.append(
                f"  {'Role':<25} {'Issue':<10} {'MR':<10} {'Pipeline':<12} {'Stage':<12}"
            )
            lines.append("  " + "-" * 58)

            for role in sorted(wave.roles, key=lambda r: r.role_name):
                issue_str = f"#{role.issue_iid}" if role.issue_iid else "-"
                mr_str = f"!{role.mr_iid}" if role.mr_iid else "-"
                pipeline_str = role.pipeline_status or "-"
                stage_str = role.progress_stage

                # Add train position indicator
                if role.train_position:
                    stage_str = f"train #{role.train_position}"

                lines.append(
                    f"  {role.role_name:<25} "
                    f"{issue_str:<10} "
                    f"{mr_str:<10} "
                    f"{pipeline_str:<12} "
                    f"{stage_str:<12}"
                )

            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def format_compact_status(self, status: FullStatus) -> str:
        """
        Format a compact single-line status summary.

        Args:
            status: FullStatus to format

        Returns:
            Compact status string
        """
        train_indicator = ""
        if status.merge_train.total_in_train > 0:
            train_indicator = f" | Train: {status.merge_train.total_in_train}"
            if status.merge_train.blocked_mrs:
                train_indicator += f" ({len(status.merge_train.blocked_mrs)} blocked)"

        return (
            f"Progress: {status.total_merged}/{status.total_roles} merged "
            f"({status.overall_completion_percentage:.1f}%) | "
            f"Issues: {status.total_issues} | MRs: {status.total_mrs}"
            f"{train_indicator}"
        )

    def format_wave_summary(self, wave_status: WaveStatus) -> str:
        """
        Format a summary for a single wave.

        Args:
            wave_status: WaveStatus to format

        Returns:
            Formatted wave summary string
        """
        lines = [
            f"Wave {wave_status.wave}: {wave_status.wave_name}",
            f"  Progress: {wave_status.completion_percentage:.1f}% complete",
            f"  Roles: {wave_status.total_roles} total, {wave_status.mrs_merged} merged",
            f"  MRs: {wave_status.roles_with_mrs} open, {wave_status.mrs_in_train} in train",
        ]

        # List incomplete roles
        incomplete = [r for r in wave_status.roles if not r.is_merged]
        if incomplete:
            lines.append(f"  Pending: {', '.join(r.role_name for r in incomplete[:5])}")
            if len(incomplete) > 5:
                lines.append(f"           ... and {len(incomplete) - 5} more")

        return "\n".join(lines)
