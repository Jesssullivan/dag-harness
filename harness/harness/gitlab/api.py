"""
GitLab API client for harness operations.

Handles:
- Iteration management (group-level)
- Issue creation and assignment
- Merge request creation
- Merge train integration
- Label management

.. deprecated::
    GitLabClient is deprecated. Use GitLabAPI from harness.gitlab.http_client instead.
    GitLabAPI provides the same functionality using async httpx without glab CLI dependency.

    Migration example::

        # Old (deprecated):
        from harness.gitlab.api import GitLabClient
        client = GitLabClient(db)
        issue = client.create_issue(role_name, title, description)

        # New (recommended):
        from harness.gitlab.http_client import GitLabAPI, GitLabAPIConfig
        config = GitLabAPIConfig.from_harness_yml(repo_root)
        async with GitLabAPI(config) as api:
            issue, created = await api.get_or_create_issue(search, title, description)
"""

import json
import logging
import os
import subprocess
import time
import urllib.parse
import warnings
from dataclasses import dataclass, field
from typing import Any

from harness.db.models import Issue, Iteration, MergeRequest
from harness.db.state import StateDB

logger = logging.getLogger(__name__)


# =============================================================================
# LABEL CONFIGURATION
# =============================================================================

# Wave labels with colors that indicate progression (cooler to warmer)
WAVE_LABEL_COLORS: dict[str, str] = {
    "wave-0": "#0033CC",  # Blue - Foundation
    "wave-1": "#00CC66",  # Green - Core Infrastructure
    "wave-2": "#FFCC00",  # Yellow - Core Application
    "wave-3": "#FF6600",  # Orange - Integration
    "wave-4": "#CC0033",  # Red - Final
}

# Wave label descriptions for documentation
WAVE_LABEL_DESCRIPTIONS: dict[str, str] = {
    "wave-0": "Foundation layer roles with no dependencies",
    "wave-1": "Core infrastructure roles",
    "wave-2": "Core application roles",
    "wave-3": "Integration and orchestration roles",
    "wave-4": "Final deployment and verification roles",
}

# Scoped labels using GitLab's :: notation for grouping
SCOPED_LABELS: dict[str, list[str]] = {
    "priority": ["high", "medium", "low"],
    "status": ["ready", "in-progress", "blocked", "review"],
    "type": ["role", "molecule", "fix", "chore"],
}

# Colors for scoped label categories
SCOPED_LABEL_COLORS: dict[str, str] = {
    "priority": "#D93F0B",  # Red-orange for priority
    "status": "#0E8A16",  # Green for status
    "type": "#1D76DB",  # Blue for type
}

# Default color for labels when not specified
DEFAULT_LABEL_COLOR = "#428BCA"


@dataclass
class GitLabConfig:
    """GitLab configuration."""

    project_path: str = "bates-ils/projects/ems/ems-mono"
    group_path: str = "bates-ils"
    default_assignee: str = "jsullivan2"
    default_labels: list[str] = None
    default_reviewers: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.default_labels is None:
            self.default_labels = ["role", "ansible", "molecule"]
        # Load default reviewers from environment if not provided
        if not self.default_reviewers:
            env_reviewers = os.environ.get("GITLAB_DEFAULT_REVIEWERS", "")
            if env_reviewers:
                self.default_reviewers = [r.strip() for r in env_reviewers.split(",") if r.strip()]

    @property
    def project_path_encoded(self) -> str:
        return self.project_path.replace("/", "%2F")


class GitLabClient:
    """
    GitLab API client using glab CLI and direct API calls.

    Uses glab CLI where possible for authentication, falls back to
    direct API calls for features not supported by glab (e.g., iterations).

    .. deprecated::
        This class is deprecated. Use :class:`GitLabAPI` from
        :mod:`harness.gitlab.http_client` instead for async HTTP operations
        without glab CLI dependency.
    """

    def __init__(self, db: StateDB, config: GitLabConfig | None = None):
        warnings.warn(
            "GitLabClient is deprecated. Use GitLabAPI from harness.gitlab.http_client instead. "
            "GitLabAPI provides async HTTP operations without glab CLI dependency.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.db = db
        self.config = config or GitLabConfig()
        self._token: str | None = None

    @property
    def token(self) -> str:
        """Get GitLab token from environment or glab config."""
        if self._token:
            return self._token

        # Try environment variables
        self._token = os.environ.get("GITLAB_TOKEN") or os.environ.get("GLAB_TOKEN")
        if self._token:
            return self._token

        # Try glab config
        try:
            result = subprocess.run(
                ["glab", "auth", "status", "-t"], capture_output=True, text=True
            )
            # Parse token from output if available
            for line in result.stdout.split("\n"):
                if "Token:" in line:
                    self._token = line.split("Token:")[1].strip()
                    return self._token
        except Exception:
            pass

        raise ValueError("No GitLab token found. Set GITLAB_TOKEN or run 'glab auth login'")

    def _glab(self, *args: str, json_output: bool = False) -> dict[str, Any] | str:
        """Run a glab command."""
        cmd = ["glab"] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"glab command failed: {result.stderr}")

        if json_output:
            return json.loads(result.stdout)
        return result.stdout

    def _api_get(self, endpoint: str) -> dict[str, Any] | list[dict]:
        """Make authenticated GET request to GitLab API."""
        result = subprocess.run(["glab", "api", endpoint], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"API request failed: {result.stderr}")
        return json.loads(result.stdout)

    def _api_post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated POST request to GitLab API."""
        # Build field arguments
        field_args = []
        for key, value in data.items():
            field_args.extend(["-f", f"{key}={value}"])

        result = subprocess.run(
            ["glab", "api", endpoint, "--method", "POST"] + field_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"API request failed: {result.stderr}")
        return json.loads(result.stdout)

    def _api_put(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated PUT request to GitLab API."""
        field_args = []
        for key, value in data.items():
            field_args.extend(["-f", f"{key}={value}"])

        result = subprocess.run(
            ["glab", "api", endpoint, "--method", "PUT"] + field_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"API request failed: {result.stderr}")
        return json.loads(result.stdout)

    # =========================================================================
    # ITERATION OPERATIONS
    # =========================================================================

    def list_iterations(self, state: str = "opened") -> list[Iteration]:
        """List group iterations."""
        data = self._api_get(f"groups/{self.config.group_path}/iterations?state={state}")

        iterations = []
        for item in data:
            iteration = Iteration(
                id=item["id"],
                title=item.get("title"),
                state=item.get("state", "opened"),
                start_date=item.get("start_date"),
                due_date=item.get("due_date"),
                group_id=item.get("group", {}).get("id"),
            )
            self.db.upsert_iteration(iteration)
            iterations.append(iteration)

        return iterations

    def find_iteration(self, search: str) -> Iteration | None:
        """Find iteration by name/title search."""
        data = self._api_get(
            f"groups/{self.config.group_path}/iterations?state=opened&search={search}"
        )
        if data:
            item = data[0]
            return Iteration(
                id=item["id"],
                title=item.get("title"),
                state=item.get("state", "opened"),
                start_date=item.get("start_date"),
                due_date=item.get("due_date"),
                group_id=item.get("group", {}).get("id"),
            )
        return None

    def get_current_iteration_by_date(self) -> Iteration | None:
        """
        Get iteration that contains today's date (not just first opened).

        This method finds the iteration whose date range includes today,
        which is more accurate than just returning the first opened iteration.

        Returns:
            Iteration that contains today's date, or None if no matching iteration found
        """
        from datetime import date

        today = date.today().isoformat()
        iterations = self.list_iterations(state="opened")

        for iteration in iterations:
            if iteration.start_date and iteration.due_date:
                if iteration.start_date <= today <= iteration.due_date:
                    return iteration

        return None

    def get_current_iteration(self) -> Iteration | None:
        """
        Get the current active iteration.

        Uses date-based selection first, falling back to first opened iteration
        if no iteration contains today's date.

        Returns:
            Current iteration or None if no iterations available
        """
        # First try to find iteration by date
        by_date = self.get_current_iteration_by_date()
        if by_date:
            return by_date

        # Fallback: return first opened iteration
        iterations = self.list_iterations(state="opened")
        return iterations[0] if iterations else None

    def ensure_iteration_exists(self, cadence_name: str | None = None) -> Iteration | None:
        """
        Ensure an iteration exists for current period. Warn if none found.

        This method checks if there's an active iteration for today's date.
        If not found, it logs a warning to alert the user.

        Args:
            cadence_name: Optional name of iteration cadence to search for

        Returns:
            Current iteration if found, None otherwise (with warning logged)
        """
        iteration = self.get_current_iteration_by_date()

        if iteration:
            logger.info(
                f"Found current iteration: {iteration.title} ({iteration.start_date} - {iteration.due_date})"
            )
            return iteration

        # No iteration for today's date found
        if cadence_name:
            # Try to find by cadence name
            iteration = self.find_iteration(cadence_name)
            if iteration:
                logger.info(f"Found iteration by cadence '{cadence_name}': {iteration.title}")
                return iteration
            logger.warning(f"No iteration found for cadence '{cadence_name}'")
        else:
            logger.warning(
                "No iteration found for current date. "
                "Consider creating an iteration in GitLab for the current sprint."
            )

        # Fall back to any open iteration
        fallback = self.list_iterations(state="opened")
        if fallback:
            logger.info(f"Using fallback iteration: {fallback[0].title}")
            return fallback[0]

        logger.warning("No open iterations available in GitLab")
        return None

    # =========================================================================
    # ISSUE OPERATIONS
    # =========================================================================

    def create_issue(
        self,
        role_name: str,
        title: str,
        description: str,
        labels: list[str] | None = None,
        iteration_id: int | None = None,
        weight: int | None = None,
    ) -> Issue:
        """Create a GitLab issue for a role."""
        role = self.db.get_role(role_name)
        if not role:
            raise ValueError(f"Role '{role_name}' not found in database")

        # Use glab issue create
        cmd = [
            "glab",
            "issue",
            "create",
            "--title",
            title,
            "--description",
            description,
            "--label",
            ",".join(labels or self.config.default_labels),
            "--assignee",
            self.config.default_assignee,
            "--repo",
            self.config.project_path,
            "--yes",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Issue creation failed: {result.stderr}")

        # Parse issue URL from output
        import re

        url_match = re.search(r"https://[^\s]+", result.stdout)
        if not url_match:
            raise RuntimeError("Could not parse issue URL from glab output")

        issue_url = url_match.group(0)
        iid_match = re.search(r"/(\d+)$", issue_url)
        issue_iid = int(iid_match.group(1)) if iid_match else 0

        # Get full issue data
        issue_data = self._api_get(
            f"projects/{self.config.project_path_encoded}/issues/{issue_iid}"
        )

        issue = Issue(
            id=issue_data["id"],
            iid=issue_data["iid"],
            role_id=role.id,
            title=issue_data["title"],
            state=issue_data["state"],
            web_url=issue_data["web_url"],
            labels=json.dumps(issue_data.get("labels", [])),
            assignee=issue_data.get("assignees", [{}])[0].get("username")
            if issue_data.get("assignees")
            else None,
            weight=issue_data.get("weight"),
        )

        # Build update data for iteration and weight (glab doesn't support these)
        update_data = {}
        if iteration_id:
            update_data["iteration_id"] = iteration_id
        if weight is not None:
            update_data["weight"] = weight

        # Apply updates via API if needed
        if update_data:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}", update_data
            )
            if iteration_id:
                issue.iteration_id = iteration_id
            if weight is not None:
                issue.weight = weight

        self.db.upsert_issue(issue)
        return issue

    def get_issue(self, iid: int) -> Issue | None:
        """Get issue by IID."""
        try:
            data = self._api_get(f"projects/{self.config.project_path_encoded}/issues/{iid}")
            return Issue(
                id=data["id"],
                iid=data["iid"],
                title=data["title"],
                state=data["state"],
                web_url=data["web_url"],
                labels=json.dumps(data.get("labels", [])),
                iteration_id=data.get("iteration", {}).get("id") if data.get("iteration") else None,
            )
        except Exception:
            return None

    def assign_issue_to_iteration(self, issue_iid: int, iteration_id: int) -> bool:
        """Assign an existing issue to an iteration."""
        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"iteration_id": iteration_id},
            )
            return True
        except Exception:
            return False

    # =========================================================================
    # ISSUE LIFECYCLE OPERATIONS
    # =========================================================================

    def close_issue(self, issue_iid: int) -> bool:
        """
        Close an issue via state_event.

        Args:
            issue_iid: Project-scoped issue IID

        Returns:
            True if successfully closed, False otherwise
        """
        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"state_event": "close"},
            )
            logger.info(f"Closed issue #{issue_iid}")
            return True
        except Exception as e:
            logger.error(f"Failed to close issue #{issue_iid}: {e}")
            return False

    def reopen_issue(self, issue_iid: int) -> bool:
        """
        Reopen a closed issue.

        Args:
            issue_iid: Project-scoped issue IID

        Returns:
            True if successfully reopened, False otherwise
        """
        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"state_event": "reopen"},
            )
            logger.info(f"Reopened issue #{issue_iid}")
            return True
        except Exception as e:
            logger.error(f"Failed to reopen issue #{issue_iid}: {e}")
            return False

    def add_issue_comment(self, issue_iid: int, body: str) -> bool:
        """
        Add a comment (note) to an issue.

        Args:
            issue_iid: Project-scoped issue IID
            body: Comment text (supports markdown)

        Returns:
            True if comment was added successfully, False otherwise
        """
        try:
            self._api_post(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}/notes",
                {"body": body},
            )
            logger.debug(f"Added comment to issue #{issue_iid}")
            return True
        except Exception as e:
            logger.error(f"Failed to add comment to issue #{issue_iid}: {e}")
            return False

    def update_issue_labels(self, issue_iid: int, labels: list[str]) -> bool:
        """
        Update issue labels (replaces all existing labels).

        Args:
            issue_iid: Project-scoped issue IID
            labels: List of label names to set

        Returns:
            True if labels were updated successfully, False otherwise
        """
        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"labels": ",".join(labels)},
            )
            logger.debug(f"Updated labels on issue #{issue_iid}: {labels}")
            return True
        except Exception as e:
            logger.error(f"Failed to update labels on issue #{issue_iid}: {e}")
            return False

    def add_issue_label(self, issue_iid: int, label: str) -> bool:
        """
        Add a single label to an issue (preserves existing labels).

        Args:
            issue_iid: Project-scoped issue IID
            label: Label name to add

        Returns:
            True if label was added successfully, False otherwise
        """
        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"add_labels": label},
            )
            logger.debug(f"Added label '{label}' to issue #{issue_iid}")
            return True
        except Exception as e:
            logger.error(f"Failed to add label '{label}' to issue #{issue_iid}: {e}")
            return False

    def remove_issue_label(self, issue_iid: int, label: str) -> bool:
        """
        Remove a single label from an issue.

        Args:
            issue_iid: Project-scoped issue IID
            label: Label name to remove

        Returns:
            True if label was removed successfully, False otherwise
        """
        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"remove_labels": label},
            )
            logger.debug(f"Removed label '{label}' from issue #{issue_iid}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove label '{label}' from issue #{issue_iid}: {e}")
            return False

    def update_issue_on_failure(self, issue_iid: int, error: str) -> bool:
        """
        Update issue with failure info: add status::blocked label, add comment.

        This method is called when a workflow fails and an associated issue
        exists. It:
        1. Adds the 'status::blocked' label to indicate the issue is blocked
        2. Adds a comment with the error details for debugging

        Args:
            issue_iid: Project-scoped issue IID
            error: Error message describing the failure

        Returns:
            True if both operations succeeded, False if either failed
        """
        # Ensure the status::blocked label exists
        self.ensure_label_exists(
            "status::blocked",
            color=SCOPED_LABEL_COLORS.get("status", DEFAULT_LABEL_COLOR),
            description="Issue is blocked due to workflow failure",
        )

        # Add the blocked label
        label_success = self.add_issue_label(issue_iid, "status::blocked")

        # Add a comment with failure details
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        comment = f"""## Workflow Failure

**Timestamp:** {timestamp}

**Error:**
```
{error}
```

This issue has been marked as `status::blocked` due to a workflow failure.
Please investigate the error and resolve the blocking issue before continuing.
"""
        comment_success = self.add_issue_comment(issue_iid, comment)

        if label_success and comment_success:
            logger.info(f"Updated issue #{issue_iid} with failure information")
        else:
            logger.warning(
                f"Partial failure updating issue #{issue_iid}: label={label_success}, comment={comment_success}"
            )

        return label_success and comment_success

    def set_issue_due_date(self, issue_iid: int, due_date: str) -> bool:
        """
        Set issue due date.

        Args:
            issue_iid: Project-scoped issue IID
            due_date: Due date in YYYY-MM-DD format

        Returns:
            True if due date was set successfully, False otherwise

        Raises:
            ValueError: If due_date format is invalid
        """
        # Validate date format
        from datetime import datetime

        try:
            datetime.strptime(due_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format '{due_date}'. Expected YYYY-MM-DD.")

        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"due_date": due_date},
            )
            logger.info(f"Set due date on issue #{issue_iid}: {due_date}")
            return True
        except Exception as e:
            logger.error(f"Failed to set due date on issue #{issue_iid}: {e}")
            return False

    def add_time_estimate(self, issue_iid: int, duration: str) -> bool:
        """
        Add time estimate to an issue.

        Args:
            issue_iid: Project-scoped issue IID
            duration: Time duration string (e.g., '3h', '2d', '1w', '1mo')
                     Supported units: mo (months), w (weeks), d (days), h (hours), m (minutes)

        Returns:
            True if time estimate was set successfully, False otherwise

        Example:
            >>> client.add_time_estimate(123, "3h")  # 3 hours
            >>> client.add_time_estimate(123, "2d")  # 2 days
            >>> client.add_time_estimate(123, "1w")  # 1 week
        """
        try:
            self._api_post(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}/time_estimate",
                {"duration": duration},
            )
            logger.info(f"Set time estimate on issue #{issue_iid}: {duration}")
            return True
        except Exception as e:
            logger.error(f"Failed to set time estimate on issue #{issue_iid}: {e}")
            return False

    def reset_time_estimate(self, issue_iid: int) -> bool:
        """
        Reset (clear) the time estimate on an issue.

        Args:
            issue_iid: Project-scoped issue IID

        Returns:
            True if time estimate was reset successfully, False otherwise
        """
        try:
            self._api_post(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}/reset_time_estimate",
                {},
            )
            logger.info(f"Reset time estimate on issue #{issue_iid}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset time estimate on issue #{issue_iid}: {e}")
            return False

    def add_time_spent(self, issue_iid: int, duration: str, summary: str | None = None) -> bool:
        """
        Add time spent on an issue.

        Args:
            issue_iid: Project-scoped issue IID
            duration: Time duration string (e.g., '3h', '2d', '1w')
            summary: Optional summary of work done

        Returns:
            True if time spent was added successfully, False otherwise
        """
        try:
            data = {"duration": duration}
            if summary:
                data["summary"] = summary

            self._api_post(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}/add_spent_time",
                data,
            )
            logger.info(f"Added time spent on issue #{issue_iid}: {duration}")
            return True
        except Exception as e:
            logger.error(f"Failed to add time spent on issue #{issue_iid}: {e}")
            return False

    # =========================================================================
    # MERGE REQUEST OPERATIONS
    # =========================================================================

    def create_merge_request(
        self,
        role_name: str,
        source_branch: str,
        title: str,
        description: str,
        issue_iid: int | None = None,
        draft: bool = False,
    ) -> MergeRequest:
        """Create a merge request."""
        role = self.db.get_role(role_name)
        if not role:
            raise ValueError(f"Role '{role_name}' not found in database")

        # Build glab command
        cmd = [
            "glab",
            "mr",
            "create",
            "--title",
            f"{'Draft: ' if draft else ''}{title}",
            "--description",
            description,
            "--source-branch",
            source_branch,
            "--target-branch",
            "main",
            "--assignee",
            self.config.default_assignee,
            "--label",
            ",".join(self.config.default_labels),
            "--squash-before-merge",
            "--remove-source-branch",
            "--repo",
            self.config.project_path,
            "--yes",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MR creation failed: {result.stderr}")

        # Parse MR URL from output
        import re

        url_match = re.search(r"https://[^\s]+", result.stdout)
        if not url_match:
            raise RuntimeError("Could not parse MR URL from glab output")

        mr_url = url_match.group(0)
        iid_match = re.search(r"/(\d+)$", mr_url)
        mr_iid = int(iid_match.group(1)) if iid_match else 0

        # Get full MR data
        mr_data = self._api_get(
            f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}"
        )

        # Get issue ID if we have an iid
        issue_id = None
        if issue_iid:
            issue = self.db.get_issue(role_name)
            if issue:
                issue_id = issue.id

        mr = MergeRequest(
            id=mr_data["id"],
            iid=mr_data["iid"],
            role_id=role.id,
            issue_id=issue_id,
            source_branch=mr_data["source_branch"],
            target_branch=mr_data["target_branch"],
            title=mr_data["title"],
            state=mr_data["state"],
            web_url=mr_data["web_url"],
            merge_status=mr_data.get("merge_status"),
            squash_on_merge=mr_data.get("squash_on_merge", True),
            remove_source_branch=mr_data.get("force_remove_source_branch", True),
        )

        self.db.upsert_merge_request(mr)
        return mr

    def get_merge_request(self, iid: int) -> MergeRequest | None:
        """Get merge request by IID."""
        try:
            data = self._api_get(
                f"projects/{self.config.project_path_encoded}/merge_requests/{iid}"
            )
            return MergeRequest(
                id=data["id"],
                iid=data["iid"],
                source_branch=data["source_branch"],
                target_branch=data["target_branch"],
                title=data["title"],
                state=data["state"],
                web_url=data["web_url"],
                merge_status=data.get("merge_status"),
            )
        except Exception:
            return None

    # =========================================================================
    # MR LIFECYCLE OPERATIONS
    # =========================================================================

    def get_user_id(self, username: str) -> int | None:
        """
        Get GitLab user ID from username.

        Args:
            username: GitLab username (e.g., 'jsullivan2')

        Returns:
            User ID or None if not found
        """
        try:
            users = self._api_get(f"users?username={username}")
            if users and len(users) > 0:
                return users[0].get("id")
            return None
        except Exception:
            return None

    def set_mr_reviewers(self, mr_iid: int, usernames: list[str]) -> bool:
        """
        Set reviewers for a merge request. Resolves usernames to IDs.

        Args:
            mr_iid: Project-scoped MR IID
            usernames: List of GitLab usernames to set as reviewers

        Returns:
            True if reviewers were set successfully, False otherwise

        Note:
            Usernames that cannot be resolved to IDs are skipped with a warning.
        """
        if not usernames:
            logger.debug("No reviewers to set")
            return True

        # Resolve usernames to user IDs
        reviewer_ids = []
        for username in usernames:
            user_id = self.get_user_id(username)
            if user_id:
                reviewer_ids.append(user_id)
            else:
                logger.warning(f"Could not resolve reviewer username: {username}")

        if not reviewer_ids:
            logger.warning("No valid reviewer IDs found")
            return False

        try:
            # Use PUT to update the MR with reviewer IDs
            # GitLab API expects reviewer_ids as comma-separated or array
            self._api_put(
                f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}",
                {"reviewer_ids": ",".join(str(rid) for rid in reviewer_ids)},
            )
            logger.info(f"Set reviewers for MR !{mr_iid}: {usernames}")
            return True
        except RuntimeError as e:
            logger.error(f"Failed to set reviewers for MR !{mr_iid}: {e}")
            return False

    def get_mr_pipeline_status(self, mr_iid: int) -> str | None:
        """
        Get the latest pipeline status for an MR.

        Args:
            mr_iid: Project-scoped MR IID

        Returns:
            Pipeline status string (e.g., 'pending', 'running', 'success', 'failed',
            'canceled', 'skipped') or None if no pipeline exists
        """
        try:
            pipelines = self._api_get(
                f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}/pipelines"
            )
            if pipelines and len(pipelines) > 0:
                # Return the most recent pipeline status (first in list)
                return pipelines[0].get("status")
            return None
        except Exception as e:
            logger.error(f"Failed to get pipeline status for MR !{mr_iid}: {e}")
            return None

    def wait_for_mr_pipeline(
        self, mr_iid: int, timeout_seconds: int = 600, poll_interval: int = 30
    ) -> str | None:
        """
        Wait for MR pipeline to complete. Returns final status or None on timeout.

        Args:
            mr_iid: Project-scoped MR IID
            timeout_seconds: Maximum time to wait (default: 600 = 10 minutes)
            poll_interval: Time between status checks (default: 30 seconds)

        Returns:
            Final pipeline status ('success', 'failed', 'canceled', etc.)
            or None if timeout is reached

        Note:
            Terminal statuses are: 'success', 'failed', 'canceled', 'skipped'
        """
        terminal_statuses = {"success", "failed", "canceled", "skipped"}
        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            status = self.get_mr_pipeline_status(mr_iid)

            if status is None:
                logger.debug(f"No pipeline found for MR !{mr_iid}, waiting...")
            elif status in terminal_statuses:
                logger.info(f"Pipeline for MR !{mr_iid} completed with status: {status}")
                return status
            else:
                logger.debug(f"Pipeline for MR !{mr_iid} status: {status}, waiting...")

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for pipeline on MR !{mr_iid}")
        return None

    def add_mr_comment(self, mr_iid: int, body: str) -> bool:
        """
        Add a comment to a merge request.

        Args:
            mr_iid: Project-scoped MR IID
            body: Comment body text (supports Markdown)

        Returns:
            True if comment was added successfully, False otherwise
        """
        try:
            self._api_post(
                f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}/notes",
                {"body": body},
            )
            logger.info(f"Added comment to MR !{mr_iid}")
            return True
        except RuntimeError as e:
            logger.error(f"Failed to add comment to MR !{mr_iid}: {e}")
            return False

    def mark_mr_ready(self, mr_iid: int) -> bool:
        """
        Mark a draft MR as ready for review (remove 'Draft:' prefix).

        Uses the GitLab API to update the MR title by removing the 'Draft:' prefix.

        Args:
            mr_iid: Project-scoped MR IID

        Returns:
            True if MR was marked ready successfully, False otherwise

        Note:
            If the MR is already not a draft, this will return True without changes.
        """
        try:
            # Get current MR data
            mr = self.get_merge_request(mr_iid)
            if not mr:
                logger.error(f"MR !{mr_iid} not found")
                return False

            # Check if it's a draft
            title = mr.title
            if not title.startswith("Draft:") and not title.startswith("WIP:"):
                logger.debug(f"MR !{mr_iid} is already not a draft")
                return True

            # Remove Draft: or WIP: prefix
            if title.startswith("Draft: "):
                new_title = title[7:]
            elif title.startswith("Draft:"):
                new_title = title[6:]
            elif title.startswith("WIP: "):
                new_title = title[5:]
            elif title.startswith("WIP:"):
                new_title = title[4:]
            else:
                new_title = title

            # Update the MR title
            self._api_put(
                f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}",
                {"title": new_title},
            )
            logger.info(f"Marked MR !{mr_iid} as ready for review")
            return True

        except RuntimeError as e:
            logger.error(f"Failed to mark MR !{mr_iid} as ready: {e}")
            return False

    def merge_immediately(
        self, mr_iid: int, skip_merge_train: bool = True, squash: bool = True
    ) -> dict[str, Any]:
        """
        Merge MR immediately, optionally skipping merge train.

        This performs an immediate merge rather than adding to the merge train.
        Useful for urgent fixes or when merge train is not desired.

        Args:
            mr_iid: Project-scoped MR IID
            skip_merge_train: If True, bypass merge train (default: True)
            squash: Squash commits on merge (default: True)

        Returns:
            Merge response data from GitLab API

        Raises:
            RuntimeError: If merge fails (e.g., conflicts, CI not passed)
        """
        data = {
            "squash": str(squash).lower(),
        }

        # skip_merge_train is a GitLab Premium feature
        if skip_merge_train:
            data["skip_merge_train"] = "true"

        try:
            result = self._api_put(
                f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}/merge", data
            )
            logger.info(f"Merged MR !{mr_iid} immediately")
            return result
        except RuntimeError as e:
            logger.error(f"Failed to merge MR !{mr_iid}: {e}")
            raise

    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================

    def sync_issues(self, role_name: str | None = None) -> int:
        """Sync issues from GitLab to database."""
        labels = "role"
        if role_name:
            labels = f"role,{role_name}"

        issues = self._api_get(
            f"projects/{self.config.project_path_encoded}/issues?labels={labels}&state=all"
        )

        synced = 0
        for data in issues:
            # Try to find associated role
            role_id = None
            if role_name:
                role = self.db.get_role(role_name)
                role_id = role.id if role else None
            else:
                # Try to extract role from title
                import re

                match = re.search(r"`(\w+)`", data["title"])
                if match:
                    role = self.db.get_role(match.group(1))
                    role_id = role.id if role else None

            issue = Issue(
                id=data["id"],
                iid=data["iid"],
                role_id=role_id,
                iteration_id=data.get("iteration", {}).get("id") if data.get("iteration") else None,
                title=data["title"],
                state=data["state"],
                web_url=data["web_url"],
                labels=json.dumps(data.get("labels", [])),
                weight=data.get("weight"),
            )
            self.db.upsert_issue(issue)
            synced += 1

        return synced

    def sync_merge_requests(self, role_name: str | None = None) -> int:
        """Sync merge requests from GitLab to database."""
        mrs = self._api_get(f"projects/{self.config.project_path_encoded}/merge_requests?state=all")

        synced = 0
        for data in mrs:
            # Filter by source branch if role_name provided
            source_branch = data["source_branch"]
            if role_name and source_branch != f"sid/{role_name}":
                continue

            # Try to find associated role
            role_id = None
            if source_branch.startswith("sid/"):
                extracted_role = source_branch.replace("sid/", "")
                role = self.db.get_role(extracted_role)
                role_id = role.id if role else None

            mr = MergeRequest(
                id=data["id"],
                iid=data["iid"],
                role_id=role_id,
                source_branch=data["source_branch"],
                target_branch=data["target_branch"],
                title=data["title"],
                state=data["state"],
                web_url=data["web_url"],
                merge_status=data.get("merge_status"),
                squash_on_merge=data.get("squash_on_merge", True),
                remove_source_branch=data.get("force_remove_source_branch", True),
            )
            self.db.upsert_merge_request(mr)
            synced += 1

        return synced

    # =========================================================================
    # MERGE TRAIN OPERATIONS
    # =========================================================================

    def add_to_merge_train(
        self, mr_iid: int, when_pipeline_succeeds: bool = True, squash: bool = True
    ) -> dict[str, Any]:
        """
        Add a merge request to the merge train using the dedicated merge train API.

        This uses the GitLab Premium merge train API endpoint:
        POST /projects/:id/merge_trains

        Args:
            mr_iid: Project-scoped MR IID
            when_pipeline_succeeds: Wait for pipeline to succeed before merging
            squash: Squash commits on merge

        Returns:
            Merge train entry data from GitLab API

        Raises:
            RuntimeError: If the MR cannot be added to the merge train

        Note:
            If merge trains are not enabled on the project, this will fall back
            to the standard merge endpoint with --when-pipeline-succeeds.
        """
        # Try the dedicated merge train API first (GitLab Premium)
        try:
            data = {"merge_request_iid": mr_iid}
            if when_pipeline_succeeds:
                data["when_pipeline_succeeds"] = "true"
            if squash:
                data["squash"] = "true"

            return self._api_post(f"projects/{self.config.project_path_encoded}/merge_trains", data)
        except RuntimeError as e:
            # Merge trains might not be enabled - fall back to standard merge
            if "merge_trains" in str(e).lower() or "not found" in str(e).lower():
                return self._merge_when_pipeline_succeeds(mr_iid, squash)
            raise

    def _merge_when_pipeline_succeeds(self, mr_iid: int, squash: bool = True) -> dict[str, Any]:
        """
        Internal: Merge when pipeline succeeds (for projects without merge trains).

        Uses: PUT /projects/:id/merge_requests/:iid/merge

        Args:
            mr_iid: Project-scoped MR IID
            squash: Squash commits on merge

        Returns:
            Merge response data
        """
        data = {
            "merge_when_pipeline_succeeds": "true",
            "squash": str(squash).lower(),
        }
        return self._api_put(
            f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}/merge", data
        )

    def merge_when_pipeline_succeeds(self, mr_iid: int, squash: bool = True) -> dict[str, Any]:
        """
        Merge MR when pipeline succeeds (fallback when merge train not enabled).

        This is the standard GitLab merge behavior without merge trains.
        The MR will be merged automatically once its pipeline passes.

        Uses: PUT /projects/:id/merge_requests/:iid/merge

        Args:
            mr_iid: Project-scoped MR IID
            squash: Squash commits on merge (default: True)

        Returns:
            Merge response data from GitLab API

        Raises:
            RuntimeError: If the merge request cannot be set to merge

        Example:
            >>> client.merge_when_pipeline_succeeds(123)
            {'id': 456, 'iid': 123, 'state': 'merged', ...}
        """
        return self._merge_when_pipeline_succeeds(mr_iid, squash)

    def get_merge_train(self, target_branch: str = "main") -> list[dict[str, Any]]:
        """
        Get the merge train queue for a target branch.

        Uses: GET /projects/:id/merge_trains

        Args:
            target_branch: Target branch of the merge train (default: main)

        Returns:
            List of merge train entries with MR details
        """
        return self._api_get(
            f"projects/{self.config.project_path_encoded}/merge_trains?target_branch={target_branch}"
        )

    def get_merge_train_status(self, mr_iid: int) -> dict[str, Any] | None:
        """
        Get merge train status for a specific MR.

        Args:
            mr_iid: Project-scoped MR IID

        Returns:
            Merge train entry data or None if not in train
        """
        try:
            # Get all trains and filter by MR
            trains = self.get_merge_train()
            for train in trains:
                if train.get("merge_request", {}).get("iid") == mr_iid:
                    return train
            return None
        except Exception:
            return None

    def remove_from_merge_train(self, train_id: int) -> bool:
        """
        Remove a merge request from the merge train by train entry ID.

        Uses: DELETE /projects/:id/merge_trains/:train_id

        Args:
            train_id: Merge train entry ID (not MR IID)

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            result = subprocess.run(
                [
                    "glab",
                    "api",
                    f"projects/{self.config.project_path_encoded}/merge_trains/{train_id}",
                    "--method",
                    "DELETE",
                ],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def remove_mr_from_merge_train(self, mr_iid: int) -> bool:
        """
        Remove a merge request from the merge train by MR IID.

        Args:
            mr_iid: Project-scoped MR IID

        Returns:
            True if successfully removed, False otherwise
        """
        train_status = self.get_merge_train_status(mr_iid)
        if train_status and "id" in train_status:
            return self.remove_from_merge_train(train_status["id"])
        return False

    def abort_merge_train(self, mr_iid: int) -> bool:
        """
        Cancel merge for an MR (abort from merge train or cancel merge when pipeline succeeds).

        Args:
            mr_iid: Project-scoped MR IID

        Returns:
            True if successfully cancelled, False otherwise
        """
        # First try to remove from merge train
        if self.remove_mr_from_merge_train(mr_iid):
            return True

        # Fall back to cancelling merge when pipeline succeeds
        try:
            self._api_post(
                f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}/cancel_merge_when_pipeline_succeeds",
                {},
            )
            return True
        except Exception:
            return False

    def list_merge_trains(self) -> list[dict[str, Any]]:
        """
        List all active merge trains in the project.

        Returns:
            List of merge trains with their queued MRs
        """
        # Get trains for common target branches
        trains = []
        for branch in ["main", "develop"]:
            try:
                branch_trains = self.get_merge_train(target_branch=branch)
                if branch_trains:
                    trains.extend(branch_trains)
            except Exception:
                continue
        return trains

    def get_merge_train_position(self, mr_iid: int) -> int | None:
        """
        Get the position of an MR in the merge train queue.

        Args:
            mr_iid: Project-scoped MR IID

        Returns:
            Position (1-indexed) or None if not in train
        """
        trains = self.get_merge_train()
        for i, train in enumerate(trains, 1):
            if train.get("merge_request", {}).get("iid") == mr_iid:
                return i
        return None

    def is_merge_train_available(self, mr_iid: int) -> dict[str, Any]:
        """
        Check if an MR can be added to the merge train.

        Args:
            mr_iid: Project-scoped MR IID

        Returns:
            Dict with 'available' boolean and 'reason' if not available
        """
        try:
            mr = self.get_merge_request(mr_iid)
            if not mr:
                return {"available": False, "reason": "MR not found"}

            if mr.state != "opened":
                return {"available": False, "reason": f"MR state is {mr.state}, must be opened"}

            if mr.merge_status in ("cannot_be_merged", "cannot_be_merged_rechecking"):
                return {"available": False, "reason": f"MR has merge conflicts: {mr.merge_status}"}

            # Check if already in merge train
            train_status = self.get_merge_train_status(mr_iid)
            if train_status:
                position = self.get_merge_train_position(mr_iid)
                return {
                    "available": False,
                    "reason": f"MR already in merge train at position {position}",
                }

            return {"available": True, "reason": None}
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def is_merge_train_enabled(self) -> bool:
        """
        Check if merge trains are enabled for this project.

        Returns:
            True if merge trains are enabled, False otherwise
        """
        try:
            # Try to get merge train - if it returns 404 or error, not enabled
            self.get_merge_train()
            return True
        except Exception:
            return False

    # =========================================================================
    # IDEMPOTENCY OPERATIONS
    # =========================================================================

    def find_existing_issue(self, role_name: str, state: str = "all") -> Issue | None:
        """
        Find existing issue for a role by searching title pattern 'Box up `{role_name}`'.

        This enables idempotent issue creation - if an issue already exists for
        the role, we can reuse it instead of creating a duplicate.

        IMPORTANT: Returns the OLDEST (lowest IID) matching issue to ensure
        consistency when duplicates exist. Default state is 'all' to find
        issues in any state.

        Args:
            role_name: Name of the role to find issue for
            state: Issue state filter ('opened', 'closed', 'all')

        Returns:
            Existing Issue if found (oldest by IID), None otherwise
        """
        # Search for issues with role name - more reliable than full title search
        search_term = urllib.parse.quote(role_name)

        try:
            issues = self._api_get(
                f"projects/{self.config.project_path_encoded}/issues"
                f"?state={state}&search={search_term}&in=title&per_page=100"
            )

            if not issues:
                return None

            # Filter for issues that have the role name in backticks and "box up" in title
            matching_issues = []
            for issue_data in issues:
                title = issue_data.get("title", "")
                # Match patterns like "Box up `role_name`" or "feat(role_name): Box up `role_name`"
                if f"`{role_name}`" in title and "box up" in title.lower():
                    matching_issues.append(issue_data)

            if not matching_issues:
                return None

            # Return the OLDEST (lowest IID) as canonical to ensure consistency
            canonical_data = min(matching_issues, key=lambda i: i.get("iid", float("inf")))

            if len(matching_issues) > 1:
                logger.warning(
                    f"Found {len(matching_issues)} issues for role '{role_name}', "
                    f"using canonical #{canonical_data.get('iid')}"
                )

            # Get role_id if available
            role = self.db.get_role(role_name)
            role_id = role.id if role else None

            issue = Issue(
                id=canonical_data["id"],
                iid=canonical_data["iid"],
                role_id=role_id,
                iteration_id=canonical_data.get("iteration", {}).get("id")
                if canonical_data.get("iteration")
                else None,
                title=canonical_data["title"],
                state=canonical_data["state"],
                web_url=canonical_data["web_url"],
                labels=json.dumps(canonical_data.get("labels", [])),
                assignee=canonical_data.get("assignees", [{}])[0].get("username")
                if canonical_data.get("assignees")
                else None,
                weight=canonical_data.get("weight"),
            )

            # Update local database
            self.db.upsert_issue(issue)
            return issue

        except Exception:
            return None

    def find_existing_mr(self, source_branch: str, state: str = "opened") -> MergeRequest | None:
        """
        Find existing MR for a source branch.

        This enables idempotent MR creation - if an MR already exists for
        the branch, we can reuse it instead of creating a duplicate.

        Args:
            source_branch: Source branch name (e.g., 'sid/common')
            state: MR state filter ('opened', 'closed', 'merged', 'all')

        Returns:
            Existing MergeRequest if found, None otherwise
        """
        try:
            mrs = self._api_get(
                f"projects/{self.config.project_path_encoded}/merge_requests"
                f"?state={state}&source_branch={source_branch}"
            )

            if not mrs:
                return None

            # Return the first (most recent) MR for this branch
            mr_data = mrs[0]

            # Try to find associated role
            role_id = None
            if source_branch.startswith("sid/"):
                extracted_role = source_branch.replace("sid/", "")
                role = self.db.get_role(extracted_role)
                role_id = role.id if role else None

            mr = MergeRequest(
                id=mr_data["id"],
                iid=mr_data["iid"],
                role_id=role_id,
                source_branch=mr_data["source_branch"],
                target_branch=mr_data["target_branch"],
                title=mr_data["title"],
                state=mr_data["state"],
                web_url=mr_data["web_url"],
                merge_status=mr_data.get("merge_status"),
                squash_on_merge=mr_data.get("squash_on_merge", True),
                remove_source_branch=mr_data.get("force_remove_source_branch", True),
            )

            # Update local database
            self.db.upsert_merge_request(mr)
            return mr

        except Exception:
            return None

    def get_or_create_issue(
        self,
        role_name: str,
        title: str,
        description: str,
        labels: list[str] | None = None,
        iteration_id: int | None = None,
        weight: int | None = None,
    ) -> tuple[Issue, bool]:
        """
        Get existing issue or create new one. Returns (Issue, created: bool).

        This is the primary idempotent issue creation method. It first checks
        if an issue already exists for the role, and only creates a new one
        if none is found.

        Args:
            role_name: Name of the role
            title: Issue title (used for creation)
            description: Issue description (used for creation)
            labels: Issue labels (used for creation)
            iteration_id: Iteration to assign (used for creation)
            weight: Issue weight (used for creation)

        Returns:
            Tuple of (Issue, created) where created is True if a new issue
            was created, False if an existing issue was found.

        Raises:
            ValueError: If role is not found in database
            RuntimeError: If issue creation fails
        """
        # First, check for existing issue
        existing = self.find_existing_issue(role_name, state="all")
        if existing:
            # If issue exists but is closed, we might want to reopen or create new
            # For now, return the existing issue regardless of state
            return (existing, False)

        # No existing issue found, create new one
        issue = self.create_issue(
            role_name=role_name,
            title=title,
            description=description,
            labels=labels,
            iteration_id=iteration_id,
            weight=weight,
        )
        return (issue, True)

    def get_or_create_mr(
        self,
        role_name: str,
        source_branch: str,
        title: str,
        description: str,
        issue_iid: int | None = None,
        draft: bool = False,
    ) -> tuple[MergeRequest, bool]:
        """
        Get existing MR or create new one. Returns (MergeRequest, created: bool).

        This is the primary idempotent MR creation method. It first checks
        if an MR already exists for the source branch, and only creates a new
        one if none is found.

        Args:
            role_name: Name of the role
            source_branch: Source branch name
            title: MR title (used for creation)
            description: MR description (used for creation)
            issue_iid: Related issue IID (used for creation)
            draft: Whether to create as draft (used for creation)

        Returns:
            Tuple of (MergeRequest, created) where created is True if a new MR
            was created, False if an existing MR was found.

        Raises:
            ValueError: If role is not found in database
            RuntimeError: If MR creation fails
        """
        # First, check for existing MR (opened or merged)
        existing = self.find_existing_mr(source_branch, state="opened")
        if existing:
            return (existing, False)

        # Also check for merged MRs - if already merged, return it
        merged = self.find_existing_mr(source_branch, state="merged")
        if merged:
            return (merged, False)

        # No existing MR found, create new one
        mr = self.create_merge_request(
            role_name=role_name,
            source_branch=source_branch,
            title=title,
            description=description,
            issue_iid=issue_iid,
            draft=draft,
        )
        return (mr, True)

    def remote_branch_exists(self, branch_name: str) -> bool:
        """
        Check if branch exists on origin using git ls-remote.

        This is useful for checking if a branch has been pushed before
        attempting to create an MR for it.

        Args:
            branch_name: Branch name to check (e.g., 'sid/common')

        Returns:
            True if branch exists on remote, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "ls-remote", "--heads", "origin", branch_name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # If output contains the branch name, it exists
            return branch_name in result.stdout

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    # =========================================================================
    # LABEL OPERATIONS
    # =========================================================================

    def get_label(self, name: str) -> dict[str, Any] | None:
        """
        Get a label by name.

        Args:
            name: Label name to look up

        Returns:
            Label data dict or None if not found
        """
        try:
            # URL-encode the label name for the API call
            encoded_name = urllib.parse.quote(name, safe="")
            data = self._api_get(
                f"projects/{self.config.project_path_encoded}/labels/{encoded_name}"
            )
            return data
        except RuntimeError as e:
            # Label not found returns 404
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            raise

    def create_label(
        self, name: str, color: str = DEFAULT_LABEL_COLOR, description: str = ""
    ) -> dict[str, Any]:
        """
        Create a new project label.

        Args:
            name: Label name
            color: Hex color code (e.g., "#FF0000")
            description: Optional label description

        Returns:
            Created label data from GitLab API

        Raises:
            RuntimeError: If label creation fails
        """
        data = {
            "name": name,
            "color": color,
        }
        if description:
            data["description"] = description

        return self._api_post(f"projects/{self.config.project_path_encoded}/labels", data)

    def ensure_label_exists(
        self, name: str, color: str = DEFAULT_LABEL_COLOR, description: str = ""
    ) -> bool:
        """
        Ensure a label exists, creating it if necessary.

        This method is idempotent - it can be called multiple times safely.
        If the label already exists, it returns True without modification.

        Args:
            name: Label name
            color: Hex color code (default: #428BCA)
            description: Optional label description

        Returns:
            True if label exists (created or already existed), False on error
        """
        try:
            # Check if label already exists
            existing = self.get_label(name)
            if existing:
                logger.debug(f"Label '{name}' already exists")
                return True

            # Create the label
            self.create_label(name, color, description)
            logger.info(f"Created label '{name}' with color {color}")
            return True

        except RuntimeError as e:
            # Handle race condition: label was created between check and create
            if "already exists" in str(e).lower() or "has already been taken" in str(e).lower():
                logger.debug(f"Label '{name}' was created by another process")
                return True
            logger.error(f"Failed to ensure label '{name}': {e}")
            return False

    def ensure_wave_labels(self) -> dict[str, bool]:
        """
        Ensure all wave labels exist with proper colors.

        Creates wave-0 through wave-4 labels with their designated colors
        and descriptions.

        Returns:
            Dict mapping label name to creation success status
        """
        results = {}
        for label_name, color in WAVE_LABEL_COLORS.items():
            description = WAVE_LABEL_DESCRIPTIONS.get(label_name, "")
            results[label_name] = self.ensure_label_exists(
                name=label_name, color=color, description=description
            )
        return results

    def ensure_scoped_labels(self, scope: str, values: list[str] | None = None) -> dict[str, bool]:
        """
        Ensure scoped labels exist (e.g., priority::high).

        GitLab scoped labels use the :: separator to create mutually
        exclusive label groups.

        Args:
            scope: Label scope/category (e.g., "priority", "status", "type")
            values: List of values for the scope. If None, uses SCOPED_LABELS.

        Returns:
            Dict mapping full label name to creation success status

        Example:
            >>> client.ensure_scoped_labels("priority", ["high", "medium", "low"])
            {'priority::high': True, 'priority::medium': True, 'priority::low': True}
        """
        if values is None:
            values = SCOPED_LABELS.get(scope, [])

        if not values:
            logger.warning(f"No values defined for scope '{scope}'")
            return {}

        # Get color for this scope
        color = SCOPED_LABEL_COLORS.get(scope, DEFAULT_LABEL_COLOR)

        results = {}
        for value in values:
            label_name = f"{scope}::{value}"
            results[label_name] = self.ensure_label_exists(
                name=label_name, color=color, description=f"{scope.title()} label: {value}"
            )
        return results

    def prepare_labels_for_role(
        self, role_name: str, wave: int, additional_labels: list[str] | None = None
    ) -> list[str]:
        """
        Prepare and ensure all required labels exist for a role.

        This method:
        1. Ensures the wave label exists with proper color
        2. Ensures default role labels exist (role, ansible, molecule)
        3. Ensures any additional labels exist
        4. Returns the complete list of labels to apply

        Args:
            role_name: Name of the role (for logging)
            wave: Wave number (0-4) for the wave label
            additional_labels: Optional list of additional labels to include

        Returns:
            List of label names to apply to the issue/MR

        Example:
            >>> labels = client.prepare_labels_for_role("common", 0)
            >>> # Creates: wave-0, role, ansible, molecule
            >>> labels
            ['role', 'ansible', 'molecule', 'wave-0']
        """
        labels_to_apply = []

        # Ensure default labels
        default_labels = ["role", "ansible", "molecule"]
        for label in default_labels:
            if self.ensure_label_exists(label):
                labels_to_apply.append(label)

        # Ensure wave label with proper color
        wave_label = f"wave-{wave}"
        if wave_label in WAVE_LABEL_COLORS:
            if self.ensure_label_exists(
                name=wave_label,
                color=WAVE_LABEL_COLORS[wave_label],
                description=WAVE_LABEL_DESCRIPTIONS.get(wave_label, ""),
            ):
                labels_to_apply.append(wave_label)
        else:
            # Wave out of range, create with default color
            logger.warning(f"Wave {wave} out of predefined range (0-4)")
            if self.ensure_label_exists(wave_label):
                labels_to_apply.append(wave_label)

        # Ensure additional labels if provided
        if additional_labels:
            for label in additional_labels:
                if self.ensure_label_exists(label):
                    if label not in labels_to_apply:
                        labels_to_apply.append(label)

        logger.info(f"Prepared labels for role '{role_name}': {labels_to_apply}")
        return labels_to_apply

    def ensure_all_standard_labels(self) -> dict[str, dict[str, bool]]:
        """
        Ensure all standard labels exist (waves and scoped labels).

        Convenience method to initialize all predefined labels at once.

        Returns:
            Dict with 'waves' and 'scoped' keys, each containing label results

        Example:
            >>> results = client.ensure_all_standard_labels()
            >>> results['waves']
            {'wave-0': True, 'wave-1': True, ...}
            >>> results['scoped']['priority']
            {'priority::high': True, 'priority::medium': True, ...}
        """
        results = {"waves": self.ensure_wave_labels(), "scoped": {}}

        for scope in SCOPED_LABELS.keys():
            results["scoped"][scope] = self.ensure_scoped_labels(scope)

        return results
