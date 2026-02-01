"""
GitLab API client for harness operations.

Handles:
- Iteration management (group-level)
- Issue creation and assignment
- Merge request creation
- Merge train integration
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from harness.db.models import Issue, Iteration, MergeRequest
from harness.db.state import StateDB


@dataclass
class GitLabConfig:
    """GitLab configuration."""
    project_path: str = "bates-ils/projects/ems/ems-mono"
    group_path: str = "bates-ils"
    default_assignee: str = "jsullivan2"
    default_labels: list[str] = None

    def __post_init__(self):
        if self.default_labels is None:
            self.default_labels = ["role", "ansible", "molecule"]

    @property
    def project_path_encoded(self) -> str:
        return self.project_path.replace("/", "%2F")


class GitLabClient:
    """
    GitLab API client using glab CLI and direct API calls.

    Uses glab CLI where possible for authentication, falls back to
    direct API calls for features not supported by glab (e.g., iterations).
    """

    def __init__(self, db: StateDB, config: Optional[GitLabConfig] = None):
        self.db = db
        self.config = config or GitLabConfig()
        self._token: Optional[str] = None

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
                ["glab", "auth", "status", "-t"],
                capture_output=True,
                text=True
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
        result = subprocess.run(
            ["glab", "api", endpoint],
            capture_output=True,
            text=True
        )
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
            text=True
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
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"API request failed: {result.stderr}")
        return json.loads(result.stdout)

    # =========================================================================
    # ITERATION OPERATIONS
    # =========================================================================

    def list_iterations(self, state: str = "opened") -> list[Iteration]:
        """List group iterations."""
        data = self._api_get(
            f"groups/{self.config.group_path}/iterations?state={state}"
        )

        iterations = []
        for item in data:
            iteration = Iteration(
                id=item["id"],
                title=item.get("title"),
                state=item.get("state", "opened"),
                start_date=item.get("start_date"),
                due_date=item.get("due_date"),
                group_id=item.get("group", {}).get("id")
            )
            self.db.upsert_iteration(iteration)
            iterations.append(iteration)

        return iterations

    def find_iteration(self, search: str) -> Optional[Iteration]:
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
                group_id=item.get("group", {}).get("id")
            )
        return None

    def get_current_iteration(self) -> Optional[Iteration]:
        """Get the current active iteration."""
        iterations = self.list_iterations(state="opened")
        # Return first opened iteration (usually current)
        return iterations[0] if iterations else None

    # =========================================================================
    # ISSUE OPERATIONS
    # =========================================================================

    def create_issue(self, role_name: str, title: str, description: str,
                     labels: Optional[list[str]] = None,
                     iteration_id: Optional[int] = None,
                     weight: Optional[int] = None) -> Issue:
        """Create a GitLab issue for a role."""
        role = self.db.get_role(role_name)
        if not role:
            raise ValueError(f"Role '{role_name}' not found in database")

        # Use glab issue create
        cmd = [
            "glab", "issue", "create",
            "--title", title,
            "--description", description,
            "--label", ",".join(labels or self.config.default_labels),
            "--assignee", self.config.default_assignee,
            "--repo", self.config.project_path,
            "--yes"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Issue creation failed: {result.stderr}")

        # Parse issue URL from output
        import re
        url_match = re.search(r'https://[^\s]+', result.stdout)
        if not url_match:
            raise RuntimeError("Could not parse issue URL from glab output")

        issue_url = url_match.group(0)
        iid_match = re.search(r'/(\d+)$', issue_url)
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
            assignee=issue_data.get("assignees", [{}])[0].get("username") if issue_data.get("assignees") else None,
            weight=issue_data.get("weight")
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
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                update_data
            )
            if iteration_id:
                issue.iteration_id = iteration_id
            if weight is not None:
                issue.weight = weight

        self.db.upsert_issue(issue)
        return issue

    def get_issue(self, iid: int) -> Optional[Issue]:
        """Get issue by IID."""
        try:
            data = self._api_get(
                f"projects/{self.config.project_path_encoded}/issues/{iid}"
            )
            return Issue(
                id=data["id"],
                iid=data["iid"],
                title=data["title"],
                state=data["state"],
                web_url=data["web_url"],
                labels=json.dumps(data.get("labels", [])),
                iteration_id=data.get("iteration", {}).get("id") if data.get("iteration") else None
            )
        except Exception:
            return None

    def assign_issue_to_iteration(self, issue_iid: int, iteration_id: int) -> bool:
        """Assign an existing issue to an iteration."""
        try:
            self._api_put(
                f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
                {"iteration_id": iteration_id}
            )
            return True
        except Exception:
            return False

    # =========================================================================
    # MERGE REQUEST OPERATIONS
    # =========================================================================

    def create_merge_request(self, role_name: str, source_branch: str,
                             title: str, description: str,
                             issue_iid: Optional[int] = None,
                             draft: bool = False) -> MergeRequest:
        """Create a merge request."""
        role = self.db.get_role(role_name)
        if not role:
            raise ValueError(f"Role '{role_name}' not found in database")

        # Build glab command
        cmd = [
            "glab", "mr", "create",
            "--title", f"{'Draft: ' if draft else ''}{title}",
            "--description", description,
            "--source-branch", source_branch,
            "--target-branch", "main",
            "--assignee", self.config.default_assignee,
            "--label", ",".join(self.config.default_labels),
            "--squash-before-merge",
            "--remove-source-branch",
            "--repo", self.config.project_path,
            "--yes"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MR creation failed: {result.stderr}")

        # Parse MR URL from output
        import re
        url_match = re.search(r'https://[^\s]+', result.stdout)
        if not url_match:
            raise RuntimeError("Could not parse MR URL from glab output")

        mr_url = url_match.group(0)
        iid_match = re.search(r'/(\d+)$', mr_url)
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
            remove_source_branch=mr_data.get("force_remove_source_branch", True)
        )

        self.db.upsert_merge_request(mr)
        return mr

    def get_merge_request(self, iid: int) -> Optional[MergeRequest]:
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
                merge_status=data.get("merge_status")
            )
        except Exception:
            return None

    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================

    def sync_issues(self, role_name: Optional[str] = None) -> int:
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
                match = re.search(r'`(\w+)`', data["title"])
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
                weight=data.get("weight")
            )
            self.db.upsert_issue(issue)
            synced += 1

        return synced

    def sync_merge_requests(self, role_name: Optional[str] = None) -> int:
        """Sync merge requests from GitLab to database."""
        mrs = self._api_get(
            f"projects/{self.config.project_path_encoded}/merge_requests?state=all"
        )

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
                remove_source_branch=data.get("force_remove_source_branch", True)
            )
            self.db.upsert_merge_request(mr)
            synced += 1

        return synced

    # =========================================================================
    # MERGE TRAIN OPERATIONS
    # =========================================================================

    def add_to_merge_train(self, mr_iid: int, when_pipeline_succeeds: bool = True,
                           squash: bool = True) -> dict[str, Any]:
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

            return self._api_post(
                f"projects/{self.config.project_path_encoded}/merge_trains",
                data
            )
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
            f"projects/{self.config.project_path_encoded}/merge_requests/{mr_iid}/merge",
            data
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

    def get_merge_train_status(self, mr_iid: int) -> Optional[dict[str, Any]]:
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
                ["glab", "api", f"projects/{self.config.project_path_encoded}/merge_trains/{train_id}",
                 "--method", "DELETE"],
                capture_output=True,
                text=True
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
                {}
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

    def get_merge_train_position(self, mr_iid: int) -> Optional[int]:
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
                return {"available": False, "reason": f"MR already in merge train at position {position}"}

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
