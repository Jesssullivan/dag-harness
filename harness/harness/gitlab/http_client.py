"""
Pure HTTP GitLab API client.

No glab CLI dependency. Uses PRIVATE-TOKEN authentication
with httpx for async HTTP requests.

This module provides a direct HTTP interface to the GitLab REST API v4,
eliminating subprocess overhead and glab CLI auth desync issues.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GitLabAPIConfig:
    """GitLab API configuration."""

    base_url: str = "https://gitlab.com/api/v4"
    project_path: str = ""  # e.g., "tinyland/archives/ems"
    token: str = ""  # GITLAB_TOKEN
    timeout: float = 30.0
    default_assignee: str = ""
    default_labels: list[str] = field(default_factory=lambda: ["role", "ansible", "molecule"])

    @property
    def project_encoded(self) -> str:
        """URL-encode the project path for API endpoints."""
        return self.project_path.replace("/", "%2F")

    @classmethod
    def from_harness_yml(cls, repo_root: Path) -> "GitLabAPIConfig":
        """
        Load configuration from harness.yml in the target repo.

        Args:
            repo_root: Path to the repository root containing harness.yml

        Returns:
            GitLabAPIConfig populated from harness.yml and environment
        """
        import yaml

        config_path = repo_root / "harness.yml"
        gitlab_config: dict[str, Any] = {}

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            gitlab_config = config.get("gitlab", {})

        # Token resolution order:
        # 1. Environment variable specified in config
        # 2. GITLAB_TOKEN env var
        # 3. Empty (will fail on first API call)
        token_env_var = gitlab_config.get("token_env_var", "GITLAB_TOKEN")
        token = os.environ.get(token_env_var, "")
        if not token:
            token = os.environ.get("GITLAB_TOKEN", "")

        return cls(
            base_url=gitlab_config.get("base_url", "https://gitlab.com/api/v4"),
            project_path=gitlab_config.get("project_path", ""),
            token=token,
            timeout=gitlab_config.get("timeout", 30.0),
            default_assignee=gitlab_config.get("default_assignee", ""),
            default_labels=gitlab_config.get(
                "default_labels", ["role", "ansible", "molecule"]
            ),
        )


class GitLabAPIError(Exception):
    """Base exception for GitLab API errors."""

    def __init__(self, message: str, status_code: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class GitLabAPI:
    """
    Async GitLab REST API client.

    Uses httpx for HTTP requests with PRIVATE-TOKEN authentication.
    All methods are async and should be used within an async context manager.

    Example:
        async with GitLabAPI(config) as api:
            issue, created = await api.get_or_create_issue(
                search="Test issue",
                title="Test issue",
                description="Testing pure HTTP client",
            )
    """

    def __init__(self, config: GitLabAPIConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GitLabAPI":
        """Enter async context, creating the HTTP client."""
        if not self.config.token:
            raise GitLabAPIError("No GitLab token configured. Set GITLAB_TOKEN environment variable.")

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"PRIVATE-TOKEN": self.config.token},
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context, closing the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _check_client(self) -> httpx.AsyncClient:
        """Ensure client is initialized."""
        if self._client is None:
            raise GitLabAPIError("GitLabAPI must be used as async context manager")
        return self._client

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        """Make authenticated GET request."""
        client = self._check_client()
        resp = await client.get(endpoint, params=params)
        if resp.status_code >= 400:
            raise GitLabAPIError(
                f"GET {endpoint} failed: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
                response=resp.json() if resp.text else None,
            )
        return resp.json()

    async def _post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated POST request."""
        client = self._check_client()
        resp = await client.post(endpoint, json=data)
        if resp.status_code >= 400:
            raise GitLabAPIError(
                f"POST {endpoint} failed: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
                response=resp.json() if resp.text else None,
            )
        return resp.json()

    async def _put(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated PUT request."""
        client = self._check_client()
        resp = await client.put(endpoint, json=data)
        if resp.status_code >= 400:
            raise GitLabAPIError(
                f"PUT {endpoint} failed: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
                response=resp.json() if resp.text else None,
            )
        return resp.json()

    async def _delete(self, endpoint: str) -> bool:
        """Make authenticated DELETE request."""
        client = self._check_client()
        resp = await client.delete(endpoint)
        if resp.status_code >= 400:
            raise GitLabAPIError(
                f"DELETE {endpoint} failed: {resp.status_code} {resp.text}",
                status_code=resp.status_code,
            )
        return True

    # =========================================================================
    # Issue Operations
    # =========================================================================

    async def find_issue(self, search: str, state: str = "opened") -> dict[str, Any] | None:
        """
        Find issue by search term.

        Args:
            search: Search term to match in issue title
            state: Issue state filter ('opened', 'closed', 'all')

        Returns:
            First matching issue dict or None if not found
        """
        issues = await self._get(
            f"/projects/{self.config.project_encoded}/issues",
            params={"search": search, "state": state, "in": "title"},
        )
        return issues[0] if issues else None

    async def find_issue_by_exact_title(
        self, title: str, state: str = "all"
    ) -> dict[str, Any] | None:
        """
        Find issue by EXACT title match.

        GitLab's search API is fuzzy. This method fetches candidates
        and filters for exact title match locally.

        Args:
            title: Exact issue title to find
            state: Issue state filter ('opened', 'closed', 'all')

        Returns:
            Issue dict if exact match found, None otherwise
        """
        import re

        # Extract a unique search term from the title
        # For "feat(common): Box up `common` Ansible role" -> use "common"
        role_match = re.search(r"`(\w+)`", title)
        if not role_match:
            # Fallback to first significant word search
            search_term = title.split()[0] if title else ""
        else:
            search_term = role_match.group(1)

        # Fetch candidates using fuzzy search
        issues = await self._get(
            f"/projects/{self.config.project_encoded}/issues",
            params={"search": search_term, "state": state, "in": "title", "per_page": 100},
        )

        if not issues:
            return None

        # Filter for EXACT title match
        for issue in issues:
            if issue.get("title") == title:
                logger.info(f"Found exact match: #{issue.get('iid')} - {title}")
                return issue

        logger.debug(f"No exact title match found for: {title}")
        return None

    async def find_issue_for_role(
        self, role_name: str, state: str = "all"
    ) -> dict[str, Any] | None:
        """
        Find issue for a role by searching for `role_name` in title.

        This method handles the case where we need to find any issue
        for a role, regardless of exact title format.

        Args:
            role_name: Name of the role to find issue for
            state: Issue state filter ('opened', 'closed', 'all')

        Returns:
            Issue dict if found (oldest by IID), None otherwise
        """
        # Fetch candidates using role name as search term
        issues = await self._get(
            f"/projects/{self.config.project_encoded}/issues",
            params={"search": role_name, "state": state, "in": "title", "per_page": 100},
        )

        if not issues:
            return None

        # Filter for issues that have the role name in backticks and "box up" in title
        matching_issues = []
        for issue in issues:
            title = issue.get("title", "").lower()
            # Match patterns like "Box up `role_name`" or "feat(role_name): Box up `role_name`"
            if f"`{role_name}`" in issue.get("title", "") and "box up" in title:
                matching_issues.append(issue)

        if not matching_issues:
            logger.debug(f"No issues found for role '{role_name}'")
            return None

        # Return the oldest (lowest IID) as canonical to ensure consistency
        canonical = min(matching_issues, key=lambda i: i.get("iid", float("inf")))
        logger.info(
            f"Found canonical issue for '{role_name}': #{canonical.get('iid')} "
            f"(of {len(matching_issues)} matching issues)"
        )
        return canonical

    async def create_issue(
        self,
        title: str,
        description: str = "",
        labels: list[str] | None = None,
        assignee_ids: list[int] | None = None,
        assignee_username: str | None = None,
        iteration_id: int | None = None,
        weight: int | None = None,
        due_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new issue.

        Args:
            title: Issue title (required)
            description: Issue description (max 1MB)
            labels: List of label names
            assignee_ids: List of user IDs to assign
            assignee_username: Username to assign (resolved to ID)
            iteration_id: Iteration ID to assign
            weight: Issue weight
            due_date: Due date in YYYY-MM-DD format

        Returns:
            Created issue data from GitLab API
        """
        data: dict[str, Any] = {"title": title, "description": description}

        if labels:
            data["labels"] = ",".join(labels)
        if assignee_ids:
            data["assignee_ids"] = assignee_ids
        if assignee_username:
            user_id = await self.get_user_id(assignee_username)
            if user_id:
                data["assignee_ids"] = [user_id]
        if iteration_id:
            data["iteration_id"] = iteration_id
        if weight is not None:
            data["weight"] = weight
        if due_date:
            data["due_date"] = due_date

        return await self._post(f"/projects/{self.config.project_encoded}/issues", data)

    async def get_or_create_issue(
        self,
        title: str,
        description: str,
        labels: list[str] | None = None,
        assignee_username: str | None = None,
        iteration_id: int | None = None,
        role_name: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """
        Idempotent issue creation using EXACT title matching. Returns (issue, created).

        Args:
            title: Issue title for both search AND creation (must be exact)
            description: Issue description for creation
            labels: Labels to apply
            assignee_username: Username to assign
            iteration_id: Iteration to assign
            role_name: Optional role name for fallback search

        Returns:
            Tuple of (issue_dict, created_bool)
        """
        # First: Try exact title match
        existing = await self.find_issue_by_exact_title(title, state="all")
        if existing:
            logger.info(f"Found existing issue by exact title: {existing.get('web_url')}")
            return existing, False

        # Second: If role_name provided, search for any issue for this role
        # This catches issues with slightly different title formats
        if role_name:
            role_issue = await self.find_issue_for_role(role_name, state="all")
            if role_issue:
                logger.info(
                    f"Found existing issue for role '{role_name}': {role_issue.get('web_url')}"
                )
                return role_issue, False

        # Use default assignee if not specified
        if not assignee_username and self.config.default_assignee:
            assignee_username = self.config.default_assignee

        # Use default labels if not specified
        if not labels:
            labels = self.config.default_labels

        issue = await self.create_issue(
            title=title,
            description=description,
            labels=labels,
            assignee_username=assignee_username,
            iteration_id=iteration_id,
        )
        logger.info(f"Created issue: {issue.get('web_url')}")
        return issue, True

    async def update_issue(self, iid: int, data: dict[str, Any]) -> dict[str, Any]:
        """
        Update an issue.

        Args:
            iid: Issue IID (project-scoped)
            data: Fields to update

        Returns:
            Updated issue data
        """
        return await self._put(f"/projects/{self.config.project_encoded}/issues/{iid}", data)

    async def close_issue(self, iid: int) -> dict[str, Any]:
        """Close an issue."""
        return await self.update_issue(iid, {"state_event": "close"})

    async def reopen_issue(self, iid: int) -> dict[str, Any]:
        """Reopen a closed issue."""
        return await self.update_issue(iid, {"state_event": "reopen"})

    async def add_issue_comment(self, iid: int, body: str) -> dict[str, Any]:
        """Add a comment/note to an issue."""
        return await self._post(
            f"/projects/{self.config.project_encoded}/issues/{iid}/notes",
            {"body": body},
        )

    # =========================================================================
    # Merge Request Operations
    # =========================================================================

    async def find_mr_by_branch(
        self, source_branch: str, state: str = "opened"
    ) -> dict[str, Any] | None:
        """
        Find MR by source branch.

        Args:
            source_branch: Source branch name
            state: MR state filter ('opened', 'closed', 'merged', 'all')

        Returns:
            First matching MR dict or None if not found
        """
        mrs = await self._get(
            f"/projects/{self.config.project_encoded}/merge_requests",
            params={"source_branch": source_branch, "state": state},
        )
        return mrs[0] if mrs else None

    async def create_mr(
        self,
        source_branch: str,
        target_branch: str = "main",
        title: str = "",
        description: str = "",
        labels: list[str] | None = None,
        assignee_username: str | None = None,
        remove_source_branch: bool = True,
        squash: bool = True,
    ) -> dict[str, Any]:
        """
        Create a merge request.

        Args:
            source_branch: Branch with changes (required)
            target_branch: Branch to merge into (default: main)
            title: MR title (required)
            description: MR description
            labels: List of label names
            assignee_username: Username to assign
            remove_source_branch: Delete branch on merge
            squash: Squash commits on merge

        Returns:
            Created MR data from GitLab API
        """
        data: dict[str, Any] = {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "title": title,
            "description": description,
            "remove_source_branch": remove_source_branch,
            "squash": squash,
        }

        if labels:
            data["labels"] = ",".join(labels)

        if assignee_username:
            user_id = await self.get_user_id(assignee_username)
            if user_id:
                data["assignee_id"] = user_id
        elif self.config.default_assignee:
            user_id = await self.get_user_id(self.config.default_assignee)
            if user_id:
                data["assignee_id"] = user_id

        return await self._post(f"/projects/{self.config.project_encoded}/merge_requests", data)

    async def get_or_create_mr(
        self,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str,
        labels: list[str] | None = None,
        assignee_username: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """
        Idempotent MR creation. Returns (mr, created).

        Args:
            source_branch: Source branch name
            target_branch: Target branch name
            title: MR title for creation
            description: MR description for creation
            labels: Labels to apply
            assignee_username: Username to assign

        Returns:
            Tuple of (mr_dict, created_bool)
        """
        # Check for existing opened MR
        existing = await self.find_mr_by_branch(source_branch, state="opened")
        if existing:
            logger.info(f"Found existing MR: {existing.get('web_url')}")
            return existing, False

        # Check for merged MR (already done)
        merged = await self.find_mr_by_branch(source_branch, state="merged")
        if merged:
            logger.info(f"Found merged MR: {merged.get('web_url')}")
            return merged, False

        # Use default labels if not specified
        if not labels:
            labels = self.config.default_labels

        mr = await self.create_mr(
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
            description=description,
            labels=labels,
            assignee_username=assignee_username,
        )
        logger.info(f"Created MR: {mr.get('web_url')}")
        return mr, True

    async def update_mr(self, iid: int, data: dict[str, Any]) -> dict[str, Any]:
        """
        Update a merge request.

        Args:
            iid: MR IID (project-scoped)
            data: Fields to update

        Returns:
            Updated MR data
        """
        return await self._put(
            f"/projects/{self.config.project_encoded}/merge_requests/{iid}", data
        )

    async def add_mr_comment(self, iid: int, body: str) -> dict[str, Any]:
        """Add a comment/note to a merge request."""
        return await self._post(
            f"/projects/{self.config.project_encoded}/merge_requests/{iid}/notes",
            {"body": body},
        )

    async def merge_mr(
        self,
        iid: int,
        squash: bool = True,
        remove_source_branch: bool = True,
        merge_when_pipeline_succeeds: bool = False,
    ) -> dict[str, Any]:
        """
        Merge a merge request.

        Args:
            iid: MR IID
            squash: Squash commits
            remove_source_branch: Delete source branch after merge
            merge_when_pipeline_succeeds: Wait for pipeline to pass

        Returns:
            Merged MR data
        """
        data: dict[str, Any] = {
            "squash": squash,
            "should_remove_source_branch": remove_source_branch,
        }
        if merge_when_pipeline_succeeds:
            data["merge_when_pipeline_succeeds"] = True

        return await self._put(
            f"/projects/{self.config.project_encoded}/merge_requests/{iid}/merge", data
        )

    # =========================================================================
    # Merge Train Operations
    # =========================================================================

    async def add_mr_to_merge_train(self, iid: int) -> dict[str, Any]:
        """
        Add MR to merge train (Premium/Ultimate only).

        Args:
            iid: MR IID

        Returns:
            Merge train entry data
        """
        return await self._post(
            f"/projects/{self.config.project_encoded}/merge_trains/merge_requests/{iid}",
            {},
        )

    async def get_merge_train(self, target_branch: str = "main") -> list[dict[str, Any]]:
        """Get merge train queue for a target branch."""
        result = await self._get(
            f"/projects/{self.config.project_encoded}/merge_trains",
            params={"target_branch": target_branch},
        )
        return result if isinstance(result, list) else []

    # =========================================================================
    # User Operations
    # =========================================================================

    async def get_user_id(self, username: str) -> int | None:
        """
        Get GitLab user ID from username.

        Args:
            username: GitLab username

        Returns:
            User ID or None if not found
        """
        try:
            users = await self._get("/users", params={"username": username})
            if users and len(users) > 0:
                return users[0].get("id")
            return None
        except GitLabAPIError:
            return None

    # =========================================================================
    # Label Operations
    # =========================================================================

    async def get_label(self, name: str) -> dict[str, Any] | None:
        """Get a label by name."""
        try:
            import urllib.parse
            encoded_name = urllib.parse.quote(name, safe="")
            return await self._get(
                f"/projects/{self.config.project_encoded}/labels/{encoded_name}"
            )
        except GitLabAPIError as e:
            if e.status_code == 404:
                return None
            raise

    async def create_label(
        self, name: str, color: str = "#428BCA", description: str = ""
    ) -> dict[str, Any]:
        """Create a project label."""
        data = {"name": name, "color": color}
        if description:
            data["description"] = description
        return await self._post(f"/projects/{self.config.project_encoded}/labels", data)

    async def ensure_label_exists(
        self, name: str, color: str = "#428BCA", description: str = ""
    ) -> bool:
        """Ensure a label exists, creating it if necessary."""
        try:
            existing = await self.get_label(name)
            if existing:
                return True
            await self.create_label(name, color, description)
            logger.info(f"Created label '{name}'")
            return True
        except GitLabAPIError as e:
            if "already exists" in str(e).lower() or "has already been taken" in str(e).lower():
                return True
            logger.error(f"Failed to ensure label '{name}': {e}")
            return False

    # =========================================================================
    # Iteration Operations
    # =========================================================================

    async def list_iterations(
        self, group_path: str, state: str = "opened"
    ) -> list[dict[str, Any]]:
        """
        List group iterations.

        Args:
            group_path: Group path (e.g., "bates-ils")
            state: Iteration state filter

        Returns:
            List of iteration dicts
        """
        encoded_group = group_path.replace("/", "%2F")
        result = await self._get(
            f"/groups/{encoded_group}/iterations",
            params={"state": state},
        )
        return result if isinstance(result, list) else []

    async def get_current_iteration(self, group_path: str) -> dict[str, Any] | None:
        """
        Get the current active iteration containing today's date.

        Args:
            group_path: Group path

        Returns:
            Current iteration dict or None
        """
        from datetime import date

        today = date.today().isoformat()
        iterations = await self.list_iterations(group_path, state="opened")

        for iteration in iterations:
            start_date = iteration.get("start_date")
            due_date = iteration.get("due_date")
            if start_date and due_date:
                if start_date <= today <= due_date:
                    return iteration

        # Fallback to first opened iteration
        return iterations[0] if iterations else None

