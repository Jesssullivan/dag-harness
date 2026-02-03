"""
Advanced GitLab API integration with async operations and GraphQL support.

Provides:
- GitLabProjectManager: Project-level operations (members, approvals, branches)
- GitLabIterationManager: Iteration/sprint management with cadence support
- GitLabMergeTrainManager: Merge train queue management and ETA estimation
- GraphQL queries for complex data retrieval
- Utility functions for GID parsing and URL encoding
"""

import asyncio
import json
import logging
import subprocess
import urllib.parse
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPHQL QUERIES
# =============================================================================

ITERATION_QUERY = """
query GetIteration($groupPath: ID!, $iterationId: ID!) {
  group(fullPath: $groupPath) {
    iteration(id: $iterationId) {
      id
      iid
      title
      description
      state
      startDate
      dueDate
      webUrl
      issues {
        count
        nodes {
          id
          iid
          title
          state
          webUrl
          labels {
            nodes {
              title
              color
            }
          }
          assignees {
            nodes {
              username
            }
          }
        }
      }
    }
  }
}
"""

PROJECT_MERGE_REQUESTS_QUERY = """
query GetProjectMergeRequests($projectPath: ID!, $state: MergeRequestState) {
  project(fullPath: $projectPath) {
    mergeRequests(state: $state, first: 50) {
      nodes {
        id
        iid
        title
        state
        webUrl
        sourceBranch
        targetBranch
        draft
        mergeStatus
        mergeableDiscussionsState
        headPipeline {
          id
          status
          duration
          finishedAt
          stages {
            nodes {
              name
              status
            }
          }
        }
        approvedBy {
          nodes {
            username
          }
        }
        labels {
          nodes {
            title
          }
        }
      }
    }
  }
}
"""

MERGE_TRAIN_QUERY = """
query GetMergeTrain($projectPath: ID!, $targetBranch: String!) {
  project(fullPath: $projectPath) {
    mergeTrains(targetBranch: $targetBranch) {
      nodes {
        id
        mergeRequest {
          id
          iid
          title
          webUrl
          sourceBranch
          headPipeline {
            id
            status
            duration
          }
        }
        targetBranch
        status
        duration
        createdAt
      }
    }
  }
}
"""

ITERATION_CADENCE_QUERY = """
query GetIterationCadence($groupPath: ID!) {
  group(fullPath: $groupPath) {
    iterationCadences(first: 10) {
      nodes {
        id
        title
        description
        startDate
        durationInWeeks
        iterationsInAdvance
        automatic
        active
        iterations(first: 5, state: current) {
          nodes {
            id
            iid
            title
            startDate
            dueDate
          }
        }
      }
    }
  }
}
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def encode_project_path(path: str) -> str:
    """
    URL encode a project path for API usage.

    GitLab API requires project paths to be URL-encoded with slashes
    converted to %2F.

    Args:
        path: Project path (e.g., "bates-ils/ems")

    Returns:
        URL-encoded path (e.g., "bates-ils%2Fems")

    Example:
        >>> encode_project_path("bates-ils/projects/ems/ems-mono")
        'bates-ils%2Fprojects%2Fems%2Fems-mono'
    """
    return urllib.parse.quote(path, safe="")


def parse_gid(gid: str) -> tuple[str, int]:
    """
    Parse a GitLab Global ID string into type and ID components.

    GitLab GraphQL uses Global IDs in format: gid://gitlab/Type/123

    Args:
        gid: GitLab Global ID string

    Returns:
        Tuple of (type_name, numeric_id)

    Raises:
        ValueError: If GID format is invalid

    Example:
        >>> parse_gid("gid://gitlab/Iteration/123")
        ('Iteration', 123)
        >>> parse_gid("gid://gitlab/MergeRequest/456")
        ('MergeRequest', 456)
    """
    if not gid.startswith("gid://gitlab/"):
        raise ValueError(f"Invalid GID format: {gid}")

    # Remove prefix and split
    parts = gid.replace("gid://gitlab/", "").split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid GID format: {gid}")

    type_name = parts[0]
    try:
        numeric_id = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid GID ID component: {parts[1]}")

    return (type_name, numeric_id)


def build_gid(type_name: str, id: int) -> str:
    """
    Build a GitLab Global ID string from type and ID.

    Args:
        type_name: GitLab type name (e.g., "Iteration", "MergeRequest")
        id: Numeric ID

    Returns:
        GitLab Global ID string

    Example:
        >>> build_gid("Iteration", 123)
        'gid://gitlab/Iteration/123'
        >>> build_gid("Project", 456)
        'gid://gitlab/Project/456'
    """
    return f"gid://gitlab/{type_name}/{id}"


# =============================================================================
# BASE ASYNC CLIENT
# =============================================================================


@dataclass
class GitLabAsyncConfig:
    """Configuration for async GitLab operations."""

    project_path: str = "bates-ils/projects/ems/ems-mono"
    group_path: str = "bates-ils"
    api_url: str = "https://gitlab.com/api/v4"
    graphql_url: str = "https://gitlab.com/api/graphql"
    timeout: int = 30

    @property
    def project_path_encoded(self) -> str:
        """Return URL-encoded project path."""
        return encode_project_path(self.project_path)


class GitLabAsyncBase:
    """
    Base class for async GitLab API operations.

    Uses glab CLI for authentication and API calls in an async context.
    """

    def __init__(self, config: GitLabAsyncConfig | None = None):
        self.config = config or GitLabAsyncConfig()

    async def _run_glab(self, *args: str) -> dict[str, Any] | list | str:
        """
        Run a glab CLI command asynchronously.

        Args:
            *args: Command arguments to pass to glab

        Returns:
            Parsed JSON response or raw output string

        Raises:
            RuntimeError: If command fails
        """
        cmd = ["glab"] + list(args)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout
            )

            if proc.returncode != 0:
                raise RuntimeError(f"glab command failed: {stderr.decode()}")

            output = stdout.decode().strip()
            if output.startswith("{") or output.startswith("["):
                return json.loads(output)
            return output

        except asyncio.TimeoutError:
            raise RuntimeError(f"glab command timed out after {self.config.timeout}s")

    async def _api_get(self, endpoint: str) -> dict[str, Any] | list:
        """Make authenticated GET request to GitLab REST API."""
        return await self._run_glab("api", endpoint)

    async def _api_post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated POST request to GitLab REST API."""
        field_args = []
        for key, value in data.items():
            field_args.extend(["-f", f"{key}={value}"])

        return await self._run_glab("api", endpoint, "--method", "POST", *field_args)

    async def _api_put(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated PUT request to GitLab REST API."""
        field_args = []
        for key, value in data.items():
            field_args.extend(["-f", f"{key}={value}"])

        return await self._run_glab("api", endpoint, "--method", "PUT", *field_args)

    async def _graphql(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query against GitLab API.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            GraphQL response data

        Raises:
            RuntimeError: If query fails or returns errors
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        # Use glab api with raw JSON body
        cmd = [
            "glab",
            "api",
            "graphql",
            "--raw-field",
            f"query={query}",
        ]

        # Add variables if present
        if variables:
            cmd.extend(["--raw-field", f"variables={json.dumps(variables)}"])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout
            )

            if proc.returncode != 0:
                raise RuntimeError(f"GraphQL query failed: {stderr.decode()}")

            result = json.loads(stdout.decode())

            if "errors" in result:
                raise RuntimeError(f"GraphQL errors: {result['errors']}")

            return result.get("data", {})

        except asyncio.TimeoutError:
            raise RuntimeError(f"GraphQL query timed out after {self.config.timeout}s")


# =============================================================================
# PROJECT MANAGER
# =============================================================================


class GitLabProjectManager(GitLabAsyncBase):
    """
    Manager for project-level GitLab operations.

    Handles:
    - Project information retrieval
    - Member listing
    - Merge request approvals
    - Protected branch management
    """

    async def get_project_info(self, project_path: str) -> dict:
        """
        Get detailed project information.

        Args:
            project_path: Full project path (e.g., "bates-ils/ems")

        Returns:
            Project data including:
            - id, name, description
            - visibility, default_branch
            - web_url, ssh_url_to_repo, http_url_to_repo
            - statistics (if available)
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(f"projects/{encoded}?statistics=true")

    async def list_project_members(self, project_path: str) -> list[dict]:
        """
        List all project members with their access levels.

        Args:
            project_path: Full project path

        Returns:
            List of member dicts containing:
            - id, username, name, avatar_url
            - access_level, access_level_description
            - expires_at (if membership is time-limited)

        Access levels:
            - 10: Guest
            - 20: Reporter
            - 30: Developer
            - 40: Maintainer
            - 50: Owner
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(f"projects/{encoded}/members/all")

    async def get_merge_request_approvals(
        self, project_path: str, mr_iid: int
    ) -> dict:
        """
        Get merge request approval status and rules.

        Args:
            project_path: Full project path
            mr_iid: Merge request IID (project-scoped)

        Returns:
            Approval data including:
            - approved: bool
            - approvals_required: int
            - approvals_left: int
            - approved_by: list of users who approved
            - approval_rules_left: list of unfulfilled rules
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(f"projects/{encoded}/merge_requests/{mr_iid}/approvals")

    async def list_protected_branches(self, project_path: str) -> list[dict]:
        """
        List all protected branches with their protection rules.

        Args:
            project_path: Full project path

        Returns:
            List of protected branch dicts containing:
            - id, name
            - push_access_levels, merge_access_levels
            - allow_force_push
            - code_owner_approval_required
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(f"projects/{encoded}/protected_branches")


# =============================================================================
# ITERATION MANAGER
# =============================================================================


class GitLabIterationManager(GitLabAsyncBase):
    """
    Manager for GitLab iteration/sprint operations.

    Iterations are group-level in GitLab and used for sprint planning.
    Supports iteration cadences for automated iteration creation.
    """

    async def list_iterations(
        self, group_path: str, state: str = "current"
    ) -> list[dict]:
        """
        List iterations for a group.

        Args:
            group_path: Group path (e.g., "bates-ils")
            state: Iteration state filter:
                - "opened": All non-closed iterations
                - "upcoming": Future iterations
                - "current": Currently active iterations
                - "closed": Completed iterations
                - "all": All iterations

        Returns:
            List of iteration dicts with id, title, start_date, due_date, state
        """
        encoded = encode_project_path(group_path)
        return await self._api_get(f"groups/{encoded}/iterations?state={state}")

    async def get_iteration_by_title(
        self, group_path: str, title: str
    ) -> dict | None:
        """
        Find an iteration by its title.

        Args:
            group_path: Group path
            title: Iteration title to search for

        Returns:
            Iteration dict if found, None otherwise
        """
        encoded_group = encode_project_path(group_path)
        encoded_title = urllib.parse.quote(title)

        iterations = await self._api_get(
            f"groups/{encoded_group}/iterations?search={encoded_title}"
        )

        for iteration in iterations:
            if iteration.get("title") == title:
                return iteration

        return None

    async def create_iteration(
        self,
        group_path: str,
        title: str,
        start_date: str,
        due_date: str,
        description: str = "",
    ) -> dict:
        """
        Create a new iteration.

        Args:
            group_path: Group path where iteration will be created
            title: Iteration title
            start_date: Start date in YYYY-MM-DD format
            due_date: Due date in YYYY-MM-DD format
            description: Optional iteration description

        Returns:
            Created iteration data

        Note:
            Requires iteration cadence to be configured for the group.
            Creating iterations manually may not work if automatic
            iteration generation is enabled.
        """
        encoded = encode_project_path(group_path)
        data = {
            "title": title,
            "start_date": start_date,
            "due_date": due_date,
        }
        if description:
            data["description"] = description

        return await self._api_post(f"groups/{encoded}/iterations", data)

    async def get_iteration_cadence(self, group_path: str) -> dict | None:
        """
        Get iteration cadence configuration for a group.

        Iteration cadences define the automatic creation schedule for
        iterations (sprints).

        Args:
            group_path: Group path

        Returns:
            Active iteration cadence dict if found, None otherwise.
            Includes:
            - id, title, description
            - start_date, duration_in_weeks
            - iterations_in_advance
            - automatic, active

        Note:
            Uses GraphQL as REST API has limited cadence support.
        """
        try:
            result = await self._graphql(
                ITERATION_CADENCE_QUERY,
                {"groupPath": group_path},
            )

            cadences = (
                result.get("group", {})
                .get("iterationCadences", {})
                .get("nodes", [])
            )

            # Return first active cadence
            for cadence in cadences:
                if cadence.get("active"):
                    return cadence

            return cadences[0] if cadences else None

        except Exception as e:
            logger.warning(f"Failed to get iteration cadence: {e}")
            return None


# =============================================================================
# MERGE TRAIN MANAGER
# =============================================================================


class GitLabMergeTrainManager(GitLabAsyncBase):
    """
    Manager for GitLab merge train operations.

    Merge trains are a GitLab Premium feature that ensures MRs are
    merged sequentially with passing pipelines.
    """

    async def get_merge_train(
        self, project_path: str, target_branch: str = "main"
    ) -> list[dict]:
        """
        Get the current merge train queue for a target branch.

        Args:
            project_path: Full project path
            target_branch: Target branch name (default: "main")

        Returns:
            List of merge train entries, each containing:
            - id: Merge train entry ID
            - merge_request: MR details (iid, title, web_url)
            - status: Train status (idle, running, fresh)
            - duration: Time in train (seconds)
            - target_branch: Target branch
        """
        encoded = encode_project_path(project_path)
        return await self._api_get(
            f"projects/{encoded}/merge_trains?target_branch={target_branch}"
        )

    async def get_mr_position_in_train(
        self, project_path: str, mr_iid: int
    ) -> int | None:
        """
        Get the position of an MR in the merge train queue.

        Args:
            project_path: Full project path
            mr_iid: Merge request IID

        Returns:
            1-indexed position in train, or None if not in train

        Example:
            >>> pos = await mgr.get_mr_position_in_train("group/project", 123)
            >>> if pos:
            ...     print(f"MR is #{pos} in the queue")
        """
        train = await self.get_merge_train(project_path)

        for i, entry in enumerate(train, 1):
            if entry.get("merge_request", {}).get("iid") == mr_iid:
                return i

        return None

    async def add_to_merge_train(
        self,
        project_path: str,
        mr_iid: int,
        when_pipeline_succeeds: bool = True,
    ) -> bool:
        """
        Add a merge request to the merge train.

        Args:
            project_path: Full project path
            mr_iid: Merge request IID
            when_pipeline_succeeds: Wait for pipeline before merging

        Returns:
            True if successfully added, False otherwise

        Note:
            Merge trains must be enabled on the project.
            The MR must be in a mergeable state (no conflicts,
            approvals met, etc.)
        """
        encoded = encode_project_path(project_path)

        data = {"merge_request_iid": str(mr_iid)}
        if when_pipeline_succeeds:
            data["when_pipeline_succeeds"] = "true"

        try:
            await self._api_post(f"projects/{encoded}/merge_trains", data)
            logger.info(f"Added MR !{mr_iid} to merge train")
            return True
        except RuntimeError as e:
            logger.error(f"Failed to add MR !{mr_iid} to merge train: {e}")
            return False

    async def get_merge_train_eta(
        self, project_path: str, mr_iid: int
    ) -> int | None:
        """
        Estimate time until MR will be merged from merge train.

        This calculates an ETA based on:
        - Position in queue
        - Average pipeline duration from recent entries
        - Current running pipeline progress

        Args:
            project_path: Full project path
            mr_iid: Merge request IID

        Returns:
            Estimated seconds until merge, or None if:
            - MR not in train
            - Cannot estimate (no historical data)

        Note:
            This is a rough estimate. Actual time depends on:
            - Pipeline failures requiring rerun
            - New MRs added ahead (if priority ordering)
            - Pipeline duration variation
        """
        train = await self.get_merge_train(project_path)

        if not train:
            return None

        # Find MR position and gather timing data
        position = None
        total_duration = 0
        count = 0

        for i, entry in enumerate(train):
            mr_data = entry.get("merge_request", {})

            # Track duration data from entries ahead in queue
            if entry.get("duration"):
                total_duration += entry["duration"]
                count += 1

            if mr_data.get("iid") == mr_iid:
                position = i  # 0-indexed position
                break

        if position is None:
            return None

        # If we have historical data, use average duration
        if count > 0:
            avg_duration = total_duration / count
        else:
            # Default estimate: 10 minutes per pipeline
            avg_duration = 600

        # Estimate: (entries ahead) * average_duration
        entries_ahead = position
        eta_seconds = int(entries_ahead * avg_duration)

        return eta_seconds


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_project_manager(
    project_path: str | None = None,
) -> GitLabProjectManager:
    """
    Create a GitLabProjectManager with optional custom project path.

    Args:
        project_path: Override default project path

    Returns:
        Configured GitLabProjectManager instance
    """
    config = GitLabAsyncConfig()
    if project_path:
        config.project_path = project_path
    return GitLabProjectManager(config)


def create_iteration_manager(
    group_path: str | None = None,
) -> GitLabIterationManager:
    """
    Create a GitLabIterationManager with optional custom group path.

    Args:
        group_path: Override default group path

    Returns:
        Configured GitLabIterationManager instance
    """
    config = GitLabAsyncConfig()
    if group_path:
        config.group_path = group_path
    return GitLabIterationManager(config)


def create_merge_train_manager(
    project_path: str | None = None,
) -> GitLabMergeTrainManager:
    """
    Create a GitLabMergeTrainManager with optional custom project path.

    Args:
        project_path: Override default project path

    Returns:
        Configured GitLabMergeTrainManager instance
    """
    config = GitLabAsyncConfig()
    if project_path:
        config.project_path = project_path
    return GitLabMergeTrainManager(config)
