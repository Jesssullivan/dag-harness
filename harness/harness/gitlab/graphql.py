"""
GitLab GraphQL client for harness operations.

Provides async GraphQL query and mutation execution with proper error handling.
Uses httpx for async HTTP requests to the GitLab GraphQL API.

Token retrieval follows the same pattern as api.py:
1. GITLAB_TOKEN or GLAB_TOKEN environment variables
2. Falls back to `glab auth token` command
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GraphQLErrorLocation:
    """Location of an error in the GraphQL query."""

    line: int
    column: int


@dataclass
class GitLabGraphQLError(Exception):
    """
    Exception raised for GitLab GraphQL API errors.

    Attributes:
        message: Error message from GraphQL response
        locations: List of error locations in the query
        path: Path to the field that caused the error
        extensions: Additional error metadata from GitLab
    """

    message: str
    locations: list[GraphQLErrorLocation] = field(default_factory=list)
    path: list[str | int] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.message]
        if self.path:
            parts.append(f"path={'.'.join(str(p) for p in self.path)}")
        if self.locations:
            loc_strs = [f"line {loc.line}, column {loc.column}" for loc in self.locations]
            parts.append(f"at {'; '.join(loc_strs)}")
        if self.extensions:
            parts.append(f"extensions={self.extensions}")
        return " | ".join(parts)


def parse_graphql_errors(errors: list[dict[str, Any]]) -> list[GitLabGraphQLError]:
    """
    Parse GraphQL errors array into GitLabGraphQLError instances.

    Args:
        errors: List of error dicts from GraphQL response

    Returns:
        List of GitLabGraphQLError instances with parsed details
    """
    parsed = []
    for error in errors:
        locations = []
        for loc in error.get("locations", []):
            locations.append(
                GraphQLErrorLocation(
                    line=loc.get("line", 0),
                    column=loc.get("column", 0),
                )
            )

        parsed.append(
            GitLabGraphQLError(
                message=error.get("message", "Unknown GraphQL error"),
                locations=locations,
                path=error.get("path", []),
                extensions=error.get("extensions", {}),
            )
        )
    return parsed


class GitLabGraphQLClient:
    """
    Async GitLab GraphQL API client.

    Uses httpx for async HTTP requests. Token is retrieved from environment
    variables or the glab CLI.

    Attributes:
        base_url: GitLab GraphQL API endpoint
        token: GitLab API token (retrieved lazily)

    Example:
        >>> client = GitLabGraphQLClient()
        >>> result = await client.execute_query('''
        ...     query {
        ...         currentUser {
        ...             username
        ...         }
        ...     }
        ... ''')
        >>> print(result['currentUser']['username'])
    """

    DEFAULT_BASE_URL = "https://gitlab.com/api/graphql"

    def __init__(self, base_url: str | None = None):
        """
        Initialize the GraphQL client.

        Args:
            base_url: GitLab GraphQL API endpoint. Defaults to https://gitlab.com/api/graphql
        """
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._token: str | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def token(self) -> str:
        """
        Get GitLab token from environment or glab config.

        Checks in order:
        1. GITLAB_TOKEN environment variable
        2. GLAB_TOKEN environment variable
        3. `glab auth token` command output

        Returns:
            GitLab API token

        Raises:
            ValueError: If no token can be found
        """
        if self._token:
            return self._token

        # Try environment variables first
        self._token = os.environ.get("GITLAB_TOKEN") or os.environ.get("GLAB_TOKEN")
        if self._token:
            logger.debug("Using GitLab token from environment variable")
            return self._token

        # Try glab auth status -t command (same pattern as api.py)
        try:
            result = subprocess.run(
                ["glab", "auth", "status", "-t"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # glab outputs to stderr, so check both stdout and stderr
            output = result.stdout + result.stderr
            # Parse token from output - looks for lines containing "Token"
            # Format: "  ✓ Token found: glpat-xxx" or "Token: glpat-xxx"
            for line in output.split("\n"):
                if "Token" in line and ":" in line:
                    # Split on last colon to get the token part
                    parts = line.rsplit(":", 1)
                    if len(parts) > 1:
                        token = parts[1].strip()
                        # Validate it looks like a GitLab token
                        if token and (token.startswith("glpat-") or token.startswith("gl")):
                            self._token = token
                            logger.debug("Using GitLab token from glab CLI")
                            return self._token
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"Failed to get token from glab CLI: {e}")

        raise ValueError(
            "No GitLab token found. Set GITLAB_TOKEN environment variable or run 'glab auth login'"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "GitLabGraphQLClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _execute(
        self, operation: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a GraphQL operation (query or mutation).

        Args:
            operation: GraphQL query or mutation string
            variables: Optional variables for the operation

        Returns:
            The 'data' portion of the GraphQL response

        Raises:
            GitLabGraphQLError: If the response contains errors
            httpx.HTTPError: If the HTTP request fails
        """
        client = await self._get_client()

        payload: dict[str, Any] = {"query": operation}
        if variables:
            payload["variables"] = variables

        response = await client.post(self.base_url, json=payload)
        response.raise_for_status()

        result = response.json()

        # Check for GraphQL errors
        if "errors" in result and result["errors"]:
            errors = parse_graphql_errors(result["errors"])
            if errors:
                # Log all errors for debugging
                for error in errors:
                    logger.error(f"GraphQL error: {error}")
                # Raise the first error
                raise errors[0]

        return result.get("data", {})

    async def execute_query(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Optional variables for the query

        Returns:
            The 'data' portion of the GraphQL response

        Raises:
            GitLabGraphQLError: If the response contains errors
            httpx.HTTPError: If the HTTP request fails

        Example:
            >>> result = await client.execute_query('''
            ...     query GetProject($path: ID!) {
            ...         project(fullPath: $path) {
            ...             name
            ...             description
            ...         }
            ...     }
            ... ''', variables={"path": "bates-ils/projects/ems/ems-mono"})
        """
        logger.debug(f"Executing GraphQL query: {query[:100]}...")
        return await self._execute(query, variables)

    async def execute_mutation(
        self, mutation: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a GraphQL mutation.

        Args:
            mutation: GraphQL mutation string
            variables: Optional variables for the mutation

        Returns:
            The 'data' portion of the GraphQL response

        Raises:
            GitLabGraphQLError: If the response contains errors
            httpx.HTTPError: If the HTTP request fails

        Example:
            >>> result = await client.execute_mutation('''
            ...     mutation CreateIssue($input: CreateIssueInput!) {
            ...         createIssue(input: $input) {
            ...             issue {
            ...                 iid
            ...                 webUrl
            ...             }
            ...             errors
            ...         }
            ...     }
            ... ''', variables={"input": {"projectPath": "...", "title": "..."}})
        """
        logger.debug(f"Executing GraphQL mutation: {mutation[:100]}...")
        return await self._execute(mutation, variables)

    # =========================================================================
    # ITERATION ASSIGNMENT MUTATIONS
    # =========================================================================

    async def assign_issue_to_iteration(
        self,
        project_path: str,
        issue_iid: int,
        iteration_id: int,
    ) -> bool:
        """
        Assign an issue to an iteration using GitLab GraphQL mutation.

        Uses the issueSetIteration mutation to assign an issue to a specific
        iteration. This is more reliable than the REST API for iteration
        operations.

        Args:
            project_path: Full project path (e.g., "bates-ils/projects/ems/ems-mono")
            issue_iid: Issue internal ID (the number shown in GitLab UI)
            iteration_id: Numeric iteration ID (will be converted to Global ID)

        Returns:
            True if assignment succeeded, False on failure

        Note:
            This method uses graceful degradation - it logs errors but does not
            raise exceptions, allowing workflows to continue even if iteration
            assignment fails.

        Example:
            >>> client = GitLabGraphQLClient()
            >>> success = await client.assign_issue_to_iteration(
            ...     "bates-ils/projects/ems/ems-mono",
            ...     issue_iid=42,
            ...     iteration_id=12345
            ... )
        """
        # Convert numeric ID to GitLab Global ID format
        iteration_gid = f"gid://gitlab/Iteration/{iteration_id}"

        mutation = """
        mutation AssignIssueToIteration($projectPath: ID!, $iid: String!, $iterationId: ID!) {
            issueSetIteration(input: {
                projectPath: $projectPath
                iid: $iid
                iterationId: $iterationId
            }) {
                issue {
                    iid
                    iteration {
                        id
                        title
                    }
                }
                errors
            }
        }
        """

        variables = {
            "projectPath": project_path,
            "iid": str(issue_iid),
            "iterationId": iteration_gid,
        }

        try:
            data = await self._execute(mutation, variables)

            result = data.get("issueSetIteration", {})
            errors = result.get("errors", [])

            if errors:
                for error in errors:
                    logger.error(f"Issue iteration assignment error: {error}")
                return False

            issue = result.get("issue")
            if issue and issue.get("iteration"):
                logger.info(
                    f"Assigned issue #{issue_iid} to iteration "
                    f"'{issue['iteration'].get('title', 'unknown')}'"
                )
                return True

            logger.warning(
                f"Issue #{issue_iid} iteration assignment returned no iteration data"
            )
            return False

        except GitLabGraphQLError as e:
            logger.error(f"Failed to assign issue #{issue_iid} to iteration: {e}")
            return False
        except httpx.HTTPError as e:
            logger.error(f"HTTP error assigning issue #{issue_iid} to iteration: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error assigning issue #{issue_iid} to iteration: {e}")
            return False

    async def assign_mr_to_iteration(
        self,
        project_path: str,
        mr_iid: int,
        iteration_id: int,
    ) -> bool:
        """
        Assign a merge request to an iteration using GitLab GraphQL mutation.

        Uses the mergeRequestSetIteration mutation. Note that MR iteration
        assignment may require GitLab Premium/Ultimate tier.

        Args:
            project_path: Full project path (e.g., "bates-ils/projects/ems/ems-mono")
            mr_iid: Merge request internal ID (the number shown in GitLab UI)
            iteration_id: Numeric iteration ID (will be converted to Global ID)

        Returns:
            True if assignment succeeded, False on failure

        Note:
            This method uses graceful degradation - it logs errors but does not
            raise exceptions. MR iteration assignment may not be available on
            all GitLab tiers.

        Example:
            >>> client = GitLabGraphQLClient()
            >>> success = await client.assign_mr_to_iteration(
            ...     "bates-ils/projects/ems/ems-mono",
            ...     mr_iid=123,
            ...     iteration_id=12345
            ... )
        """
        # Convert numeric ID to GitLab Global ID format
        iteration_gid = f"gid://gitlab/Iteration/{iteration_id}"

        mutation = """
        mutation AssignMRToIteration($projectPath: ID!, $iid: String!, $iterationId: ID!) {
            mergeRequestSetIteration(input: {
                projectPath: $projectPath
                iid: $iid
                iterationId: $iterationId
            }) {
                mergeRequest {
                    iid
                    iteration {
                        id
                        title
                    }
                }
                errors
            }
        }
        """

        variables = {
            "projectPath": project_path,
            "iid": str(mr_iid),
            "iterationId": iteration_gid,
        }

        try:
            data = await self._execute(mutation, variables)

            result = data.get("mergeRequestSetIteration", {})
            errors = result.get("errors", [])

            if errors:
                for error in errors:
                    logger.error(f"MR iteration assignment error: {error}")
                return False

            mr = result.get("mergeRequest")
            if mr and mr.get("iteration"):
                logger.info(
                    f"Assigned MR !{mr_iid} to iteration "
                    f"'{mr['iteration'].get('title', 'unknown')}'"
                )
                return True

            logger.warning(
                f"MR !{mr_iid} iteration assignment returned no iteration data "
                "(may require GitLab Premium)"
            )
            return False

        except GitLabGraphQLError as e:
            logger.error(f"Failed to assign MR !{mr_iid} to iteration: {e}")
            return False
        except httpx.HTTPError as e:
            logger.error(f"HTTP error assigning MR !{mr_iid} to iteration: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error assigning MR !{mr_iid} to iteration: {e}")
            return False

    # =========================================================================
    # ITERATION QUERIES
    # =========================================================================

    async def get_current_iteration(self, group_path: str) -> dict | None:
        """
        Query for the current active iteration in a group.

        Finds the iteration whose date range includes today's date. Returns
        the first matching iteration if multiple iterations overlap.

        Args:
            group_path: Full group path (e.g., "bates-ils")

        Returns:
            Dict with iteration details or None if no current iteration:
            {
                "id": 12345,           # Numeric ID
                "iid": "1",            # Internal ID
                "title": "Sprint 1",   # Iteration title
                "start_date": "2026-02-01",
                "due_date": "2026-02-14"
            }

        Note:
            This method uses graceful degradation - returns None on errors
            rather than raising exceptions.

        Example:
            >>> client = GitLabGraphQLClient()
            >>> iteration = await client.get_current_iteration("bates-ils")
            >>> if iteration:
            ...     print(f"Current: {iteration['title']}")
        """
        from datetime import date

        query = """
        query GetCurrentIteration($groupPath: ID!) {
            group(fullPath: $groupPath) {
                iterations(state: opened, first: 50) {
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
        """

        variables = {"groupPath": group_path}

        try:
            data = await self._execute(query, variables)

            group = data.get("group")
            if not group:
                logger.warning(f"Group not found: {group_path}")
                return None

            iterations = group.get("iterations", {}).get("nodes", [])
            if not iterations:
                logger.info(f"No open iterations found in group {group_path}")
                return None

            # Find iteration containing today's date
            today = date.today().isoformat()

            for iteration in iterations:
                start = iteration.get("startDate")
                due = iteration.get("dueDate")

                if start and due and start <= today <= due:
                    # Extract numeric ID from Global ID
                    gid = iteration.get("id", "")
                    numeric_id = None
                    if gid.startswith("gid://gitlab/Iteration/"):
                        try:
                            numeric_id = int(gid.split("/")[-1])
                        except ValueError:
                            logger.warning(f"Could not parse iteration ID from: {gid}")

                    return {
                        "id": numeric_id,
                        "iid": iteration.get("iid"),
                        "title": iteration.get("title"),
                        "start_date": start,
                        "due_date": due,
                    }

            logger.info(f"No iteration contains today's date in group {group_path}")
            return None

        except GitLabGraphQLError as e:
            logger.error(f"Failed to query current iteration for {group_path}: {e}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"HTTP error querying current iteration for {group_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying current iteration for {group_path}: {e}")
            return None
