"""
GitLab idempotency utilities for harness operations.

Provides cached, state-aware lookups for issues, MRs, branches, and worktrees
to enable safe, idempotent workflow operations.

Key features:
- Searches ALL states (opened, closed, merged) by default
- Caching with configurable TTL to reduce API calls
- Comprehensive artifact discovery for roles
- Integration with worktree manager for local state
"""

from __future__ import annotations

import functools
import logging
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, ParamSpec

from harness.db.models import Issue, MergeRequest
from harness.db.state import StateDB

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# CACHING UTILITIES
# =============================================================================


@dataclass
class CacheEntry:
    """Cache entry with expiration."""

    value: Any
    expires_at: float


# Module-level cache for expensive lookups
_cache: dict[str, CacheEntry] = {}


def cache_result(ttl_seconds: int = 300):
    """
    Decorator to cache function results with TTL.

    Caches based on the function name and all arguments (both positional and keyword).
    Cache key is a string representation of (func_name, args, sorted_kwargs).

    Args:
        ttl_seconds: Time-to-live in seconds (default: 300 = 5 minutes)

    Returns:
        Decorated function with caching

    Example:
        @cache_result(ttl_seconds=300)
        def expensive_api_call(project_id: str, state: str = "all") -> dict:
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            # Skip 'self' argument for methods
            cache_args = args[1:] if args and hasattr(args[0], "__class__") else args
            cache_key = f"{func.__module__}.{func.__qualname__}:{cache_args}:{sorted(kwargs.items())}"

            # Check cache
            now = time.time()
            entry = _cache.get(cache_key)
            if entry and entry.expires_at > now:
                logger.debug(f"Cache hit for {func.__name__}")
                return entry.value

            # Call function and cache result
            result = func(*args, **kwargs)
            _cache[cache_key] = CacheEntry(value=result, expires_at=now + ttl_seconds)
            logger.debug(f"Cache miss for {func.__name__}, cached for {ttl_seconds}s")

            return result

        return wrapper

    return decorator


def clear_cache(pattern: str | None = None) -> int:
    """
    Clear cached results.

    Args:
        pattern: Optional pattern to match cache keys. If None, clears all.

    Returns:
        Number of entries cleared
    """
    global _cache

    if pattern is None:
        count = len(_cache)
        _cache.clear()
        return count

    keys_to_remove = [k for k in _cache if pattern in k]
    for key in keys_to_remove:
        del _cache[key]

    return len(keys_to_remove)


def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dict with cache size, expired entries, and entry details
    """
    now = time.time()
    expired = sum(1 for entry in _cache.values() if entry.expires_at <= now)

    return {
        "total_entries": len(_cache),
        "expired_entries": expired,
        "active_entries": len(_cache) - expired,
    }


# =============================================================================
# ROLE ARTIFACTS DATA CLASS
# =============================================================================


@dataclass
class RoleArtifacts:
    """
    Comprehensive collection of all artifacts for a role.

    This class represents the complete state of a role's GitLab/git artifacts,
    enabling idempotent workflow decisions.
    """

    role_name: str
    existing_issue: Issue | None = None
    existing_mr: MergeRequest | None = None
    remote_branch_exists: bool = False
    worktree_exists: bool = False
    worktree_path: str | None = None

    # Additional metadata
    issue_state: str | None = None
    mr_state: str | None = None
    branch_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role_name": self.role_name,
            "existing_issue": self.existing_issue.model_dump() if self.existing_issue else None,
            "existing_mr": self.existing_mr.model_dump() if self.existing_mr else None,
            "remote_branch_exists": self.remote_branch_exists,
            "worktree_exists": self.worktree_exists,
            "worktree_path": self.worktree_path,
            "issue_state": self.issue_state,
            "mr_state": self.mr_state,
            "branch_name": self.branch_name,
        }

    @property
    def has_any_artifacts(self) -> bool:
        """Check if any artifacts exist for this role."""
        return (
            self.existing_issue is not None
            or self.existing_mr is not None
            or self.remote_branch_exists
            or self.worktree_exists
        )

    @property
    def is_complete(self) -> bool:
        """Check if all expected artifacts exist (issue, MR, branch)."""
        return (
            self.existing_issue is not None
            and self.existing_mr is not None
            and self.remote_branch_exists
        )

    @property
    def needs_issue(self) -> bool:
        """Check if an issue needs to be created."""
        return self.existing_issue is None

    @property
    def needs_mr(self) -> bool:
        """Check if an MR needs to be created."""
        return self.existing_mr is None and self.remote_branch_exists

    @property
    def needs_branch_push(self) -> bool:
        """Check if branch needs to be pushed."""
        return self.worktree_exists and not self.remote_branch_exists


# =============================================================================
# IDEMPOTENCY HELPER CLASS
# =============================================================================


class IdempotencyHelper:
    """
    Helper class for idempotent GitLab operations.

    Provides methods to find existing artifacts and determine what actions
    are needed, enabling safe re-execution of workflows.
    """

    def __init__(self, db: StateDB, project_path: str = "bates-ils/projects/ems/ems-mono"):
        """
        Initialize the idempotency helper.

        Args:
            db: StateDB instance for database access
            project_path: GitLab project path
        """
        self.db = db
        self.project_path = project_path
        self._project_path_encoded = project_path.replace("/", "%2F")

    def _api_get(self, endpoint: str) -> dict[str, Any] | list[dict]:
        """Make authenticated GET request to GitLab API."""
        result = subprocess.run(["glab", "api", endpoint], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"API request failed: {result.stderr}")
        import json

        return json.loads(result.stdout)

    @cache_result(ttl_seconds=300)
    def find_existing_issue(self, role_name: str, state: str = "all") -> Issue | None:
        """
        Find existing issue for a role by searching title pattern 'Box up `{role_name}`'.

        Searches ALL states by default (opened, closed) to find any existing issue
        for the role, enabling idempotent issue creation.

        Args:
            role_name: Name of the role to find issue for
            state: Issue state filter ('opened', 'closed', 'all')
                   Default is 'all' to find issues in any state.

        Returns:
            Existing Issue if found, None otherwise
        """
        # Search for issues with title pattern
        search_term = urllib.parse.quote(f"Box up `{role_name}`")

        try:
            issues = self._api_get(
                f"projects/{self._project_path_encoded}/issues"
                f"?state={state}&search={search_term}&in=title"
            )

            if not issues:
                logger.debug(f"No issues found for role '{role_name}' with state={state}")
                return None

            # Filter to exact match (GitLab search is fuzzy, we need exact)
            for issue_data in issues:
                title = issue_data.get("title", "")
                # Match patterns like "Box up `role_name`" or "feat(role_name): Box up `role_name`"
                if f"`{role_name}`" in title and "box up" in title.lower():
                    # Get role_id if available
                    role = self.db.get_role(role_name)
                    role_id = role.id if role else None

                    issue = Issue(
                        id=issue_data["id"],
                        iid=issue_data["iid"],
                        role_id=role_id,
                        iteration_id=(
                            issue_data.get("iteration", {}).get("id")
                            if issue_data.get("iteration")
                            else None
                        ),
                        title=issue_data["title"],
                        state=issue_data["state"],
                        web_url=issue_data["web_url"],
                        labels=str(issue_data.get("labels", [])),
                        assignee=(
                            issue_data.get("assignees", [{}])[0].get("username")
                            if issue_data.get("assignees")
                            else None
                        ),
                        weight=issue_data.get("weight"),
                    )

                    # Update local database
                    self.db.upsert_issue(issue)

                    logger.info(
                        f"Found existing issue for '{role_name}': #{issue.iid} ({issue.state})"
                    )
                    return issue

            logger.debug(f"No exact match found for role '{role_name}'")
            return None

        except RuntimeError as e:
            logger.warning(f"API error searching for issue: {e}")
            return None

    @cache_result(ttl_seconds=300)
    def find_existing_mr(
        self, role_name: str, source_branch: str | None = None, state: str = "all"
    ) -> MergeRequest | None:
        """
        Find existing MR for a role/source branch.

        Searches ALL states by default (opened, merged, closed) to find any
        existing MR, enabling idempotent MR creation.

        Args:
            role_name: Name of the role to find MR for
            source_branch: Source branch name. If None, defaults to f"sid/{role_name}"
            state: MR state filter ('opened', 'closed', 'merged', 'all')
                   Default is 'all' to find MRs in any state.

        Returns:
            Existing MergeRequest if found, None otherwise
        """
        if source_branch is None:
            source_branch = f"sid/{role_name}"

        try:
            mrs = self._api_get(
                f"projects/{self._project_path_encoded}/merge_requests"
                f"?state={state}&source_branch={source_branch}"
            )

            if not mrs:
                logger.debug(
                    f"No MRs found for source_branch '{source_branch}' with state={state}"
                )
                return None

            # Return the first (most recent) MR for this branch
            mr_data = mrs[0]

            # Try to find associated role
            role_id = None
            role = self.db.get_role(role_name)
            if role:
                role_id = role.id

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

            logger.info(f"Found existing MR for '{role_name}': !{mr.iid} ({mr.state})")
            return mr

        except RuntimeError as e:
            logger.warning(f"API error searching for MR: {e}")
            return None

    @cache_result(ttl_seconds=60)
    def remote_branch_exists(self, branch_name: str) -> bool:
        """
        Check if branch exists on origin using git ls-remote.

        Uses a shorter cache TTL (60s) since branch state can change frequently.

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

            exists = branch_name in result.stdout
            logger.debug(f"Remote branch '{branch_name}' exists: {exists}")
            return exists

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Error checking remote branch: {e}")
            return False

    def worktree_exists(self, role_name: str, base_path: str = "..") -> tuple[bool, str | None]:
        """
        Check if a worktree exists for the role.

        Args:
            role_name: Name of the role
            base_path: Base path for worktrees (default: parent directory)

        Returns:
            Tuple of (exists, path) where path is None if not found
        """
        # Compute expected worktree path
        worktree_path = Path(base_path).resolve() / f"sid-{role_name}"

        if worktree_path.exists() and (worktree_path / ".git").exists():
            logger.debug(f"Worktree exists for '{role_name}' at {worktree_path}")
            return True, str(worktree_path)

        # Also check git worktree list
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"], capture_output=True, text=True
            )

            branch_name = f"sid/{role_name}"
            for line in result.stdout.split("\n"):
                if line.startswith("worktree "):
                    path = line.split(" ", 1)[1]
                elif line.startswith("branch ") and branch_name in line:
                    logger.debug(f"Found worktree for '{role_name}' via git worktree list")
                    return True, path

        except subprocess.SubprocessError:
            pass

        logger.debug(f"No worktree found for '{role_name}'")
        return False, None

    def find_all_role_artifacts(
        self, role_name: str, worktree_base_path: str = ".."
    ) -> RoleArtifacts:
        """
        Find all existing artifacts for a role.

        Returns a comprehensive view of what already exists, enabling
        workflows to make idempotent decisions about what to create.

        Args:
            role_name: Name of the role
            worktree_base_path: Base path for worktrees

        Returns:
            RoleArtifacts with all discovered artifacts
        """
        branch_name = f"sid/{role_name}"

        # Find existing issue (all states)
        existing_issue = self.find_existing_issue(role_name, state="all")

        # Find existing MR (all states)
        existing_mr = self.find_existing_mr(role_name, source_branch=branch_name, state="all")

        # Check remote branch
        remote_exists = self.remote_branch_exists(branch_name)

        # Check worktree
        wt_exists, wt_path = self.worktree_exists(role_name, base_path=worktree_base_path)

        artifacts = RoleArtifacts(
            role_name=role_name,
            existing_issue=existing_issue,
            existing_mr=existing_mr,
            remote_branch_exists=remote_exists,
            worktree_exists=wt_exists,
            worktree_path=wt_path,
            issue_state=existing_issue.state if existing_issue else None,
            mr_state=existing_mr.state if existing_mr else None,
            branch_name=branch_name,
        )

        logger.info(
            f"Artifacts for '{role_name}': "
            f"issue={'#' + str(existing_issue.iid) if existing_issue else 'None'}, "
            f"mr={'!' + str(existing_mr.iid) if existing_mr else 'None'}, "
            f"branch={remote_exists}, worktree={wt_exists}"
        )

        return artifacts


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

# These functions provide a simpler API for common operations without
# needing to instantiate IdempotencyHelper directly.

_helper: IdempotencyHelper | None = None


def _get_helper(db: StateDB | None = None) -> IdempotencyHelper:
    """Get or create the module-level helper instance."""
    global _helper
    if _helper is None:
        if db is None:
            from harness.config import HarnessConfig

            config = HarnessConfig.load()
            db = StateDB(config.database.path)
        _helper = IdempotencyHelper(db)
    return _helper


def find_existing_issue(role_name: str, state: str = "all", db: StateDB | None = None) -> Issue | None:
    """
    Find existing issue for a role.

    Convenience function that creates/reuses a module-level helper.

    Args:
        role_name: Name of the role
        state: Issue state filter (default: 'all')
        db: Optional StateDB instance

    Returns:
        Issue if found, None otherwise
    """
    helper = _get_helper(db)
    return helper.find_existing_issue(role_name, state=state)


def find_existing_mr(
    role_name: str, source_branch: str | None = None, state: str = "all", db: StateDB | None = None
) -> MergeRequest | None:
    """
    Find existing MR for a role.

    Convenience function that creates/reuses a module-level helper.

    Args:
        role_name: Name of the role
        source_branch: Source branch (default: sid/{role_name})
        state: MR state filter (default: 'all')
        db: Optional StateDB instance

    Returns:
        MergeRequest if found, None otherwise
    """
    helper = _get_helper(db)
    return helper.find_existing_mr(role_name, source_branch=source_branch, state=state)


def find_all_role_artifacts(
    role_name: str, worktree_base_path: str = "..", db: StateDB | None = None
) -> RoleArtifacts:
    """
    Find all artifacts for a role.

    Convenience function that creates/reuses a module-level helper.

    Args:
        role_name: Name of the role
        worktree_base_path: Base path for worktrees
        db: Optional StateDB instance

    Returns:
        RoleArtifacts with all discovered artifacts
    """
    helper = _get_helper(db)
    return helper.find_all_role_artifacts(role_name, worktree_base_path=worktree_base_path)
