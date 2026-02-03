"""
Dependency-aware merge request ordering for merge trains.

Provides intelligent MR ordering based on role dependencies to ensure
dependent changes are merged in the correct order.

Features:
- Wave-based merge train placement
- Dependency-aware ordering analysis
- Ordering violation detection
- Merge train position estimation
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from harness.db.state import StateDB

logger = logging.getLogger(__name__)


@dataclass
class PlacementResult:
    """
    Result of a merge train placement operation.

    Attributes:
        success: Whether the operation succeeded (MR was added or dry_run completed)
        position: Actual position in train (1-indexed) if added, None if failed
        blocking_mrs: List of MR IIDs ahead in the train that block this one
        blocked_by: List of role names this MR's role depends on
        estimated_merge_time: Estimated time until this MR would merge
        warnings: List of ordering violation warnings (e.g., "Should be before MR #123")
        dry_run: Whether this was a dry run (no actual train modification)
    """

    success: bool
    position: int | None
    blocking_mrs: list[int] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    estimated_merge_time: timedelta | None = None
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False


@dataclass
class MergeOrder:
    """
    Represents the merge order information for a single MR.

    Attributes:
        mr_iid: The merge request IID (project-scoped identifier)
        role_name: Name of the Ansible role associated with this MR
        wave: The wave number the role belongs to (0-8)
        dependency_depth: How many dependency levels deep (0 = no deps)
        suggested_position: Suggested position in merge train queue
        blocking_mrs: List of MR IIDs that should merge before this one
    """

    mr_iid: int
    role_name: str
    wave: int
    dependency_depth: int
    suggested_position: int
    blocking_mrs: list[int] = field(default_factory=list)


class MergeOrderingService:
    """
    Service for dependency-aware MR ordering in merge trains.

    Uses role dependency information from StateDB to determine optimal
    merge order, ensuring that roles with dependencies are merged after
    their dependencies.

    Example:
        >>> db = StateDB("harness.db")
        >>> service = MergeOrderingService(db)
        >>> order = service.calculate_merge_order([101, 102, 103])
        >>> for mo in order:
        ...     print(f"MR !{mo.mr_iid} ({mo.role_name}): position {mo.suggested_position}")
    """

    def __init__(
        self,
        db: StateDB,
        project_path: str = "bates-ils/projects/ems/ems-mono",
    ):
        """
        Initialize the MergeOrderingService.

        Args:
            db: StateDB instance for accessing role and dependency information
            project_path: GitLab project path for API calls
        """
        self.db = db
        self.project_path = project_path
        self._project_path_encoded = project_path.replace("/", "%2F")
        self._mr_to_role_cache: dict[int, str | None] = {}
        self._role_to_mr_cache: dict[str, int | None] = {}

    def _get_role_for_mr(self, mr_iid: int) -> str | None:
        """
        Get the role name associated with an MR.

        Args:
            mr_iid: Merge request IID

        Returns:
            Role name if found, None otherwise
        """
        if mr_iid in self._mr_to_role_cache:
            return self._mr_to_role_cache[mr_iid]

        # Query the database for MR -> role mapping
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT r.name
                FROM merge_requests mr
                JOIN roles r ON mr.role_id = r.id
                WHERE mr.iid = ?
                ORDER BY mr.created_at DESC
                LIMIT 1
                """,
                (mr_iid,),
            ).fetchone()

            role_name = row["name"] if row else None
            self._mr_to_role_cache[mr_iid] = role_name
            return role_name

    def _get_mr_for_role(self, role_name: str) -> int | None:
        """
        Get the most recent open MR IID for a role.

        Args:
            role_name: Name of the role

        Returns:
            MR IID if found, None otherwise
        """
        if role_name in self._role_to_mr_cache:
            return self._role_to_mr_cache[role_name]

        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT mr.iid
                FROM merge_requests mr
                JOIN roles r ON mr.role_id = r.id
                WHERE r.name = ? AND mr.state = 'opened'
                ORDER BY mr.created_at DESC
                LIMIT 1
                """,
                (role_name,),
            ).fetchone()

            mr_iid = row["iid"] if row else None
            self._role_to_mr_cache[role_name] = mr_iid
            return mr_iid

    def get_dependency_depth(self, role_name: str) -> int:
        """
        Calculate how many levels deep in the dependency chain a role is.

        A role with no dependencies has depth 0. A role that depends on
        roles with no dependencies has depth 1, and so on.

        Args:
            role_name: Name of the role

        Returns:
            Dependency depth (0 = no dependencies)
        """
        deps = self.db.get_dependencies(role_name, transitive=True)
        if not deps:
            return 0

        # The depth is the maximum depth of any dependency
        return max(depth for _, depth in deps)

    def get_blocking_mrs(self, mr_iid: int) -> list[int]:
        """
        Find MRs for roles that this MR depends on.

        These are the MRs that should be merged before this one
        to maintain proper dependency order.

        Args:
            mr_iid: Merge request IID

        Returns:
            List of MR IIDs that should merge before this one
        """
        role_name = self._get_role_for_mr(mr_iid)
        if not role_name:
            return []

        # Get transitive dependencies
        deps = self.db.get_dependencies(role_name, transitive=True)
        blocking = []

        for dep_role, _ in deps:
            dep_mr = self._get_mr_for_role(dep_role)
            if dep_mr is not None and dep_mr != mr_iid:
                blocking.append(dep_mr)

        return sorted(set(blocking))

    def _topological_sort(self, mr_iids: list[int]) -> list[int]:
        """
        Perform topological sort on MRs based on role dependencies.

        Uses Kahn's algorithm for topological sorting with cycle detection.
        If cycles are detected, logs a warning and returns a partial order.

        Args:
            mr_iids: List of MR IIDs to sort

        Returns:
            Sorted list of MR IIDs (dependencies first)
        """
        if not mr_iids:
            return []

        # Build adjacency list and in-degree count
        # Edge from A to B means B depends on A (A should merge first)
        in_degree: dict[int, int] = {iid: 0 for iid in mr_iids}
        graph: dict[int, list[int]] = {iid: [] for iid in mr_iids}
        mr_set = set(mr_iids)

        for mr_iid in mr_iids:
            blocking = self.get_blocking_mrs(mr_iid)
            for blocker in blocking:
                if blocker in mr_set:
                    graph[blocker].append(mr_iid)
                    in_degree[mr_iid] += 1

        # Kahn's algorithm
        queue = [iid for iid, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by wave and mr_iid for consistent ordering within same level
            queue.sort(key=lambda iid: (self._get_wave_for_mr(iid), iid))
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(mr_iids):
            cycle_nodes = [iid for iid, degree in in_degree.items() if degree > 0]
            cycle_roles = [self._get_role_for_mr(iid) or f"MR!{iid}" for iid in cycle_nodes]
            logger.warning(
                f"Cyclic dependency detected among MRs: {cycle_roles}. "
                f"Returning partial order with remaining MRs appended."
            )
            # Append remaining nodes in wave/iid order
            remaining = sorted(
                cycle_nodes,
                key=lambda iid: (self._get_wave_for_mr(iid), iid)
            )
            result.extend(remaining)

        return result

    def _get_wave_for_mr(self, mr_iid: int) -> int:
        """Get the wave number for an MR's associated role."""
        role_name = self._get_role_for_mr(mr_iid)
        if not role_name:
            return 999  # Unknown roles go last

        role = self.db.get_role(role_name)
        return role.wave if role else 999

    def calculate_merge_order(self, mr_iids: list[int]) -> list[MergeOrder]:
        """
        Calculate the optimal merge order for a set of MRs.

        Orders MRs by:
        1. Wave number (ascending) - earlier waves merge first
        2. Dependency depth (ascending) - roles with fewer deps merge first
        3. MR IID (ascending) - tie-breaker for consistent ordering

        Args:
            mr_iids: List of merge request IIDs to order

        Returns:
            List of MergeOrder objects in suggested merge order
        """
        if not mr_iids:
            return []

        # Clear caches for fresh calculation
        self._mr_to_role_cache.clear()
        self._role_to_mr_cache.clear()

        # Pre-populate caches
        for mr_iid in mr_iids:
            self._get_role_for_mr(mr_iid)

        # Get topologically sorted order
        sorted_iids = self._topological_sort(mr_iids)

        # Build MergeOrder objects
        result = []
        for position, mr_iid in enumerate(sorted_iids, start=1):
            role_name = self._get_role_for_mr(mr_iid)
            if not role_name:
                # MR not associated with a role - put at end
                result.append(
                    MergeOrder(
                        mr_iid=mr_iid,
                        role_name="<unknown>",
                        wave=999,
                        dependency_depth=0,
                        suggested_position=position,
                        blocking_mrs=[],
                    )
                )
                continue

            role = self.db.get_role(role_name)
            wave = role.wave if role else 0
            depth = self.get_dependency_depth(role_name)
            blocking = [b for b in self.get_blocking_mrs(mr_iid) if b in mr_iids]

            result.append(
                MergeOrder(
                    mr_iid=mr_iid,
                    role_name=role_name,
                    wave=wave,
                    dependency_depth=depth,
                    suggested_position=position,
                    blocking_mrs=blocking,
                )
            )

        return result

    def suggest_train_position(self, mr_iid: int, current_train: list[dict]) -> int:
        """
        Calculate ideal position in current merge train for an MR.

        Analyzes the current merge train queue and suggests where to
        insert the given MR based on dependencies.

        Args:
            mr_iid: Merge request IID to position
            current_train: List of current merge train entries, each with
                          at minimum a 'merge_request' dict containing 'iid'

        Returns:
            Suggested 0-indexed position in the train (0 = first)
        """
        if not current_train:
            return 0

        # Get MRs currently in train
        train_iids = [
            entry.get("merge_request", {}).get("iid")
            for entry in current_train
            if entry.get("merge_request", {}).get("iid") is not None
        ]

        if not train_iids:
            return 0

        # Get blocking MRs that are in the train
        blocking = self.get_blocking_mrs(mr_iid)
        blocking_in_train = [b for b in blocking if b in train_iids]

        if not blocking_in_train:
            # No dependencies in train - can go at position 0
            # But prefer after same-wave MRs for consistency
            role_name = self._get_role_for_mr(mr_iid)
            if role_name:
                role = self.db.get_role(role_name)
                if role:
                    # Find position after all lower waves
                    for pos, train_iid in enumerate(train_iids):
                        train_role = self._get_role_for_mr(train_iid)
                        if train_role:
                            train_role_obj = self.db.get_role(train_role)
                            if train_role_obj and train_role_obj.wave > role.wave:
                                return pos
            return 0

        # Find the position of the last blocking MR
        last_blocker_pos = -1
        for blocker in blocking_in_train:
            try:
                pos = train_iids.index(blocker)
                last_blocker_pos = max(last_blocker_pos, pos)
            except ValueError:
                continue

        # Position should be after the last blocker
        return last_blocker_pos + 1

    # =========================================================================
    # API HELPERS
    # =========================================================================

    def _api_get(self, endpoint: str) -> dict[str, Any] | list:
        """Make authenticated GET request to GitLab API using glab."""
        result = subprocess.run(
            ["glab", "api", endpoint],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"API GET failed: {result.stderr}")
        return json.loads(result.stdout)

    def _api_post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated POST request to GitLab API using glab."""
        field_args = []
        for key, value in data.items():
            field_args.extend(["-f", f"{key}={value}"])

        result = subprocess.run(
            ["glab", "api", endpoint, "--method", "POST"] + field_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"API POST failed: {result.stderr}")
        return json.loads(result.stdout)

    # =========================================================================
    # MERGE TRAIN HELPERS
    # =========================================================================

    def _extract_role_from_branch(self, source_branch: str) -> str | None:
        """Extract role name from branch name (e.g., 'sid/common' -> 'common')."""
        if source_branch.startswith("sid/"):
            return source_branch[4:]
        return None

    def get_merge_train_from_api(self, target_branch: str = "main") -> list[dict]:
        """
        Fetch current merge train state from GitLab API.

        Args:
            target_branch: Target branch of the merge train

        Returns:
            List of merge train entry dicts from the API
        """
        try:
            return self._api_get(
                f"projects/{self._project_path_encoded}/merge_trains"
                f"?target_branch={target_branch}"
            )
        except RuntimeError as e:
            logger.warning(f"Failed to get merge train: {e}")
            return []

    def get_mr_details(self, mr_iid: int) -> dict[str, Any] | None:
        """
        Get merge request details from GitLab API.

        Args:
            mr_iid: Merge request IID

        Returns:
            MR data dict or None if not found
        """
        try:
            return self._api_get(
                f"projects/{self._project_path_encoded}/merge_requests/{mr_iid}"
            )
        except RuntimeError:
            return None

    # =========================================================================
    # ORDERING VIOLATION DETECTION
    # =========================================================================

    def check_ordering_violations(
        self, mr_iid: int, current_train: list[dict]
    ) -> list[str]:
        """
        Check if an MR should be merged before others already in the train.

        GitLab doesn't support repositioning MRs in the merge train, so this
        returns warnings when the ordering is suboptimal based on:
        - Wave hierarchy (lower waves should merge first)
        - Role dependencies (dependencies should merge before dependents)

        Args:
            mr_iid: MR IID to check
            current_train: Current merge train entries from GitLab API

        Returns:
            List of warning messages for ordering violations

        Example warnings:
            - "MR !123 (wave-2) should merge before !456 (wave-3)"
            - "MR !123 (common) is a dependency of !456 (ems_web_app)"
        """
        warnings = []

        # Get MR details
        mr_data = self.get_mr_details(mr_iid)
        if not mr_data:
            return warnings

        source_branch = mr_data.get("source_branch", "")
        mr_role = self._extract_role_from_branch(source_branch)
        if not mr_role:
            return warnings

        role_obj = self.db.get_role(mr_role)
        mr_wave = role_obj.wave if role_obj else None

        if mr_wave is None:
            return warnings

        # Check each entry in the train
        for entry in current_train:
            entry_mr = entry.get("merge_request", {})
            entry_iid = entry_mr.get("iid")
            entry_branch = entry_mr.get("source_branch", "")
            entry_role = self._extract_role_from_branch(entry_branch)

            if not entry_role:
                continue

            entry_role_obj = self.db.get_role(entry_role)
            entry_wave = entry_role_obj.wave if entry_role_obj else None

            # Check wave ordering: lower waves should merge first
            if entry_wave is not None and mr_wave < entry_wave:
                warnings.append(
                    f"MR !{mr_iid} (wave-{mr_wave}) should merge before "
                    f"!{entry_iid} (wave-{entry_wave})"
                )

            # Check dependency ordering: dependencies should merge first
            if entry_role:
                entry_deps = self.db.get_dependencies(entry_role, transitive=True)
                entry_dep_names = [name for name, _ in entry_deps]
                if mr_role in entry_dep_names:
                    warnings.append(
                        f"MR !{mr_iid} ({mr_role}) is a dependency of "
                        f"!{entry_iid} ({entry_role}) - should merge first"
                    )

        return warnings

    # =========================================================================
    # MERGE TRAIN PLACEMENT
    # =========================================================================

    def _estimate_merge_time(
        self,
        position: int,
        current_train: list[dict],
        avg_pipeline_duration: int = 600,
    ) -> timedelta:
        """
        Estimate time until an MR at a given position would merge.

        Args:
            position: 1-indexed position in the train
            current_train: Current train entries
            avg_pipeline_duration: Average pipeline duration in seconds (default: 10 min)

        Returns:
            Estimated timedelta until merge
        """
        entries_ahead = position - 1
        if entries_ahead <= 0:
            return timedelta(seconds=0)

        return timedelta(seconds=entries_ahead * avg_pipeline_duration)

    async def place_in_merge_train(
        self,
        project_path: str,
        mr_iid: int,
        dry_run: bool = False,
    ) -> PlacementResult:
        """
        Calculate ideal position and optionally add MR to merge train.

        This method:
        1. Gets the current merge train state from GitLab API
        2. Calculates the ideal position based on wave and dependencies
        3. Checks for ordering violations (MR should be before existing entries)
        4. If not dry_run, adds the MR to the merge train via API
        5. Returns comprehensive result with warnings

        Note: GitLab adds MRs to the END of the merge train. We cannot
        reposition entries, so warnings indicate suboptimal ordering.

        Args:
            project_path: GitLab project path
            mr_iid: Merge request IID to place
            dry_run: If True, only calculate placement without adding to train

        Returns:
            PlacementResult with position, warnings, and status
        """
        # Use provided project_path if different
        original_path = None
        if project_path != self.project_path:
            original_path = self.project_path
            original_encoded = self._project_path_encoded
            self.project_path = project_path
            self._project_path_encoded = project_path.replace("/", "%2F")

        try:
            # Get current merge train from API
            current_train = self.get_merge_train_from_api()

            # Check if MR is already in train
            for entry in current_train:
                entry_mr = entry.get("merge_request", {})
                if entry_mr.get("iid") == mr_iid:
                    train_iids = [
                        e.get("merge_request", {}).get("iid")
                        for e in current_train
                    ]
                    position = train_iids.index(mr_iid) + 1 if mr_iid in train_iids else None
                    return PlacementResult(
                        success=True,
                        position=position,
                        warnings=[f"MR !{mr_iid} is already in merge train at position {position}"],
                        dry_run=dry_run,
                    )

            # Get MR details for dependency analysis
            mr_data = self.get_mr_details(mr_iid)
            if not mr_data:
                return PlacementResult(
                    success=False,
                    position=None,
                    warnings=[f"MR !{mr_iid} not found"],
                    dry_run=dry_run,
                )

            # Check MR state
            if mr_data.get("state") != "opened":
                return PlacementResult(
                    success=False,
                    position=None,
                    warnings=[f"MR !{mr_iid} is not open (state: {mr_data.get('state')})"],
                    dry_run=dry_run,
                )

            # Extract role and get dependencies
            source_branch = mr_data.get("source_branch", "")
            mr_role = self._extract_role_from_branch(source_branch)
            blocked_by: list[str] = []
            if mr_role:
                deps = self.db.get_dependencies(mr_role, transitive=True)
                blocked_by = [name for name, _ in deps]

            # Find which MRs ahead in the train would block this one
            blocking_mrs = []
            for entry in current_train:
                entry_mr = entry.get("merge_request", {})
                entry_branch = entry_mr.get("source_branch", "")
                entry_role = self._extract_role_from_branch(entry_branch)
                if entry_role and entry_role in blocked_by:
                    blocking_mrs.append(entry_mr.get("iid"))

            # Calculate ideal position using existing method
            ideal_position = self.suggest_train_position(mr_iid, current_train) + 1  # Convert to 1-indexed

            # Check for ordering violations
            warnings = self.check_ordering_violations(mr_iid, current_train)

            # Actual position will be end of train (GitLab limitation)
            actual_position = len(current_train) + 1

            # Add warning if ideal position differs significantly
            if ideal_position < actual_position and not warnings:
                warnings.append(
                    f"Ideal position is {ideal_position}, but GitLab will place at {actual_position}"
                )

            # Estimate merge time
            estimated_merge_time = self._estimate_merge_time(actual_position, current_train)

            # If dry run, return without adding
            if dry_run:
                return PlacementResult(
                    success=True,
                    position=actual_position,
                    blocking_mrs=blocking_mrs,
                    blocked_by=blocked_by,
                    estimated_merge_time=estimated_merge_time,
                    warnings=warnings,
                    dry_run=True,
                )

            # Add to merge train via API
            try:
                self._api_post(
                    f"projects/{self._project_path_encoded}/merge_trains",
                    {
                        "merge_request_iid": str(mr_iid),
                        "when_pipeline_succeeds": "true",
                        "squash": "true",
                    },
                )
                logger.info(f"Added MR !{mr_iid} to merge train at position {actual_position}")

                return PlacementResult(
                    success=True,
                    position=actual_position,
                    blocking_mrs=blocking_mrs,
                    blocked_by=blocked_by,
                    estimated_merge_time=estimated_merge_time,
                    warnings=warnings,
                    dry_run=False,
                )

            except RuntimeError as e:
                error_msg = str(e)
                # Check for common errors
                if "merge_trains" in error_msg.lower() or "not enabled" in error_msg.lower():
                    warnings.append("Merge trains may not be enabled for this project")
                elif "cannot be merged" in error_msg.lower():
                    warnings.append("MR has merge conflicts or unmet requirements")
                else:
                    warnings.append(f"Failed to add to merge train: {error_msg}")

                return PlacementResult(
                    success=False,
                    position=None,
                    blocking_mrs=blocking_mrs,
                    blocked_by=blocked_by,
                    estimated_merge_time=None,
                    warnings=warnings,
                    dry_run=False,
                )

        finally:
            # Restore original project path if changed
            if original_path is not None:
                self.project_path = original_path
                self._project_path_encoded = original_encoded
