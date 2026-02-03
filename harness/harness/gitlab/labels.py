"""
Label and milestone management for GitLab.

Provides dedicated managers for:
- Label creation and validation (including scoped labels)
- Milestone management and issue assignment
- Wave label helpers for DAG workflow organization

These classes operate independently of the main GitLabClient and can be
used for focused label/milestone operations with caching for efficiency.
"""

import json
import logging
import subprocess
import urllib.parse
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# WAVE LABEL CONFIGURATION
# =============================================================================

# Wave labels for DAG workflow organization (waves 0-8 to match molecule-sequential.sh)
WAVE_LABELS: list[str] = [
    "wave-0",
    "wave-1",
    "wave-2",
    "wave-3",
    "wave-4",
    "wave-5",
    "wave-6",
    "wave-7",
    "wave-8",
]

# Wave label colors with progression from cool to warm
WAVE_LABEL_COLORS: dict[str, str] = {
    "wave-0": "#0033CC",  # Deep blue - Foundation
    "wave-1": "#0066FF",  # Blue - Core infrastructure
    "wave-2": "#00CC66",  # Green - Core application
    "wave-3": "#66CC00",  # Yellow-green - Extended application
    "wave-4": "#FFCC00",  # Yellow - Integration
    "wave-5": "#FF9900",  # Orange - Advanced features
    "wave-6": "#FF6600",  # Dark orange - Utilities
    "wave-7": "#FF3300",  # Red-orange - Final stages
    "wave-8": "#CC0033",  # Red - Completion
}

# Wave label descriptions
WAVE_LABEL_DESCRIPTIONS: dict[str, str] = {
    "wave-0": "Foundation layer roles with no dependencies",
    "wave-1": "Core infrastructure roles",
    "wave-2": "Core application roles",
    "wave-3": "Extended application roles",
    "wave-4": "Integration and orchestration roles",
    "wave-5": "Advanced feature roles",
    "wave-6": "Utility and support roles",
    "wave-7": "Final deployment stages",
    "wave-8": "Completion and verification roles",
}

# Default label color
DEFAULT_LABEL_COLOR = "#428BCA"


class LabelManager:
    """
    Manages GitLab project labels with caching and batch operations.

    Features:
    - Local cache to minimize API calls
    - Batch label creation
    - Scoped label validation (e.g., priority::high)
    - Idempotent operations
    """

    def __init__(self, project_id: str, token: str | None = None):
        """
        Initialize the LabelManager.

        Args:
            project_id: GitLab project path (e.g., 'bates-ils/projects/ems/ems-mono')
            token: GitLab API token (optional, uses glab auth if not provided)
        """
        self.project_id = project_id
        self._token = token
        self._label_cache: dict[str, bool] = {}  # Cache for existence checks

    @property
    def project_id_encoded(self) -> str:
        """URL-encoded project ID for API calls."""
        return self.project_id.replace("/", "%2F")

    def _api_get(self, endpoint: str) -> dict[str, Any] | list[dict]:
        """Make authenticated GET request to GitLab API using glab."""
        result = subprocess.run(["glab", "api", endpoint], capture_output=True, text=True)
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

    def _api_put(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated PUT request to GitLab API using glab."""
        field_args = []
        for key, value in data.items():
            field_args.extend(["-f", f"{key}={value}"])

        result = subprocess.run(
            ["glab", "api", endpoint, "--method", "PUT"] + field_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"API PUT failed: {result.stderr}")
        return json.loads(result.stdout)

    def clear_cache(self) -> None:
        """Clear the label existence cache."""
        self._label_cache.clear()

    def _label_exists_cached(self, name: str) -> bool | None:
        """Check cache for label existence. Returns None if not cached."""
        return self._label_cache.get(name)

    def _cache_label(self, name: str, exists: bool) -> None:
        """Add label existence to cache."""
        self._label_cache[name] = exists

    async def ensure_labels_exist(self, labels: list[str]) -> list[str]:
        """
        Batch check/create labels in a single session.

        This method efficiently processes multiple labels, using the cache
        to minimize API calls and creating any missing labels.

        Args:
            labels: List of label names to ensure exist

        Returns:
            List of label names that were created (not already existing)
        """
        created_labels: list[str] = []

        for label in labels:
            # Check cache first
            cached = self._label_exists_cached(label)
            if cached is True:
                continue

            # Check if it's a scoped label and validate format
            if "::" in label and not self.validate_scoped_label(label):
                logger.warning(f"Invalid scoped label format: {label}")
                continue

            # Determine color and description
            color = DEFAULT_LABEL_COLOR
            description = ""

            if label in WAVE_LABEL_COLORS:
                color = WAVE_LABEL_COLORS[label]
                description = WAVE_LABEL_DESCRIPTIONS.get(label, "")

            # Try to get or create the label
            result = await self.get_or_create_label(label, color, description)
            if result and result.get("_created"):
                created_labels.append(label)

        return created_labels

    async def get_or_create_label(
        self, name: str, color: str = DEFAULT_LABEL_COLOR, description: str = ""
    ) -> dict[str, Any]:
        """
        Get existing label or create if not exists.

        Uses cache for efficiency, falling back to API if not cached.

        Args:
            name: Label name
            color: Hex color code (e.g., '#428BCA')
            description: Optional label description

        Returns:
            Label data dict with '_created' key indicating if newly created
        """
        # Check cache first
        cached = self._label_exists_cached(name)
        if cached is True:
            return {"name": name, "_created": False}

        # Check API for existing label
        try:
            encoded_name = urllib.parse.quote(name, safe="")
            label_data = self._api_get(
                f"projects/{self.project_id_encoded}/labels/{encoded_name}"
            )
            self._cache_label(name, True)
            label_data["_created"] = False
            return label_data
        except RuntimeError as e:
            if "404" not in str(e) and "not found" not in str(e).lower():
                raise

        # Label doesn't exist, create it
        try:
            data = {"name": name, "color": color}
            if description:
                data["description"] = description

            result = self._api_post(f"projects/{self.project_id_encoded}/labels", data)
            self._cache_label(name, True)
            result["_created"] = True
            logger.info(f"Created label '{name}' with color {color}")
            return result
        except RuntimeError as e:
            # Handle race condition: label was created between check and create
            if "already exists" in str(e).lower() or "has already been taken" in str(e).lower():
                self._cache_label(name, True)
                return {"name": name, "_created": False}
            raise

    def validate_scoped_label(self, label: str) -> bool:
        """
        Check if a scoped label has valid format.

        Scoped labels in GitLab use '::' as separator (e.g., 'priority::high').
        This creates mutually exclusive label groups within the scope.

        Args:
            label: Label string to validate

        Returns:
            True if valid scoped label format, False otherwise

        Examples:
            >>> validate_scoped_label("priority::high")
            True
            >>> validate_scoped_label("priority:high")  # Single colon - invalid
            False
            >>> validate_scoped_label("simple-label")  # Not scoped - not validated here
            False
        """
        if "::" not in label:
            return False

        parts = label.split("::")
        if len(parts) != 2:
            return False

        scope, value = parts
        # Both scope and value must be non-empty
        if not scope or not value:
            return False

        # Check for invalid characters (basic validation)
        # GitLab labels allow most characters, but scope/value shouldn't be empty or whitespace
        if scope.strip() != scope or value.strip() != value:
            return False

        return True

    async def ensure_wave_labels_exist(self) -> None:
        """
        Ensure all wave labels (wave-0 through wave-8) exist with proper colors.

        This is a convenience method for initializing all wave labels at once.
        """
        await self.ensure_labels_exist(WAVE_LABELS)
        logger.info(f"Ensured {len(WAVE_LABELS)} wave labels exist")


class MilestoneManager:
    """
    Manages GitLab project milestones.

    Features:
    - Find milestones by title
    - Create milestones with dates
    - Assign issues to milestones
    """

    def __init__(self, project_id: str, token: str | None = None):
        """
        Initialize the MilestoneManager.

        Args:
            project_id: GitLab project path (e.g., 'bates-ils/projects/ems/ems-mono')
            token: GitLab API token (optional, uses glab auth if not provided)
        """
        self.project_id = project_id
        self._token = token

    @property
    def project_id_encoded(self) -> str:
        """URL-encoded project ID for API calls."""
        return self.project_id.replace("/", "%2F")

    def _api_get(self, endpoint: str) -> dict[str, Any] | list[dict]:
        """Make authenticated GET request to GitLab API using glab."""
        result = subprocess.run(["glab", "api", endpoint], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"API GET failed: {result.stderr}")
        return json.loads(result.stdout)

    def _api_post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated POST request to GitLab API using glab."""
        field_args = []
        for key, value in data.items():
            if value is not None:
                field_args.extend(["-f", f"{key}={value}"])

        result = subprocess.run(
            ["glab", "api", endpoint, "--method", "POST"] + field_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"API POST failed: {result.stderr}")
        return json.loads(result.stdout)

    def _api_put(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated PUT request to GitLab API using glab."""
        field_args = []
        for key, value in data.items():
            if value is not None:
                field_args.extend(["-f", f"{key}={value}"])

        result = subprocess.run(
            ["glab", "api", endpoint, "--method", "PUT"] + field_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"API PUT failed: {result.stderr}")
        return json.loads(result.stdout)

    async def find_milestone(self, title: str) -> dict[str, Any] | None:
        """
        Find a milestone by its title.

        Args:
            title: Milestone title to search for

        Returns:
            Milestone data dict or None if not found
        """
        try:
            # URL encode the search term
            encoded_title = urllib.parse.quote(title, safe="")
            milestones = self._api_get(
                f"projects/{self.project_id_encoded}/milestones?search={encoded_title}"
            )

            if not milestones:
                return None

            # Find exact match
            for milestone in milestones:
                if milestone.get("title") == title:
                    return milestone

            return None
        except RuntimeError as e:
            logger.error(f"Failed to find milestone '{title}': {e}")
            return None

    async def ensure_milestone_exists(
        self,
        title: str,
        start_date: str | None = None,
        due_date: str | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        """
        Ensure a milestone exists, creating if necessary.

        Args:
            title: Milestone title
            start_date: Start date in YYYY-MM-DD format (optional)
            due_date: Due date in YYYY-MM-DD format (optional)
            description: Milestone description (optional)

        Returns:
            Milestone data dict (existing or newly created)

        Raises:
            RuntimeError: If milestone creation fails
        """
        # Check if milestone already exists
        existing = await self.find_milestone(title)
        if existing:
            logger.debug(f"Milestone '{title}' already exists (id: {existing.get('id')})")
            return existing

        # Create new milestone
        data: dict[str, Any] = {"title": title}
        if start_date:
            data["start_date"] = start_date
        if due_date:
            data["due_date"] = due_date
        if description:
            data["description"] = description

        try:
            milestone = self._api_post(f"projects/{self.project_id_encoded}/milestones", data)
            logger.info(f"Created milestone '{title}' (id: {milestone.get('id')})")
            return milestone
        except RuntimeError as e:
            # Handle race condition
            if "already exists" in str(e).lower():
                existing = await self.find_milestone(title)
                if existing:
                    return existing
            raise

    async def assign_issue_to_milestone(self, issue_iid: int, milestone_id: int) -> bool:
        """
        Assign an issue to a milestone.

        Args:
            issue_iid: Project-scoped issue IID
            milestone_id: Milestone ID to assign

        Returns:
            True if assignment succeeded, False otherwise
        """
        try:
            self._api_put(
                f"projects/{self.project_id_encoded}/issues/{issue_iid}",
                {"milestone_id": milestone_id},
            )
            logger.info(f"Assigned issue #{issue_iid} to milestone {milestone_id}")
            return True
        except RuntimeError as e:
            logger.error(f"Failed to assign issue #{issue_iid} to milestone: {e}")
            return False

    async def get_milestone_issues(
        self, milestone_id: int, state: str = "all"
    ) -> list[dict[str, Any]]:
        """
        Get all issues in a milestone.

        Args:
            milestone_id: Milestone ID
            state: Issue state filter ('opened', 'closed', 'all')

        Returns:
            List of issue data dicts
        """
        try:
            issues = self._api_get(
                f"projects/{self.project_id_encoded}/milestones/{milestone_id}/issues?state={state}"
            )
            return issues if isinstance(issues, list) else []
        except RuntimeError as e:
            logger.error(f"Failed to get issues for milestone {milestone_id}: {e}")
            return []

    async def close_milestone(self, milestone_id: int) -> bool:
        """
        Close a milestone.

        Args:
            milestone_id: Milestone ID to close

        Returns:
            True if successfully closed, False otherwise
        """
        try:
            self._api_put(
                f"projects/{self.project_id_encoded}/milestones/{milestone_id}",
                {"state_event": "close"},
            )
            logger.info(f"Closed milestone {milestone_id}")
            return True
        except RuntimeError as e:
            logger.error(f"Failed to close milestone {milestone_id}: {e}")
            return False
