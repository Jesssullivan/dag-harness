"""Wave-based role queue for HOTL mode.

Manages a queue of Ansible roles to process through box-up-role workflows,
ordered by wave number with configurable concurrency limits.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from harness.db.state import StateDB

logger = logging.getLogger(__name__)


class QueueItemStatus(Enum):
    """Status of a queued role."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # Awaiting human approval
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class QueueItem:
    """A role queued for processing."""

    role_name: str
    wave: int
    status: QueueItemStatus = QueueItemStatus.PENDING
    execution_id: Optional[int] = None
    issue_iid: Optional[int] = None
    mr_iid: Optional[int] = None
    error: Optional[str] = None


class RoleQueue:
    """Manages a queue of roles to process, ordered by wave.

    Roles are processed in wave order (lower waves first). Within a wave,
    roles can run concurrently up to max_concurrent. A wave must fully
    complete (all items completed, failed, or skipped) before the next
    wave begins.

    Args:
        db: StateDB instance for looking up role metadata.
        max_concurrent: Maximum number of roles to process concurrently.
    """

    def __init__(self, db: StateDB, max_concurrent: int = 2):
        self._db = db
        self._max_concurrent = max_concurrent
        self._items: list[QueueItem] = []

    @property
    def items(self) -> list[QueueItem]:
        """Read-only access to queue items."""
        return list(self._items)

    @property
    def max_concurrent(self) -> int:
        """Maximum concurrent processing limit."""
        return self._max_concurrent

    def add_wave(self, wave: int) -> int:
        """Add all roles in a wave to the queue.

        Looks up roles for the given wave number from the database and
        adds them to the queue. Roles already in the queue are skipped.

        Args:
            wave: Wave number to add roles from.

        Returns:
            Count of roles added.
        """
        roles = self._db.list_roles(wave=wave)
        existing_names = {item.role_name for item in self._items}
        added = 0

        for role in roles:
            if role.name not in existing_names:
                item = QueueItem(role_name=role.name, wave=role.wave)
                self._items.append(item)
                added += 1
                logger.info(f"Queued role '{role.name}' (wave {role.wave})")

        # Keep items sorted by wave then name
        self._items.sort(key=lambda i: (i.wave, i.role_name))
        logger.info(f"Added {added} roles from wave {wave}")
        return added

    def add_role(self, role_name: str) -> QueueItem:
        """Add a single role to the queue.

        Looks up the role from the database to determine its wave number.
        If the role is already queued, returns the existing item.

        Args:
            role_name: Name of the Ansible role.

        Returns:
            The QueueItem for the role.

        Raises:
            ValueError: If the role is not found in the database.
        """
        # Check if already queued
        for item in self._items:
            if item.role_name == role_name:
                return item

        role = self._db.get_role(role_name)
        if role is None:
            raise ValueError(f"Role '{role_name}' not found in database")

        item = QueueItem(role_name=role.name, wave=role.wave)
        self._items.append(item)

        # Keep items sorted by wave then name
        self._items.sort(key=lambda i: (i.wave, i.role_name))
        logger.info(f"Queued role '{role.name}' (wave {role.wave})")
        return item

    def get_next_batch(self) -> list[QueueItem]:
        """Get next batch of roles to process.

        Returns up to max_concurrent PENDING items from the lowest
        incomplete wave. Respects wave ordering: all items in a wave
        must be terminal (completed/failed/skipped) before the next
        wave's items are returned.

        Items currently RUNNING or PAUSED count against the concurrency
        limit.

        Returns:
            List of QueueItems to start processing.
        """
        if not self._items:
            return []

        # Find the lowest wave that still has non-terminal items
        active_wave = None
        for item in self._items:
            if item.status not in (
                QueueItemStatus.COMPLETED,
                QueueItemStatus.FAILED,
                QueueItemStatus.SKIPPED,
            ):
                active_wave = item.wave
                break

        if active_wave is None:
            return []

        # Count currently running/paused items in this wave
        in_progress_count = sum(
            1
            for item in self._items
            if item.wave == active_wave
            and item.status in (QueueItemStatus.RUNNING, QueueItemStatus.PAUSED)
        )

        # How many more can we start?
        available_slots = self._max_concurrent - in_progress_count
        if available_slots <= 0:
            return []

        # Get pending items from the active wave
        batch = []
        for item in self._items:
            if item.wave == active_wave and item.status == QueueItemStatus.PENDING:
                batch.append(item)
                if len(batch) >= available_slots:
                    break

        return batch

    def update_status(self, role_name: str, status: QueueItemStatus, **kwargs) -> None:
        """Update status of a queued role.

        Args:
            role_name: Name of the role to update.
            status: New status.
            **kwargs: Additional fields to update (execution_id, issue_iid,
                mr_iid, error).

        Raises:
            KeyError: If the role is not in the queue.
        """
        for item in self._items:
            if item.role_name == role_name:
                item.status = status
                for key, value in kwargs.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                logger.info(f"Role '{role_name}' status -> {status.value}")
                return

        raise KeyError(f"Role '{role_name}' not in queue")

    def get_paused(self) -> list[QueueItem]:
        """Get all roles paused at human approval.

        Returns:
            List of QueueItems with PAUSED status.
        """
        return [item for item in self._items if item.status == QueueItemStatus.PAUSED]

    def get_status(self) -> dict:
        """Get queue status summary.

        Returns:
            Dict with counts by status and overall progress info.
        """
        status_counts = {}
        for status in QueueItemStatus:
            status_counts[status.value] = sum(
                1 for item in self._items if item.status == status
            )

        waves = sorted({item.wave for item in self._items}) if self._items else []

        # Determine active wave
        active_wave = None
        for item in self._items:
            if item.status not in (
                QueueItemStatus.COMPLETED,
                QueueItemStatus.FAILED,
                QueueItemStatus.SKIPPED,
            ):
                active_wave = item.wave
                break

        return {
            "total": len(self._items),
            "max_concurrent": self._max_concurrent,
            "waves": waves,
            "active_wave": active_wave,
            "complete": self.is_complete(),
            **status_counts,
        }

    def is_complete(self) -> bool:
        """True if all items are completed, failed, or skipped.

        An empty queue is considered complete.
        """
        if not self._items:
            return True

        terminal_statuses = {
            QueueItemStatus.COMPLETED,
            QueueItemStatus.FAILED,
            QueueItemStatus.SKIPPED,
        }
        return all(item.status in terminal_statuses for item in self._items)
