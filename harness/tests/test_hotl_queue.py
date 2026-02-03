"""Tests for HOTL wave-based role queue."""

import pytest

from harness.db.models import Role
from harness.db.state import StateDB
from harness.hotl.queue import QueueItem, QueueItemStatus, RoleQueue


@pytest.fixture
def queue_db(in_memory_db: StateDB) -> StateDB:
    """Database with roles across multiple waves for queue testing."""
    roles = [
        # Wave 1
        Role(name="common", wave=1, has_molecule_tests=True),
        Role(name="windows_prerequisites", wave=1, has_molecule_tests=True),
        Role(name="iis_config", wave=1, has_molecule_tests=False),
        # Wave 2
        Role(name="ems_platform_services", wave=2, has_molecule_tests=True),
        Role(name="ems_download_artifacts", wave=2, has_molecule_tests=True),
        Role(name="database_clone", wave=2, has_molecule_tests=True),
        # Wave 3
        Role(name="ems_web_app", wave=3, has_molecule_tests=True),
        Role(name="ems_master_calendar", wave=3, has_molecule_tests=False),
    ]
    for role in roles:
        in_memory_db.upsert_role(role)
    return in_memory_db


class TestAddWave:
    """Tests for RoleQueue.add_wave()."""

    def test_add_wave(self, queue_db: StateDB):
        """Adding a wave populates the queue with all roles in that wave."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        count = queue.add_wave(1)

        assert count == 3
        assert len(queue.items) == 3
        names = {item.role_name for item in queue.items}
        assert names == {"common", "windows_prerequisites", "iis_config"}

    def test_add_wave_idempotent(self, queue_db: StateDB):
        """Adding the same wave twice does not duplicate items."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)
        count = queue.add_wave(1)

        assert count == 0
        assert len(queue.items) == 3

    def test_add_multiple_waves(self, queue_db: StateDB):
        """Adding multiple waves populates items from all waves."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)
        queue.add_wave(2)

        assert len(queue.items) == 6

    def test_add_empty_wave(self, queue_db: StateDB):
        """Adding a wave with no roles returns 0."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        count = queue.add_wave(99)

        assert count == 0
        assert len(queue.items) == 0


class TestAddRole:
    """Tests for RoleQueue.add_role()."""

    def test_add_role(self, queue_db: StateDB):
        """Adding a single role creates a QueueItem with correct metadata."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        item = queue.add_role("common")

        assert item.role_name == "common"
        assert item.wave == 1
        assert item.status == QueueItemStatus.PENDING
        assert len(queue.items) == 1

    def test_add_role_idempotent(self, queue_db: StateDB):
        """Adding the same role twice returns the existing item."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        item1 = queue.add_role("common")
        item2 = queue.add_role("common")

        assert item1 is item2
        assert len(queue.items) == 1

    def test_add_role_not_found(self, queue_db: StateDB):
        """Adding a role that doesn't exist raises ValueError."""
        queue = RoleQueue(queue_db, max_concurrent=2)

        with pytest.raises(ValueError, match="not found"):
            queue.add_role("nonexistent_role")


class TestGetNextBatch:
    """Tests for RoleQueue.get_next_batch()."""

    def test_get_next_batch(self, queue_db: StateDB):
        """Returns up to max_concurrent pending items from the lowest wave."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)
        queue.add_wave(2)

        batch = queue.get_next_batch()

        assert len(batch) == 2
        # All items should be from wave 1
        assert all(item.wave == 1 for item in batch)

    def test_get_next_batch_empty_queue(self, queue_db: StateDB):
        """Returns empty list for empty queue."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        batch = queue.get_next_batch()

        assert batch == []

    def test_get_next_batch_respects_running(self, queue_db: StateDB):
        """Running items count against the concurrency limit."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)

        # Mark one as running
        queue.update_status("common", QueueItemStatus.RUNNING)

        batch = queue.get_next_batch()

        # Should only return 1 more (2 max - 1 running = 1 available)
        assert len(batch) == 1
        assert batch[0].role_name != "common"

    def test_get_next_batch_all_complete(self, queue_db: StateDB):
        """Returns empty list when all items are terminal."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_role("common")
        queue.update_status("common", QueueItemStatus.COMPLETED)

        batch = queue.get_next_batch()

        assert batch == []


class TestWaveOrdering:
    """Tests for wave-based ordering."""

    def test_wave_ordering(self, queue_db: StateDB):
        """Wave 1 roles are returned before wave 2 roles."""
        queue = RoleQueue(queue_db, max_concurrent=10)
        queue.add_wave(2)
        queue.add_wave(1)

        batch = queue.get_next_batch()

        # All returned items should be wave 1 even though we added wave 2 first
        assert all(item.wave == 1 for item in batch)

    def test_wave_gate(self, queue_db: StateDB):
        """Wave 2 items are not returned until wave 1 is fully terminal."""
        queue = RoleQueue(queue_db, max_concurrent=10)
        queue.add_wave(1)
        queue.add_wave(2)

        # Complete two of three wave 1 items
        queue.update_status("common", QueueItemStatus.COMPLETED)
        queue.update_status("windows_prerequisites", QueueItemStatus.COMPLETED)

        batch = queue.get_next_batch()

        # Still wave 1 (iis_config is pending)
        assert len(batch) == 1
        assert batch[0].role_name == "iis_config"
        assert batch[0].wave == 1

    def test_wave_gate_opens(self, queue_db: StateDB):
        """Wave 2 items appear once wave 1 is fully terminal."""
        queue = RoleQueue(queue_db, max_concurrent=10)
        queue.add_wave(1)
        queue.add_wave(2)

        # Complete all wave 1 items
        queue.update_status("common", QueueItemStatus.COMPLETED)
        queue.update_status("windows_prerequisites", QueueItemStatus.FAILED)
        queue.update_status("iis_config", QueueItemStatus.SKIPPED)

        batch = queue.get_next_batch()

        # Now we should get wave 2 items
        assert all(item.wave == 2 for item in batch)
        assert len(batch) == 3


class TestUpdateStatus:
    """Tests for RoleQueue.update_status()."""

    def test_update_status(self, queue_db: StateDB):
        """Status transitions update the item correctly."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_role("common")

        queue.update_status("common", QueueItemStatus.RUNNING)

        item = queue.items[0]
        assert item.status == QueueItemStatus.RUNNING

    def test_update_status_with_kwargs(self, queue_db: StateDB):
        """Extra kwargs update corresponding fields."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_role("common")

        queue.update_status(
            "common",
            QueueItemStatus.FAILED,
            error="Connection timeout",
            execution_id=42,
        )

        item = queue.items[0]
        assert item.status == QueueItemStatus.FAILED
        assert item.error == "Connection timeout"
        assert item.execution_id == 42

    def test_update_status_not_found(self, queue_db: StateDB):
        """Updating a role not in queue raises KeyError."""
        queue = RoleQueue(queue_db, max_concurrent=2)

        with pytest.raises(KeyError, match="not in queue"):
            queue.update_status("nonexistent", QueueItemStatus.RUNNING)


class TestGetPaused:
    """Tests for RoleQueue.get_paused()."""

    def test_get_paused(self, queue_db: StateDB):
        """Returns only paused items."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)

        queue.update_status("common", QueueItemStatus.PAUSED)
        queue.update_status("windows_prerequisites", QueueItemStatus.RUNNING)

        paused = queue.get_paused()

        assert len(paused) == 1
        assert paused[0].role_name == "common"

    def test_get_paused_empty(self, queue_db: StateDB):
        """Returns empty list when nothing is paused."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)

        paused = queue.get_paused()

        assert paused == []


class TestIsComplete:
    """Tests for RoleQueue.is_complete()."""

    def test_is_complete_empty(self, queue_db: StateDB):
        """Empty queue is considered complete."""
        queue = RoleQueue(queue_db, max_concurrent=2)

        assert queue.is_complete() is True

    def test_is_complete_all_done(self, queue_db: StateDB):
        """True when all items are in terminal states."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)

        queue.update_status("common", QueueItemStatus.COMPLETED)
        queue.update_status("windows_prerequisites", QueueItemStatus.FAILED)
        queue.update_status("iis_config", QueueItemStatus.SKIPPED)

        assert queue.is_complete() is True

    def test_is_complete_not_done(self, queue_db: StateDB):
        """False when any item is not in a terminal state."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)

        queue.update_status("common", QueueItemStatus.COMPLETED)
        # windows_prerequisites still PENDING, iis_config still PENDING

        assert queue.is_complete() is False


class TestMaxConcurrent:
    """Tests for max_concurrent enforcement."""

    def test_max_concurrent(self, queue_db: StateDB):
        """Never returns more items than max_concurrent."""
        queue = RoleQueue(queue_db, max_concurrent=1)
        queue.add_wave(1)  # 3 roles in wave 1

        batch = queue.get_next_batch()

        assert len(batch) == 1

    def test_max_concurrent_with_running(self, queue_db: StateDB):
        """Already running items reduce available slots."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)

        # Mark 2 as running (at max)
        queue.update_status("common", QueueItemStatus.RUNNING)
        queue.update_status("windows_prerequisites", QueueItemStatus.RUNNING)

        batch = queue.get_next_batch()

        assert len(batch) == 0

    def test_max_concurrent_large(self, queue_db: StateDB):
        """When max_concurrent exceeds available items, returns all available."""
        queue = RoleQueue(queue_db, max_concurrent=100)
        queue.add_wave(1)  # Only 3 roles

        batch = queue.get_next_batch()

        assert len(batch) == 3


class TestGetStatus:
    """Tests for RoleQueue.get_status()."""

    def test_get_status(self, queue_db: StateDB):
        """Status summary includes counts and metadata."""
        queue = RoleQueue(queue_db, max_concurrent=2)
        queue.add_wave(1)
        queue.update_status("common", QueueItemStatus.COMPLETED)

        status = queue.get_status()

        assert status["total"] == 3
        assert status["completed"] == 1
        assert status["pending"] == 2
        assert status["max_concurrent"] == 2
        assert status["waves"] == [1]
        assert status["active_wave"] == 1
        assert status["complete"] is False
