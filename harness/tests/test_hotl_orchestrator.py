"""Tests for HOTL orchestrator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.db.models import Role
from harness.db.state import StateDB
from harness.hotl.orchestrator import HOTLOrchestrator
from harness.hotl.queue import QueueItemStatus, RoleQueue


@pytest.fixture
def orch_db(in_memory_db: StateDB) -> StateDB:
    """Database with roles for orchestrator testing."""
    roles = [
        Role(name="common", wave=1, has_molecule_tests=True),
        Role(name="windows_prerequisites", wave=1, has_molecule_tests=True),
        Role(name="ems_platform_services", wave=2, has_molecule_tests=True),
        Role(name="database_clone", wave=2, has_molecule_tests=True),
    ]
    for role in roles:
        in_memory_db.upsert_role(role)
    return in_memory_db


@pytest.fixture
def orch_config() -> dict:
    """Configuration for orchestrator tests."""
    return {
        "repo_root": "/tmp/test-repo",
        "repo_python": "/usr/bin/python3",
    }


@pytest.fixture
def orchestrator(orch_db: StateDB, orch_config: dict) -> HOTLOrchestrator:
    """Create an orchestrator with a fresh queue."""
    queue = RoleQueue(orch_db, max_concurrent=2)
    return HOTLOrchestrator(orch_db, config=orch_config, queue=queue)


def _make_mock_graph(result_map: dict | None = None):
    """Create a mock graph that returns specified results per role.

    Args:
        result_map: Dict mapping role_name -> result dict. If None,
            all roles return completed status.
    """
    default_result = {"status": "completed"}

    mock_graph = MagicMock()

    async def mock_execute(role_name, **kwargs):
        if result_map and role_name in result_map:
            return result_map[role_name]
        return default_result

    mock_graph.execute = mock_execute
    return mock_graph


class TestStartProcessesQueue:
    """Tests for HOTLOrchestrator.start()."""

    @pytest.mark.asyncio
    async def test_start_processes_queue(self, orchestrator: HOTLOrchestrator):
        """Starting the orchestrator processes all queued roles."""
        mock_graph = _make_mock_graph()

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            await orchestrator.start(waves=[1], max_concurrent=2)

        # All wave 1 roles should be completed
        queue = orchestrator.queue
        assert queue.is_complete()
        completed = [
            i for i in queue.items if i.status == QueueItemStatus.COMPLETED
        ]
        assert len(completed) == 2  # common, windows_prerequisites

    @pytest.mark.asyncio
    async def test_start_adds_waves(self, orchestrator: HOTLOrchestrator):
        """Starting with waves param populates the queue from database."""
        mock_graph = _make_mock_graph()

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            await orchestrator.start(waves=[1, 2], max_concurrent=4)

        assert len(orchestrator.queue.items) == 4
        assert orchestrator.queue.is_complete()


class TestStopGraceful:
    """Tests for HOTLOrchestrator.stop()."""

    @pytest.mark.asyncio
    async def test_stop_graceful(self, orchestrator: HOTLOrchestrator):
        """Stop waits for current items to finish before returning."""
        # Use a slow mock to simulate processing time
        call_count = 0

        async def slow_execute(role_name, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return {"status": "completed"}

        mock_graph = MagicMock()
        mock_graph.execute = slow_execute

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            # Start in background
            task = asyncio.create_task(
                orchestrator.start(waves=[1], max_concurrent=1)
            )

            # Give it time to start processing
            await asyncio.sleep(0.02)

            # Request stop
            await orchestrator.stop()

            # Wait for task to finish
            await asyncio.wait_for(task, timeout=5.0)

        # Should have processed at least 1 role
        assert call_count >= 1
        assert not orchestrator.running


class TestApproveResumes:
    """Tests for approval workflow."""

    @pytest.mark.asyncio
    async def test_approve_resumes(self, orchestrator: HOTLOrchestrator):
        """Approving a paused role sets it back to pending for processing."""
        queue = orchestrator.queue
        queue.add_role("common")
        queue.update_status("common", QueueItemStatus.PAUSED)

        await orchestrator.approve("common", approved=True)

        item = queue.items[0]
        assert item.status == QueueItemStatus.PENDING

    @pytest.mark.asyncio
    async def test_approve_full_workflow(self, orchestrator: HOTLOrchestrator):
        """Approved roles get processed through the workflow."""
        execution_count = 0

        async def mock_execute(role_name, **kwargs):
            nonlocal execution_count
            execution_count += 1
            if execution_count == 1:
                # First call pauses
                return {"status": "paused", "execution_id": 42}
            # Second call completes
            return {"status": "completed"}

        mock_graph = MagicMock()
        mock_graph.execute = mock_execute

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            # Start processing in background
            queue = orchestrator.queue
            queue.add_role("common")

            async def approve_after_pause():
                """Wait for pause then approve."""
                for _ in range(50):
                    await asyncio.sleep(0.05)
                    paused = queue.get_paused()
                    if paused:
                        await orchestrator.approve("common", approved=True)
                        return

            task = asyncio.create_task(
                orchestrator.start(max_concurrent=1)
            )
            approve_task = asyncio.create_task(approve_after_pause())

            await asyncio.wait_for(
                asyncio.gather(task, approve_task), timeout=10.0
            )

        assert queue.is_complete()


class TestRejectSkips:
    """Tests for rejection workflow."""

    @pytest.mark.asyncio
    async def test_reject_skips(self, orchestrator: HOTLOrchestrator):
        """Rejecting a paused role marks it as skipped."""
        queue = orchestrator.queue
        queue.add_role("common")
        queue.update_status("common", QueueItemStatus.PAUSED)

        await orchestrator.approve("common", approved=False, reason="Not ready")

        item = queue.items[0]
        assert item.status == QueueItemStatus.SKIPPED
        assert item.error == "Not ready"


class TestFailureContinues:
    """Tests for failure resilience."""

    @pytest.mark.asyncio
    async def test_failure_continues(self, orchestrator: HOTLOrchestrator):
        """One role failure does not stop the queue from processing others."""
        result_map = {
            "common": {"status": "failed", "error": "Connection refused"},
            "windows_prerequisites": {"status": "completed"},
        }
        mock_graph = _make_mock_graph(result_map)

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            await orchestrator.start(waves=[1], max_concurrent=2)

        queue = orchestrator.queue
        assert queue.is_complete()

        items_by_name = {i.role_name: i for i in queue.items}
        assert items_by_name["common"].status == QueueItemStatus.FAILED
        assert items_by_name["windows_prerequisites"].status == QueueItemStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_exception_handled(self, orchestrator: HOTLOrchestrator):
        """Exceptions in role processing are caught and role is marked failed."""

        async def exploding_execute(role_name, **kwargs):
            raise RuntimeError("Unexpected explosion")

        mock_graph = MagicMock()
        mock_graph.execute = exploding_execute

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            await orchestrator.start(waves=[1], max_concurrent=2)

        queue = orchestrator.queue
        assert queue.is_complete()

        for item in queue.items:
            assert item.status == QueueItemStatus.FAILED
            assert item.error is not None


class TestConcurrentProcessing:
    """Tests for concurrent role processing."""

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, orchestrator: HOTLOrchestrator):
        """Multiple roles process in parallel up to max_concurrent."""
        max_simultaneous = 0
        current_active = 0
        lock = asyncio.Lock()

        async def tracking_execute(role_name, **kwargs):
            nonlocal max_simultaneous, current_active
            async with lock:
                current_active += 1
                if current_active > max_simultaneous:
                    max_simultaneous = current_active
            await asyncio.sleep(0.05)  # Simulate work
            async with lock:
                current_active -= 1
            return {"status": "completed"}

        mock_graph = MagicMock()
        mock_graph.execute = tracking_execute

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            await orchestrator.start(waves=[1], max_concurrent=2)

        # Should have had 2 concurrent at some point (wave 1 has 2 roles)
        assert max_simultaneous == 2

    @pytest.mark.asyncio
    async def test_concurrent_limited(self, orchestrator: HOTLOrchestrator):
        """Concurrency never exceeds max_concurrent."""
        max_simultaneous = 0
        current_active = 0
        lock = asyncio.Lock()

        async def tracking_execute(role_name, **kwargs):
            nonlocal max_simultaneous, current_active
            async with lock:
                current_active += 1
                if current_active > max_simultaneous:
                    max_simultaneous = current_active
            await asyncio.sleep(0.05)
            async with lock:
                current_active -= 1
            return {"status": "completed"}

        mock_graph = MagicMock()
        mock_graph.execute = tracking_execute

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            # max_concurrent=1 means strictly sequential
            await orchestrator.start(waves=[1, 2], max_concurrent=1)

        assert max_simultaneous == 1


class TestStatus:
    """Tests for orchestrator status."""

    @pytest.mark.asyncio
    async def test_status(self, orchestrator: HOTLOrchestrator):
        """Status returns current orchestrator state."""
        mock_graph = _make_mock_graph()

        with patch(
            "harness.dag.graph.create_box_up_role_graph",
            return_value=mock_graph,
        ):
            await orchestrator.start(waves=[1], max_concurrent=2)

        status = await orchestrator.status()

        assert status["running"] is False
        assert status["completed"] == 2
        assert status["failed"] == 0
        assert "queue" in status
        assert status["queue"]["complete"] is True
