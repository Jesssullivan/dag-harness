"""HOTL orchestrator that processes the role queue.

Coordinates wave-based role processing in HOTL mode, managing concurrent
box-up-role workflows with human approval checkpoints.
"""

import asyncio
import logging
import time
from typing import Any

from harness.db.state import StateDB
from harness.hotl.queue import QueueItem, QueueItemStatus, RoleQueue

logger = logging.getLogger(__name__)


class HOTLOrchestrator:
    """Orchestrates box-up-role workflows in HOTL mode.

    Processes a RoleQueue by running box-up-role workflows for each role,
    handling pauses for human approval, and managing failures gracefully.

    Args:
        db: StateDB instance for persistence.
        config: Configuration dict (repo_root, etc.).
        queue: RoleQueue to process.
    """

    def __init__(self, db: StateDB, config: dict[str, Any], queue: RoleQueue):
        self._db = db
        self._config = config
        self._queue = queue
        self._running = False
        self._stop_event: asyncio.Event | None = None
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._started_at: float | None = None
        self._completed_count = 0
        self._failed_count = 0

    @property
    def running(self) -> bool:
        """Whether the orchestrator is currently processing."""
        return self._running

    @property
    def queue(self) -> RoleQueue:
        """The role queue being processed."""
        return self._queue

    async def start(
        self,
        waves: list[int] | None = None,
        max_concurrent: int = 2,
    ) -> None:
        """Start processing the queue.

        Populates the queue from the specified waves (if provided) and
        begins processing roles through box-up-role workflows.

        Args:
            waves: List of wave numbers to process. If None, processes
                whatever is already in the queue.
            max_concurrent: Maximum concurrent role processing (overrides
                queue setting).
        """
        if self._running:
            logger.warning("Orchestrator is already running")
            return

        self._running = True
        self._stop_event = asyncio.Event()
        self._started_at = time.time()
        self._completed_count = 0
        self._failed_count = 0

        # Update concurrency limit
        self._queue._max_concurrent = max_concurrent

        # Add waves to queue if specified
        if waves:
            for wave in sorted(waves):
                self._queue.add_wave(wave)

        logger.info(
            f"HOTL orchestrator starting: {len(self._queue.items)} roles, "
            f"max_concurrent={max_concurrent}"
        )

        try:
            await self._process_loop()
        finally:
            self._running = False
            self._stop_event = None
            logger.info(
                f"HOTL orchestrator stopped: {self._completed_count} completed, "
                f"{self._failed_count} failed"
            )

    async def stop(self) -> None:
        """Stop processing (graceful, waits for current items).

        Sets the stop flag and waits for currently running workflows
        to complete before returning.
        """
        if not self._running or self._stop_event is None:
            logger.warning("Orchestrator is not running")
            return

        logger.info("Stop requested, waiting for current items to complete...")
        self._stop_event.set()

        # Wait for active tasks to finish
        if self._active_tasks:
            await asyncio.gather(
                *self._active_tasks.values(), return_exceptions=True
            )

    async def status(self) -> dict:
        """Get orchestrator status.

        Returns:
            Dict with orchestrator state, queue status, and timing info.
        """
        queue_status = self._queue.get_status()
        elapsed = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "elapsed_seconds": elapsed,
            "completed": self._completed_count,
            "failed": self._failed_count,
            "active_roles": list(self._active_tasks.keys()),
            "queue": queue_status,
        }

    async def approve(
        self, role_name: str, approved: bool = True, reason: str = ""
    ) -> None:
        """Approve or reject a paused workflow.

        Args:
            role_name: Name of the role to approve/reject.
            approved: True to approve and resume, False to reject and skip.
            reason: Optional reason for rejection.
        """
        if approved:
            logger.info(f"Approved role '{role_name}'")
            self._queue.update_status(role_name, QueueItemStatus.PENDING)
        else:
            logger.info(f"Rejected role '{role_name}': {reason}")
            self._queue.update_status(
                role_name, QueueItemStatus.SKIPPED, error=reason or "Rejected"
            )

    async def _process_loop(self) -> None:
        """Main processing loop.

        Continuously fetches batches from the queue and processes them
        until the queue is complete or stop is requested.
        """
        while self._running and not self._queue.is_complete():
            # Check for stop
            if self._stop_event and self._stop_event.is_set():
                logger.info("Stop event received, exiting process loop")
                break

            # Get next batch
            batch = self._queue.get_next_batch()
            if not batch:
                # No items ready; either waiting for running items or paused
                if self._active_tasks:
                    # Wait a bit for active tasks
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Check if there are paused items blocking progress
                    paused = self._queue.get_paused()
                    if paused:
                        logger.info(
                            f"Waiting for approval on {len(paused)} role(s): "
                            f"{[p.role_name for p in paused]}"
                        )
                        await asyncio.sleep(1.0)
                        continue
                    else:
                        # Nothing left to process
                        break

            # Start processing batch items
            for item in batch:
                self._queue.update_status(item.role_name, QueueItemStatus.RUNNING)
                task = asyncio.create_task(
                    self._process_role_safe(item),
                    name=f"process-{item.role_name}",
                )
                self._active_tasks[item.role_name] = task

            # Wait for at least one task to complete before checking again
            if self._active_tasks:
                done, _ = await asyncio.wait(
                    self._active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Clean up completed tasks
                completed_names = []
                for role_name, task in self._active_tasks.items():
                    if task.done():
                        completed_names.append(role_name)
                for name in completed_names:
                    del self._active_tasks[name]

    async def _process_role_safe(self, item: QueueItem) -> None:
        """Process a role with error handling wrapper.

        Args:
            item: QueueItem to process.
        """
        try:
            await self._process_role(item)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Role '{item.role_name}' processing failed: {error_msg}")
            await self._on_failure(item, error_msg)

    async def _process_role(self, item: QueueItem) -> None:
        """Process a single role through the box-up-role workflow.

        This method runs the box-up-role DAG workflow for the given role.
        If the workflow pauses for human approval, the item status is set
        to PAUSED. On completion or failure, the appropriate callback is
        invoked.

        Args:
            item: QueueItem to process.
        """
        logger.info(f"Processing role '{item.role_name}' (wave {item.wave})")

        try:
            from pathlib import Path

            from harness.dag.graph import create_box_up_role_graph

            repo_root = Path(self._config.get("repo_root", "."))
            graph = create_box_up_role_graph(self._db)

            result = await graph.execute(
                item.role_name,
                repo_root=repo_root,
                repo_python=self._config.get("repo_python"),
            )

            if result.get("status") == "completed":
                await self._on_complete(item)
            elif result.get("status") in ("paused", "human_needed"):
                item.execution_id = result.get("execution_id")
                await self._on_pause(item)
            else:
                error = result.get("error", "Unknown error")
                await self._on_failure(item, error)

        except ImportError as e:
            # Graph module may not be available in test environments
            logger.warning(f"Graph module not available: {e}")
            await self._on_failure(item, f"Import error: {e}")
        except Exception as e:
            await self._on_failure(item, str(e))

    async def _on_pause(self, item: QueueItem) -> None:
        """Handle workflow pause (human approval needed).

        Args:
            item: QueueItem that was paused.
        """
        logger.info(
            f"Role '{item.role_name}' paused for human approval "
            f"(execution_id={item.execution_id})"
        )
        self._queue.update_status(
            item.role_name,
            QueueItemStatus.PAUSED,
            execution_id=item.execution_id,
        )

    async def _on_complete(self, item: QueueItem) -> None:
        """Handle workflow completion.

        Args:
            item: QueueItem that completed successfully.
        """
        logger.info(f"Role '{item.role_name}' completed successfully")
        self._queue.update_status(item.role_name, QueueItemStatus.COMPLETED)
        self._completed_count += 1

    async def _on_failure(self, item: QueueItem, error: str) -> None:
        """Handle workflow failure.

        Marks the item as failed but does not stop the queue -- other
        roles continue processing.

        Args:
            item: QueueItem that failed.
            error: Error message describing the failure.
        """
        logger.error(f"Role '{item.role_name}' failed: {error}")
        self._queue.update_status(
            item.role_name, QueueItemStatus.FAILED, error=error
        )
        self._failed_count += 1
