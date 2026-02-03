"""
Parallel execution support for wave-based workflows.

Provides:
- Wave grouping of independent nodes
- Concurrent execution with asyncio (legacy)
- LangGraph Send API for parallel role execution with checkpointing
- Progress tracking across parallel tasks
- Error aggregation and reporting

Wave Execution Architecture:
- WaveExecutionState: TypedDict for wave context
- create_wave_execution_graph(): Creates StateGraph for wave execution
- route_to_wave_roles(): Returns list[Send] for parallel role execution
- merge_wave_results_node(): Aggregates results from parallel roles
"""

import asyncio
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from harness.dag.langgraph_engine import (
    LangGraphWorkflowRunner,
)
from harness.db.state import StateDB


class WaveStatus(str, Enum):
    """Status of a wave execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some roles completed, some failed


@dataclass
class RoleExecutionResult:
    """Result of a single role execution."""

    role_name: str
    wave: int
    status: str
    execution_id: int | None = None
    error: str | None = None
    duration_seconds: float | None = None
    summary: dict | None = None


@dataclass
class WaveExecutionResult:
    """Result of a wave execution."""

    wave: int
    wave_name: str
    status: WaveStatus
    roles: list[RoleExecutionResult]
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.roles if r.status == "completed")

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.roles if r.status in ("failed", "error"))


@dataclass
class WaveProgress:
    """Progress tracking for parallel wave execution."""

    wave: int
    total_roles: int
    completed_roles: int = 0
    failed_roles: int = 0
    current_roles: list[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        if self.total_roles == 0:
            return 100.0
        return (self.completed_roles + self.failed_roles) / self.total_roles * 100


# =============================================================================
# LANGGRAPH SEND API FOR WAVE EXECUTION
# =============================================================================


class WaveExecutionState(TypedDict, total=False):
    """
    TypedDict state schema for wave-level parallel execution.

    Uses Annotated types with operator.add reducers for proper list accumulation.
    This ensures results from parallel role executions are aggregated correctly.

    Fields:
        wave_number: Current wave being executed (0-8)
        wave_name: Human-readable wave name (e.g., "Infrastructure Foundation")
        role_names: List of roles to execute in this wave
        max_concurrent: Maximum concurrent role executions
        results: Accumulated results from each role execution
        errors: Accumulated errors across all roles
        completed_roles: List of roles that have completed (success or failure)
        pending_roles: Roles not yet started
        start_time: Wave execution start timestamp
        duration_seconds: Total wave execution duration
        status: Overall wave status
    """

    # Wave identification
    wave_number: int
    wave_name: str

    # Role configuration
    role_names: list[str]
    max_concurrent: int

    # Results accumulation (use reducers for parallel updates)
    results: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    completed_roles: Annotated[list[str], operator.add]

    # Execution tracking
    pending_roles: list[str]
    start_time: float | None
    duration_seconds: float | None
    status: str  # "pending" | "running" | "completed" | "failed" | "partial"


class RoleExecutionState(TypedDict, total=False):
    """
    State for individual role execution within a wave.

    This is passed to each parallel role execution via Send API.
    Contains the role-specific context plus wave context for tracking.
    """

    # Role identification
    role_name: str
    wave_number: int
    wave_name: str

    # Execution tracking
    start_time: float | None
    execution_id: int | None

    # Results
    status: str  # "pending" | "running" | "completed" | "failed"
    error: str | None
    duration_seconds: float | None
    summary: dict | None


def create_initial_wave_state(
    wave_number: int,
    wave_name: str,
    role_names: list[str],
    max_concurrent: int = 3,
) -> WaveExecutionState:
    """Create initial state for wave execution."""
    return WaveExecutionState(
        wave_number=wave_number,
        wave_name=wave_name,
        role_names=role_names,
        max_concurrent=max_concurrent,
        results=[],
        errors=[],
        completed_roles=[],
        pending_roles=list(role_names),
        start_time=None,
        duration_seconds=None,
        status="pending",
    )


# =============================================================================
# WAVE EXECUTION NODES
# =============================================================================


async def start_wave_node(state: WaveExecutionState) -> dict:
    """
    Initialize wave execution and record start time.

    This node prepares the wave for parallel role execution.
    """
    return {
        "start_time": time.time(),
        "status": "running",
        "pending_roles": list(state.get("role_names", [])),
    }


async def execute_role_node(state: RoleExecutionState) -> dict:
    """
    Execute a single role workflow within a wave.

    This node is invoked via Send API for each role in parallel.
    It runs the full box-up-role workflow for the given role.

    Returns:
        Dict with role execution results to be merged into wave state.
    """
    role_name = state["role_name"]
    wave_number = state.get("wave_number", 0)
    start_time = time.time()

    try:
        # Import here to avoid circular imports
        from harness.db.state import StateDB

        # Get or create database connection
        # Note: In production, this should be passed via config
        db = StateDB()

        runner = LangGraphWorkflowRunner(db)
        result = await runner.execute(role_name)

        duration = time.time() - start_time
        status = result.get("status", "unknown")

        return {
            "results": [
                {
                    "role_name": role_name,
                    "wave": wave_number,
                    "status": status,
                    "execution_id": result.get("execution_id"),
                    "error": result.get("error"),
                    "duration_seconds": duration,
                    "summary": result.get("summary"),
                }
            ],
            "completed_roles": [role_name],
            "errors": [f"{role_name}: {result.get('error')}"] if result.get("error") else [],
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "results": [
                {
                    "role_name": role_name,
                    "wave": wave_number,
                    "status": "error",
                    "error": str(e),
                    "duration_seconds": duration,
                }
            ],
            "completed_roles": [role_name],
            "errors": [f"{role_name}: {e}"],
        }


async def merge_wave_results_node(state: WaveExecutionState) -> dict:
    """
    Merge results from parallel role executions and determine wave status.

    This node runs after all parallel role executions complete.
    It aggregates results and calculates final wave status.

    Returns:
        Dict with final wave status and duration.
    """
    results = state.get("results", [])
    role_names = state.get("role_names", [])
    start_time = state.get("start_time")

    # Calculate duration
    duration = time.time() - start_time if start_time else None

    # Count successes and failures
    success_count = sum(1 for r in results if r.get("status") == "completed")
    failure_count = len(results) - success_count

    # Determine wave status
    if failure_count == 0:
        status = "completed"
    elif success_count == 0:
        status = "failed"
    else:
        status = "partial"

    # Log completion
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"Wave execution completed: {success_count}/{len(role_names)} roles succeeded, "
        f"duration={duration:.1f}s" if duration else ""
    )

    return {
        "status": status,
        "duration_seconds": duration,
        "pending_roles": [],
    }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_to_wave_roles(state: WaveExecutionState) -> list[Send]:
    """
    Fan out to parallel role execution nodes using Send API.

    Uses LangGraph's Send API to execute multiple roles in parallel.
    Each role gets its own execution context while sharing the wave context.

    This enables:
    - Proper checkpoint integration for each role
    - Parallel execution with semaphore-like control via max_concurrent
    - Result aggregation via reducer pattern

    Args:
        state: Current wave execution state

    Returns:
        List of Send objects targeting execute_role nodes.
    """
    role_names = state.get("role_names", [])
    wave_number = state.get("wave_number", 0)
    wave_name = state.get("wave_name", "")

    if not role_names:
        # No roles to execute, skip to merge
        return [Send("merge_wave_results", state)]

    sends = []
    for role_name in role_names:
        # Create role-specific execution state
        role_state: RoleExecutionState = {
            "role_name": role_name,
            "wave_number": wave_number,
            "wave_name": wave_name,
            "start_time": time.time(),
            "status": "pending",
        }
        sends.append(Send("execute_role", role_state))

    return sends


def should_continue_after_wave_start(
    state: WaveExecutionState,
) -> Literal["route_to_roles", "merge_wave_results"]:
    """
    Route after wave start - either fan out to roles or skip if no roles.
    """
    role_names = state.get("role_names", [])
    if not role_names:
        return "merge_wave_results"
    return "route_to_roles"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def create_wave_execution_graph() -> StateGraph:
    """
    Create a LangGraph StateGraph for wave-level parallel execution.

    This graph enables running multiple roles in parallel within a wave,
    with proper checkpoint support via LangGraph's Send API.

    Graph Structure:
        start_wave -> [parallel: execute_role x N] -> merge_wave_results -> END

    The parallel execution uses Send API to fan out to execute_role nodes,
    one for each role in the wave. Results are merged via the reducer pattern
    defined in WaveExecutionState.

    Returns:
        StateGraph configured for wave execution (call .compile() to use).

    Usage:
        graph = create_wave_execution_graph()
        compiled = graph.compile(checkpointer=checkpointer)
        result = await compiled.ainvoke(initial_state)
    """
    graph = StateGraph(WaveExecutionState)

    # Add nodes
    graph.add_node("start_wave", start_wave_node)
    graph.add_node("execute_role", execute_role_node)
    graph.add_node("merge_wave_results", merge_wave_results_node)

    # Set entry point
    graph.set_entry_point("start_wave")

    # Add edges
    # After start_wave, fan out to parallel role execution
    graph.add_conditional_edges(
        "start_wave",
        route_to_wave_roles,
        ["execute_role", "merge_wave_results"],
    )

    # All role executions converge at merge_wave_results
    graph.add_edge("execute_role", "merge_wave_results")

    # After merge, we're done
    graph.add_edge("merge_wave_results", END)

    return graph


async def create_compiled_wave_graph(db_path: str = "harness.db"):
    """
    Create and compile the wave execution graph with checkpointer.

    Args:
        db_path: Path to SQLite database for checkpointing.

    Returns:
        Compiled graph ready for execution with .ainvoke().
    """
    graph = create_wave_execution_graph()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        return compiled


class ParallelWaveExecutor:
    """
    Execute workflow for multiple roles in parallel within a wave.

    Supports:
    - Concurrent execution up to max_concurrent limit
    - Progress tracking and callbacks
    - Error aggregation without stopping other roles
    - Wave-level success/failure determination
    """

    # Wave definitions matching CLAUDE.md
    WAVES = {
        0: {"name": "Foundation", "roles": ["common"]},
        1: {
            "name": "Infrastructure Foundation",
            "roles": ["windows_prerequisites", "ems_registry_urls", "iis-config"],
        },
        2: {
            "name": "Core Platform",
            "roles": [
                "ems_platform_services",
                "ems_web_app",
                "database_clone",
                "ems_download_artifacts",
            ],
        },
        3: {
            "name": "Web Applications",
            "roles": [
                "ems_master_calendar",
                "ems_master_calendar_api",
                "ems_campus_webservice",
                "ems_desktop_deploy",
            ],
        },
        4: {
            "name": "Supporting Services",
            "roles": [
                "grafana_alloy_windows",
                "email_infrastructure",
                "hrtk_protected_users",
                "ems_environment_util",
            ],
        },
    }

    def __init__(
        self,
        db: StateDB,
        max_concurrent: int = 3,
        progress_callback: Callable[[WaveProgress], None] | None = None,
    ):
        self.db = db
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback
        self._runner = LangGraphWorkflowRunner(db)

    async def _execute_role(
        self, role_name: str, wave: int, semaphore: asyncio.Semaphore, progress: WaveProgress
    ) -> RoleExecutionResult:
        """Execute a single role with semaphore control."""
        async with semaphore:
            progress.current_roles.append(role_name)
            if self.progress_callback:
                self.progress_callback(progress)

            start_time = datetime.utcnow()
            try:
                result = await self._runner.execute(role_name)

                duration = (datetime.utcnow() - start_time).total_seconds()

                if result.get("status") == "completed":
                    progress.completed_roles += 1
                else:
                    progress.failed_roles += 1

                return RoleExecutionResult(
                    role_name=role_name,
                    wave=wave,
                    status=result.get("status", "unknown"),
                    execution_id=result.get("execution_id"),
                    error=result.get("error"),
                    duration_seconds=duration,
                    summary=result.get("summary"),
                )

            except Exception as e:
                progress.failed_roles += 1
                duration = (datetime.utcnow() - start_time).total_seconds()

                return RoleExecutionResult(
                    role_name=role_name,
                    wave=wave,
                    status="error",
                    error=str(e),
                    duration_seconds=duration,
                )
            finally:
                progress.current_roles.remove(role_name)
                if self.progress_callback:
                    self.progress_callback(progress)

    async def execute_wave(self, wave: int, roles: list[str] | None = None) -> WaveExecutionResult:
        """
        Execute all roles in a wave concurrently.

        Args:
            wave: Wave number (0-4)
            roles: Optional override of roles to execute (uses WAVES if not provided)

        Returns:
            WaveExecutionResult with all role results
        """
        wave_config = self.WAVES.get(wave, {"name": f"Wave {wave}", "roles": []})
        wave_name = wave_config["name"]
        roles_to_execute = roles or wave_config["roles"]

        if not roles_to_execute:
            return WaveExecutionResult(
                wave=wave,
                wave_name=wave_name,
                status=WaveStatus.COMPLETED,
                roles=[],
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_seconds=0,
            )

        progress = WaveProgress(wave=wave, total_roles=len(roles_to_execute))

        semaphore = asyncio.Semaphore(self.max_concurrent)
        started_at = datetime.utcnow()

        # Execute all roles concurrently
        tasks = [self._execute_role(role, wave, semaphore, progress) for role in roles_to_execute]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        # Process results (handle any exceptions that weren't caught)
        role_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                role_results.append(
                    RoleExecutionResult(
                        role_name=roles_to_execute[i], wave=wave, status="error", error=str(result)
                    )
                )
            else:
                role_results.append(result)

        # Determine wave status
        success_count = sum(1 for r in role_results if r.status == "completed")
        failure_count = len(role_results) - success_count

        if failure_count == 0:
            status = WaveStatus.COMPLETED
        elif success_count == 0:
            status = WaveStatus.FAILED
        else:
            status = WaveStatus.PARTIAL

        return WaveExecutionResult(
            wave=wave,
            wave_name=wave_name,
            status=status,
            roles=role_results,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    async def execute_all_waves(
        self, start_wave: int = 0, end_wave: int = 4, stop_on_wave_failure: bool = True
    ) -> list[WaveExecutionResult]:
        """
        Execute all waves sequentially, with roles within each wave in parallel.

        Args:
            start_wave: First wave to execute (default: 0)
            end_wave: Last wave to execute (default: 4)
            stop_on_wave_failure: If True, stop if any role in a wave fails

        Returns:
            List of WaveExecutionResult for all executed waves
        """
        results = []

        for wave in range(start_wave, end_wave + 1):
            wave_result = await self.execute_wave(wave)
            results.append(wave_result)

            # Check if we should stop
            if stop_on_wave_failure and wave_result.status in (
                WaveStatus.FAILED,
                WaveStatus.PARTIAL,
            ):
                break

        return results


async def execute_roles_parallel(
    db: StateDB,
    roles: list[str],
    max_concurrent: int = 3,
    progress_callback: Callable[[WaveProgress], None] | None = None,
) -> list[RoleExecutionResult]:
    """
    Convenience function to execute arbitrary roles in parallel.

    Args:
        db: StateDB instance
        roles: List of role names to execute
        max_concurrent: Maximum concurrent executions
        progress_callback: Optional progress callback

    Returns:
        List of RoleExecutionResult
    """
    executor = ParallelWaveExecutor(db, max_concurrent, progress_callback)

    # Execute as a single "wave"
    result = await executor.execute_wave(wave=-1, roles=roles)
    return result.roles
