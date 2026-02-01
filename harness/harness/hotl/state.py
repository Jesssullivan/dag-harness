"""State definitions for HOTL mode."""

from enum import Enum
from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
import operator


class HOTLPhase(str, Enum):
    """Current phase of HOTL execution."""
    IDLE = "idle"
    RESEARCHING = "researching"
    PLANNING = "planning"
    GAP_ANALYZING = "gap_analyzing"
    EXECUTING = "executing"
    AGENT_EXECUTING = "agent_executing"  # Claude agent is running
    TESTING = "testing"
    NOTIFYING = "notifying"
    PAUSED = "paused"
    STOPPED = "stopped"


class HOTLState(TypedDict, total=False):
    """
    State for HOTL supervisor.

    Uses Annotated types with operator.add for list accumulation
    following LangGraph reducer patterns.
    """
    # Current phase
    phase: HOTLPhase

    # Task tracking
    current_task_id: Optional[int]
    pending_tasks: list[int]
    completed_tasks: Annotated[list[int], operator.add]
    failed_tasks: Annotated[list[int], operator.add]

    # Research context
    research_findings: Annotated[list[dict], operator.add]
    web_search_results: Annotated[list[dict], operator.add]
    codebase_insights: Annotated[list[str], operator.add]

    # Planning context
    current_plan: Optional[str]
    plan_file_path: Optional[str]
    plan_gaps: list[str]
    plan_revision: int

    # Execution tracking
    iteration_count: int
    max_iterations: int
    last_notification_time: float
    notification_interval: int  # seconds between notifications

    # Error tracking
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]

    # Control flags
    should_continue: bool
    pause_requested: bool
    stop_requested: bool

    # Session metadata
    session_id: str
    started_at: str
    config: dict[str, Any]

    # Agent execution tracking
    active_agent_sessions: list[str]  # Session IDs of active Claude agents
    completed_agent_sessions: Annotated[list[str], operator.add]
    pending_interventions: list[str]  # Session IDs needing human help


def create_initial_state(
    max_iterations: int = 100,
    notification_interval: int = 300,  # 5 minutes
    config: Optional[dict] = None
) -> HOTLState:
    """Create initial HOTL state with defaults."""
    import time
    import uuid
    from datetime import datetime

    return HOTLState(
        phase=HOTLPhase.IDLE,
        current_task_id=None,
        pending_tasks=[],
        completed_tasks=[],
        failed_tasks=[],
        research_findings=[],
        web_search_results=[],
        codebase_insights=[],
        current_plan=None,
        plan_file_path=None,
        plan_gaps=[],
        plan_revision=0,
        iteration_count=0,
        max_iterations=max_iterations,
        last_notification_time=time.time(),
        notification_interval=notification_interval,
        errors=[],
        warnings=[],
        should_continue=True,
        pause_requested=False,
        stop_requested=False,
        session_id=str(uuid.uuid4()),
        started_at=datetime.utcnow().isoformat(),
        config=config or {},
        active_agent_sessions=[],
        completed_agent_sessions=[],
        pending_interventions=[],
    )
