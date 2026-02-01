"""HOTL Supervisor Agent using LangGraph for autonomous operation."""

import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from harness.db.state import StateDB
from harness.hotl.state import HOTLState, HOTLPhase, create_initial_state
from harness.hotl.notifications import NotificationService, NotificationConfig
from harness.hotl.agent_session import AgentStatus, AgentSessionManager
from harness.hotl.claude_integration import HOTLClaudeIntegration, ClaudeAgentConfig

logger = logging.getLogger(__name__)


class HOTLSupervisor:
    """
    Supervisor that orchestrates HOTL autonomous operation.

    Implements a LangGraph-based state machine that cycles through:
    - Research: Explore codebase and gather information
    - Planning: Review and update plans
    - Gap Analysis: Identify missing work
    - Execution: Run tasks from the queue
    - Testing: Validate changes
    - Notification: Send status updates

    The supervisor continues until max_iterations is reached,
    stop is requested, or a critical error occurs.

    NOTE: All node functions are synchronous because LangGraph's standard
    compile() produces a sync graph. We use sync invocation throughout.
    """

    def __init__(
        self,
        db: StateDB,
        config: Optional[dict] = None
    ):
        """
        Initialize the HOTL supervisor.

        Args:
            db: StateDB instance for persistence
            config: Optional configuration dict with keys:
                - discord_webhook_url: Discord webhook for notifications
                - email_*: Email configuration
                - claude_*: Claude agent configuration
                - repo_root: Repository root for agent working directory
        """
        self.db = db
        self.config = config or {}

        # Initialize notification service
        notif_config = NotificationConfig(
            discord_webhook_url=self.config.get("discord_webhook_url"),
            email_smtp_host=self.config.get("email_smtp_host"),
            email_smtp_port=self.config.get("email_smtp_port", 587),
            email_from=self.config.get("email_from"),
            email_to=self.config.get("email_to"),
            email_username=self.config.get("email_username"),
            email_password=self.config.get("email_password"),
        )
        self.notification_service = NotificationService(notif_config)

        # Initialize Claude agent integration
        agent_config = ClaudeAgentConfig(
            claude_cli_path=self.config.get("claude_cli_path", "claude"),
            default_timeout=self.config.get("claude_timeout", 600),
            max_concurrent_agents=self.config.get("claude_max_concurrent", 3),
            skip_permissions=self.config.get("claude_skip_permissions", False),
            model=self.config.get("claude_model"),
        )
        session_manager = AgentSessionManager(db=db)
        self.claude_integration = HOTLClaudeIntegration(
            config=agent_config,
            session_manager=session_manager,
            db=db,
        )

        # Set up agent callbacks
        self.claude_integration.set_callbacks(
            on_complete=self._on_agent_complete,
            on_progress=self._on_agent_progress,
            on_intervention=self._on_agent_intervention,
        )

        # Repository root for agent working directory
        self.repo_root = Path(self.config.get("repo_root", "."))

        # Initialize checkpointer for state persistence
        # Use direct connection for SQLite checkpoint saver
        if db._is_memory:
            # For in-memory databases, create a separate checkpoint connection
            self._checkpoint_conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            # For file-based databases, create a separate connection with WAL mode
            checkpoint_db_path = str(db.db_path).replace(".db", "_checkpoint.db")
            self._checkpoint_conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
            self._checkpoint_conn.execute("PRAGMA journal_mode=WAL")

        self.checkpointer = SqliteSaver(self._checkpoint_conn)

        # Custom node functions can be registered here
        self._custom_nodes: dict[str, Callable] = {}

        # Track current session for external stop/status requests
        self._current_session_id: Optional[str] = None
        self._stop_requested: bool = False

        # Build workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the HOTL supervisor workflow graph."""
        graph = StateGraph(HOTLState)

        # Add nodes
        graph.add_node("check_status", self._check_status_node)
        graph.add_node("research", self._research_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("gap_analysis", self._gap_analysis_node)
        graph.add_node("execute_task", self._execute_task_node)
        graph.add_node("agent_execute", self._agent_execute_node)
        graph.add_node("test", self._test_node)
        graph.add_node("notify", self._notify_node)
        graph.add_node("decide_next", self._decide_next_node)

        # Set entry point
        graph.set_entry_point("check_status")

        # Add edges
        graph.add_edge("check_status", "decide_next")

        # Conditional routing from decide_next
        graph.add_conditional_edges(
            "decide_next",
            self._route_next,
            {
                "research": "research",
                "plan": "plan",
                "gap_analysis": "gap_analysis",
                "execute": "execute_task",
                "agent_execute": "agent_execute",
                "test": "test",
                "notify": "notify",
                "end": END
            }
        )

        # All action nodes loop back to decide_next
        for node in ["research", "plan", "gap_analysis", "execute_task", "agent_execute", "test", "notify"]:
            graph.add_edge(node, "decide_next")

        return graph.compile(checkpointer=self.checkpointer)

    def _route_next(self, state: HOTLState) -> str:
        """
        Decide next action based on current state.

        Routing logic:
        1. If stop requested -> end
        2. If max iterations reached -> notify then end
        3. If notification due -> notify
        4. If pending interventions -> notify (alert human)
        5. If gaps identified and agents enabled -> agent_execute
        6. Otherwise cycle through phases
        """
        if state.get("stop_requested", False):
            return "end"

        if state.get("iteration_count", 0) >= state.get("max_iterations", 100):
            # Check if we just notified
            if state.get("phase") == HOTLPhase.NOTIFYING:
                return "end"
            return "notify"

        # Check if any agents need human intervention
        pending_interventions = state.get("pending_interventions", [])
        if pending_interventions:
            return "notify"  # Alert human about pending interventions

        # Check if notification is due
        time_since_notify = time.time() - state.get("last_notification_time", 0)
        notification_interval = state.get("notification_interval", 300)
        if time_since_notify >= notification_interval:
            return "notify"

        # If paused, stay in current state
        if state.get("pause_requested", False):
            return "notify"  # Send pause notification

        # Check for active agent sessions
        active_agents = state.get("active_agent_sessions", [])
        if active_agents:
            # Continue monitoring agents
            return "agent_execute"

        current_phase = state.get("phase", HOTLPhase.IDLE)

        # Map phase to route - now includes agent execution path
        phase_to_route = {
            HOTLPhase.IDLE: "research",
            HOTLPhase.RESEARCHING: "plan",
            HOTLPhase.PLANNING: "gap_analysis",
            HOTLPhase.GAP_ANALYZING: "agent_execute",  # Spawn agents for gaps
            HOTLPhase.AGENT_EXECUTING: "agent_execute",  # Continue monitoring
            HOTLPhase.EXECUTING: "test",
            HOTLPhase.TESTING: "research",  # Loop back
            HOTLPhase.NOTIFYING: "research",
            HOTLPhase.PAUSED: "notify",
        }

        return phase_to_route.get(current_phase, "research")

    def _check_status_node(self, state: HOTLState) -> dict:
        """Check current status and prepare for execution."""
        logger.info(f"HOTL check_status: iteration {state.get('iteration_count', 0)}")

        # Check if external stop was requested
        if self._stop_requested:
            logger.info("External stop requested")
            return {"stop_requested": True}

        # Check database health
        try:
            stats = self.db.get_statistics()
            logger.debug(f"DB stats: {stats}")
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"errors": [f"Database error: {e}"]}

        return {}

    def _research_node(self, state: HOTLState) -> dict:
        """
        Run deep research phase.

        This node explores the codebase, searches for relevant information,
        and gathers insights for planning.
        """
        logger.info("HOTL research phase starting")

        findings = []
        insights = []

        # Check for active regressions
        try:
            regressions = self.db.get_active_regressions()
            if regressions:
                findings.append({
                    "type": "regressions",
                    "count": len(regressions),
                    "roles": [r.role_name for r in regressions[:5]]
                })
                insights.append(f"Found {len(regressions)} active test regressions")
        except Exception as e:
            logger.warning(f"Failed to check regressions: {e}")

        # Check role statistics
        try:
            roles = self.db.list_roles()
            roles_with_tests = sum(1 for r in roles if r.has_molecule_tests)
            findings.append({
                "type": "role_coverage",
                "total_roles": len(roles),
                "with_tests": roles_with_tests,
                "coverage_pct": (roles_with_tests / len(roles) * 100) if roles else 0
            })
        except Exception as e:
            logger.warning(f"Failed to get role stats: {e}")

        # Check pending executions
        try:
            stats = self.db.get_statistics()
            pending = stats.get("pending_executions", 0)
            if pending > 0:
                insights.append(f"Found {pending} pending workflow executions")
        except Exception as e:
            logger.warning(f"Failed to check executions: {e}")

        return {
            "phase": HOTLPhase.RESEARCHING,
            "research_findings": findings,
            "codebase_insights": insights,
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    def _plan_node(self, state: HOTLState) -> dict:
        """
        Review and update plans based on research.

        This node analyzes findings and updates the current plan
        or creates a new one if needed.
        """
        logger.info("HOTL plan phase starting")

        findings = state.get("research_findings", [])
        insights = state.get("codebase_insights", [])

        # Generate plan summary
        plan_items = []

        for finding in findings[-5:]:  # Recent findings
            if finding.get("type") == "regressions":
                plan_items.append(f"- Investigate {finding.get('count', 0)} test regressions")
            elif finding.get("type") == "role_coverage":
                coverage = finding.get("coverage_pct", 0)
                if coverage < 80:
                    plan_items.append(f"- Improve test coverage (currently {coverage:.1f}%)")

        for insight in insights[-5:]:
            plan_items.append(f"- {insight}")

        current_plan = "\n".join(plan_items) if plan_items else "No action items identified"

        return {
            "phase": HOTLPhase.PLANNING,
            "current_plan": current_plan,
            "plan_revision": state.get("plan_revision", 0) + 1
        }

    def _gap_analysis_node(self, state: HOTLState) -> dict:
        """
        Analyze gaps between expected and actual state.

        This node compares the current state against the plan
        and identifies what needs to be done.
        """
        logger.info("HOTL gap_analysis phase starting")

        gaps = []

        # Check for roles without tests
        try:
            roles = self.db.list_roles()
            for role in roles:
                if not role.has_molecule_tests:
                    gaps.append(f"Role '{role.name}' missing molecule tests")
        except Exception as e:
            logger.warning(f"Failed to analyze role coverage: {e}")

        # Check for incomplete executions
        try:
            stats = self.db.get_statistics()
            pending = stats.get("pending_executions", 0)
            if pending > 0:
                gaps.append(f"{pending} workflow executions pending")
        except Exception as e:
            logger.warning(f"Failed to check executions: {e}")

        # Limit gaps to prevent overwhelming
        gaps = gaps[:10]

        return {
            "phase": HOTLPhase.GAP_ANALYZING,
            "plan_gaps": gaps
        }

    def _execute_task_node(self, state: HOTLState) -> dict:
        """
        Execute a pending task from the queue.

        This node processes one task at a time, updating
        the completed/failed lists accordingly.
        """
        logger.info("HOTL execute_task phase starting")

        pending = state.get("pending_tasks", [])

        if not pending:
            # No tasks to execute, but that's not an error
            return {"phase": HOTLPhase.EXECUTING}

        task_id = pending[0]
        remaining = pending[1:]

        # Execute the task
        try:
            logger.info(f"Executing task {task_id}")

            # Get task details from database if available
            # For now, mark as completed - in future, integrate with
            # actual task execution system
            return {
                "phase": HOTLPhase.EXECUTING,
                "pending_tasks": remaining,
                "completed_tasks": [task_id]
            }
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return {
                "phase": HOTLPhase.EXECUTING,
                "pending_tasks": remaining,
                "failed_tasks": [task_id],
                "errors": [f"Task {task_id} failed: {e}"]
            }

    def _agent_execute_node(self, state: HOTLState) -> dict:
        """
        Manage Claude Code agent execution for gap/fix tasks.

        This node:
        1. Spawns new agents for identified gaps
        2. Monitors active agent sessions
        3. Handles agent completion or intervention requests
        """
        logger.info("HOTL agent_execute phase starting")

        gaps = state.get("plan_gaps", [])
        active_sessions = list(state.get("active_agent_sessions", []))
        pending_interventions = list(state.get("pending_interventions", []))
        completed_sessions: list[str] = []

        # Check status of active agents
        sessions_to_remove = []
        for session_id in active_sessions:
            try:
                session = self.claude_integration.get_session(session_id)
                if not session:
                    sessions_to_remove.append(session_id)
                    continue

                if session.status == AgentStatus.COMPLETED:
                    logger.info(f"Agent {session_id} completed successfully")
                    sessions_to_remove.append(session_id)
                    completed_sessions.append(session_id)

                elif session.status == AgentStatus.FAILED:
                    logger.warning(f"Agent {session_id} failed: {session.error_message}")
                    sessions_to_remove.append(session_id)

                elif session.status == AgentStatus.NEEDS_HUMAN:
                    logger.info(f"Agent {session_id} needs intervention: {session.intervention_reason}")
                    if session_id not in pending_interventions:
                        pending_interventions.append(session_id)
                    sessions_to_remove.append(session_id)

                elif session.status == AgentStatus.CANCELLED:
                    sessions_to_remove.append(session_id)

            except Exception as e:
                logger.error(f"Error checking agent {session_id}: {e}")
                sessions_to_remove.append(session_id)

        # Remove completed/failed sessions from active list
        for session_id in sessions_to_remove:
            if session_id in active_sessions:
                active_sessions.remove(session_id)

        # Spawn new agents for gaps (if we have capacity)
        max_concurrent = self.config.get("claude_max_concurrent", 3)
        while gaps and len(active_sessions) < max_concurrent:
            gap = gaps.pop(0)
            try:
                # Create task from gap description
                task = self._create_agent_task(gap, state)
                session = self.claude_integration.spawn_agent(
                    task=task,
                    working_dir=self.repo_root,
                    context={
                        "gap": gap,
                        "plan": state.get("current_plan"),
                        "iteration": state.get("iteration_count", 0),
                    },
                )
                active_sessions.append(session.id)
                logger.info(f"Spawned agent {session.id} for gap: {gap[:50]}...")

            except Exception as e:
                logger.error(f"Failed to spawn agent for gap '{gap}': {e}")

        # Determine phase based on agent status
        if active_sessions:
            phase = HOTLPhase.AGENT_EXECUTING
        elif pending_interventions:
            phase = HOTLPhase.AGENT_EXECUTING  # Stay here, will route to notify
        else:
            phase = HOTLPhase.EXECUTING  # Move to regular execution

        return {
            "phase": phase,
            "plan_gaps": gaps,
            "active_agent_sessions": active_sessions,
            "completed_agent_sessions": completed_sessions,
            "pending_interventions": pending_interventions,
        }

    def _create_agent_task(self, gap: str, state: HOTLState) -> str:
        """
        Create a task prompt for a Claude agent based on a gap.

        Args:
            gap: Gap description from gap analysis
            state: Current HOTL state for context

        Returns:
            Formatted task string for the agent
        """
        current_plan = state.get("current_plan", "No plan available")

        return f"""You are working on addressing the following gap in our Ansible role codebase:

## Gap to Address
{gap}

## Current Plan Context
{current_plan[:2000]}

## Instructions
1. Analyze the gap and determine what changes are needed
2. Make the necessary code changes to address this gap
3. Report your progress using the agent_report_progress tool
4. If you encounter blockers, use agent_request_intervention
5. Log all file changes using agent_log_file_operation
6. Provide a summary when complete

Work autonomously to resolve this gap while maintaining code quality and following existing patterns.
"""

    # Agent lifecycle callbacks
    def _on_agent_complete(self, session) -> None:
        """Called when an agent completes."""
        logger.info(f"Agent {session.id} completed with status: {session.status.value}")
        if session.file_changes:
            logger.info(f"  Made {len(session.file_changes)} file changes")

    def _on_agent_progress(self, session_id: str, progress: str) -> None:
        """Called when an agent reports progress."""
        logger.debug(f"Agent {session_id} progress: {progress[:100]}")

    def _on_agent_intervention(self, session) -> None:
        """Called when an agent needs human intervention."""
        logger.warning(f"Agent {session.id} needs intervention: {session.intervention_reason}")
        # Could trigger immediate notification here

    def _test_node(self, state: HOTLState) -> dict:
        """
        Run tests to validate changes.

        This node runs validation checks and reports any failures.
        """
        logger.info("HOTL test phase starting")

        errors = []
        warnings = []

        # Validate database integrity
        try:
            validation = self.db.validate_data_integrity()
            if not validation.get("valid"):
                for issue in validation.get("issues", []):
                    warnings.append(f"Data integrity: {issue}")
        except Exception as e:
            errors.append(f"Database validation failed: {e}")

        # Validate dependency graph
        try:
            dep_validation = self.db.validate_dependencies()
            if not dep_validation.get("valid"):
                cycles = dep_validation.get("cycles", [])
                if cycles:
                    errors.append(f"Dependency cycles detected: {len(cycles)}")
        except Exception as e:
            errors.append(f"Dependency validation failed: {e}")

        return {
            "phase": HOTLPhase.TESTING,
            "errors": errors if errors else [],
            "warnings": warnings if warnings else []
        }

    def _notify_node(self, state: HOTLState) -> dict:
        """
        Send notification with status summary.

        This node sends updates to configured notification channels
        (Discord, email) with the current status.
        """
        logger.info("HOTL notify phase starting")

        # Generate summary
        summary = self._generate_summary(state)

        # Send notifications synchronously
        try:
            results = self.notification_service.send_status_update_sync(
                dict(state),
                summary
            )
            logger.info(f"Notification results: {results}")
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

        return {
            "phase": HOTLPhase.NOTIFYING,
            "last_notification_time": time.time()
        }

    def _decide_next_node(self, state: HOTLState) -> dict:
        """Decision node - no state changes, just routing."""
        return {}

    def _generate_summary(self, state: HOTLState) -> str:
        """Generate a status summary from the current state."""
        phase = state.get("phase", HOTLPhase.IDLE)
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 100)

        completed = len(state.get("completed_tasks", []))
        failed = len(state.get("failed_tasks", []))
        pending = len(state.get("pending_tasks", []))

        errors = state.get("errors", [])
        warnings = state.get("warnings", [])
        insights = state.get("codebase_insights", [])
        gaps = state.get("plan_gaps", [])

        # Agent session info
        active_agents = state.get("active_agent_sessions", [])
        completed_agents = state.get("completed_agent_sessions", [])
        pending_interventions = state.get("pending_interventions", [])

        summary = f"""## HOTL Execution Summary

**Phase**: {phase}
**Iteration**: {iteration} / {max_iter}

### Task Status
- Completed: {completed}
- Failed: {failed}
- Pending: {pending}

### Claude Agent Status
- Active Agents: {len(active_agents)}
- Completed Sessions: {len(completed_agents)}
- Pending Interventions: {len(pending_interventions)}
"""

        # Add intervention alerts if any
        if pending_interventions:
            summary += "\n**ATTENTION: Agent Interventions Required**\n"
            for session_id in pending_interventions[:5]:
                session = self.claude_integration.get_session(session_id)
                if session:
                    summary += f"- {session_id[:8]}: {session.intervention_reason or 'Unknown reason'}\n"

        summary += f"""
### Recent Insights
{chr(10).join(f"- {i}" for i in insights[-3:]) if insights else "None"}

### Identified Gaps
{chr(10).join(f"- {g}" for g in gaps[-5:]) if gaps else "None"}

### Current Plan
{state.get('current_plan', 'No plan set')[:500]}

### Issues
**Errors**: {len(errors)}
{chr(10).join(f"- {e}" for e in errors[-3:]) if errors else "None"}

**Warnings**: {len(warnings)}
{chr(10).join(f"- {w}" for w in warnings[-3:]) if warnings else "None"}
"""
        return summary

    def run(
        self,
        max_iterations: int = 100,
        notification_interval: int = 300,
        resume_from: Optional[str] = None
    ) -> HOTLState:
        """
        Run the HOTL supervisor synchronously.

        Args:
            max_iterations: Maximum number of iterations before stopping
            notification_interval: Seconds between status notifications
            resume_from: Optional thread ID to resume from checkpoint

        Returns:
            Final state after execution
        """
        logger.info(f"Starting HOTL supervisor (max_iterations={max_iterations})")

        # Reset stop flag
        self._stop_requested = False

        try:
            if resume_from:
                # Resume from checkpoint
                thread_id = resume_from
                config = {"configurable": {"thread_id": thread_id}}

                # Get last checkpoint state
                # The graph will automatically restore from checkpoint
                initial_state = None
            else:
                # Create new state
                thread_id = f"hotl-{int(time.time())}"
                config = {"configurable": {"thread_id": thread_id}}

                initial_state = create_initial_state(
                    max_iterations=max_iterations,
                    notification_interval=notification_interval,
                    config=self.config
                )

            # Track session ID for external status/stop
            self._current_session_id = thread_id

            # Execute the workflow synchronously
            final_state = self.workflow.invoke(
                initial_state,
                config=config
            )

            logger.info("HOTL supervisor completed")
            return final_state

        finally:
            self._current_session_id = None
            # Close notification service
            self.notification_service.close_sync()
            # Terminate any running agents
            if self.claude_integration:
                self.claude_integration.terminate_all_agents("Supervisor shutdown")

    def request_stop(self) -> bool:
        """
        Request the supervisor to stop after current iteration.

        Returns:
            True if a session is running and stop was requested
        """
        if self._current_session_id:
            logger.info(f"Stop requested for HOTL session {self._current_session_id}")
            self._stop_requested = True
            return True
        logger.warning("No active HOTL session to stop")
        return False

    def request_pause(self) -> bool:
        """
        Request the supervisor to pause.

        Returns:
            True if pause was requested (not yet implemented)
        """
        logger.info("Pause requested for HOTL supervisor")
        # Pause functionality can be implemented via state flag
        return False

    def get_status(self) -> dict[str, Any]:
        """
        Get current supervisor status.

        Returns:
            Status dict with session info
        """
        return {
            "running": self._current_session_id is not None,
            "session_id": self._current_session_id,
            "stop_requested": self._stop_requested,
        }

    def register_custom_node(
        self,
        name: str,
        func: Callable[[HOTLState], dict]
    ) -> None:
        """
        Register a custom node function.

        Args:
            name: Node name
            func: Function that takes state and returns state update
        """
        self._custom_nodes[name] = func

    # =========================================================================
    # AGENT MANAGEMENT METHODS
    # =========================================================================

    def get_active_agents(self) -> list:
        """Get all currently active Claude agent sessions."""
        return self.claude_integration.get_active_agents()

    def get_pending_interventions(self) -> list:
        """Get agent sessions that need human intervention."""
        return self.claude_integration.get_pending_interventions()

    def resolve_intervention(
        self,
        session_id: str,
        resolution: str,
        continue_agent: bool = False
    ) -> None:
        """
        Resolve a pending agent intervention.

        Args:
            session_id: Agent session ID
            resolution: Description of the resolution
            continue_agent: If True, spawn a continuation agent
        """
        self.claude_integration.resolve_intervention(
            session_id=session_id,
            resolution=resolution,
            continue_agent=continue_agent
        )
        logger.info(f"Resolved intervention for agent {session_id}")

    def terminate_agent(self, session_id: str, reason: str = "Manual termination") -> bool:
        """
        Terminate a running agent.

        Args:
            session_id: Agent session ID
            reason: Reason for termination

        Returns:
            True if terminated successfully
        """
        return self.claude_integration.terminate_agent(session_id, reason)

    def terminate_all_agents(self, reason: str = "Shutdown") -> int:
        """
        Terminate all running agents.

        Args:
            reason: Reason for termination

        Returns:
            Number of agents terminated
        """
        return self.claude_integration.terminate_all_agents(reason)

    def get_agent_stats(self) -> dict[str, Any]:
        """Get Claude agent integration statistics."""
        return self.claude_integration.get_stats()

    def spawn_agent_manually(
        self,
        task: str,
        working_dir: Optional[Path] = None,
        context: Optional[dict] = None
    ):
        """
        Manually spawn a Claude agent outside the normal workflow.

        Args:
            task: Task description for the agent
            working_dir: Working directory (uses repo_root if not specified)
            context: Optional context dict

        Returns:
            AgentSession for the spawned agent
        """
        return self.claude_integration.spawn_agent(
            task=task,
            working_dir=working_dir or self.repo_root,
            context=context or {}
        )
