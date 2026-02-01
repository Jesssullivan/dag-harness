"""Claude Code integration for HOTL autonomous operation.

Provides the ability to spawn Claude Code subagents for autonomous
code generation, modification, and analysis tasks.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from harness.hotl.agent_session import (
    AgentSession,
    AgentSessionManager,
    AgentStatus,
    FileChange,
    FileChangeType,
)

logger = logging.getLogger(__name__)


@dataclass
class ClaudeAgentConfig:
    """Configuration for Claude Code agent spawning."""
    # Path to claude CLI executable
    claude_cli_path: str = "claude"

    # Default timeout for agent execution (seconds)
    default_timeout: int = 600  # 10 minutes

    # Maximum concurrent agents
    max_concurrent_agents: int = 3

    # Working directory for agent output
    agent_output_dir: Optional[Path] = None

    # Environment variables to pass to agent
    env_vars: dict[str, str] = field(default_factory=dict)

    # Whether to use --dangerously-skip-permissions
    skip_permissions: bool = False

    # Model to use (if not default)
    model: Optional[str] = None

    # MCP config file path
    mcp_config: Optional[Path] = None


class HOTLClaudeIntegration:
    """
    Manages Claude Code subagent spawning and lifecycle for HOTL mode.

    Integrates with the HOTL supervisor to enable autonomous code execution
    by spawning Claude Code CLI processes as subagents.
    """

    def __init__(
        self,
        config: Optional[ClaudeAgentConfig] = None,
        session_manager: Optional[AgentSessionManager] = None,
        db: Optional[Any] = None,
    ):
        """
        Initialize Claude integration.

        Args:
            config: Agent configuration
            session_manager: Optional session manager (created if not provided)
            db: Optional StateDB for persistence
        """
        self.config = config or ClaudeAgentConfig()
        self.session_manager = session_manager or AgentSessionManager(db=db)
        self.db = db

        # Track running processes
        self._processes: dict[str, subprocess.Popen] = {}
        self._process_threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

        # Semaphore for limiting concurrent agents
        self._semaphore = threading.Semaphore(self.config.max_concurrent_agents)

        # Callbacks for status updates
        self._on_complete: Optional[Callable[[AgentSession], None]] = None
        self._on_progress: Optional[Callable[[str, str], None]] = None
        self._on_intervention: Optional[Callable[[AgentSession], None]] = None

    def set_callbacks(
        self,
        on_complete: Optional[Callable[[AgentSession], None]] = None,
        on_progress: Optional[Callable[[str, str], None]] = None,
        on_intervention: Optional[Callable[[AgentSession], None]] = None,
    ) -> None:
        """
        Set callbacks for agent events.

        Args:
            on_complete: Called when agent completes (success or failure)
            on_progress: Called with (session_id, progress_message)
            on_intervention: Called when agent needs human help
        """
        self._on_complete = on_complete
        self._on_progress = on_progress
        self._on_intervention = on_intervention

    def spawn_agent(
        self,
        task: str,
        working_dir: Path,
        context: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
        execution_id: Optional[int] = None,
    ) -> AgentSession:
        """
        Spawn a new Claude Code subagent for a task.

        Args:
            task: Task description/prompt for the agent
            working_dir: Working directory for the agent
            context: Optional context information
            timeout: Optional timeout in seconds (uses default if not specified)
            execution_id: Optional link to workflow execution

        Returns:
            Created AgentSession
        """
        # Create session
        session = self.session_manager.create_session(
            task=task,
            working_dir=working_dir,
            context=context or {},
            execution_id=execution_id,
        )

        # Start agent in background thread
        thread = threading.Thread(
            target=self._run_agent,
            args=(session, timeout or self.config.default_timeout),
            daemon=True,
            name=f"claude-agent-{session.id[:8]}",
        )

        with self._lock:
            self._process_threads[session.id] = thread

        thread.start()
        logger.info(f"Spawned Claude agent {session.id} for task: {task[:50]}...")

        return session

    def _run_agent(self, session: AgentSession, timeout: int) -> None:
        """
        Run agent in background thread.

        Args:
            session: Agent session
            timeout: Timeout in seconds
        """
        # Acquire semaphore to limit concurrency
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            session.mark_failed("Timeout waiting for available agent slot")
            self.session_manager.update_session(session)
            return

        try:
            self._execute_agent(session, timeout)
        finally:
            self._semaphore.release()
            with self._lock:
                self._process_threads.pop(session.id, None)
                self._processes.pop(session.id, None)

    def _execute_agent(self, session: AgentSession, timeout: int) -> None:
        """
        Execute the Claude CLI agent.

        Args:
            session: Agent session
            timeout: Timeout in seconds
        """
        # Build command
        cmd = self._build_command(session)

        # Prepare environment
        env = os.environ.copy()
        env.update(self.config.env_vars)

        # Add session info for MCP tools to use
        env["HOTL_SESSION_ID"] = session.id
        if session.execution_id:
            env["HOTL_EXECUTION_ID"] = str(session.execution_id)

        try:
            session.mark_started()
            self.session_manager.update_session(session)

            logger.debug(f"Executing: {' '.join(cmd)}")

            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=session.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            with self._lock:
                self._processes[session.id] = process
                session.pid = process.pid

            self.session_manager.update_session(session)

            # Collect output with timeout
            output_lines = []
            start_time = time.time()

            try:
                while True:
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        session.mark_failed(f"Agent timed out after {timeout}s")
                        break

                    # Read output
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break

                    if line:
                        output_lines.append(line)
                        session.add_progress(line.strip())

                        # Parse for special markers
                        self._parse_output_line(session, line)

                        # Call progress callback
                        if self._on_progress:
                            try:
                                self._on_progress(session.id, line.strip())
                            except Exception as e:
                                logger.warning(f"Progress callback error: {e}")

                # Process finished
                return_code = process.returncode
                session.output = "".join(output_lines)

                if session.status == AgentStatus.RUNNING:
                    if return_code == 0:
                        session.mark_completed(session.output)
                    else:
                        session.mark_failed(f"Agent exited with code {return_code}")

            except Exception as e:
                logger.error(f"Error running agent {session.id}: {e}")
                session.mark_failed(str(e))

        except Exception as e:
            logger.error(f"Failed to start agent {session.id}: {e}")
            session.mark_failed(f"Failed to start: {e}")

        finally:
            self.session_manager.update_session(session)

            # Call completion callback
            if self._on_complete:
                try:
                    self._on_complete(session)
                except Exception as e:
                    logger.warning(f"Complete callback error: {e}")

            # Check if intervention was requested
            if session.status == AgentStatus.NEEDS_HUMAN and self._on_intervention:
                try:
                    self._on_intervention(session)
                except Exception as e:
                    logger.warning(f"Intervention callback error: {e}")

    def _build_command(self, session: AgentSession) -> list[str]:
        """
        Build the Claude CLI command.

        Args:
            session: Agent session

        Returns:
            Command as list of strings
        """
        cmd = [self.config.claude_cli_path]

        # Add common flags
        cmd.extend(["--print"])  # Print output to stdout

        if self.config.skip_permissions:
            cmd.extend(["--dangerously-skip-permissions"])

        if self.config.model:
            cmd.extend(["--model", self.config.model])

        if self.config.mcp_config:
            cmd.extend(["--mcp-config", str(self.config.mcp_config)])

        # Add the task as a prompt
        # Build a structured prompt with context
        prompt = self._build_prompt(session)
        cmd.extend(["--prompt", prompt])

        return cmd

    def _build_prompt(self, session: AgentSession) -> str:
        """
        Build a structured prompt for the agent.

        Args:
            session: Agent session

        Returns:
            Formatted prompt string
        """
        parts = [
            "You are a Claude Code subagent running in HOTL (Human Out of The Loop) mode.",
            "Your session ID is: " + session.id,
            "",
            "## Task",
            session.task,
            "",
        ]

        # Add context if provided
        if session.context:
            parts.append("## Context")
            for key, value in session.context.items():
                if isinstance(value, (dict, list)):
                    parts.append(f"**{key}**: {json.dumps(value, indent=2)}")
                else:
                    parts.append(f"**{key}**: {value}")
            parts.append("")

        # Add instructions for MCP tools
        parts.extend([
            "## Instructions",
            "1. Work autonomously to complete the task",
            "2. Use the agent_report_progress MCP tool periodically to report status",
            "3. If you encounter a problem you cannot solve, use agent_request_intervention",
            "4. Track all file changes using agent_log_file_operation",
            "5. When complete, provide a summary of changes made",
            "",
            "Begin working on the task now.",
        ])

        return "\n".join(parts)

    def _parse_output_line(self, session: AgentSession, line: str) -> None:
        """
        Parse output line for special markers and file changes.

        Args:
            session: Agent session
            line: Output line to parse
        """
        line = line.strip()

        # Check for intervention request marker
        if "[NEEDS_INTERVENTION]" in line or "agent_request_intervention" in line:
            reason = line.split(":", 1)[-1].strip() if ":" in line else "Agent requested help"
            session.request_intervention(reason)

        # Try to detect file operations from output
        # These patterns match common Claude Code output
        if line.startswith("Created ") or line.startswith("Wrote "):
            path = line.split(" ", 1)[-1].strip()
            session.add_file_change(FileChange(
                file_path=path,
                change_type=FileChangeType.CREATE,
            ))
        elif line.startswith("Modified ") or line.startswith("Updated "):
            path = line.split(" ", 1)[-1].strip()
            session.add_file_change(FileChange(
                file_path=path,
                change_type=FileChangeType.MODIFY,
            ))
        elif line.startswith("Deleted ") or line.startswith("Removed "):
            path = line.split(" ", 1)[-1].strip()
            session.add_file_change(FileChange(
                file_path=path,
                change_type=FileChangeType.DELETE,
            ))

    def poll_agent(self, session_id: str) -> AgentStatus:
        """
        Poll the status of an agent.

        Args:
            session_id: Session identifier

        Returns:
            Current agent status
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        return session.status

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """
        Get full session details.

        Args:
            session_id: Session identifier

        Returns:
            AgentSession if found
        """
        return self.session_manager.get_session(session_id)

    def send_feedback(self, session_id: str, feedback: str) -> None:
        """
        Send feedback to a running agent.

        Note: Currently agents run non-interactively. This stores feedback
        for potential future interactive mode or for logging purposes.

        Args:
            session_id: Session identifier
            feedback: Feedback message
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Store feedback as progress update
        session.add_progress(f"[HUMAN_FEEDBACK] {feedback}")
        self.session_manager.update_session(session)

        logger.info(f"Feedback recorded for session {session_id}: {feedback[:50]}...")

    def terminate_agent(self, session_id: str, reason: str = "Manual termination") -> bool:
        """
        Terminate a running agent.

        Args:
            session_id: Session identifier
            reason: Reason for termination

        Returns:
            True if agent was terminated, False if not running
        """
        with self._lock:
            process = self._processes.get(session_id)

            if not process:
                logger.warning(f"No process found for session {session_id}")
                return False

            try:
                # Try graceful termination first
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    process.kill()
                    process.wait()

                logger.info(f"Terminated agent {session_id}: {reason}")

            except Exception as e:
                logger.error(f"Error terminating agent {session_id}: {e}")
                return False

        # Update session status
        session = self.session_manager.get_session(session_id)
        if session:
            session.status = AgentStatus.CANCELLED
            session.error_message = reason
            session.completed_at = datetime.utcnow()
            self.session_manager.update_session(session)

        return True

    def terminate_all_agents(self, reason: str = "Shutdown") -> int:
        """
        Terminate all running agents.

        Args:
            reason: Reason for termination

        Returns:
            Number of agents terminated
        """
        with self._lock:
            session_ids = list(self._processes.keys())

        count = 0
        for session_id in session_ids:
            if self.terminate_agent(session_id, reason):
                count += 1

        return count

    def get_active_agents(self) -> list[AgentSession]:
        """Get all currently active agent sessions."""
        return self.session_manager.get_active_sessions()

    def get_pending_interventions(self) -> list[AgentSession]:
        """Get sessions waiting for human intervention."""
        return self.session_manager.get_pending_interventions()

    def resolve_intervention(
        self,
        session_id: str,
        resolution: str,
        continue_agent: bool = False,
    ) -> None:
        """
        Resolve a pending intervention request.

        Args:
            session_id: Session identifier
            resolution: Resolution description
            continue_agent: Whether to spawn a new agent to continue
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.status != AgentStatus.NEEDS_HUMAN:
            raise ValueError(f"Session {session_id} does not need intervention")

        session.add_progress(f"[INTERVENTION_RESOLVED] {resolution}")

        if continue_agent:
            # Spawn continuation agent with resolution context
            new_context = session.context.copy()
            new_context["previous_session_id"] = session_id
            new_context["intervention_resolution"] = resolution
            new_context["previous_output"] = session.output[-5000:]  # Last 5K chars

            continuation_task = f"""Continue the previous task after human intervention.

Previous task: {session.task}

Intervention reason: {session.intervention_reason}

Resolution provided: {resolution}

Previous work summary:
{session.output[-2000:]}

Continue from where the previous agent left off, incorporating the resolution.
"""
            new_session = self.spawn_agent(
                task=continuation_task,
                working_dir=session.working_dir,
                context=new_context,
                execution_id=session.execution_id,
            )
            session.add_progress(f"[CONTINUATION] Spawned new session: {new_session.id}")

        # Mark original session as completed (with intervention)
        session.status = AgentStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        self.session_manager.update_session(session)

    def get_stats(self) -> dict[str, Any]:
        """Get integration statistics."""
        with self._lock:
            active_processes = len(self._processes)

        session_stats = self.session_manager.get_stats()
        session_stats["active_processes"] = active_processes
        session_stats["max_concurrent"] = self.config.max_concurrent_agents

        return session_stats


# Async wrapper for compatibility with async code
class AsyncHOTLClaudeIntegration:
    """Async wrapper around HOTLClaudeIntegration."""

    def __init__(self, integration: HOTLClaudeIntegration):
        """
        Initialize async wrapper.

        Args:
            integration: Sync integration to wrap
        """
        self._integration = integration
        self._executor = None

    async def spawn_agent(
        self,
        task: str,
        working_dir: Path,
        context: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
        execution_id: Optional[int] = None,
    ) -> AgentSession:
        """Async spawn_agent."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._integration.spawn_agent(
                task, working_dir, context, timeout, execution_id
            )
        )

    async def poll_agent(self, session_id: str) -> AgentStatus:
        """Async poll_agent."""
        return self._integration.poll_agent(session_id)

    async def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Async get_session."""
        return self._integration.get_session(session_id)

    async def send_feedback(self, session_id: str, feedback: str) -> None:
        """Async send_feedback."""
        self._integration.send_feedback(session_id, feedback)

    async def terminate_agent(self, session_id: str, reason: str = "Manual termination") -> bool:
        """Async terminate_agent."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._integration.terminate_agent(session_id, reason)
        )

    async def wait_for_completion(
        self,
        session_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> AgentSession:
        """
        Wait for an agent to complete.

        Args:
            session_id: Session to wait for
            poll_interval: Seconds between polls
            timeout: Optional timeout in seconds

        Returns:
            Completed session

        Raises:
            TimeoutError: If timeout exceeded
            ValueError: If session not found
        """
        start = time.time()

        while True:
            session = self._integration.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")

            if session.status in (
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.CANCELLED,
                AgentStatus.NEEDS_HUMAN,
            ):
                return session

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Timeout waiting for session {session_id}")

            await asyncio.sleep(poll_interval)

    def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        return self._integration.get_stats()
