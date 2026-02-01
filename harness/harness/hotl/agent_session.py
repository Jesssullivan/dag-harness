"""Agent session management for HOTL Claude Code integration.

Provides data structures and session management for Claude Code subagents
spawned during HOTL autonomous operation.
"""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of a Claude Code agent session."""
    PENDING = "pending"        # Created but not yet started
    RUNNING = "running"        # Actively executing
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Terminated with error
    NEEDS_HUMAN = "needs_human"  # Agent requested human intervention
    CANCELLED = "cancelled"    # Externally terminated


class FileChangeType(str, Enum):
    """Type of file change made by an agent."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class FileChange:
    """Record of a file change made by an agent."""
    file_path: str
    change_type: FileChangeType
    diff: Optional[str] = None
    old_path: Optional[str] = None  # For renames
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "change_type": self.change_type.value,
            "diff": self.diff,
            "old_path": self.old_path,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileChange":
        """Create from dictionary."""
        return cls(
            file_path=data["file_path"],
            change_type=FileChangeType(data["change_type"]),
            diff=data.get("diff"),
            old_path=data.get("old_path"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
        )


@dataclass
class AgentSession:
    """
    Represents a Claude Code agent session.

    Tracks the lifecycle, output, and file changes of a spawned agent.
    """
    id: str
    task: str
    working_dir: Path
    status: AgentStatus = AgentStatus.PENDING
    context: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error_message: Optional[str] = None
    file_changes: list[FileChange] = field(default_factory=list)
    progress_updates: list[str] = field(default_factory=list)
    intervention_reason: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Process tracking
    pid: Optional[int] = None
    execution_id: Optional[int] = None  # Link to workflow execution

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "task": self.task,
            "working_dir": str(self.working_dir),
            "status": self.status.value,
            "context": self.context,
            "output": self.output,
            "error_message": self.error_message,
            "file_changes": [fc.to_dict() for fc in self.file_changes],
            "progress_updates": self.progress_updates,
            "intervention_reason": self.intervention_reason,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "pid": self.pid,
            "execution_id": self.execution_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSession":
        """Create from dictionary."""
        session = cls(
            id=data["id"],
            task=data["task"],
            working_dir=Path(data["working_dir"]),
            status=AgentStatus(data["status"]),
            context=data.get("context", {}),
            output=data.get("output", ""),
            error_message=data.get("error_message"),
            progress_updates=data.get("progress_updates", []),
            intervention_reason=data.get("intervention_reason"),
            pid=data.get("pid"),
            execution_id=data.get("execution_id"),
        )

        if data.get("created_at"):
            session.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            session.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            session.completed_at = datetime.fromisoformat(data["completed_at"])

        if data.get("file_changes"):
            session.file_changes = [FileChange.from_dict(fc) for fc in data["file_changes"]]

        return session

    def mark_started(self, pid: Optional[int] = None) -> None:
        """Mark the session as started."""
        self.status = AgentStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.pid = pid

    def mark_completed(self, output: str = "") -> None:
        """Mark the session as completed successfully."""
        self.status = AgentStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if output:
            self.output = output

    def mark_failed(self, error_message: str) -> None:
        """Mark the session as failed."""
        self.status = AgentStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message

    def request_intervention(self, reason: str) -> None:
        """Mark the session as needing human intervention."""
        self.status = AgentStatus.NEEDS_HUMAN
        self.intervention_reason = reason

    def add_file_change(self, file_change: FileChange) -> None:
        """Record a file change made by the agent."""
        self.file_changes.append(file_change)

    def add_progress(self, update: str) -> None:
        """Add a progress update from the agent."""
        self.progress_updates.append(f"[{datetime.utcnow().isoformat()}] {update}")

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get session duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()


class AgentSessionManager:
    """
    Manages active and historical Claude Code agent sessions.

    Thread-safe session tracking with persistence support.
    """

    def __init__(self, db: Optional[Any] = None):
        """
        Initialize the session manager.

        Args:
            db: Optional StateDB instance for persistence
        """
        self.db = db
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        task: str,
        working_dir: Path,
        context: Optional[dict[str, Any]] = None,
        execution_id: Optional[int] = None,
    ) -> AgentSession:
        """
        Create a new agent session.

        Args:
            task: Task description for the agent
            working_dir: Working directory for the agent
            context: Optional context dict with additional info
            execution_id: Optional link to workflow execution

        Returns:
            Created AgentSession
        """
        session_id = str(uuid.uuid4())
        session = AgentSession(
            id=session_id,
            task=task,
            working_dir=working_dir,
            context=context or {},
            execution_id=execution_id,
        )

        with self._lock:
            self._sessions[session_id] = session

        # Persist to database if available
        if self.db:
            self._persist_session(session)

        logger.info(f"Created agent session {session_id}: {task[:50]}...")
        return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            AgentSession if found, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session

        # Try loading from database
        if self.db:
            session = self._load_session(session_id)
            if session:
                with self._lock:
                    self._sessions[session_id] = session
                return session

        return None

    def update_session(self, session: AgentSession) -> None:
        """
        Update a session in the manager.

        Args:
            session: Session to update
        """
        with self._lock:
            self._sessions[session.id] = session

        if self.db:
            self._persist_session(session)

    def list_sessions(
        self,
        status: Optional[AgentStatus] = None,
        execution_id: Optional[int] = None,
        limit: int = 100,
    ) -> list[AgentSession]:
        """
        List sessions with optional filtering.

        Args:
            status: Filter by status
            execution_id: Filter by execution ID
            limit: Maximum number to return

        Returns:
            List of matching sessions
        """
        with self._lock:
            sessions = list(self._sessions.values())

        if status:
            sessions = [s for s in sessions if s.status == status]
        if execution_id is not None:
            sessions = [s for s in sessions if s.execution_id == execution_id]

        # Sort by created_at descending
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    def get_active_sessions(self) -> list[AgentSession]:
        """Get all currently active sessions."""
        return self.list_sessions(status=AgentStatus.RUNNING)

    def get_pending_interventions(self) -> list[AgentSession]:
        """Get sessions that need human intervention."""
        return self.list_sessions(status=AgentStatus.NEEDS_HUMAN)

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from the manager.

        Args:
            session_id: Session to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Remove sessions older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of sessions removed
        """
        cutoff = datetime.utcnow()
        removed = 0

        with self._lock:
            to_remove = []
            for session_id, session in self._sessions.items():
                if session.status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED):
                    age_hours = (cutoff - session.created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        to_remove.append(session_id)

            for session_id in to_remove:
                del self._sessions[session_id]
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old agent sessions")

        return removed

    def _persist_session(self, session: AgentSession) -> None:
        """Persist session to database."""
        if not self.db:
            return

        try:
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_sessions (
                        id, execution_id, task, status, output,
                        error_message, intervention_reason,
                        context_json, progress_json,
                        created_at, started_at, completed_at,
                        working_dir, pid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status = excluded.status,
                        output = excluded.output,
                        error_message = excluded.error_message,
                        intervention_reason = excluded.intervention_reason,
                        progress_json = excluded.progress_json,
                        started_at = excluded.started_at,
                        completed_at = excluded.completed_at,
                        pid = excluded.pid
                    """,
                    (
                        session.id,
                        session.execution_id,
                        session.task,
                        session.status.value,
                        session.output,
                        session.error_message,
                        session.intervention_reason,
                        json.dumps(session.context),
                        json.dumps(session.progress_updates),
                        session.created_at.isoformat(),
                        session.started_at.isoformat() if session.started_at else None,
                        session.completed_at.isoformat() if session.completed_at else None,
                        str(session.working_dir),
                        session.pid,
                    )
                )

                # Persist file changes
                for fc in session.file_changes:
                    conn.execute(
                        """
                        INSERT INTO agent_file_changes (
                            session_id, file_path, change_type, diff, old_path, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, file_path, change_type) DO UPDATE SET
                            diff = excluded.diff
                        """,
                        (
                            session.id,
                            fc.file_path,
                            fc.change_type.value,
                            fc.diff,
                            fc.old_path,
                            fc.timestamp.isoformat(),
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to persist session {session.id}: {e}")

    def _load_session(self, session_id: str) -> Optional[AgentSession]:
        """Load session from database."""
        if not self.db:
            return None

        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM agent_sessions WHERE id = ?",
                    (session_id,)
                ).fetchone()

                if not row:
                    return None

                # Load file changes
                fc_rows = conn.execute(
                    "SELECT * FROM agent_file_changes WHERE session_id = ?",
                    (session_id,)
                ).fetchall()

                session = AgentSession(
                    id=row["id"],
                    task=row["task"],
                    working_dir=Path(row["working_dir"]),
                    status=AgentStatus(row["status"]),
                    output=row["output"] or "",
                    error_message=row["error_message"],
                    intervention_reason=row["intervention_reason"],
                    context=json.loads(row["context_json"]) if row["context_json"] else {},
                    progress_updates=json.loads(row["progress_json"]) if row["progress_json"] else [],
                    execution_id=row["execution_id"],
                    pid=row["pid"],
                )

                if row["created_at"]:
                    session.created_at = datetime.fromisoformat(row["created_at"])
                if row["started_at"]:
                    session.started_at = datetime.fromisoformat(row["started_at"])
                if row["completed_at"]:
                    session.completed_at = datetime.fromisoformat(row["completed_at"])

                for fc_row in fc_rows:
                    session.file_changes.append(FileChange(
                        file_path=fc_row["file_path"],
                        change_type=FileChangeType(fc_row["change_type"]),
                        diff=fc_row["diff"],
                        old_path=fc_row["old_path"],
                        timestamp=datetime.fromisoformat(fc_row["created_at"]) if fc_row["created_at"] else datetime.utcnow(),
                    ))

                return session

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics."""
        with self._lock:
            sessions = list(self._sessions.values())

        status_counts = {}
        for status in AgentStatus:
            status_counts[status.value] = sum(1 for s in sessions if s.status == status)

        total_file_changes = sum(len(s.file_changes) for s in sessions)

        completed = [s for s in sessions if s.duration_seconds is not None]
        avg_duration = (
            sum(s.duration_seconds for s in completed) / len(completed)
            if completed else 0.0
        )

        return {
            "total_sessions": len(sessions),
            "status_counts": status_counts,
            "total_file_changes": total_file_changes,
            "average_duration_seconds": avg_duration,
            "pending_interventions": status_counts.get(AgentStatus.NEEDS_HUMAN.value, 0),
        }
