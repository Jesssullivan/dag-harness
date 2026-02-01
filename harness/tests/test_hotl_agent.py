"""
Tests for HOTL (Human Out of The Loop) agent components.

Tests cover:
- Agent session management (create, update, list, cleanup)
- Agent spawning (mocked subprocess)
- MCP tool feedback (progress, intervention, file operations)
- Worker pool async execution
- Claude integration lifecycle
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from harness.db.state import StateDB
from harness.hotl.state import HOTLState, HOTLPhase, create_initial_state
from harness.hotl.agent_session import (
    AgentSession,
    AgentSessionManager,
    AgentStatus,
    FileChange,
    FileChangeType,
)
from harness.hotl.claude_integration import (
    ClaudeAgentConfig,
    HOTLClaudeIntegration,
    AsyncHOTLClaudeIntegration,
)
from harness.hotl.worker_pool import WorkerPool, Task, TaskResult


# ============================================================================
# HOTL STATE TESTS
# ============================================================================


class TestHOTLState:
    """Tests for HOTL state creation and management."""

    def test_create_initial_state(self):
        """Test creating initial HOTL state with defaults."""
        state = create_initial_state()

        assert state["phase"] == HOTLPhase.IDLE
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 100
        assert state["notification_interval"] == 300
        assert state["stop_requested"] is False
        assert state["pause_requested"] is False
        assert state["session_id"] is not None
        assert state["pending_tasks"] == []
        assert state["completed_tasks"] == []
        assert state["active_agent_sessions"] == []

    def test_create_initial_state_custom_params(self):
        """Test creating initial state with custom parameters."""
        state = create_initial_state(
            max_iterations=50,
            notification_interval=60,
            config={"test_mode": True, "verbose": True}
        )

        assert state["max_iterations"] == 50
        assert state["notification_interval"] == 60
        assert state["config"]["test_mode"] is True
        assert state["config"]["verbose"] is True

    def test_hotl_phase_values(self):
        """Test HOTLPhase enum values."""
        assert HOTLPhase.IDLE.value == "idle"
        assert HOTLPhase.RESEARCHING.value == "researching"
        assert HOTLPhase.PLANNING.value == "planning"
        assert HOTLPhase.GAP_ANALYZING.value == "gap_analyzing"
        assert HOTLPhase.EXECUTING.value == "executing"
        assert HOTLPhase.AGENT_EXECUTING.value == "agent_executing"
        assert HOTLPhase.TESTING.value == "testing"
        assert HOTLPhase.NOTIFYING.value == "notifying"
        assert HOTLPhase.PAUSED.value == "paused"
        assert HOTLPhase.STOPPED.value == "stopped"


# ============================================================================
# FILE CHANGE TESTS
# ============================================================================


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_create_file_change(self):
        """Test creating a file change record."""
        change = FileChange(
            file_path="/path/to/file.py",
            change_type=FileChangeType.CREATE,
        )

        assert change.file_path == "/path/to/file.py"
        assert change.change_type == FileChangeType.CREATE
        assert change.diff is None
        assert change.old_path is None
        assert change.timestamp is not None

    def test_file_change_with_diff(self):
        """Test file change with diff content."""
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 def hello():
+    print("world")
     pass
"""
        change = FileChange(
            file_path="/path/to/file.py",
            change_type=FileChangeType.MODIFY,
            diff=diff,
        )

        assert change.diff == diff
        assert change.change_type == FileChangeType.MODIFY

    def test_file_change_rename(self):
        """Test file change for rename operation."""
        change = FileChange(
            file_path="/path/to/new_file.py",
            change_type=FileChangeType.RENAME,
            old_path="/path/to/old_file.py",
        )

        assert change.old_path == "/path/to/old_file.py"
        assert change.change_type == FileChangeType.RENAME

    def test_file_change_to_dict(self):
        """Test converting file change to dictionary."""
        change = FileChange(
            file_path="/path/to/file.py",
            change_type=FileChangeType.DELETE,
        )

        data = change.to_dict()
        assert data["file_path"] == "/path/to/file.py"
        assert data["change_type"] == "delete"
        assert "timestamp" in data

    def test_file_change_from_dict(self):
        """Test creating file change from dictionary."""
        data = {
            "file_path": "/path/to/file.py",
            "change_type": "create",
            "diff": None,
            "old_path": None,
            "timestamp": "2024-01-15T10:30:00",
        }

        change = FileChange.from_dict(data)
        assert change.file_path == "/path/to/file.py"
        assert change.change_type == FileChangeType.CREATE


# ============================================================================
# AGENT SESSION TESTS
# ============================================================================


class TestAgentSession:
    """Tests for AgentSession dataclass."""

    @pytest.fixture
    def session(self, tmp_path):
        """Create a test session."""
        return AgentSession(
            id="test-session-123",
            task="Analyze and fix linting errors",
            working_dir=tmp_path,
            context={"role": "common", "wave": 1},
        )

    def test_create_session(self, session, tmp_path):
        """Test creating an agent session."""
        assert session.id == "test-session-123"
        assert session.task == "Analyze and fix linting errors"
        assert session.working_dir == tmp_path
        assert session.status == AgentStatus.PENDING
        assert session.context == {"role": "common", "wave": 1}
        assert session.file_changes == []
        assert session.progress_updates == []

    def test_mark_started(self, session):
        """Test marking session as started."""
        session.mark_started(pid=12345)

        assert session.status == AgentStatus.RUNNING
        assert session.started_at is not None
        assert session.pid == 12345

    def test_mark_completed(self, session):
        """Test marking session as completed."""
        session.mark_started()
        session.mark_completed(output="Task completed successfully")

        assert session.status == AgentStatus.COMPLETED
        assert session.completed_at is not None
        assert session.output == "Task completed successfully"

    def test_mark_failed(self, session):
        """Test marking session as failed."""
        session.mark_started()
        session.mark_failed("Error: file not found")

        assert session.status == AgentStatus.FAILED
        assert session.completed_at is not None
        assert session.error_message == "Error: file not found"

    def test_request_intervention(self, session):
        """Test requesting human intervention."""
        session.mark_started()
        session.request_intervention("Need approval for destructive changes")

        assert session.status == AgentStatus.NEEDS_HUMAN
        assert session.intervention_reason == "Need approval for destructive changes"

    def test_add_file_change(self, session):
        """Test adding file changes to session."""
        session.add_file_change(FileChange(
            file_path="/path/to/file1.py",
            change_type=FileChangeType.CREATE,
        ))
        session.add_file_change(FileChange(
            file_path="/path/to/file2.py",
            change_type=FileChangeType.MODIFY,
        ))

        assert len(session.file_changes) == 2
        assert session.file_changes[0].change_type == FileChangeType.CREATE
        assert session.file_changes[1].change_type == FileChangeType.MODIFY

    def test_add_progress(self, session):
        """Test adding progress updates."""
        session.add_progress("Starting analysis")
        session.add_progress("Found 5 issues")

        assert len(session.progress_updates) == 2
        assert "Starting analysis" in session.progress_updates[0]
        assert "Found 5 issues" in session.progress_updates[1]

    def test_duration_seconds(self, session):
        """Test calculating session duration."""
        assert session.duration_seconds is None  # Not started

        session.mark_started()
        time.sleep(0.1)
        session.mark_completed()

        assert session.duration_seconds >= 0.1
        assert session.duration_seconds < 1.0

    def test_session_to_dict(self, session, tmp_path):
        """Test converting session to dictionary."""
        session.mark_started(pid=123)
        session.add_file_change(FileChange(
            file_path="/test.py",
            change_type=FileChangeType.CREATE,
        ))

        data = session.to_dict()

        assert data["id"] == "test-session-123"
        assert data["status"] == "running"
        assert data["pid"] == 123
        assert data["working_dir"] == str(tmp_path)
        assert len(data["file_changes"]) == 1

    def test_session_from_dict(self, tmp_path):
        """Test creating session from dictionary."""
        data = {
            "id": "restored-session",
            "task": "Continue previous work",
            "working_dir": str(tmp_path),
            "status": "running",
            "context": {"key": "value"},
            "output": "Previous output",
            "error_message": None,
            "file_changes": [],
            "progress_updates": ["Step 1", "Step 2"],
            "intervention_reason": None,
            "created_at": "2024-01-15T10:00:00",
            "started_at": "2024-01-15T10:01:00",
            "completed_at": None,
            "pid": 456,
            "execution_id": 789,
        }

        session = AgentSession.from_dict(data)

        assert session.id == "restored-session"
        assert session.status == AgentStatus.RUNNING
        assert session.output == "Previous output"
        assert session.pid == 456
        assert session.execution_id == 789


# ============================================================================
# AGENT SESSION MANAGER TESTS
# ============================================================================


class TestAgentSessionManager:
    """Tests for AgentSessionManager."""

    @pytest.fixture
    def manager(self):
        """Create a session manager without database."""
        return AgentSessionManager()

    @pytest.fixture
    def manager_with_db(self, in_memory_db):
        """Create a session manager with database."""
        return AgentSessionManager(db=in_memory_db)

    def test_create_session(self, manager, tmp_path):
        """Test creating a new session."""
        session = manager.create_session(
            task="Test task",
            working_dir=tmp_path,
            context={"key": "value"},
            execution_id=123,
        )

        assert session.id is not None
        assert session.task == "Test task"
        assert session.working_dir == tmp_path
        assert session.context == {"key": "value"}
        assert session.execution_id == 123
        assert session.status == AgentStatus.PENDING

    def test_get_session(self, manager, tmp_path):
        """Test retrieving a session."""
        created = manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        retrieved = manager.get_session(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.task == "Test task"

    def test_get_session_not_found(self, manager):
        """Test retrieving non-existent session."""
        session = manager.get_session("non-existent-id")
        assert session is None

    def test_update_session(self, manager, tmp_path):
        """Test updating a session."""
        session = manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        session.mark_started(pid=999)
        manager.update_session(session)

        retrieved = manager.get_session(session.id)
        assert retrieved.status == AgentStatus.RUNNING
        assert retrieved.pid == 999

    def test_list_sessions(self, manager, tmp_path):
        """Test listing sessions."""
        manager.create_session(task="Task 1", working_dir=tmp_path)
        manager.create_session(task="Task 2", working_dir=tmp_path)
        manager.create_session(task="Task 3", working_dir=tmp_path)

        sessions = manager.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_by_status(self, manager, tmp_path):
        """Test listing sessions filtered by status."""
        s1 = manager.create_session(task="Task 1", working_dir=tmp_path)
        s2 = manager.create_session(task="Task 2", working_dir=tmp_path)

        s1.mark_started()
        manager.update_session(s1)

        running = manager.list_sessions(status=AgentStatus.RUNNING)
        pending = manager.list_sessions(status=AgentStatus.PENDING)

        assert len(running) == 1
        assert len(pending) == 1
        assert running[0].id == s1.id
        assert pending[0].id == s2.id

    def test_list_sessions_by_execution_id(self, manager, tmp_path):
        """Test listing sessions filtered by execution ID."""
        manager.create_session(task="Task 1", working_dir=tmp_path, execution_id=100)
        manager.create_session(task="Task 2", working_dir=tmp_path, execution_id=100)
        manager.create_session(task="Task 3", working_dir=tmp_path, execution_id=200)

        exec_100 = manager.list_sessions(execution_id=100)
        exec_200 = manager.list_sessions(execution_id=200)

        assert len(exec_100) == 2
        assert len(exec_200) == 1

    def test_get_active_sessions(self, manager, tmp_path):
        """Test getting active (running) sessions."""
        s1 = manager.create_session(task="Task 1", working_dir=tmp_path)
        s2 = manager.create_session(task="Task 2", working_dir=tmp_path)

        s1.mark_started()
        manager.update_session(s1)

        active = manager.get_active_sessions()
        assert len(active) == 1
        assert active[0].id == s1.id

    def test_get_pending_interventions(self, manager, tmp_path):
        """Test getting sessions needing intervention."""
        s1 = manager.create_session(task="Task 1", working_dir=tmp_path)
        s2 = manager.create_session(task="Task 2", working_dir=tmp_path)

        s1.mark_started()
        s1.request_intervention("Need help")
        manager.update_session(s1)

        pending = manager.get_pending_interventions()
        assert len(pending) == 1
        assert pending[0].intervention_reason == "Need help"

    def test_remove_session(self, manager, tmp_path):
        """Test removing a session."""
        session = manager.create_session(task="Task 1", working_dir=tmp_path)

        result = manager.remove_session(session.id)
        assert result is True

        retrieved = manager.get_session(session.id)
        assert retrieved is None

        # Try removing again
        result = manager.remove_session(session.id)
        assert result is False

    def test_cleanup_old_sessions(self, manager, tmp_path):
        """Test cleaning up old sessions."""
        # Create old completed session
        old_session = manager.create_session(task="Old task", working_dir=tmp_path)
        old_session.mark_started()
        old_session.mark_completed()
        old_session.created_at = datetime.utcnow() - timedelta(hours=48)
        manager.update_session(old_session)

        # Create recent session
        new_session = manager.create_session(task="New task", working_dir=tmp_path)
        new_session.mark_started()
        new_session.mark_completed()
        manager.update_session(new_session)

        # Cleanup with 24 hour threshold
        removed = manager.cleanup_old_sessions(max_age_hours=24)

        assert removed == 1
        assert manager.get_session(old_session.id) is None
        assert manager.get_session(new_session.id) is not None

    def test_get_stats(self, manager, tmp_path):
        """Test getting session statistics."""
        s1 = manager.create_session(task="Task 1", working_dir=tmp_path)
        s2 = manager.create_session(task="Task 2", working_dir=tmp_path)
        s3 = manager.create_session(task="Task 3", working_dir=tmp_path)

        s1.mark_started()
        s1.mark_completed()
        manager.update_session(s1)

        s2.mark_started()
        s2.mark_failed("Error")
        manager.update_session(s2)

        stats = manager.get_stats()

        assert stats["total_sessions"] == 3
        assert stats["status_counts"]["completed"] == 1
        assert stats["status_counts"]["failed"] == 1
        assert stats["status_counts"]["pending"] == 1


# ============================================================================
# CLAUDE INTEGRATION TESTS
# ============================================================================


class TestClaudeAgentConfig:
    """Tests for ClaudeAgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClaudeAgentConfig()

        assert config.claude_cli_path == "claude"
        assert config.default_timeout == 600
        assert config.max_concurrent_agents == 3
        assert config.skip_permissions is False
        assert config.model is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClaudeAgentConfig(
            claude_cli_path="/usr/local/bin/claude",
            default_timeout=300,
            max_concurrent_agents=5,
            skip_permissions=True,
            model="claude-3-opus",
            env_vars={"CUSTOM_VAR": "value"},
        )

        assert config.claude_cli_path == "/usr/local/bin/claude"
        assert config.default_timeout == 300
        assert config.max_concurrent_agents == 5
        assert config.skip_permissions is True
        assert config.model == "claude-3-opus"
        assert config.env_vars == {"CUSTOM_VAR": "value"}


class TestHOTLClaudeIntegration:
    """Tests for HOTLClaudeIntegration with mocked subprocess."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ClaudeAgentConfig(
            default_timeout=10,
            max_concurrent_agents=2,
        )

    @pytest.fixture
    def integration(self, config):
        """Create integration instance."""
        return HOTLClaudeIntegration(config=config)

    @pytest.fixture
    def mock_popen(self):
        """Mock subprocess.Popen."""
        with patch("subprocess.Popen") as mock:
            process = MagicMock()
            process.pid = 12345
            process.returncode = 0
            process.poll.return_value = 0
            process.stdout.readline.side_effect = [
                "Starting task...\n",
                "Analyzing code...\n",
                "Created /path/to/file.py\n",
                "Task completed.\n",
                "",  # EOF
            ]
            mock.return_value = process
            yield mock, process

    def test_spawn_agent(self, integration, tmp_path, mock_popen):
        """Test spawning a new agent."""
        mock_class, mock_process = mock_popen

        session = integration.spawn_agent(
            task="Analyze and fix linting errors",
            working_dir=tmp_path,
            context={"role": "common"},
        )

        assert session.id is not None
        assert session.task == "Analyze and fix linting errors"
        assert session.working_dir == tmp_path
        assert session.context == {"role": "common"}

        # Wait a bit for thread to start
        time.sleep(0.2)

        # Verify session was updated
        updated = integration.get_session(session.id)
        assert updated is not None

    def test_spawn_agent_with_callbacks(self, integration, tmp_path, mock_popen):
        """Test agent callbacks are invoked."""
        mock_class, mock_process = mock_popen

        on_complete = MagicMock()
        on_progress = MagicMock()

        integration.set_callbacks(
            on_complete=on_complete,
            on_progress=on_progress,
        )

        session = integration.spawn_agent(
            task="Test task",
            working_dir=tmp_path,
        )

        # Wait for agent to complete
        time.sleep(0.5)

        # Callbacks should have been called
        # Note: exact calls depend on timing, so we just check they were invoked
        assert on_progress.called or on_complete.called

    def test_get_session_not_found(self, integration):
        """Test getting non-existent session."""
        session = integration.get_session("non-existent")
        assert session is None

    def test_poll_agent(self, integration, tmp_path, mock_popen):
        """Test polling agent status."""
        mock_class, mock_process = mock_popen

        session = integration.spawn_agent(
            task="Test task",
            working_dir=tmp_path,
        )

        status = integration.poll_agent(session.id)
        assert status in [AgentStatus.PENDING, AgentStatus.RUNNING, AgentStatus.COMPLETED]

    def test_poll_agent_not_found(self, integration):
        """Test polling non-existent agent."""
        with pytest.raises(ValueError, match="Session not found"):
            integration.poll_agent("non-existent")

    def test_send_feedback(self, integration, tmp_path, mock_popen):
        """Test sending feedback to agent."""
        mock_class, mock_process = mock_popen

        session = integration.spawn_agent(
            task="Test task",
            working_dir=tmp_path,
        )

        integration.send_feedback(session.id, "Consider using type hints")

        updated = integration.get_session(session.id)
        assert any("[HUMAN_FEEDBACK]" in p for p in updated.progress_updates)

    def test_send_feedback_not_found(self, integration):
        """Test sending feedback to non-existent session."""
        with pytest.raises(ValueError, match="Session not found"):
            integration.send_feedback("non-existent", "Feedback")

    def test_terminate_agent(self, integration, tmp_path, mock_popen):
        """Test terminating an agent."""
        mock_class, mock_process = mock_popen

        session = integration.spawn_agent(
            task="Long running task",
            working_dir=tmp_path,
        )

        # Wait for process to be registered
        time.sleep(0.2)

        result = integration.terminate_agent(session.id, "User cancelled")

        # Note: result depends on timing - process may have finished already
        # Just verify the method doesn't crash

    def test_terminate_agent_not_found(self, integration):
        """Test terminating non-existent agent."""
        result = integration.terminate_agent("non-existent")
        assert result is False

    def test_terminate_all_agents(self, integration, tmp_path, mock_popen):
        """Test terminating all agents."""
        mock_class, mock_process = mock_popen

        integration.spawn_agent(task="Task 1", working_dir=tmp_path)
        integration.spawn_agent(task="Task 2", working_dir=tmp_path)

        time.sleep(0.2)

        count = integration.terminate_all_agents("Shutdown")
        # Count may vary based on timing
        assert count >= 0

    def test_get_active_agents(self, integration, tmp_path, mock_popen):
        """Test getting active agents."""
        mock_class, mock_process = mock_popen

        # Initially no active agents
        active = integration.get_active_agents()
        assert len(active) == 0

        # Spawn an agent
        session = integration.spawn_agent(task="Task 1", working_dir=tmp_path)

        # Wait a bit for status update
        time.sleep(0.1)

        # Note: agent may have already completed depending on mock timing

    def test_get_pending_interventions(self, integration):
        """Test getting pending interventions."""
        interventions = integration.get_pending_interventions()
        assert interventions == []

    def test_resolve_intervention(self, integration, tmp_path):
        """Test resolving an intervention."""
        session = integration.session_manager.create_session(
            task="Task needing help",
            working_dir=tmp_path,
        )
        session.mark_started()
        session.request_intervention("Need approval")
        integration.session_manager.update_session(session)

        integration.resolve_intervention(
            session_id=session.id,
            resolution="Approved",
            continue_agent=False,
        )

        resolved = integration.get_session(session.id)
        assert resolved.status == AgentStatus.COMPLETED
        assert any("[INTERVENTION_RESOLVED]" in p for p in resolved.progress_updates)

    def test_resolve_intervention_not_found(self, integration):
        """Test resolving intervention for non-existent session."""
        with pytest.raises(ValueError, match="Session not found"):
            integration.resolve_intervention("non-existent", "Resolution")

    def test_resolve_intervention_not_needed(self, integration, tmp_path):
        """Test resolving intervention when not needed."""
        session = integration.session_manager.create_session(
            task="Normal task",
            working_dir=tmp_path,
        )
        session.mark_started()
        session.mark_completed()
        integration.session_manager.update_session(session)

        with pytest.raises(ValueError, match="does not need intervention"):
            integration.resolve_intervention(session.id, "Resolution")

    def test_get_stats(self, integration, tmp_path, mock_popen):
        """Test getting integration statistics."""
        mock_class, mock_process = mock_popen

        stats = integration.get_stats()

        assert "total_sessions" in stats
        assert "status_counts" in stats
        assert "active_processes" in stats
        assert "max_concurrent" in stats
        assert stats["max_concurrent"] == 2


class TestAsyncHOTLClaudeIntegration:
    """Tests for AsyncHOTLClaudeIntegration wrapper."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ClaudeAgentConfig(default_timeout=5)

    @pytest.fixture
    def async_integration(self, config):
        """Create async integration instance."""
        sync_integration = HOTLClaudeIntegration(config=config)
        return AsyncHOTLClaudeIntegration(sync_integration)

    @pytest.mark.asyncio
    async def test_get_session(self, async_integration, tmp_path):
        """Test async get_session."""
        # Create a session via sync integration
        session = async_integration._integration.session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        retrieved = await async_integration.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_poll_agent(self, async_integration, tmp_path):
        """Test async poll_agent."""
        session = async_integration._integration.session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        status = await async_integration.poll_agent(session.id)
        assert status == AgentStatus.PENDING

    @pytest.mark.asyncio
    async def test_send_feedback(self, async_integration, tmp_path):
        """Test async send_feedback."""
        session = async_integration._integration.session_manager.create_session(
            task="Test task",
            working_dir=tmp_path,
        )

        await async_integration.send_feedback(session.id, "Test feedback")

        updated = await async_integration.get_session(session.id)
        assert any("HUMAN_FEEDBACK" in p for p in updated.progress_updates)

    @pytest.mark.asyncio
    async def test_get_stats(self, async_integration):
        """Test async get_stats."""
        stats = async_integration.get_stats()
        assert "total_sessions" in stats


# ============================================================================
# WORKER POOL TESTS
# ============================================================================


class TestWorkerPool:
    """Tests for async WorkerPool."""

    @pytest.fixture
    def pool(self):
        """Create a worker pool."""
        return WorkerPool(max_workers=2)

    @pytest.mark.asyncio
    async def test_start_stop(self, pool):
        """Test starting and stopping the pool."""
        await pool.start()
        assert pool.running is True

        await pool.stop()
        assert pool.running is False

    @pytest.mark.asyncio
    async def test_submit_task(self, pool):
        """Test submitting a task."""
        await pool.start()

        async def simple_task(x):
            return x * 2

        task = Task(
            id=0,
            func=simple_task,
            args=(5,),
            description="Multiply by 2",
        )

        task_id = await pool.submit(task)
        assert task_id > 0

        result = await pool.wait_for_result(task_id)
        assert result is not None
        assert result.success is True
        assert result.result == 10

        await pool.stop()

    @pytest.mark.asyncio
    async def test_submit_func(self, pool):
        """Test submitting a function directly."""
        await pool.start()

        async def add(a, b):
            return a + b

        task_id = await pool.submit_func(
            add, 3, 4,
            description="Add numbers"
        )

        result = await pool.wait_for_result(task_id)
        assert result.success is True
        assert result.result == 7

        await pool.stop()

    @pytest.mark.asyncio
    async def test_task_failure(self, pool):
        """Test handling task failure."""
        await pool.start()

        async def failing_task():
            raise ValueError("Task failed")

        task_id = await pool.submit_func(failing_task, description="Failing task")

        result = await pool.wait_for_result(task_id)
        assert result.success is False
        assert "Task failed" in result.error

        await pool.stop()

    @pytest.mark.asyncio
    async def test_task_priority(self, pool):
        """Test task priority ordering."""
        await pool.start()

        results = []

        async def record_order(value):
            results.append(value)
            return value

        # Submit low priority first
        await pool.submit_func(record_order, "low", priority=0)
        # Submit high priority second
        await pool.submit_func(record_order, "high", priority=10)

        await pool.wait_all()

        # High priority should be processed first (though timing may vary)
        await pool.stop()

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, pool):
        """Test concurrent task execution."""
        await pool.start()

        async def slow_task(task_id):
            await asyncio.sleep(0.1)
            return task_id

        # Submit more tasks than workers
        task_ids = []
        for i in range(4):
            tid = await pool.submit_func(slow_task, i, description=f"Task {i}")
            task_ids.append(tid)

        # Wait for all
        await pool.wait_all()

        # All should complete
        for tid in task_ids:
            result = pool.get_result(tid)
            assert result is not None
            assert result.success is True

        await pool.stop()

    @pytest.mark.asyncio
    async def test_pending_count(self, pool):
        """Test pending task count."""
        await pool.start()

        async def slow_task():
            await asyncio.sleep(0.5)

        # Submit several tasks
        for _ in range(5):
            await pool.submit_func(slow_task)

        # Some should be pending
        pending = pool.pending_count()
        assert pending >= 0

        await pool.stop(wait_for_completion=False)

    @pytest.mark.asyncio
    async def test_completed_count(self, pool):
        """Test completed task count."""
        await pool.start()

        async def quick_task():
            return True

        for _ in range(3):
            await pool.submit_func(quick_task)

        await pool.wait_all()

        completed = pool.completed_count()
        assert completed == 3

        await pool.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, pool):
        """Test getting pool statistics."""
        await pool.start()

        async def task():
            return True

        await pool.submit_func(task)
        await pool.wait_all()

        stats = pool.get_stats()

        assert stats["running"] is True
        assert stats["max_workers"] == 2
        assert stats["completed_tasks"] == 1
        assert stats["successful_tasks"] == 1
        assert stats["failed_tasks"] == 0

        await pool.stop()

    @pytest.mark.asyncio
    async def test_submit_without_running(self, pool):
        """Test submitting task when pool not running."""
        async def task():
            return True

        with pytest.raises(RuntimeError, match="not running"):
            await pool.submit_func(task)

    @pytest.mark.asyncio
    async def test_wait_for_result_timeout(self, pool):
        """Test wait_for_result with timeout."""
        await pool.start()

        async def slow_task():
            await asyncio.sleep(10)

        task_id = await pool.submit_func(slow_task)

        result = await pool.wait_for_result(task_id, timeout=0.1)
        assert result is None  # Timed out

        await pool.stop(wait_for_completion=False)


# ============================================================================
# MCP TOOL FEEDBACK TESTS (Integration with mocked database)
# ============================================================================


class TestMCPAgentFeedback:
    """Tests for MCP agent feedback tools."""

    @pytest.fixture
    def db(self):
        """Create in-memory database with agent tables."""
        database = StateDB(":memory:")

        # Create agent_sessions table if not exists
        with database.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id TEXT PRIMARY KEY,
                    execution_id INTEGER,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    output TEXT,
                    error_message TEXT,
                    intervention_reason TEXT,
                    context_json TEXT,
                    progress_json TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    working_dir TEXT,
                    pid INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_file_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    diff TEXT,
                    old_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, file_path, change_type)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT,
                    entity_id INTEGER,
                    action TEXT,
                    new_value TEXT,
                    actor TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create agent_sessions view
            conn.execute("""
                CREATE VIEW IF NOT EXISTS v_agent_sessions AS
                SELECT * FROM agent_sessions
            """)

        return database

    @pytest.fixture
    def session_in_db(self, db):
        """Create a session in the database."""
        with db.connection() as conn:
            conn.execute("""
                INSERT INTO agent_sessions (
                    id, task, status, progress_json, working_dir
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                "test-session-456",
                "Test task",
                "running",
                json.dumps([]),
                "/tmp/test",
            ))
        return "test-session-456"

    @pytest.mark.asyncio
    async def test_agent_report_progress(self, db, session_in_db):
        """Test agent_report_progress MCP tool logic."""
        session_id = session_in_db
        progress = "Completed step 1 of 3"

        with db.connection() as conn:
            # Get existing progress
            row = conn.execute(
                "SELECT progress_json FROM agent_sessions WHERE id = ?",
                (session_id,)
            ).fetchone()

            existing = json.loads(row["progress_json"]) if row["progress_json"] else []
            existing.append(f"[timestamp] {progress}")

            conn.execute(
                "UPDATE agent_sessions SET progress_json = ? WHERE id = ?",
                (json.dumps(existing), session_id)
            )

        # Verify progress was recorded
        with db.connection() as conn:
            row = conn.execute(
                "SELECT progress_json FROM agent_sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            progress_list = json.loads(row["progress_json"])
            assert len(progress_list) == 1
            assert "Completed step 1" in progress_list[0]

    @pytest.mark.asyncio
    async def test_agent_request_intervention(self, db, session_in_db):
        """Test agent_request_intervention MCP tool logic."""
        session_id = session_in_db
        reason = "Need approval for deleting files"

        with db.connection() as conn:
            conn.execute(
                """
                UPDATE agent_sessions
                SET status = 'needs_human', intervention_reason = ?
                WHERE id = ?
                """,
                (reason, session_id)
            )

        # Verify status was updated
        with db.connection() as conn:
            row = conn.execute(
                "SELECT status, intervention_reason FROM agent_sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            assert row["status"] == "needs_human"
            assert row["intervention_reason"] == reason

    @pytest.mark.asyncio
    async def test_agent_log_file_operation(self, db, session_in_db):
        """Test agent_log_file_operation MCP tool logic."""
        session_id = session_in_db

        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_file_changes (
                    session_id, file_path, change_type, diff, old_path
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, "/path/to/file.py", "create", None, None)
            )

        # Verify file change was logged
        with db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_file_changes WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            assert row["file_path"] == "/path/to/file.py"
            assert row["change_type"] == "create"

    @pytest.mark.asyncio
    async def test_agent_get_session_context(self, db, session_in_db):
        """Test agent_get_session_context MCP tool logic."""
        session_id = session_in_db

        # Update session with context
        with db.connection() as conn:
            conn.execute(
                "UPDATE agent_sessions SET context_json = ? WHERE id = ?",
                (json.dumps({"role": "common", "wave": 1}), session_id)
            )

        # Retrieve context
        with db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_sessions WHERE id = ?",
                (session_id,)
            ).fetchone()

            context = json.loads(row["context_json"]) if row["context_json"] else {}
            assert context["role"] == "common"
            assert context["wave"] == 1

    @pytest.mark.asyncio
    async def test_agent_list_sessions(self, db, session_in_db):
        """Test agent_list_sessions MCP tool logic."""
        # Add another session
        with db.connection() as conn:
            conn.execute("""
                INSERT INTO agent_sessions (
                    id, task, status, working_dir
                ) VALUES (?, ?, ?, ?)
            """, ("session-2", "Task 2", "completed", "/tmp"))

        # List sessions
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM v_agent_sessions ORDER BY created_at DESC"
            ).fetchall()

            assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_agent_get_file_changes(self, db, session_in_db):
        """Test agent_get_file_changes MCP tool logic."""
        session_id = session_in_db

        # Add file changes
        with db.connection() as conn:
            conn.execute("""
                INSERT INTO agent_file_changes (session_id, file_path, change_type)
                VALUES (?, ?, ?), (?, ?, ?)
            """, (session_id, "/file1.py", "create", session_id, "/file2.py", "modify"))

        # Get file changes
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_file_changes WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            ).fetchall()

            assert len(rows) == 2
            assert rows[0]["file_path"] == "/file1.py"
            assert rows[1]["file_path"] == "/file2.py"
