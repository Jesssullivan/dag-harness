"""Tests for the hooks framework.

Tests cover:
- Base hook classes (PreToolUse, PostToolUse, SubagentStart, SubagentStop)
- HookManager registration and execution
- FileChangeTrackerHook
- AuditHook and AuditLogger
- VerificationHook
- ToolLogger (database-backed audit trail)
- Pre/Post tool use hook scripts (Claude Code protocol)
"""

import json
import os
import subprocess
import sys
from datetime import datetime

import pytest

from harness.hooks.audit import (
    AuditEntry,
    AuditHook,
    AuditLevel,
    AuditLogger,
)
from harness.hooks.base import (
    HookContext,
    HookManager,
    HookPriority,
    HookResult,
    PostToolUseHook,
    PreToolUseHook,
    SubagentStartHook,
    SubagentStopHook,
    create_simple_pre_tool_hook,
    create_simple_subagent_stop_hook,
)
from harness.hooks.file_tracker import (
    FileChange,
    FileChangeTrackerHook,
    FileChangeType,
    TrackedFile,
)
from harness.hooks.verification import (
    VerificationHook,
    VerificationResult,
    VerificationStatus,
    create_file_exists_verification,
)

# ============================================================================
# BASE HOOK TESTS
# ============================================================================


class TestHookContext:
    """Tests for HookContext."""

    def test_create_context(self):
        """Test creating a hook context."""
        context = HookContext(agent_id="agent-123")
        assert context.agent_id == "agent-123"
        assert context.session_id is None
        assert context.metadata == {}
        assert isinstance(context.timestamp, datetime)

    def test_context_with_all_fields(self):
        """Test context with all fields."""
        context = HookContext(
            agent_id="agent-123",
            session_id="session-456",
            execution_id=789,
            parent_agent_id="parent-111",
            tool_name="Write",
            tool_input={"file_path": "/tmp/test.txt"},
            metadata={"custom": "data"},
        )
        assert context.agent_id == "agent-123"
        assert context.session_id == "session-456"
        assert context.execution_id == 789
        assert context.parent_agent_id == "parent-111"
        assert context.tool_name == "Write"
        assert context.tool_input == {"file_path": "/tmp/test.txt"}
        assert context.metadata == {"custom": "data"}


class TestHookPriority:
    """Tests for HookPriority."""

    def test_priority_ordering(self):
        """Test that priorities are ordered correctly."""
        assert HookPriority.CRITICAL < HookPriority.HIGH
        assert HookPriority.HIGH < HookPriority.NORMAL
        assert HookPriority.NORMAL < HookPriority.LOW
        assert HookPriority.LOW < HookPriority.AUDIT


class TestHookResult:
    """Tests for HookResult."""

    def test_result_values(self):
        """Test HookResult enum values."""
        assert HookResult.CONTINUE.value == "continue"
        assert HookResult.BLOCK.value == "block"
        assert HookResult.MODIFY.value == "modify"


# ============================================================================
# CONCRETE HOOK IMPLEMENTATIONS FOR TESTING
# ============================================================================


class TestPreToolHook(PreToolUseHook):
    """Test implementation of PreToolUseHook."""

    def __init__(self, name="test_pre_hook", should_block=False, modify_input=None):
        super().__init__(name=name)
        self.should_block = should_block
        self.modify_input = modify_input
        self.calls = []

    async def on_pre_tool_use(
        self, tool_name: str, tool_input: dict, context: HookContext
    ) -> tuple[HookResult, dict | None]:
        self.calls.append((tool_name, tool_input, context))
        if self.should_block:
            return HookResult.BLOCK, None
        if self.modify_input:
            return HookResult.MODIFY, self.modify_input
        return HookResult.CONTINUE, None


class TestPostToolHook(PostToolUseHook):
    """Test implementation of PostToolUseHook."""

    def __init__(self, name="test_post_hook"):
        super().__init__(name=name)
        self.calls = []

    async def on_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: dict,
        context: HookContext,
    ) -> None:
        self.calls.append((tool_name, tool_input, tool_output, context))


class TestSubagentStartHook(SubagentStartHook):
    """Test implementation of SubagentStartHook."""

    def __init__(self, name="test_start_hook", should_block=False, inject_context=None):
        super().__init__(name=name)
        self.should_block = should_block
        self.inject_context = inject_context
        self.calls = []

    async def on_subagent_start(
        self, agent_id: str, task: str, context: HookContext
    ) -> tuple[HookResult, dict | None]:
        self.calls.append((agent_id, task, context))
        if self.should_block:
            return HookResult.BLOCK, None
        return HookResult.CONTINUE, self.inject_context


class TestSubagentStopHook(SubagentStopHook):
    """Test implementation of SubagentStopHook."""

    def __init__(self, name="test_stop_hook"):
        super().__init__(name=name)
        self.calls = []

    async def on_subagent_stop(self, agent_id: str, result: dict, context: HookContext) -> None:
        self.calls.append((agent_id, result, context))


# ============================================================================
# HOOK MANAGER TESTS
# ============================================================================


class TestHookManager:
    """Tests for HookManager."""

    @pytest.fixture
    def manager(self):
        """Create a hook manager."""
        return HookManager()

    @pytest.fixture
    def context(self):
        """Create a hook context."""
        return HookContext(agent_id="test-agent")

    def test_register_pre_tool_hook(self, manager):
        """Test registering a PreToolUseHook."""
        hook = TestPreToolHook()
        manager.register(hook)
        assert "test_pre_hook" in manager.list_hooks()["pre_tool_use"]

    def test_register_post_tool_hook(self, manager):
        """Test registering a PostToolUseHook."""
        hook = TestPostToolHook()
        manager.register(hook)
        assert "test_post_hook" in manager.list_hooks()["post_tool_use"]

    def test_register_subagent_start_hook(self, manager):
        """Test registering a SubagentStartHook."""
        hook = TestSubagentStartHook()
        manager.register(hook)
        assert "test_start_hook" in manager.list_hooks()["subagent_start"]

    def test_register_subagent_stop_hook(self, manager):
        """Test registering a SubagentStopHook."""
        hook = TestSubagentStopHook()
        manager.register(hook)
        assert "test_stop_hook" in manager.list_hooks()["subagent_stop"]

    def test_unregister_hook(self, manager):
        """Test unregistering a hook."""
        hook = TestPreToolHook()
        manager.register(hook)
        assert manager.unregister("test_pre_hook") is True
        assert "test_pre_hook" not in manager.list_hooks()["pre_tool_use"]

    def test_unregister_nonexistent(self, manager):
        """Test unregistering a non-existent hook."""
        assert manager.unregister("nonexistent") is False

    def test_get_hook(self, manager):
        """Test getting a hook by name."""
        hook = TestPreToolHook()
        manager.register(hook)
        retrieved = manager.get_hook("test_pre_hook")
        assert retrieved is hook

    def test_get_hook_not_found(self, manager):
        """Test getting a non-existent hook."""
        assert manager.get_hook("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_pre_tool_use(self, manager, context):
        """Test executing PreToolUse hooks."""
        hook = TestPreToolHook()
        manager.register(hook)

        result, final_input = await manager.execute_pre_tool_use(
            "Write", {"file_path": "/tmp/test.txt"}, context
        )

        assert result == HookResult.CONTINUE
        assert len(hook.calls) == 1
        assert hook.calls[0][0] == "Write"

    @pytest.mark.asyncio
    async def test_execute_pre_tool_use_blocking(self, manager, context):
        """Test blocking in PreToolUse hooks."""
        hook = TestPreToolHook(should_block=True)
        manager.register(hook)

        result, _ = await manager.execute_pre_tool_use(
            "Write", {"file_path": "/tmp/test.txt"}, context
        )

        assert result == HookResult.BLOCK

    @pytest.mark.asyncio
    async def test_execute_pre_tool_use_modify(self, manager, context):
        """Test modifying input in PreToolUse hooks."""
        modified = {"file_path": "/tmp/modified.txt"}
        hook = TestPreToolHook(modify_input=modified)
        manager.register(hook)

        result, final_input = await manager.execute_pre_tool_use(
            "Write", {"file_path": "/tmp/test.txt"}, context
        )

        assert result == HookResult.CONTINUE
        assert final_input == modified

    @pytest.mark.asyncio
    async def test_execute_post_tool_use(self, manager, context):
        """Test executing PostToolUse hooks."""
        hook = TestPostToolHook()
        manager.register(hook)

        await manager.execute_post_tool_use(
            "Write",
            {"file_path": "/tmp/test.txt"},
            {"success": True},
            context,
        )

        assert len(hook.calls) == 1
        assert hook.calls[0][0] == "Write"
        assert hook.calls[0][2] == {"success": True}

    @pytest.mark.asyncio
    async def test_execute_subagent_start(self, manager, context):
        """Test executing SubagentStart hooks."""
        hook = TestSubagentStartHook()
        manager.register(hook)

        result, ctx = await manager.execute_subagent_start("agent-123", "Test task", context)

        assert result == HookResult.CONTINUE
        assert len(hook.calls) == 1
        assert hook.calls[0][0] == "agent-123"
        assert hook.calls[0][1] == "Test task"

    @pytest.mark.asyncio
    async def test_execute_subagent_start_inject_context(self, manager, context):
        """Test injecting context in SubagentStart hooks."""
        hook = TestSubagentStartHook(inject_context={"injected": True})
        manager.register(hook)

        result, ctx = await manager.execute_subagent_start("agent-123", "Test task", context)

        assert result == HookResult.CONTINUE
        assert ctx == {"injected": True}

    @pytest.mark.asyncio
    async def test_execute_subagent_stop(self, manager, context):
        """Test executing SubagentStop hooks."""
        hook = TestSubagentStopHook()
        manager.register(hook)

        await manager.execute_subagent_stop("agent-123", {"status": "completed"}, context)

        assert len(hook.calls) == 1
        assert hook.calls[0][0] == "agent-123"
        assert hook.calls[0][1] == {"status": "completed"}

    @pytest.mark.asyncio
    async def test_hook_priority_ordering(self, manager, context):
        """Test that hooks execute in priority order."""
        order = []

        class OrderedHook(PreToolUseHook):
            def __init__(self, name, priority, order_list):
                super().__init__(name=name, priority=priority)
                self.order_list = order_list

            async def on_pre_tool_use(self, tool_name, tool_input, context):
                self.order_list.append(self.name)
                return HookResult.CONTINUE, None

        manager.register(OrderedHook("low", HookPriority.LOW, order))
        manager.register(OrderedHook("high", HookPriority.HIGH, order))
        manager.register(OrderedHook("normal", HookPriority.NORMAL, order))

        await manager.execute_pre_tool_use("test", {}, context)

        assert order == ["high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_disabled_hook_not_executed(self, manager, context):
        """Test that disabled hooks are not executed."""
        hook = TestPreToolHook()
        hook.enabled = False
        manager.register(hook)

        await manager.execute_pre_tool_use("Write", {}, context)

        assert len(hook.calls) == 0

    def test_get_stats(self, manager):
        """Test getting hook statistics."""
        hook = TestPreToolHook()
        manager.register(hook)
        stats = manager.get_stats()
        assert "test_pre_hook" in stats


# ============================================================================
# FILE TRACKER HOOK TESTS
# ============================================================================


class TestFileChangeTrackerHook:
    """Tests for FileChangeTrackerHook."""

    @pytest.fixture
    def context(self):
        """Create a hook context."""
        return HookContext(agent_id="test-agent", session_id="test-session")

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a file change tracker."""
        return FileChangeTrackerHook(
            backup_dir=tmp_path / "backups",
            store_content=True,
            log_file=tmp_path / "changes.jsonl",
        )

    def test_tracker_initialization(self, tracker, tmp_path):
        """Test tracker initialization."""
        assert tracker.name == "file_change_tracker"
        assert tracker.backup_dir == tmp_path / "backups"
        assert tracker.store_content is True

    @pytest.mark.asyncio
    async def test_track_file_tool(self, tracker, context, tmp_path):
        """Test tracking a file write tool."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        result, _ = await tracker.on_pre_tool_use(
            "Write",
            {"file_path": str(test_file)},
            context,
        )

        assert result == HookResult.CONTINUE
        assert str(test_file) in tracker._tracked_files

        # Verify original content is stored
        tracked = tracker._tracked_files[str(test_file)]
        assert tracked.original_content == "original content"

    @pytest.mark.asyncio
    async def test_record_change(self, tracker, context, tmp_path):
        """Test recording a file change."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original")

        # Pre-hook
        await tracker.on_pre_tool_use("Write", {"file_path": str(test_file)}, context)

        # Simulate file modification
        test_file.write_text("modified")

        # Post-hook
        await tracker.on_post_tool_use(
            "Write",
            {"file_path": str(test_file)},
            {"success": True},
            context,
        )

        changes = tracker.get_changes()
        assert len(changes) == 1
        assert changes[0].path == str(test_file)
        assert changes[0].agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_generate_diff(self, tracker, context, tmp_path):
        """Test diff generation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\n")

        await tracker.on_pre_tool_use("Edit", {"file_path": str(test_file)}, context)

        test_file.write_text("line1\nmodified\n")

        await tracker.on_post_tool_use(
            "Edit",
            {"file_path": str(test_file)},
            {"success": True},
            context,
        )

        changes = tracker.get_changes()
        assert len(changes) == 1
        assert changes[0].diff is not None
        assert "-line2" in changes[0].diff
        assert "+modified" in changes[0].diff

    def test_get_stats(self, tracker):
        """Test getting tracker statistics."""
        stats = tracker.get_stats()
        assert "total_tracked_files" in stats
        assert "total_changes" in stats
        assert "changes_by_type" in stats

    def test_clear(self, tracker, tmp_path):
        """Test clearing tracker state."""
        tracker._tracked_files["test"] = TrackedFile(path="test")
        tracker.clear()
        assert len(tracker._tracked_files) == 0


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_to_dict(self):
        """Test FileChange serialization."""
        change = FileChange(
            path="/tmp/test.txt",
            change_type=FileChangeType.CREATE,
            agent_id="agent-123",
            tool_name="Write",
        )
        data = change.to_dict()
        assert data["path"] == "/tmp/test.txt"
        assert data["change_type"] == "create"
        assert data["agent_id"] == "agent-123"

    def test_from_dict(self):
        """Test FileChange deserialization."""
        data = {
            "path": "/tmp/test.txt",
            "change_type": "modify",
            "agent_id": "agent-456",
            "timestamp": "2024-01-15T12:00:00",
        }
        change = FileChange.from_dict(data)
        assert change.path == "/tmp/test.txt"
        assert change.change_type == FileChangeType.MODIFY
        assert change.agent_id == "agent-456"


# ============================================================================
# AUDIT HOOK TESTS
# ============================================================================


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create an audit logger."""
        return AuditLogger(log_file=tmp_path / "audit.jsonl")

    def test_log_entry(self, logger):
        """Test logging an entry."""
        entry = AuditEntry(
            event_type="test",
            agent_id="agent-123",
            message="Test message",
        )
        logger.log(entry)
        assert len(logger._entries) == 1

    def test_get_entries(self, logger):
        """Test getting entries."""
        for i in range(5):
            logger.log(
                AuditEntry(
                    event_type="test",
                    agent_id=f"agent-{i}",
                    message=f"Message {i}",
                )
            )

        entries = logger.get_entries()
        assert len(entries) == 5

    def test_get_entries_filtered(self, logger):
        """Test filtering entries."""
        logger.log(AuditEntry(event_type="start", agent_id="agent-1", message=""))
        logger.log(AuditEntry(event_type="stop", agent_id="agent-1", message=""))
        logger.log(AuditEntry(event_type="start", agent_id="agent-2", message=""))

        entries = logger.get_entries(agent_id="agent-1")
        assert len(entries) == 2

        entries = logger.get_entries(event_type="start")
        assert len(entries) == 2

    def test_write_to_file(self, logger, tmp_path):
        """Test writing to log file."""
        logger.log(AuditEntry(event_type="test", agent_id="agent-1", message="Test"))
        logger.close()

        log_file = tmp_path / "audit.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["event_type"] == "test"


class TestAuditHook:
    """Tests for AuditHook."""

    @pytest.fixture
    def context(self):
        """Create a hook context."""
        return HookContext(agent_id="test-agent", session_id="test-session")

    @pytest.fixture
    def hook(self, tmp_path):
        """Create an audit hook."""
        return AuditHook(
            log_file=tmp_path / "audit.jsonl",
            level=AuditLevel.DETAILED,
        )

    @pytest.mark.asyncio
    async def test_log_subagent_start(self, hook, context):
        """Test logging subagent start."""
        await hook.on_subagent_start("agent-123", "Test task", context)

        entries = hook.get_audit_entries()
        assert len(entries) == 1
        assert entries[0].event_type == "subagent_start"
        assert entries[0].agent_id == "agent-123"

    @pytest.mark.asyncio
    async def test_log_subagent_stop(self, hook, context):
        """Test logging subagent stop."""
        # Start first to track duration
        await hook.on_subagent_start("agent-123", "Test task", context)

        await hook.on_subagent_stop("agent-123", {"status": "completed"}, context)

        entries = hook.get_audit_entries()
        assert len(entries) == 2
        stop_entry = [e for e in entries if e.event_type == "subagent_stop"][0]
        assert stop_entry.agent_id == "agent-123"
        assert stop_entry.duration_ms is not None

    @pytest.mark.asyncio
    async def test_log_tool_use(self, hook, context):
        """Test logging tool use."""
        await hook.on_pre_tool_use("Write", {"file_path": "/tmp/test.txt"}, context)

        entries = hook.get_audit_entries(event_type="tool_use")
        assert len(entries) == 1
        assert entries[0].details["tool_name"] == "Write"

    @pytest.mark.asyncio
    async def test_redact_sensitive(self, hook, context):
        """Test redacting sensitive data."""
        await hook.on_pre_tool_use(
            "Bash",
            {"command": "echo test", "password": "secret123"},
            context,
        )

        entries = hook.get_audit_entries(event_type="tool_use")
        assert entries[0].details["input"]["password"] == "[REDACTED]"


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_to_dict(self):
        """Test AuditEntry serialization."""
        entry = AuditEntry(
            event_type="test",
            agent_id="agent-123",
            message="Test message",
            level=AuditLevel.NORMAL,
        )
        data = entry.to_dict()
        assert data["event_type"] == "test"
        assert data["agent_id"] == "agent-123"
        assert data["level"] == "normal"

    def test_from_dict(self):
        """Test AuditEntry deserialization."""
        data = {
            "event_type": "subagent_start",
            "agent_id": "agent-456",
            "timestamp": "2024-01-15T12:00:00",
            "level": "detailed",
            "message": "Started",
        }
        entry = AuditEntry.from_dict(data)
        assert entry.event_type == "subagent_start"
        assert entry.agent_id == "agent-456"
        assert entry.level == AuditLevel.DETAILED

    def test_to_log_line(self):
        """Test formatting as log line."""
        entry = AuditEntry(
            event_type="test",
            agent_id="agent-12345678",
            message="Test message",
        )
        line = entry.to_log_line()
        assert "[TEST]" in line
        assert "agent=agent-12" in line
        assert "Test message" in line


# ============================================================================
# VERIFICATION HOOK TESTS
# ============================================================================


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_to_dict(self):
        """Test VerificationResult serialization."""
        result = VerificationResult(
            name="test",
            status=VerificationStatus.PASSED,
            message="All tests passed",
        )
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "passed"

    def test_from_dict(self):
        """Test VerificationResult deserialization."""
        data = {
            "name": "lint",
            "status": "failed",
            "message": "Lint failed",
            "errors": ["Error 1", "Error 2"],
        }
        result = VerificationResult.from_dict(data)
        assert result.name == "lint"
        assert result.status == VerificationStatus.FAILED
        assert len(result.errors) == 2


class TestVerificationHook:
    """Tests for VerificationHook."""

    @pytest.fixture
    def context(self):
        """Create a hook context."""
        return HookContext(agent_id="test-agent")

    def test_create_verification_hook(self):
        """Test creating a verification hook."""

        async def verify_fn(agent_id, result, context):
            return VerificationResult(
                name="test",
                status=VerificationStatus.PASSED,
            )

        hook = VerificationHook(
            name="test_verification",
            verification_fn=verify_fn,
        )
        assert hook.name == "test_verification"
        assert hook.required is True

    @pytest.mark.asyncio
    async def test_verification_passed(self, context):
        """Test successful verification."""

        async def verify_fn(agent_id, result, context):
            return VerificationResult(
                name="test",
                status=VerificationStatus.PASSED,
                message="OK",
            )

        hook = VerificationHook(name="test", verification_fn=verify_fn)
        await hook.on_subagent_stop("agent-123", {"status": "completed"}, context)

        history = hook.get_history()
        assert len(history) == 1
        assert history[0].status == VerificationStatus.PASSED

    @pytest.mark.asyncio
    async def test_verification_failed(self, context):
        """Test failed verification."""

        async def verify_fn(agent_id, result, context):
            return VerificationResult(
                name="test",
                status=VerificationStatus.FAILED,
                message="Test failed",
                errors=["Error 1"],
            )

        hook = VerificationHook(
            name="test",
            verification_fn=verify_fn,
            retry_on_failure=True,
            max_retries=2,
        )
        await hook.on_subagent_stop("agent-123", {"status": "completed"}, context)

        assert context.metadata.get("retry_requested") is True

    def test_pass_rate(self):
        """Test pass rate calculation."""

        async def verify_fn(agent_id, result, context):
            return VerificationResult(name="test", status=VerificationStatus.PASSED)

        hook = VerificationHook(name="test", verification_fn=verify_fn)
        hook._history = [
            VerificationResult(name="test", status=VerificationStatus.PASSED),
            VerificationResult(name="test", status=VerificationStatus.PASSED),
            VerificationResult(name="test", status=VerificationStatus.FAILED),
        ]

        assert hook.get_pass_rate() == pytest.approx(66.67, rel=0.01)


# ============================================================================
# SIMPLE HOOK CREATION TESTS
# ============================================================================


class TestSimpleHookCreation:
    """Tests for simple hook creation functions."""

    @pytest.fixture
    def context(self):
        """Create a hook context."""
        return HookContext(agent_id="test-agent")

    @pytest.mark.asyncio
    async def test_create_simple_pre_tool_hook(self, context):
        """Test creating a simple PreToolUse hook."""

        def callback(tool_name, tool_input, ctx):
            return HookResult.CONTINUE, None

        hook = create_simple_pre_tool_hook("simple", callback)
        result, _ = await hook.on_pre_tool_use("Write", {}, context)
        assert result == HookResult.CONTINUE

    @pytest.mark.asyncio
    async def test_create_simple_subagent_stop_hook(self, context):
        """Test creating a simple SubagentStop hook."""
        called = {"value": False}

        def callback(agent_id, result, ctx):
            called["value"] = True

        hook = create_simple_subagent_stop_hook("simple", callback)
        await hook.on_subagent_stop("agent-123", {}, context)
        assert called["value"] is True


# ============================================================================
# VERIFICATION FUNCTION TESTS
# ============================================================================


class TestVerificationFunctions:
    """Tests for built-in verification functions."""

    @pytest.fixture
    def context(self, tmp_path):
        """Create a hook context with working dir."""
        ctx = HookContext(agent_id="test-agent")
        ctx.metadata["working_dir"] = str(tmp_path)
        return ctx

    @pytest.mark.asyncio
    async def test_file_exists_verification_pass(self, context, tmp_path):
        """Test file exists verification passing."""
        (tmp_path / "required.txt").write_text("content")

        verify_fn = create_file_exists_verification(["required.txt"])
        result = await verify_fn("agent-123", {}, context)

        assert result.status == VerificationStatus.PASSED

    @pytest.mark.asyncio
    async def test_file_exists_verification_fail(self, context, tmp_path):
        """Test file exists verification failing."""
        verify_fn = create_file_exists_verification(["missing.txt"])
        result = await verify_fn("agent-123", {}, context)

        assert result.status == VerificationStatus.FAILED
        assert "missing.txt" in result.errors[0]


# ============================================================================
# TOOL LOGGER TESTS
# ============================================================================


class TestToolLogger:
    """Tests for the tool_logger module (database-backed audit trail)."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database path for tool logger tests."""
        db_dir = tmp_path / ".harness"
        db_dir.mkdir()
        return str(db_dir / "harness.db")

    def test_log_tool_invocation(self, db_path):
        """Test that log_tool_invocation writes to the database correctly."""
        from harness.hooks.tool_logger import log_tool_invocation

        row_id = log_tool_invocation(
            tool_name="Bash",
            tool_input={"command": "git status"},
            phase="pre",
            db_path=db_path,
        )

        assert row_id is not None
        assert row_id > 0

        # Verify record exists in database
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tool_invocations WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["tool_name"] == "Bash"
        assert row["status"] == "pending"
        assert '"command": "git status"' in row["arguments"]

    def test_log_tool_invocation_post_phase(self, db_path):
        """Test that post-phase logging updates the pending record."""
        from harness.hooks.tool_logger import log_tool_invocation

        # First create a pre record
        pre_id = log_tool_invocation(
            tool_name="Write",
            tool_input={"file_path": "/tmp/test.txt"},
            phase="pre",
            db_path=db_path,
        )
        assert pre_id is not None

        # Then log the post phase
        post_id = log_tool_invocation(
            tool_name="Write",
            tool_input={"file_path": "/tmp/test.txt"},
            phase="post",
            result="File written successfully",
            db_path=db_path,
        )

        assert post_id == pre_id  # Should update the same record

        # Verify the record was updated
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tool_invocations WHERE id = ?", (pre_id,)
        ).fetchone()
        conn.close()

        assert row["status"] == "completed"
        assert row["result"] == "File written successfully"
        assert row["completed_at"] is not None

    def test_check_capability_allows(self):
        """Test that normal operations are allowed."""
        from harness.hooks.tool_logger import check_capability

        # Normal bash commands should be allowed
        assert check_capability("Bash", {"command": "git status"}) is True
        assert check_capability("Bash", {"command": "ls -la"}) is True
        assert check_capability("Bash", {"command": "npm run test"}) is True

        # Non-Bash tools should always be allowed
        assert check_capability("Write", {"file_path": "/tmp/test.txt"}) is True
        assert check_capability("Edit", {"file_path": "/tmp/test.txt"}) is True
        assert check_capability("Read", {"file_path": "/tmp/test.txt"}) is True

    def test_check_capability_blocks_dangerous(self):
        """Test that dangerous commands are blocked without capability."""
        from harness.hooks.tool_logger import check_capability

        # Should block dangerous patterns
        assert check_capability("Bash", {"command": "rm -rf /"}) is False
        assert check_capability("Bash", {"command": "git push --force origin main"}) is False
        assert check_capability("Bash", {"command": "DROP TABLE users;"}) is False
        assert check_capability("Bash", {"command": "git reset --hard HEAD~5"}) is False

    def test_check_capability_allows_with_grant(self):
        """Test that dangerous commands are allowed with explicit capability."""
        from harness.hooks.tool_logger import check_capability

        # With destructive capability, dangerous patterns should be allowed
        assert (
            check_capability("Bash", {"command": "rm -rf /tmp/old"}, ["destructive"])
            is True
        )
        assert (
            check_capability(
                "Bash", {"command": "git push --force"}, ["destructive"]
            )
            is True
        )

    def test_track_file_change(self, db_path):
        """Test that file changes are recorded in the database."""
        from harness.hooks.tool_logger import track_file_change

        row_id = track_file_change(
            file_path="/tmp/test.txt",
            change_type="create",
            tool_name="Write",
            db_path=db_path,
        )

        assert row_id is not None
        assert row_id > 0

        # Verify in database
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM file_changes WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["file_path"] == "/tmp/test.txt"
        assert row["change_type"] == "create"
        assert row["tool_name"] == "Write"

    def test_track_file_change_invalid_type(self, db_path):
        """Test that invalid change types are rejected."""
        from harness.hooks.tool_logger import track_file_change

        row_id = track_file_change(
            file_path="/tmp/test.txt",
            change_type="invalid",
            db_path=db_path,
        )

        assert row_id is None

    def test_log_blocked_invocation(self, db_path):
        """Test logging a blocked tool invocation."""
        from harness.hooks.tool_logger import log_blocked_invocation

        row_id = log_blocked_invocation(
            tool_name="Bash",
            tool_input={"command": "rm -rf /"},
            reason="Blocked: rm -rf",
            db_path=db_path,
        )

        assert row_id is not None

        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tool_invocations WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()

        assert row["status"] == "blocked"
        assert row["blocked_reason"] == "Blocked: rm -rf"

    def test_get_dangerous_pattern(self):
        """Test dangerous pattern detection."""
        from harness.hooks.tool_logger import get_dangerous_pattern

        assert get_dangerous_pattern("rm -rf /tmp/old") == "rm -rf"
        assert get_dangerous_pattern("git push --force") == "git push --force"
        assert get_dangerous_pattern("DROP TABLE users") == "DROP TABLE"
        assert get_dangerous_pattern("git status") is None
        assert get_dangerous_pattern("ls -la") is None


# ============================================================================
# PRE/POST HOOK SCRIPT TESTS
# ============================================================================


class TestPreToolUseScript:
    """Tests for the pre_tool_use.py hook script."""

    def test_pre_hook_allows_normal(self, tmp_path):
        """Test that pre-hook exits 0 for normal tool invocations."""
        import subprocess

        hook_data = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "git status"},
        })

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.pre_tool_use",
            ],
            input=hook_data,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": str(tmp_path / "harness.db")},
        )

        assert result.returncode == 0

    def test_pre_hook_blocks_dangerous(self, tmp_path):
        """Test that pre-hook exits 2 for dangerous commands."""
        import subprocess

        hook_data = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /important/data"},
        })

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.pre_tool_use",
            ],
            input=hook_data,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": str(tmp_path / "harness.db")},
        )

        assert result.returncode == 2
        output = json.loads(result.stdout)
        assert output["blocked"] is True
        assert "rm -rf" in output["reason"]

    def test_pre_hook_allows_write_tool(self, tmp_path):
        """Test that pre-hook allows non-Bash tools."""
        import subprocess

        hook_data = json.dumps({
            "tool_name": "Write",
            "tool_input": {"file_path": "/tmp/test.txt", "content": "hello"},
        })

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.pre_tool_use",
            ],
            input=hook_data,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": str(tmp_path / "harness.db")},
        )

        assert result.returncode == 0

    def test_pre_hook_handles_empty_input(self, tmp_path):
        """Test that pre-hook handles empty stdin gracefully."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.pre_tool_use",
            ],
            input="",
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": str(tmp_path / "harness.db")},
        )

        assert result.returncode == 0


class TestPostToolUseScript:
    """Tests for the post_tool_use.py hook script."""

    def test_post_hook_logs_result(self, tmp_path):
        """Test that post-hook logs tool results and exits 0."""
        import subprocess

        db_path = str(tmp_path / "harness.db")
        hook_data = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "git status"},
            "tool_output": "On branch main\nnothing to commit",
        })

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.post_tool_use",
            ],
            input=hook_data,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": db_path},
        )

        assert result.returncode == 0

    def test_post_hook_tracks_file_write(self, tmp_path):
        """Test that post-hook tracks Write tool file changes."""
        import sqlite3
        import subprocess

        db_path = str(tmp_path / "harness.db")
        hook_data = json.dumps({
            "tool_name": "Write",
            "tool_input": {"file_path": "/tmp/new_file.py", "content": "print('hello')"},
            "tool_output": "File written",
        })

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.post_tool_use",
            ],
            input=hook_data,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": db_path},
        )

        assert result.returncode == 0

        # Verify file change was tracked
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM file_changes").fetchall()
        conn.close()

        assert len(rows) >= 1
        file_change = rows[0]
        assert file_change["file_path"] == "/tmp/new_file.py"
        assert file_change["change_type"] == "create"
        assert file_change["tool_name"] == "Write"

    def test_post_hook_tracks_edit_tool(self, tmp_path):
        """Test that post-hook tracks Edit tool file changes."""
        import sqlite3
        import subprocess

        db_path = str(tmp_path / "harness.db")
        hook_data = json.dumps({
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/tmp/existing.py",
                "old_string": "foo",
                "new_string": "bar",
            },
            "tool_output": "Edit applied",
        })

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.post_tool_use",
            ],
            input=hook_data,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": db_path},
        )

        assert result.returncode == 0

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM file_changes").fetchall()
        conn.close()

        assert len(rows) >= 1
        assert rows[0]["change_type"] == "edit"

    def test_post_hook_handles_empty_input(self, tmp_path):
        """Test that post-hook handles empty stdin gracefully."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.hooks.post_tool_use",
            ],
            input="",
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ, "HARNESS_DB_PATH": str(tmp_path / "harness.db")},
        )

        assert result.returncode == 0
