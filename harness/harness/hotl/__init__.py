"""HOTL (Human Out of The Loop) mode for autonomous harness operation."""

from harness.hotl.state import HOTLState, HOTLPhase
from harness.hotl.supervisor import HOTLSupervisor
from harness.hotl.worker_pool import WorkerPool, Task
from harness.hotl.agent_session import (
    AgentSession,
    AgentSessionManager,
    AgentStatus,
    FileChange,
    FileChangeType,
)
from harness.hotl.claude_integration import (
    HOTLClaudeIntegration,
    AsyncHOTLClaudeIntegration,
    ClaudeAgentConfig,
)

__all__ = [
    # State management
    "HOTLState",
    "HOTLPhase",
    "HOTLSupervisor",
    # Worker pool
    "WorkerPool",
    "Task",
    # Agent sessions
    "AgentSession",
    "AgentSessionManager",
    "AgentStatus",
    "FileChange",
    "FileChangeType",
    # Claude integration
    "HOTLClaudeIntegration",
    "AsyncHOTLClaudeIntegration",
    "ClaudeAgentConfig",
]
