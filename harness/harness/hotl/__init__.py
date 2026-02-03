"""HOTL (Human Out of The Loop) mode for autonomous harness operation."""

from harness.hotl.agent_session import (
    AgentSession,
    AgentSessionManager,
    AgentStatus,
    FileChange,
    FileChangeType,
)
from harness.hotl.claude_integration import (
    AsyncHOTLClaudeIntegration,
    ClaudeAgentConfig,
    HOTLClaudeIntegration,
)
from harness.hotl.claude_sdk_integration import (
    AsyncSDKClaudeIntegration,
    HookEvent,
    PermissionMode,
    SDKAgentConfig,
    SDKClaudeIntegration,
    create_claude_integration,
    create_hotl_mcp_tools,
    sdk_available,
)
from harness.hotl.context import (
    ContextAwareSession,
    ContextConfig,
    ContextManager,
    ContextStats,
    FileMemory,
)
from harness.hotl.orchestrator import HOTLOrchestrator
from harness.hotl.queue import QueueItem, QueueItemStatus, RoleQueue
from harness.hotl.state import HOTLPhase, HOTLState
from harness.hotl.supervisor import HOTLSupervisor
from harness.hotl.worker_pool import Task, WorkerPool

__all__ = [
    # State management
    "HOTLState",
    "HOTLPhase",
    "HOTLSupervisor",
    # Queue and orchestration
    "RoleQueue",
    "QueueItem",
    "QueueItemStatus",
    "HOTLOrchestrator",
    # Worker pool
    "WorkerPool",
    "Task",
    # Agent sessions
    "AgentSession",
    "AgentSessionManager",
    "AgentStatus",
    "FileChange",
    "FileChangeType",
    # Claude integration (subprocess-based, legacy)
    "HOTLClaudeIntegration",
    "AsyncHOTLClaudeIntegration",
    "ClaudeAgentConfig",
    # Claude SDK integration (native, preferred)
    "SDKClaudeIntegration",
    "AsyncSDKClaudeIntegration",
    "SDKAgentConfig",
    "HookEvent",
    "PermissionMode",
    "sdk_available",
    "create_claude_integration",
    "create_hotl_mcp_tools",
    # Context management (Manus patterns)
    "ContextConfig",
    "ContextManager",
    "ContextStats",
    "FileMemory",
    "ContextAwareSession",
]
