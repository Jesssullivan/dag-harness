"""
EMS Harness - DAG-based orchestration for Ansible role deployment workflow.

This harness provides:
- SQLite state management with graph-queryable relationships
- DAG-based workflow execution following LangGraph patterns
- MCP server integration for MCP client with lifecycle hooks
- GitLab API integration (iterations, issues, MRs, merge trains)
- Git worktree management for parallel development
- Parallel wave execution for batch processing
- Notification service (Discord, Email)
- Human-in-the-loop (HITL) integration
- SEE/ACP context control
- Test regression tracking
"""

__version__ = "0.3.0"

from harness.config import HarnessConfig
from harness.dag.graph import WorkflowGraph
from harness.dag.langgraph_engine import BoxUpRoleState, LangGraphWorkflowRunner
from harness.dag.parallel import ParallelWaveExecutor
from harness.db.state import StateDB
from harness.hitl import BreakpointManager, HumanInputHandler
from harness.notifications import NotificationService

__all__ = [
    "StateDB",
    "WorkflowGraph",
    "LangGraphWorkflowRunner",
    "BoxUpRoleState",
    "ParallelWaveExecutor",
    "HarnessConfig",
    "NotificationService",
    "HumanInputHandler",
    "BreakpointManager",
    "__version__",
]
