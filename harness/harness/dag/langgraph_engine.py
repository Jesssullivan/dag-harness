"""
LangGraph-based workflow execution engine.

Implements proper LangGraph patterns:
- TypedDict-based state schema with Annotated reducers
- StateGraph with node and edge definitions
- SqliteSaver checkpointer for persistence
- Conditional edges for failure routing
- Parallel wave execution support
- RetryPolicy for external API and subprocess nodes (LangGraph 1.0.x)
- Human-in-the-loop via interrupt() pattern (Task #18)
- Parallel test execution via Send API (Task #21)

Migration Notes (LangGraph 0.2.x -> 1.0.x, Task #13):
- RetryPolicy imported from langgraph.types (new in 1.0.x)
- AsyncSqliteSaver path unchanged (langgraph.checkpoint.sqlite.aio)
- StateGraph, END imports unchanged from langgraph.graph
- add_node() now accepts 'retry_policy' parameter for RetryPolicy
- Default retry behavior: retries on 5xx HTTP errors, skips 4xx
- langgraph-checkpoint-sqlite upgraded to 3.0.0+ for compatibility

HITL Pattern (Task #18):
- interrupt() from langgraph.types pauses execution and stores state
- Command(resume=...) provides human input to resume execution
- Checkpointer required for interrupt to work (SqliteSaver)
- interrupt_before parameter in compile() for automatic breakpoints

Parallel Test Execution (Task #21):
- Send API from langgraph.types enables parallel node execution
- route_to_parallel_tests() fans out to run_molecule and run_pytest
- merge_test_results_node() aggregates results from parallel tests
- Performance tracking: logs time savings from parallel execution
- Partial failure handling: workflow tracks which tests failed
- Target: 30%+ time reduction for test phase when both test types exist

Module Organization:
This file serves as the backwards-compatible entry point for the LangGraph engine.
The implementation is split across multiple modules for maintainability:
- langgraph_state.py: State schema, reducers, module-level DB access
- langgraph_nodes.py: Node functions (validate, test, commit, etc.)
- langgraph_routing.py: Conditional edge routing functions
- langgraph_policies.py: Retry policies for nodes
- langgraph_builder.py: Graph construction
- langgraph_runner.py: Workflow execution wrapper
"""

# Re-export from langgraph.types for convenience
from langgraph.types import Command, Send, interrupt  # noqa: F401

# =============================================================================
# STATE MODULE EXPORTS
# =============================================================================
from harness.dag.langgraph_state import (
    BREAKPOINTS_ENV_VAR,
    DEFAULT_BREAKPOINTS,
    BoxUpRoleState,
    _record_test_result,
    create_initial_state,
    get_breakpoints_enabled,
    get_module_config,
    get_module_db,
    keep_last_n,
    set_module_config,
    set_module_db,
)

# =============================================================================
# POLICIES MODULE EXPORTS
# =============================================================================
from harness.dag.langgraph_policies import (
    GIT_RETRY_POLICY,
    GITLAB_API_RETRY_POLICY,
    SUBPROCESS_RETRY_POLICY,
)

# =============================================================================
# ROUTING MODULE EXPORTS
# =============================================================================
from harness.dag.langgraph_routing import (
    route_to_parallel_tests,
    should_continue_after_commit,
    should_continue_after_deploy,
    should_continue_after_deps,
    should_continue_after_human_approval,
    should_continue_after_issue,
    should_continue_after_merge,
    should_continue_after_molecule,
    should_continue_after_mr,
    should_continue_after_push,
    should_continue_after_pytest,
    should_continue_after_reverse_deps,
    should_continue_after_validation,
    should_continue_after_worktree,
)

# =============================================================================
# NODES MODULE EXPORTS
# =============================================================================
from harness.dag.langgraph_nodes import (
    add_to_merge_train_node,
    analyze_deps_node,
    check_reverse_deps_node,
    create_commit_node,
    create_issue_node,
    create_mr_node,
    create_worktree_node,
    human_approval_node,
    merge_test_results_node,
    notify_failure_node,
    push_branch_node,
    report_summary_node,
    run_molecule_node,
    run_pytest_node,
    validate_deploy_node,
    validate_role_node,
)

# =============================================================================
# BUILDER MODULE EXPORTS
# =============================================================================
from harness.dag.langgraph_builder import (
    create_box_up_role_graph,
    create_compiled_graph,
)

# =============================================================================
# RUNNER MODULE EXPORTS
# =============================================================================
from harness.dag.langgraph_runner import LangGraphWorkflowRunner

# =============================================================================
# NOTIFICATION EXPORTS (for test mocking compatibility)
# =============================================================================
from harness.notifications import (
    notify_workflow_completed,
    notify_workflow_failed,
    notify_workflow_started,
)

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # LangGraph types (re-exported for convenience)
    "Command",
    "Send",
    "interrupt",
    # Custom reducers
    "keep_last_n",
    # Breakpoints
    "DEFAULT_BREAKPOINTS",
    "BREAKPOINTS_ENV_VAR",
    "get_breakpoints_enabled",
    # Module DB access
    "set_module_db",
    "set_module_config",
    "get_module_db",
    "get_module_config",
    "_record_test_result",
    # State
    "BoxUpRoleState",
    "create_initial_state",
    # Retry policies
    "GITLAB_API_RETRY_POLICY",
    "SUBPROCESS_RETRY_POLICY",
    "GIT_RETRY_POLICY",
    # Parallel test routing
    "route_to_parallel_tests",
    "should_continue_after_merge",
    # Conditional routing
    "should_continue_after_validation",
    "should_continue_after_deps",
    "should_continue_after_reverse_deps",
    "should_continue_after_worktree",
    "should_continue_after_molecule",
    "should_continue_after_pytest",
    "should_continue_after_deploy",
    "should_continue_after_commit",
    "should_continue_after_push",
    "should_continue_after_issue",
    "should_continue_after_mr",
    "should_continue_after_human_approval",
    # Node functions
    "validate_role_node",
    "analyze_deps_node",
    "check_reverse_deps_node",
    "create_worktree_node",
    "run_molecule_node",
    "run_pytest_node",
    "merge_test_results_node",
    "validate_deploy_node",
    "create_commit_node",
    "push_branch_node",
    "create_issue_node",
    "create_mr_node",
    "add_to_merge_train_node",
    "human_approval_node",
    "report_summary_node",
    "notify_failure_node",
    # Graph construction
    "create_box_up_role_graph",
    "create_compiled_graph",
    # Runner
    "LangGraphWorkflowRunner",
]
