"""
LangGraph workflow graph construction.

Creates the StateGraph for the box-up-role workflow with:
- Node registration
- Edge definitions
- Conditional routing
- Retry policies
"""

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

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
from harness.dag.langgraph_policies import (
    GIT_RETRY_POLICY,
    GITLAB_API_RETRY_POLICY,
    SUBPROCESS_RETRY_POLICY,
)
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
)
from harness.dag.langgraph_state import (
    DEFAULT_BREAKPOINTS,
    BoxUpRoleState,
    get_breakpoints_enabled,
)


def create_box_up_role_graph(
    db_path: str = "harness.db",
    parallel_tests: bool = True,
    enable_breakpoints: bool | None = None,
) -> tuple[StateGraph, list[str]]:
    """
    Create the LangGraph-based box-up-role workflow.

    This implements the full workflow with:
    - TypedDict state schema
    - Conditional routing on failures
    - Checkpointing via SqliteSaver
    - All gates (molecule, pytest, deploy)
    - RetryPolicy for external API and subprocess nodes (LangGraph 1.0.x)
    - Parallel test execution via Send API (Task #21)
    - Static breakpoints via interrupt_before (Task #23)

    Args:
        db_path: Path to the SQLite database for checkpointing
        parallel_tests: If True (default), run molecule and pytest in parallel.
                       If False, run tests sequentially for debugging.
        enable_breakpoints: If True, return breakpoint nodes for interrupt_before.
                           If None, check HARNESS_BREAKPOINTS env var.
                           If False, return empty breakpoints list.

    Returns:
        Tuple of (StateGraph, breakpoint_nodes) where breakpoint_nodes is
        a list of node names to pass to compile(interrupt_before=...).

    Graph Structure (parallel_tests=True):
        validate_role -> analyze_deps -> check_reverse_deps -> create_worktree
        -> [parallel: run_molecule, run_pytest] -> merge_test_results
        -> validate_deploy -> create_commit -> push_branch -> create_issue
        -> create_mr -> human_approval -> add_to_merge_train -> report_summary

    Graph Structure (parallel_tests=False, legacy):
        validate_role -> analyze_deps -> check_reverse_deps -> create_worktree
        -> run_molecule -> run_pytest -> validate_deploy -> ...

    Retry Policies Applied:
    - GitLab API nodes: 3 attempts with exponential backoff (1s, 2s, 4s)
    - Subprocess nodes: 2 attempts for timeout recovery
    - Git operations: 3 attempts with backoff for push reliability
    """
    # Create the graph
    graph = StateGraph(BoxUpRoleState)

    # Add nodes without retry policies (local operations)
    graph.add_node("validate_role", validate_role_node)
    graph.add_node("analyze_deps", analyze_deps_node)
    graph.add_node("check_reverse_deps", check_reverse_deps_node)
    graph.add_node("create_worktree", create_worktree_node)

    # Add subprocess nodes with retry policy for timeout handling
    graph.add_node("run_molecule", run_molecule_node, retry_policy=SUBPROCESS_RETRY_POLICY)
    graph.add_node("run_pytest", run_pytest_node, retry_policy=SUBPROCESS_RETRY_POLICY)

    # Add test results merger for parallel execution (Task #21)
    graph.add_node("merge_test_results", merge_test_results_node)

    graph.add_node("validate_deploy", validate_deploy_node)

    # Add git nodes with retry policy for push reliability
    graph.add_node("create_commit", create_commit_node)
    graph.add_node("push_branch", push_branch_node, retry_policy=GIT_RETRY_POLICY)

    # Add GitLab API nodes with retry policy for network resilience
    graph.add_node("create_issue", create_issue_node, retry_policy=GITLAB_API_RETRY_POLICY)
    graph.add_node("create_mr", create_mr_node, retry_policy=GITLAB_API_RETRY_POLICY)

    # Human-in-the-loop node (uses interrupt() for approval)
    graph.add_node("human_approval", human_approval_node)

    # Add merge train node with retry policy
    graph.add_node(
        "add_to_merge_train", add_to_merge_train_node, retry_policy=GITLAB_API_RETRY_POLICY
    )

    # Add terminal nodes (no retry needed)
    graph.add_node("report_summary", report_summary_node)
    graph.add_node("notify_failure", notify_failure_node)

    # Set entry point
    graph.set_entry_point("validate_role")

    # Add edges with conditional routing
    graph.add_conditional_edges("validate_role", should_continue_after_validation)
    graph.add_conditional_edges("analyze_deps", should_continue_after_deps)
    graph.add_conditional_edges("check_reverse_deps", should_continue_after_reverse_deps)

    if parallel_tests:
        # Task #21: Parallel test execution using Send API
        # create_worktree -> [parallel: run_molecule, run_pytest] -> merge_test_results
        graph.add_conditional_edges(
            "create_worktree",
            route_to_parallel_tests,
            ["run_molecule", "run_pytest", "merge_test_results"],
        )

        # Both parallel test nodes converge at merge_test_results
        graph.add_edge("run_molecule", "merge_test_results")
        graph.add_edge("run_pytest", "merge_test_results")

        # After merge, route based on combined test results
        graph.add_conditional_edges("merge_test_results", should_continue_after_merge)
    else:
        # Legacy sequential test execution (for debugging or backwards compatibility)
        graph.add_conditional_edges(
            "create_worktree",
            lambda state: "run_molecule" if not state.get("errors") else "notify_failure",
        )
        graph.add_conditional_edges("run_molecule", should_continue_after_molecule)
        graph.add_conditional_edges("run_pytest", should_continue_after_pytest)

    graph.add_conditional_edges("validate_deploy", should_continue_after_deploy)
    graph.add_conditional_edges("create_commit", should_continue_after_commit)
    graph.add_conditional_edges("push_branch", should_continue_after_push)
    graph.add_conditional_edges("create_issue", should_continue_after_issue)
    graph.add_conditional_edges("create_mr", should_continue_after_mr)
    graph.add_conditional_edges("human_approval", should_continue_after_human_approval)

    # Terminal edges
    graph.add_edge("add_to_merge_train", "report_summary")
    graph.add_edge("report_summary", END)
    graph.add_edge("notify_failure", END)

    # Determine breakpoints
    if enable_breakpoints is None:
        enable_breakpoints = get_breakpoints_enabled()

    breakpoints = DEFAULT_BREAKPOINTS if enable_breakpoints else []

    return graph, breakpoints


async def create_compiled_graph(
    db_path: str = "harness.db",
    enable_breakpoints: bool | None = None,
):
    """
    Create and compile the graph with SqliteSaver checkpointer.

    Args:
        db_path: Path to SQLite database for checkpointing.
        enable_breakpoints: If True, enable static breakpoints (interrupt_before).
                           If None, check HARNESS_BREAKPOINTS env var.

    Returns a compiled graph that can be executed with .ainvoke().
    """
    graph, breakpoints = create_box_up_role_graph(db_path, enable_breakpoints=enable_breakpoints)

    # Create async SQLite checkpointer
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compile_kwargs = {"checkpointer": checkpointer}
        if breakpoints:
            compile_kwargs["interrupt_before"] = breakpoints
        compiled = graph.compile(**compile_kwargs)
        return compiled


__all__ = [
    "create_box_up_role_graph",
    "create_compiled_graph",
]
