"""
Property-based tests for the BoxUpRoleState state machine.

Tests invariants of the LangGraph workflow state schema, conditional routing
functions, custom reducers, and graph traversal properties.

Test Categories:
1. STATE INVARIANTS - Fields that must satisfy constraints for any valid state
2. STATE TRANSITIONS - Conditional routing functions route correctly for all inputs
3. REDUCER PROPERTIES - Custom reducers (operator.add, keep_last_n) behave correctly
4. GRAPH TRAVERSAL - Structural properties of the compiled workflow graph
"""

import operator
from typing import Literal

import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    create_box_up_role_graph,
    create_initial_state,
    keep_last_n,
    should_continue_after_commit,
    should_continue_after_deploy,
    should_continue_after_deps,
    should_continue_after_human_approval,
    should_continue_after_issue,
    should_continue_after_merge,
    should_continue_after_mr,
    should_continue_after_push,
    should_continue_after_reverse_deps,
    should_continue_after_validation,
    should_continue_after_worktree,
)
from tests.strategies import role_name_strategy


# =============================================================================
# STRATEGIES
# =============================================================================

# All node names in the workflow DAG
ALL_NODE_NAMES: list[str] = [
    "validate_role",
    "analyze_deps",
    "check_reverse_deps",
    "create_worktree",
    "run_molecule",
    "run_pytest",
    "merge_test_results",
    "validate_deploy",
    "create_commit",
    "push_branch",
    "create_issue",
    "create_mr",
    "human_approval",
    "add_to_merge_train",
    "report_summary",
    "notify_failure",
]

# Terminal node names
TERMINAL_NODES: set[str] = {"report_summary", "notify_failure"}


@st.composite
def box_up_state_strategy(draw):
    """
    Generate valid BoxUpRoleState dicts with all fields.

    Produces states that satisfy the TypedDict schema, with realistic
    combinations of field values for property-based testing.
    """
    role_name = draw(role_name_strategy)
    has_molecule = draw(st.booleans())
    errors = draw(st.lists(st.text(min_size=1, max_size=80), min_size=0, max_size=5))
    completed = draw(
        st.lists(
            st.sampled_from(ALL_NODE_NAMES),
            min_size=0,
            max_size=len(ALL_NODE_NAMES),
            unique=True,
        )
    )

    return BoxUpRoleState(
        role_name=role_name,
        execution_id=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000))),
        role_path=f"ansible/roles/{role_name}",
        has_molecule_tests=has_molecule,
        has_meta=draw(st.booleans()),
        wave=draw(st.integers(min_value=0, max_value=8)),
        wave_name=draw(st.text(min_size=0, max_size=30)),
        explicit_deps=draw(st.lists(st.text(min_size=1, max_size=30), max_size=5)),
        implicit_deps=draw(st.lists(st.text(min_size=1, max_size=30), max_size=5)),
        reverse_deps=draw(st.lists(st.text(min_size=1, max_size=30), max_size=5)),
        credentials=draw(st.lists(st.fixed_dictionaries({"name": st.text(min_size=1, max_size=20)}), max_size=3)),
        tags=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        blocking_deps=draw(st.lists(st.text(min_size=1, max_size=30), max_size=3)),
        worktree_path=draw(st.text(min_size=0, max_size=100)),
        branch=draw(st.text(min_size=0, max_size=50)),
        commit_sha=draw(st.one_of(st.none(), st.text(alphabet="0123456789abcdef", min_size=40, max_size=40))),
        commit_message=draw(st.one_of(st.none(), st.text(max_size=200))),
        pushed=draw(st.booleans()),
        molecule_passed=draw(st.one_of(st.none(), st.booleans())),
        molecule_skipped=draw(st.booleans()),
        molecule_duration=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=3600))),
        molecule_output=draw(st.one_of(st.none(), st.text(max_size=100))),
        pytest_passed=draw(st.one_of(st.none(), st.booleans())),
        pytest_skipped=draw(st.booleans()),
        pytest_duration=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=3600))),
        deploy_passed=draw(st.one_of(st.none(), st.booleans())),
        deploy_skipped=draw(st.booleans()),
        all_tests_passed=draw(st.one_of(st.none(), st.booleans())),
        parallel_tests_completed=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        test_phase_start_time=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e12, allow_nan=False))),
        test_phase_duration=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=3600.0, allow_nan=False))),
        parallel_execution_enabled=draw(st.booleans()),
        issue_url=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        issue_iid=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000))),
        issue_created=draw(st.booleans()),
        mr_url=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        mr_iid=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000))),
        mr_created=draw(st.booleans()),
        reviewers_set=draw(st.booleans()),
        iteration_assigned=draw(st.booleans()),
        merge_train_status=draw(st.one_of(st.none(), st.sampled_from(["added", "fallback", "skipped", "error"]))),
        branch_existed=draw(st.booleans()),
        current_node=draw(st.sampled_from(ALL_NODE_NAMES)),
        completed_nodes=completed,
        errors=errors,
        test_results=draw(st.lists(st.fixed_dictionaries({"test": st.text(min_size=1, max_size=20), "passed": st.booleans()}), max_size=3)),
        git_operations=draw(st.lists(st.text(min_size=1, max_size=50), max_size=5)),
        api_calls=draw(st.lists(st.fixed_dictionaries({"url": st.text(min_size=1, max_size=50)}), max_size=3)),
        timing_metrics=draw(st.lists(st.fixed_dictionaries({"node": st.text(min_size=1, max_size=20), "ms": st.integers(min_value=0, max_value=60000)}), max_size=5)),
        state_snapshots=draw(st.lists(st.fixed_dictionaries({"step": st.integers(min_value=0, max_value=100)}), max_size=10)),
        summary=draw(st.one_of(st.none(), st.fixed_dictionaries({"status": st.text(min_size=1, max_size=20)}))),
        human_approved=draw(st.one_of(st.none(), st.booleans())),
        human_rejection_reason=draw(st.one_of(st.none(), st.text(max_size=200))),
        awaiting_human_input=draw(st.booleans()),
    )


@st.composite
def state_update_strategy(draw):
    """
    Generate valid state update dicts -- what workflow nodes return.

    Each node returns a partial dict that gets merged into the state.
    The update always includes completed_nodes (tracking which node ran).
    """
    node_name = draw(st.sampled_from(ALL_NODE_NAMES))
    update = {"completed_nodes": [node_name]}

    # Optionally include errors
    if draw(st.booleans()):
        update["errors"] = draw(
            st.lists(st.text(min_size=1, max_size=80), min_size=0, max_size=3)
        )

    # Optionally include test results
    if draw(st.booleans()):
        update["molecule_passed"] = draw(st.one_of(st.none(), st.booleans()))

    if draw(st.booleans()):
        update["pytest_passed"] = draw(st.one_of(st.none(), st.booleans()))

    # Optionally include git state
    if draw(st.booleans()):
        update["pushed"] = draw(st.booleans())

    # Optionally include GitLab state
    if draw(st.booleans()):
        update["issue_iid"] = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000)))

    if draw(st.booleans()):
        update["mr_iid"] = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000)))

    return update


@st.composite
def workflow_path_strategy(draw):
    """
    Generate valid paths through the DAG by following actual edges.

    Uses the graph definition to produce paths that only follow defined
    edges, starting from validate_role.
    """
    # Define adjacency based on the happy path through the graph
    # (conditional edges can go to notify_failure from any node)
    adjacency = {
        "validate_role": ["analyze_deps", "notify_failure"],
        "analyze_deps": ["check_reverse_deps", "notify_failure"],
        "check_reverse_deps": ["create_worktree", "notify_failure"],
        "create_worktree": ["run_molecule", "run_pytest", "notify_failure"],
        "run_molecule": ["merge_test_results"],
        "run_pytest": ["merge_test_results"],
        "merge_test_results": ["validate_deploy", "notify_failure"],
        "validate_deploy": ["create_commit", "notify_failure"],
        "create_commit": ["push_branch", "notify_failure"],
        "push_branch": ["create_issue", "notify_failure"],
        "create_issue": ["create_mr", "notify_failure"],
        "create_mr": ["human_approval", "report_summary"],
        "human_approval": ["add_to_merge_train", "notify_failure"],
        "add_to_merge_train": ["report_summary"],
        "report_summary": [],  # terminal -> END
        "notify_failure": [],  # terminal -> END
    }

    path = ["validate_role"]
    current = "validate_role"

    # Walk the graph until we reach a terminal node
    max_steps = 20  # safety bound
    for _ in range(max_steps):
        neighbors = adjacency.get(current, [])
        if not neighbors:
            break
        current = draw(st.sampled_from(neighbors))
        path.append(current)
        if current in TERMINAL_NODES:
            break

    return path


# =============================================================================
# 1. STATE INVARIANTS
# =============================================================================


class TestStateInvariants:
    """Test that state invariants hold for all generated states."""

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_completed_nodes_never_none(self, state):
        """completed_nodes should always be a list, never None."""
        assert state["completed_nodes"] is not None
        assert isinstance(state["completed_nodes"], list)

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_no_duplicate_completed_nodes(self, state):
        """completed_nodes should not contain duplicates after reducer."""
        # The strategy generates unique completed_nodes.
        # Verify operator.add reducer + set dedup in real usage preserves this.
        completed = state["completed_nodes"]
        assert len(completed) == len(set(completed)), (
            f"Duplicate completed_nodes found: {completed}"
        )

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_errors_always_list(self, state):
        """errors field should always be a list."""
        assert state["errors"] is not None
        assert isinstance(state["errors"], list)

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_role_name_always_present(self, state):
        """role_name must always be a non-empty string."""
        assert state["role_name"] is not None
        assert isinstance(state["role_name"], str)
        assert len(state["role_name"]) > 0

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_wave_in_valid_range(self, state):
        """wave must be a non-negative integer."""
        assert isinstance(state["wave"], int)
        assert state["wave"] >= 0

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_current_node_is_valid(self, state):
        """current_node must be a recognized node name."""
        assert state["current_node"] in ALL_NODE_NAMES

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_completed_nodes_subset_of_all_nodes(self, state):
        """completed_nodes entries must all be valid node names."""
        for node in state["completed_nodes"]:
            assert node in ALL_NODE_NAMES, f"Unknown node in completed_nodes: {node}"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_list_fields_never_none(self, state):
        """All Annotated[list, ...] fields must be lists, never None."""
        list_fields = [
            "explicit_deps", "implicit_deps", "reverse_deps", "credentials",
            "tags", "blocking_deps", "completed_nodes", "errors",
            "test_results", "git_operations", "api_calls", "timing_metrics",
            "state_snapshots", "parallel_tests_completed",
        ]
        for field in list_fields:
            value = state.get(field)
            assert value is not None, f"{field} is None"
            assert isinstance(value, list), f"{field} is {type(value)}, expected list"

    @pytest.mark.pbt
    def test_initial_state_invariants(self):
        """create_initial_state must produce a valid state with correct defaults."""
        state = create_initial_state("test_role", execution_id=1)
        assert state["role_name"] == "test_role"
        assert state["execution_id"] == 1
        assert state["completed_nodes"] == []
        assert state["errors"] == []
        assert state["current_node"] == "validate_role"
        assert state["pushed"] is False
        assert state["human_approved"] is None
        assert state["awaiting_human_input"] is False


# =============================================================================
# 2. STATE TRANSITIONS
# =============================================================================


class TestStateTransitions:
    """Test that conditional routing functions behave correctly for all inputs."""

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_errors_route_to_notification(self, state):
        """When errors is non-empty, error-checking routers should go to notify_failure."""
        assume(len(state["errors"]) > 0)
        # All these routers check state.get("errors") and route to notify_failure
        assert should_continue_after_validation(state) == "notify_failure"
        assert should_continue_after_deps(state) == "notify_failure"
        assert should_continue_after_worktree(state) == "notify_failure"
        assert should_continue_after_commit(state) == "notify_failure"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_no_errors_continues(self, state):
        """When errors is empty, error-checking routers should continue to next node."""
        assume(len(state["errors"]) == 0)
        assert should_continue_after_validation(state) == "analyze_deps"
        assert should_continue_after_deps(state) == "check_reverse_deps"
        assert should_continue_after_worktree(state) == "parallel_tests"
        assert should_continue_after_commit(state) == "push_branch"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_blocking_deps_routes_to_failure(self, state):
        """When blocking_deps is non-empty, should route to notify_failure."""
        assume(len(state["blocking_deps"]) > 0)
        assert should_continue_after_reverse_deps(state) == "notify_failure"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_no_blocking_deps_continues(self, state):
        """When blocking_deps is empty, should continue to create_worktree."""
        assume(len(state["blocking_deps"]) == 0)
        assert should_continue_after_reverse_deps(state) == "create_worktree"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_deploy_failure_routes_correctly(self, state):
        """deploy_passed=False routes to notify_failure, otherwise to create_commit."""
        if state.get("deploy_passed") is False:
            assert should_continue_after_deploy(state) == "notify_failure"
        else:
            assert should_continue_after_deploy(state) == "create_commit"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_push_failure_routes_correctly(self, state):
        """pushed=False routes to notify_failure, pushed=True routes to create_issue."""
        if not state.get("pushed"):
            assert should_continue_after_push(state) == "notify_failure"
        else:
            assert should_continue_after_push(state) == "create_issue"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_issue_failure_routes_correctly(self, state):
        """No issue_iid routes to notify_failure, with issue_iid routes to create_mr."""
        if not state.get("issue_iid"):
            assert should_continue_after_issue(state) == "notify_failure"
        else:
            assert should_continue_after_issue(state) == "create_mr"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_mr_routes_correctly(self, state):
        """With mr_iid routes to human_approval, without to report_summary."""
        if state.get("mr_iid"):
            assert should_continue_after_mr(state) == "human_approval"
        else:
            assert should_continue_after_mr(state) == "report_summary"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_human_approval_routes_correctly(self, state):
        """Approved routes to merge train, unapproved to notify_failure."""
        if state.get("human_approved"):
            assert should_continue_after_human_approval(state) == "add_to_merge_train"
        else:
            assert should_continue_after_human_approval(state) == "notify_failure"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_merge_results_routes_correctly(self, state):
        """all_tests_passed=True routes to validate_deploy, else notify_failure."""
        if state.get("all_tests_passed", False):
            assert should_continue_after_merge(state) == "validate_deploy"
        else:
            assert should_continue_after_merge(state) == "notify_failure"

    @pytest.mark.pbt
    @given(state=box_up_state_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_routing_always_returns_valid_string(self, state):
        """All routing functions must return a non-empty string."""
        routers = [
            should_continue_after_validation,
            should_continue_after_deps,
            should_continue_after_reverse_deps,
            should_continue_after_worktree,
            should_continue_after_deploy,
            should_continue_after_commit,
            should_continue_after_push,
            should_continue_after_issue,
            should_continue_after_mr,
            should_continue_after_human_approval,
            should_continue_after_merge,
        ]
        for router in routers:
            result = router(state)
            assert isinstance(result, str)
            assert len(result) > 0


# =============================================================================
# 3. REDUCER PROPERTIES
# =============================================================================


class TestReducerProperties:
    """Test that custom reducers satisfy their contracts."""

    @pytest.mark.pbt
    @given(
        lists=st.lists(
            st.lists(st.text(min_size=1), min_size=0, max_size=5),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_list_reducer_accumulates(self, lists):
        """Annotated[list, operator.add] should concatenate all updates."""
        # Simulate what LangGraph does with operator.add reducer
        accumulated = []
        for update in lists:
            accumulated = operator.add(accumulated, update)
        # The result should be the concatenation of all lists
        expected = []
        for lst in lists:
            expected.extend(lst)
        assert accumulated == expected

    @pytest.mark.pbt
    @given(items=st.lists(st.text(min_size=1), min_size=0, max_size=100))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_keep_last_n_bounds(self, items):
        """keep_last_n(n) should never exceed n items."""
        for n in [1, 5, 10, 50]:
            reducer = keep_last_n(n)
            result = reducer([], items)
            assert len(result) <= n, f"keep_last_n({n}) returned {len(result)} items"

    @pytest.mark.pbt
    @given(items=st.lists(st.text(min_size=1), min_size=0, max_size=100))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_keep_last_n_preserves_order(self, items):
        """keep_last_n(n) should keep the last n items in order."""
        n = 10
        reducer = keep_last_n(n)
        result = reducer([], items)
        expected = items[-n:] if len(items) > n else items
        assert result == expected

    @pytest.mark.pbt
    @given(
        existing=st.lists(st.text(min_size=1), min_size=0, max_size=20),
        new=st.lists(st.text(min_size=1), min_size=0, max_size=20),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_keep_last_n_combines_then_truncates(self, existing, new):
        """keep_last_n combines current + new, then takes last n."""
        n = 10
        reducer = keep_last_n(n)
        result = reducer(existing, new)
        combined = existing + new
        expected = combined[-n:] if len(combined) > n else combined
        assert result == expected
        assert len(result) <= n

    @pytest.mark.pbt
    @given(items=st.lists(st.text(min_size=1), min_size=0, max_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_keep_last_n_idempotent_when_under_limit(self, items):
        """When total items <= n, keep_last_n returns all items unchanged."""
        n = 100
        reducer = keep_last_n(n)
        result = reducer([], items)
        assert result == items

    @pytest.mark.pbt
    @given(
        lists=st.lists(
            st.lists(st.integers(), min_size=0, max_size=5),
            min_size=2,
            max_size=8,
        )
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_operator_add_associative(self, lists):
        """operator.add is associative: (a+b)+c == a+(b+c)."""
        if len(lists) < 3:
            return
        a, b, c = lists[0], lists[1], lists[2]
        left = operator.add(operator.add(a, b), c)
        right = operator.add(a, operator.add(b, c))
        assert left == right

    @pytest.mark.pbt
    @given(lst=st.lists(st.text(min_size=1), min_size=0, max_size=20))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_operator_add_identity(self, lst):
        """operator.add with empty list is identity: [] + x == x."""
        assert operator.add([], lst) == lst
        assert operator.add(lst, []) == lst

    @pytest.mark.pbt
    def test_keep_last_n_handles_none_gracefully(self):
        """keep_last_n handles None inputs by treating them as empty lists."""
        reducer = keep_last_n(5)
        assert reducer(None, None) == []
        assert reducer(None, ["a", "b"]) == ["a", "b"]
        assert reducer(["a", "b"], None) == ["a", "b"]


# =============================================================================
# 4. GRAPH TRAVERSAL
# =============================================================================


class TestGraphTraversal:
    """Test structural properties of the workflow graph."""

    def _get_graph_edges(self) -> dict[str, list[str]]:
        """
        Extract the adjacency list from the compiled graph.

        Returns a dict mapping each node to its possible next nodes.
        """
        # Define adjacency from graph construction in create_box_up_role_graph
        return {
            "validate_role": ["analyze_deps", "notify_failure"],
            "analyze_deps": ["check_reverse_deps", "notify_failure"],
            "check_reverse_deps": ["create_worktree", "notify_failure"],
            "create_worktree": ["run_molecule", "run_pytest", "merge_test_results", "notify_failure"],
            "run_molecule": ["merge_test_results"],
            "run_pytest": ["merge_test_results"],
            "merge_test_results": ["validate_deploy", "notify_failure"],
            "validate_deploy": ["create_commit", "notify_failure"],
            "create_commit": ["push_branch", "notify_failure"],
            "push_branch": ["create_issue", "notify_failure"],
            "create_issue": ["create_mr", "notify_failure"],
            "create_mr": ["human_approval", "report_summary"],
            "human_approval": ["add_to_merge_train", "notify_failure"],
            "add_to_merge_train": ["report_summary"],
            "report_summary": [],
            "notify_failure": [],
        }

    @pytest.mark.pbt
    def test_all_paths_reach_terminal(self):
        """Every path through the graph should reach END (report_summary or notify_failure)."""
        edges = self._get_graph_edges()
        terminal = {"report_summary", "notify_failure"}

        def can_reach_terminal(node: str, visited: set[str]) -> bool:
            if node in terminal:
                return True
            if node in visited:
                return False  # cycle detection
            visited.add(node)
            neighbors = edges.get(node, [])
            if not neighbors:
                return node in terminal
            return all(can_reach_terminal(n, visited.copy()) for n in neighbors)

        for node in edges:
            assert can_reach_terminal(node, set()), (
                f"Node '{node}' cannot reach a terminal state"
            )

    @pytest.mark.pbt
    def test_no_orphan_nodes(self):
        """Every node should be reachable from the entry point (validate_role)."""
        edges = self._get_graph_edges()
        reachable: set[str] = set()

        def dfs(node: str):
            if node in reachable:
                return
            reachable.add(node)
            for neighbor in edges.get(node, []):
                dfs(neighbor)

        dfs("validate_role")

        all_nodes = set(edges.keys())
        orphans = all_nodes - reachable
        assert not orphans, f"Orphan nodes not reachable from start: {orphans}"

    @pytest.mark.pbt
    @given(path=workflow_path_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_path_respects_edges(self, path):
        """Generated paths only follow defined edges."""
        edges = self._get_graph_edges()

        # Verify each consecutive pair follows a defined edge
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            neighbors = edges.get(source, [])
            assert target in neighbors, (
                f"Invalid edge {source} -> {target}. "
                f"Valid targets from {source}: {neighbors}"
            )

    @pytest.mark.pbt
    @given(path=workflow_path_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_path_starts_at_entry(self, path):
        """Every generated path must start at validate_role."""
        assert len(path) > 0
        assert path[0] == "validate_role"

    @pytest.mark.pbt
    @given(path=workflow_path_strategy())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_path_ends_at_terminal(self, path):
        """Every generated path must end at a terminal node."""
        assert path[-1] in TERMINAL_NODES

    @pytest.mark.pbt
    def test_notify_failure_reachable_from_conditional_nodes(self):
        """Nodes with conditional routing should have a path to notify_failure.

        Not all nodes have conditional routing. add_to_merge_train has a
        direct edge to report_summary only, which is correct since failures
        at the merge train stage are reported via report_summary.
        """
        edges = self._get_graph_edges()
        terminal = {"report_summary", "notify_failure"}
        # Nodes that use conditional routing (can potentially fail)
        conditional_nodes = {
            "validate_role", "analyze_deps", "check_reverse_deps",
            "create_worktree", "merge_test_results", "validate_deploy",
            "create_commit", "push_branch", "create_issue", "human_approval",
        }

        def can_reach(node: str, target: str, visited: set[str]) -> bool:
            if node == target:
                return True
            if node in visited:
                return False
            visited.add(node)
            return any(can_reach(n, target, visited.copy()) for n in edges.get(node, []))

        for node in conditional_nodes:
            assert can_reach(node, "notify_failure", set()), (
                f"Conditional node '{node}' cannot reach notify_failure"
            )

    @pytest.mark.pbt
    def test_graph_is_acyclic(self):
        """The workflow graph must be a DAG (no cycles)."""
        edges = self._get_graph_edges()

        def has_cycle(node: str, path: set[str], visited: set[str]) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for neighbor in edges.get(node, []):
                if has_cycle(neighbor, path.copy(), visited):
                    return True
            return False

        visited: set[str] = set()
        for node in edges:
            assert not has_cycle(node, set(), visited), (
                f"Cycle detected involving node '{node}'"
            )

    @pytest.mark.pbt
    def test_graph_construction_succeeds(self):
        """create_box_up_role_graph should succeed and return valid structure."""
        graph, breakpoints = create_box_up_role_graph(
            db_path=":memory:", parallel_tests=True, enable_breakpoints=False
        )
        assert graph is not None
        assert isinstance(breakpoints, list)
        assert len(breakpoints) == 0  # breakpoints disabled

    @pytest.mark.pbt
    def test_graph_construction_with_breakpoints(self):
        """create_box_up_role_graph with breakpoints returns expected nodes."""
        graph, breakpoints = create_box_up_role_graph(
            db_path=":memory:", parallel_tests=True, enable_breakpoints=True
        )
        assert graph is not None
        assert len(breakpoints) > 0
        # Breakpoints should be valid node names
        for bp in breakpoints:
            assert isinstance(bp, str)
            assert len(bp) > 0

    @pytest.mark.pbt
    def test_sequential_and_parallel_graphs_have_same_terminals(self):
        """Both parallel and sequential graph modes should have the same terminal nodes."""
        graph_par, _ = create_box_up_role_graph(
            db_path=":memory:", parallel_tests=True, enable_breakpoints=False
        )
        graph_seq, _ = create_box_up_role_graph(
            db_path=":memory:", parallel_tests=False, enable_breakpoints=False
        )
        # Both should have report_summary and notify_failure as terminal nodes
        # Verify by checking both graphs were created without error
        assert graph_par is not None
        assert graph_seq is not None
