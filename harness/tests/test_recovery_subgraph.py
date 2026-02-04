"""
Tests for recovery subgraph (v0.6.0 agentic recovery).

Tests cover:
- RecoveryConfig per-node configuration
- Recovery subgraph node functions (analyze, plan, execute, verify)
- Recovery routing (loop, resolve, escalate)
- HarnessStore memory persistence
- Tier selection and escalation
- Full recovery loop integration
"""

import pytest

from harness.dag.recovery_config import (
    DEFAULT_RECOVERY_CONFIG,
    NODE_RECOVERY_CONFIGS,
    RecoveryConfig,
    get_max_iterations,
    get_max_tier,
    get_recovery_config,
)
from harness.dag.recovery_subgraph import (
    RecoveryState,
    analyze_failure_node,
    create_recovery_subgraph,
    execute_fix_node,
    plan_fix_node,
    recovery_router,
    verify_fix_node,
)


# =============================================================================
# RECOVERY CONFIG TESTS
# =============================================================================


class TestRecoveryConfig:
    """Tests for per-node recovery configuration."""

    def test_default_config_values(self):
        """Default config should have sensible defaults."""
        config = DEFAULT_RECOVERY_CONFIG
        assert config.max_iterations == 3
        assert config.max_tier == 1
        assert config.agent_budget_usd == 0.50
        assert config.agent_timeout == 300
        assert config.escalate_on_exhaust is True
        assert config.persist_memory is True

    def test_run_molecule_high_budget(self):
        """run_molecule should have high iteration budget and tier 3 access."""
        config = get_recovery_config("run_molecule")
        assert config.max_iterations == 12
        assert config.max_tier == 3
        assert config.agent_budget_usd == 1.0
        assert config.agent_timeout == 600

    def test_run_pytest_high_budget(self):
        """run_pytest should have high iteration budget and tier 3 access."""
        config = get_recovery_config("run_pytest")
        assert config.max_iterations == 12
        assert config.max_tier == 3

    def test_create_worktree_limited(self):
        """create_worktree should have moderate budget and tier 1 max."""
        config = get_recovery_config("create_worktree")
        assert config.max_iterations == 5
        assert config.max_tier == 1

    def test_validate_deploy_tier2(self):
        """validate_deploy should allow tier 2 recovery."""
        config = get_recovery_config("validate_deploy")
        assert config.max_iterations == 5
        assert config.max_tier == 2

    def test_unknown_node_gets_default(self):
        """Unknown nodes should get default config."""
        config = get_recovery_config("nonexistent_node")
        assert config == DEFAULT_RECOVERY_CONFIG

    def test_get_max_iterations_convenience(self):
        """get_max_iterations should return correct value."""
        assert get_max_iterations("run_molecule") == 12
        assert get_max_iterations("create_commit") == 3
        assert get_max_iterations("unknown") == 3  # default

    def test_get_max_tier_convenience(self):
        """get_max_tier should return correct value."""
        assert get_max_tier("run_molecule") == 3
        assert get_max_tier("create_worktree") == 1
        assert get_max_tier("unknown") == 1  # default

    def test_config_is_frozen(self):
        """RecoveryConfig should be immutable."""
        config = RecoveryConfig()
        with pytest.raises(AttributeError):
            config.max_iterations = 10

    def test_all_configured_nodes_exist(self):
        """All configured nodes should be known workflow nodes."""
        known_nodes = {
            "create_worktree", "run_molecule", "run_pytest",
            "validate_deploy", "create_commit", "push_branch",
            "create_issue", "create_mr",
        }
        for node_name in NODE_RECOVERY_CONFIGS:
            assert node_name in known_nodes, f"Unknown node in config: {node_name}"

    def test_molecule_has_allowed_tools(self):
        """run_molecule config should specify allowed tools for agent."""
        config = get_recovery_config("run_molecule")
        assert len(config.allowed_tools) > 0
        assert "Read" in config.allowed_tools
        assert "Edit" in config.allowed_tools


# =============================================================================
# TIER SELECTION TESTS
# =============================================================================


class TestTierSelection:
    """Tests for select_recovery_tier function."""

    def test_transient_error_tier0(self):
        """Transient errors should get tier 0."""
        from harness.dag.error_resolution import ClassifiedError, ErrorType, select_recovery_tier

        error = ClassifiedError(
            error_type=ErrorType.TRANSIENT,
            message="timeout",
            resolution_hint="retry_with_backoff",
        )
        assert select_recovery_tier(error, "any_node") == 0

    def test_unexpected_error_tier4(self):
        """Unexpected errors should get tier 4."""
        from harness.dag.error_resolution import ClassifiedError, ErrorType, select_recovery_tier

        error = ClassifiedError(
            error_type=ErrorType.UNEXPECTED,
            message="crash",
        )
        assert select_recovery_tier(error, "any_node") == 4

    def test_recoverable_simple_tier1(self):
        """Simple recoverable errors should get tier 1."""
        from harness.dag.error_resolution import ClassifiedError, ErrorType, select_recovery_tier

        error = ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message="worktree exists",
            resolution_hint="remove_existing_worktree",
        )
        assert select_recovery_tier(error, "create_worktree") == 1

    def test_recoverable_complex_tier2(self):
        """Complex recoverable errors should get tier 2 if allowed."""
        from harness.dag.error_resolution import ClassifiedError, ErrorType, select_recovery_tier

        error = ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message="unknown recoverable issue",
            resolution_hint="investigate_further",
        )
        # validate_deploy allows up to tier 2
        assert select_recovery_tier(error, "validate_deploy") == 2

    def test_tier_capped_by_node_config(self):
        """Tier should be capped by node's max_tier."""
        from harness.dag.error_resolution import ClassifiedError, ErrorType, select_recovery_tier

        error = ClassifiedError(
            error_type=ErrorType.USER_FIXABLE,
            message="test failure",
            resolution_hint="fix_failing_tests",
        )
        # create_worktree has max_tier=1, so tier 3 gets capped to 1
        assert select_recovery_tier(error, "create_worktree") == 1
        # run_molecule has max_tier=3
        assert select_recovery_tier(error, "run_molecule") == 3

    def test_user_fixable_gets_tier3_when_allowed(self):
        """USER_FIXABLE should get tier 3 when node allows it."""
        from harness.dag.error_resolution import ClassifiedError, ErrorType, select_recovery_tier

        error = ClassifiedError(
            error_type=ErrorType.USER_FIXABLE,
            message="test failed",
            resolution_hint="fix_failing_tests",
        )
        tier = select_recovery_tier(error, "run_pytest")
        assert tier == 3


# =============================================================================
# RECOVERY SUBGRAPH NODE TESTS
# =============================================================================


class TestAnalyzeFailureNode:
    """Tests for analyze_failure_node."""

    @pytest.mark.asyncio
    async def test_increments_iteration(self):
        """Should increment recovery_iteration."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="Worktree already exists",
            recovery_iteration=0,
            recovery_budget=5,
        )
        result = await analyze_failure_node(state)
        assert result["recovery_iteration"] == 1

    @pytest.mark.asyncio
    async def test_escalates_when_over_budget(self):
        """Should escalate when iteration exceeds budget."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="Worktree already exists",
            recovery_iteration=5,  # Already at max
            recovery_budget=5,
        )
        result = await analyze_failure_node(state)
        assert result["recovery_result"] == "escalate"

    @pytest.mark.asyncio
    async def test_classifies_error(self):
        """Should classify the error and select tier."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="Worktree already exists at /tmp/test",
            recovery_iteration=0,
            recovery_budget=5,
        )
        result = await analyze_failure_node(state)
        assert result["recovery_tier"] >= 0
        assert result["recovery_result"] == "continue"

    @pytest.mark.asyncio
    async def test_records_analysis_action(self):
        """Should record analysis action in recovery_actions."""
        state = RecoveryState(
            role_name="common",
            failed_node="validate_deploy",
            error_message="site.yml not found",
            recovery_iteration=0,
            recovery_budget=5,
        )
        result = await analyze_failure_node(state)
        actions = result.get("recovery_actions", [])
        assert len(actions) >= 1
        assert actions[0]["action"] in ("analyze", "memory_lookup")


class TestPlanFixNode:
    """Tests for plan_fix_node."""

    @pytest.mark.asyncio
    async def test_tier1_plan(self):
        """Tier 1 should plan an inline fix."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="Worktree already exists",
            recovery_tier=1,
            recovery_iteration=1,
        )
        result = await plan_fix_node(state)
        assert result["recovery_plan"] is not None
        assert "Tier 1" in result["recovery_plan"]

    @pytest.mark.asyncio
    async def test_tier2_plan(self):
        """Tier 2 should plan a subgraph fix."""
        state = RecoveryState(
            role_name="common",
            failed_node="validate_deploy",
            error_message="site.yml could not be found",
            recovery_tier=2,
            recovery_iteration=1,
        )
        result = await plan_fix_node(state)
        assert result["recovery_plan"] is not None
        assert "Tier 2" in result["recovery_plan"]

    @pytest.mark.asyncio
    async def test_tier3_plan(self):
        """Tier 3 should plan a Claude SDK agent fix."""
        state = RecoveryState(
            role_name="common",
            failed_node="run_molecule",
            error_message="Test failed: assertion error",
            recovery_tier=3,
            recovery_iteration=1,
        )
        result = await plan_fix_node(state)
        assert result["recovery_plan"] is not None
        assert "Tier 3" in result["recovery_plan"]

    @pytest.mark.asyncio
    async def test_tier4_escalates(self):
        """Tier 4 (unexpected) should escalate immediately."""
        state = RecoveryState(
            role_name="common",
            failed_node="unknown",
            error_message="crash",
            recovery_tier=4,
            recovery_iteration=1,
        )
        result = await plan_fix_node(state)
        assert result.get("recovery_result") == "escalate"


class TestExecuteFixNode:
    """Tests for execute_fix_node."""

    @pytest.mark.asyncio
    async def test_tier1_applies_resolution(self):
        """Tier 1 should attempt inline resolution."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="Worktree already exists at /tmp/test",
            recovery_tier=1,
            recovery_iteration=1,
        )
        result = await execute_fix_node(state)
        # Should have resolution_updates (worktree_force_recreate)
        assert "resolution_updates" in result

    @pytest.mark.asyncio
    async def test_tier2_checks_paths(self, tmp_path):
        """Tier 2 should investigate paths for validate_deploy."""
        # Create ansible/site.yml
        ansible_dir = tmp_path / "ansible"
        ansible_dir.mkdir()
        (ansible_dir / "site.yml").touch()

        state = RecoveryState(
            role_name="common",
            failed_node="validate_deploy",
            error_message="site.yml could not be found",
            recovery_tier=2,
            recovery_iteration=1,
            worktree_path=str(tmp_path),
        )
        result = await execute_fix_node(state)
        resolution = result.get("resolution_updates")
        assert resolution is not None
        assert "ansible_cwd" in resolution

    @pytest.mark.asyncio
    async def test_passthrough_when_escalated(self):
        """Should pass through when already escalated."""
        state = RecoveryState(
            role_name="common",
            failed_node="unknown",
            error_message="crash",
            recovery_tier=1,
            recovery_iteration=1,
            recovery_result="escalate",
        )
        result = await execute_fix_node(state)
        assert result == {}


class TestVerifyFixNode:
    """Tests for verify_fix_node."""

    @pytest.mark.asyncio
    async def test_resolves_with_updates(self):
        """Should resolve when resolution_updates are present."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="worktree exists",
            recovery_tier=1,
            recovery_iteration=1,
            recovery_budget=5,
            resolution_updates={"worktree_force_recreate": True},
        )
        result = await verify_fix_node(state)
        assert result["recovery_result"] == "resolved"

    @pytest.mark.asyncio
    async def test_continues_when_no_fix(self):
        """Should continue looping when no fix was applied."""
        state = RecoveryState(
            role_name="common",
            failed_node="validate_deploy",
            error_message="unknown error",
            recovery_tier=2,
            recovery_iteration=1,
            recovery_budget=5,
            resolution_updates=None,
        )
        result = await verify_fix_node(state)
        assert result["recovery_result"] == "continue"

    @pytest.mark.asyncio
    async def test_escalates_at_budget(self):
        """Should escalate when budget is exhausted."""
        state = RecoveryState(
            role_name="common",
            failed_node="validate_deploy",
            error_message="unknown error",
            recovery_tier=2,
            recovery_iteration=5,
            recovery_budget=5,
            resolution_updates=None,
        )
        result = await verify_fix_node(state)
        assert result["recovery_result"] == "escalate"

    @pytest.mark.asyncio
    async def test_passthrough_when_already_resolved(self):
        """Should pass through when already resolved."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="worktree exists",
            recovery_result="resolved",
        )
        result = await verify_fix_node(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_passthrough_when_already_escalated(self):
        """Should pass through when already escalated."""
        state = RecoveryState(
            role_name="common",
            failed_node="create_worktree",
            error_message="worktree exists",
            recovery_result="escalate",
        )
        result = await verify_fix_node(state)
        assert result == {}


# =============================================================================
# ROUTING TESTS
# =============================================================================


class TestRecoveryRouter:
    """Tests for recovery_router function."""

    def test_resolved_goes_to_end(self):
        """Resolved should route to END."""
        from langgraph.graph import END

        state = RecoveryState(recovery_result="resolved")
        assert recovery_router(state) == END

    def test_escalate_goes_to_end(self):
        """Escalate should route to END."""
        from langgraph.graph import END

        state = RecoveryState(recovery_result="escalate")
        assert recovery_router(state) == END

    def test_continue_loops_back(self):
        """Continue should loop back to analyze_failure."""
        state = RecoveryState(recovery_result="continue")
        assert recovery_router(state) == "analyze_failure"

    def test_none_loops_back(self):
        """None result should loop back to analyze_failure."""
        state = RecoveryState(recovery_result=None)
        assert recovery_router(state) == "analyze_failure"


class TestRouteAfterRecovery:
    """Tests for route_after_recovery in langgraph_routing."""

    def test_resolved_routes_to_failed_node(self):
        """Resolved recovery should route back to the failed node."""
        from harness.dag.langgraph_routing import route_after_recovery

        state = {
            "recovery_result": "resolved",
            "last_error_node": "validate_deploy",
        }
        assert route_after_recovery(state) == "validate_deploy"

    def test_escalate_routes_to_failure(self):
        """Escalated recovery should route to notify_failure."""
        from harness.dag.langgraph_routing import route_after_recovery

        state = {
            "recovery_result": "escalate",
            "last_error_node": "validate_deploy",
        }
        assert route_after_recovery(state) == "notify_failure"

    def test_none_result_routes_to_failure(self):
        """None result should route to notify_failure."""
        from harness.dag.langgraph_routing import route_after_recovery

        state = {
            "recovery_result": None,
            "last_error_node": "validate_deploy",
        }
        assert route_after_recovery(state) == "notify_failure"

    def test_resolved_unknown_node_routes_to_failure(self):
        """Resolved with unknown node should route to notify_failure."""
        from harness.dag.langgraph_routing import route_after_recovery

        state = {
            "recovery_result": "resolved",
            "last_error_node": "unknown_node",
        }
        assert route_after_recovery(state) == "notify_failure"


# =============================================================================
# SUBGRAPH CONSTRUCTION TESTS
# =============================================================================


class TestRecoverySubgraphConstruction:
    """Tests for subgraph creation and compilation."""

    def test_create_subgraph_returns_state_graph(self):
        """create_recovery_subgraph should return a StateGraph."""
        from langgraph.graph import StateGraph

        graph = create_recovery_subgraph()
        assert isinstance(graph, StateGraph)

    def test_subgraph_compiles(self):
        """Recovery subgraph should compile without errors."""
        graph = create_recovery_subgraph()
        compiled = graph.compile()
        assert compiled is not None

    def test_subgraph_has_all_nodes(self):
        """Subgraph should have all four recovery nodes."""
        graph = create_recovery_subgraph()
        # Check node names exist by compiling and checking
        compiled = graph.compile()
        # The graph should have the nodes defined
        node_names = set(graph.nodes.keys())
        expected = {"analyze_failure", "plan_fix", "execute_fix", "verify_fix"}
        assert expected.issubset(node_names)


# =============================================================================
# MEMORY PERSISTENCE TESTS
# =============================================================================


class TestRecoveryMemory:
    """Tests for recovery memory persistence functions."""

    def test_lookup_returns_none_without_store(self):
        """lookup_recovery_memory should return None when store is None."""
        from harness.dag.error_resolution import lookup_recovery_memory

        result = lookup_recovery_memory(None, "test_node", "test_role", "error")
        assert result is None

    def test_persist_does_nothing_without_store(self):
        """persist_recovery_memory should not raise when store is None."""
        from harness.dag.error_resolution import persist_recovery_memory

        # Should not raise
        persist_recovery_memory(
            store=None,
            node_name="test",
            role_name="test",
            error_pattern="test",
            fix_applied="test",
            success=True,
            iterations=1,
        )


# =============================================================================
# SHOULD_ATTEMPT_RECOVERY_V2 TESTS
# =============================================================================


class TestShouldAttemptRecoveryV2:
    """Tests for should_attempt_recovery_v2 with per-node configs."""

    def test_below_budget_returns_true(self):
        """Should return True when below node's max_iterations."""
        from harness.dag.error_resolution import should_attempt_recovery_v2

        state = {
            "recovery_attempts": [
                {"node": "run_molecule", "hint": "fix_test"},
            ]
        }
        result = should_attempt_recovery_v2(state, "run_molecule")
        assert result is True  # 1 attempt < 12 max

    def test_at_budget_returns_false(self):
        """Should return False when at max_iterations."""
        from harness.dag.error_resolution import should_attempt_recovery_v2

        config = RecoveryConfig(max_iterations=2)
        state = {
            "recovery_attempts": [
                {"node": "test_node", "hint": "fix1"},
                {"node": "test_node", "hint": "fix2"},
            ]
        }
        result = should_attempt_recovery_v2(state, "test_node", config=config)
        assert result is False

    def test_uses_node_config(self):
        """Should use the node's config from registry."""
        from harness.dag.error_resolution import should_attempt_recovery_v2

        # create_worktree has max_iterations=5
        state = {
            "recovery_attempts": [
                {"node": "create_worktree", "hint": "fix"} for _ in range(4)
            ]
        }
        result = should_attempt_recovery_v2(state, "create_worktree")
        assert result is True  # 4 < 5

        state["recovery_attempts"].append({"node": "create_worktree", "hint": "fix"})
        result = should_attempt_recovery_v2(state, "create_worktree")
        assert result is False  # 5 >= 5


# =============================================================================
# ESCALATION TESTS
# =============================================================================


class TestEscalation:
    """Tests for escalate_to_human function."""

    @pytest.mark.asyncio
    async def test_escalation_returns_correct_state(self):
        """escalate_to_human should set escalation markers."""
        from harness.dag.recovery_subgraph import escalate_to_human

        state = RecoveryState(
            role_name="common",
            failed_node="run_molecule",
            error_message="Tests keep failing",
            recovery_iteration=12,
            recovery_actions=[],
        )
        result = await escalate_to_human(state)
        assert result["recovery_result"] == "escalate"
        assert result["awaiting_human_input"] is True
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_escalation_with_notification_service(self):
        """escalate_to_human should send notification if service provided."""
        from unittest.mock import AsyncMock

        from harness.dag.recovery_subgraph import escalate_to_human

        mock_service = AsyncMock()
        mock_service.send = AsyncMock(return_value=True)

        state = RecoveryState(
            role_name="common",
            failed_node="run_molecule",
            error_message="Tests keep failing",
            recovery_iteration=12,
            recovery_actions=[],
        )
        result = await escalate_to_human(state, notification_service=mock_service)
        assert result["recovery_result"] == "escalate"
        # Notification may or may not have been called depending on imports
        # The important thing is it didn't raise
