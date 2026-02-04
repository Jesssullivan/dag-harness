"""
Per-node recovery configuration for agentic recovery loops (v0.6.0).

Defines configurable recovery budgets, tiers, and escalation rules
for each workflow node. Recovery tiers determine what strategies
are available:

- Tier 0: LangGraph RetryPolicy (transient errors, existing)
- Tier 1: Inline fix via state update (v0.5.0 pattern)
- Tier 2: Recovery subgraph with tool execution
- Tier 3: Claude SDK agent spawned to fix code
- Tier 4: Fail fast, persist diagnostics, escalate (UNEXPECTED)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecoveryConfig:
    """Configuration for a node's recovery behavior.

    Attributes:
        max_iterations: Maximum recovery loop iterations before escalation.
        max_tier: Highest recovery tier this node can use (0-3).
        agent_budget_usd: Budget cap for Claude SDK agent (tier 3).
        agent_timeout: Timeout in seconds for agent execution (tier 3).
        escalate_on_exhaust: Whether to escalate via HOTL when budget exhausted.
        persist_memory: Whether to save recovery learnings to HarnessStore.
        allowed_tools: Tools available to tier 3 agents. Empty list means default set.
    """

    max_iterations: int = 3
    max_tier: int = 1
    agent_budget_usd: float = 0.50
    agent_timeout: int = 300
    escalate_on_exhaust: bool = True
    persist_memory: bool = True
    allowed_tools: tuple[str, ...] = ()


# Per-node recovery configurations.
# Nodes not listed here get DEFAULT_RECOVERY_CONFIG.
NODE_RECOVERY_CONFIGS: dict[str, RecoveryConfig] = {
    "create_worktree": RecoveryConfig(
        max_iterations=5,
        max_tier=1,
    ),
    "run_molecule": RecoveryConfig(
        max_iterations=12,
        max_tier=3,
        agent_budget_usd=1.0,
        agent_timeout=600,
        allowed_tools=("Read", "Write", "Edit", "Bash", "Glob", "Grep"),
    ),
    "run_pytest": RecoveryConfig(
        max_iterations=12,
        max_tier=3,
        agent_budget_usd=1.0,
        agent_timeout=600,
        allowed_tools=("Read", "Write", "Edit", "Bash", "Glob", "Grep"),
    ),
    "validate_deploy": RecoveryConfig(
        max_iterations=5,
        max_tier=2,
    ),
    "create_commit": RecoveryConfig(
        max_iterations=3,
        max_tier=1,
    ),
    "push_branch": RecoveryConfig(
        max_iterations=5,
        max_tier=1,
    ),
    "create_issue": RecoveryConfig(
        max_iterations=3,
        max_tier=1,
    ),
    "create_mr": RecoveryConfig(
        max_iterations=3,
        max_tier=1,
    ),
}

DEFAULT_RECOVERY_CONFIG = RecoveryConfig()


def get_recovery_config(node_name: str) -> RecoveryConfig:
    """Get the recovery configuration for a node.

    Args:
        node_name: Name of the workflow node.

    Returns:
        RecoveryConfig for the node, or DEFAULT_RECOVERY_CONFIG if not configured.
    """
    return NODE_RECOVERY_CONFIGS.get(node_name, DEFAULT_RECOVERY_CONFIG)


def get_max_iterations(node_name: str) -> int:
    """Get the maximum recovery iterations for a node.

    Convenience function for routing decisions.
    """
    return get_recovery_config(node_name).max_iterations


def get_max_tier(node_name: str) -> int:
    """Get the maximum recovery tier for a node.

    Convenience function for tier selection.
    """
    return get_recovery_config(node_name).max_tier


__all__ = [
    "RecoveryConfig",
    "NODE_RECOVERY_CONFIGS",
    "DEFAULT_RECOVERY_CONFIG",
    "get_recovery_config",
    "get_max_iterations",
    "get_max_tier",
]
