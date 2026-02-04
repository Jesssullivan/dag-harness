"""
Recovery subgraph for agentic error recovery (v0.6.0).

Implements a LangGraph StateGraph with an analyze -> plan -> execute -> verify
cycle. This subgraph is embedded as a compiled node in the parent workflow
graph. When a node fails with a recoverable or user-fixable error, routing
enters this subgraph which loops internally until the fix works, the budget
is exhausted, or escalation is needed.

Architecture:
    analyze_failure -> plan_fix -> execute_fix -> verify_fix
         ^                                          |
         └──────────── (loop if not resolved) ──────┘
                          |
                   [resolved -> END]
                   [budget exhausted -> END (escalate)]
"""

from __future__ import annotations

import logging
import operator
import subprocess
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


# =============================================================================
# RECOVERY STATE SCHEMA
# =============================================================================


class RecoveryState(TypedDict, total=False):
    """State schema for the recovery subgraph.

    Fields prefixed with the parent graph's state are passed through
    from the parent and available for context.
    """

    # Inherited from parent (read-only context)
    role_name: str
    worktree_path: str
    execution_id: int | None
    has_molecule_tests: bool

    # Recovery identification
    failed_node: str
    error_message: str
    error_type: str  # ErrorType value

    # Recovery loop control
    recovery_iteration: int
    recovery_budget: int  # Max iterations (from RecoveryConfig)
    recovery_tier: int  # Current tier (0-4)
    recovery_plan: str | None  # Natural language plan from analyze step
    recovery_actions: Annotated[list[dict], operator.add]  # Actions taken
    recovery_result: str | None  # "resolved" | "escalate" | "continue"

    # Agent state (tier 3)
    agent_session_id: str | None
    agent_output: str | None
    agent_budget_usd: float

    # Memory / store
    store_namespace: str | None  # Serialized namespace for HarnessStore

    # Resolution state (passed back to parent)
    resolution_updates: dict | None  # State updates to apply on success

    # Parent state fields that may be updated by recovery
    worktree_force_recreate: bool
    branch_force_recreate: bool
    ansible_cwd: str | None
    recovery_context: dict | None
    recovery_attempts: Annotated[list[dict], operator.add]
    last_error_type: str | None
    last_error_message: str | None
    last_error_node: str | None

    # HOTL escalation
    awaiting_human_input: bool
    errors: Annotated[list[str], operator.add]


# =============================================================================
# RECOVERY SUBGRAPH NODES
# =============================================================================


async def analyze_failure_node(state: RecoveryState) -> dict:
    """
    Classify error, load prior recovery memory, determine recovery tier.

    For tier 3 (agent), checks if Claude SDK is available.
    If a matching pattern exists in HarnessStore, returns the known fix
    immediately (skipping plan/execute).
    """
    from harness.dag.error_resolution import (
        ClassifiedError,
        ErrorType,
        classify_error,
        lookup_recovery_memory,
        select_recovery_tier,
    )
    from harness.dag.langgraph_state import get_module_db
    from harness.dag.recovery_config import get_recovery_config

    failed_node = state.get("failed_node", "unknown")
    error_message = state.get("error_message", "")
    role_name = state.get("role_name", "unknown")
    iteration = state.get("recovery_iteration", 0) + 1

    config = get_recovery_config(failed_node)

    # Classify the error
    error = classify_error(error_message, failed_node, {"role_name": role_name})
    tier = select_recovery_tier(error, failed_node)

    # Check budget
    if iteration > config.max_iterations:
        return {
            "recovery_iteration": iteration,
            "recovery_tier": tier,
            "recovery_result": "escalate",
            "recovery_plan": None,
        }

    # Try to look up prior recovery memory
    db = get_module_db()
    store = None
    if db is not None:
        try:
            from harness.dag.store import create_harness_store

            store = create_harness_store(db)
        except Exception:
            pass

    # Extract error pattern for memory lookup (first 100 chars, normalized)
    error_pattern = error_message[:100].strip().lower()
    memory = lookup_recovery_memory(store, failed_node, role_name, error_pattern)

    if memory and memory.get("success"):
        # Known fix from prior run - apply directly
        logger.info(
            f"Applying known fix from memory for {failed_node}: {memory.get('fix_applied')}"
        )
        return {
            "recovery_iteration": iteration,
            "recovery_tier": min(tier, 1),  # Known fixes are tier 1
            "recovery_plan": f"Apply known fix: {memory.get('fix_applied')}",
            "recovery_result": "continue",
            "recovery_actions": [
                {
                    "action": "memory_lookup",
                    "iteration": iteration,
                    "result": f"Found known fix: {memory.get('fix_applied')}",
                }
            ],
        }

    return {
        "recovery_iteration": iteration,
        "recovery_tier": tier,
        "recovery_plan": None,
        "recovery_result": "continue",
        "error_type": error.error_type.value,
        "recovery_actions": [
            {
                "action": "analyze",
                "iteration": iteration,
                "error_type": error.error_type.value,
                "tier": tier,
                "hint": error.resolution_hint,
            }
        ],
    }


async def plan_fix_node(state: RecoveryState) -> dict:
    """
    Plan the recovery fix based on the selected tier.

    - Tier 1: Select inline resolution from error_resolution.py
    - Tier 2: Determine tool commands to run
    - Tier 3: Build system prompt + task description for Claude SDK agent
    """
    from harness.dag.error_resolution import (
        ClassifiedError,
        ErrorType,
        classify_error,
    )

    tier = state.get("recovery_tier", 1)
    failed_node = state.get("failed_node", "unknown")
    error_message = state.get("error_message", "")
    role_name = state.get("role_name", "unknown")
    iteration = state.get("recovery_iteration", 1)

    # Re-classify for resolution hint (idempotent)
    error = classify_error(error_message, failed_node, {"role_name": role_name})

    if tier <= 1:
        # Tier 1: inline fix from existing resolution strategies
        plan = f"Tier 1: Apply inline fix '{error.resolution_hint}' for {failed_node}"
        return {
            "recovery_plan": plan,
            "recovery_actions": [
                {
                    "action": "plan",
                    "iteration": iteration,
                    "tier": tier,
                    "strategy": error.resolution_hint,
                }
            ],
        }

    if tier == 2:
        # Tier 2: subgraph tool execution
        plan = _build_tier2_plan(failed_node, error, state)
        return {
            "recovery_plan": plan,
            "recovery_actions": [
                {
                    "action": "plan",
                    "iteration": iteration,
                    "tier": tier,
                    "plan": plan,
                }
            ],
        }

    if tier == 3:
        # Tier 3: Claude SDK agent
        plan = _build_tier3_plan(failed_node, error, state)
        return {
            "recovery_plan": plan,
            "recovery_actions": [
                {
                    "action": "plan",
                    "iteration": iteration,
                    "tier": tier,
                    "plan": plan,
                }
            ],
        }

    # Tier 4 / unexpected: no plan, will escalate
    return {
        "recovery_plan": None,
        "recovery_result": "escalate",
    }


async def execute_fix_node(state: RecoveryState) -> dict:
    """
    Execute the planned fix based on tier.

    - Tier 1: Apply state updates via attempt_resolution()
    - Tier 2: Execute commands via subprocess (sandboxed)
    - Tier 3: Spawn Claude SDK agent
    """
    from harness.dag.error_resolution import (
        attempt_resolution,
        classify_error,
        create_recovery_state_update,
    )

    tier = state.get("recovery_tier", 1)
    failed_node = state.get("failed_node", "unknown")
    error_message = state.get("error_message", "")
    role_name = state.get("role_name", "unknown")
    iteration = state.get("recovery_iteration", 1)

    # If already escalated, pass through
    if state.get("recovery_result") == "escalate":
        return {}

    error = classify_error(error_message, failed_node, {"role_name": role_name})

    if tier <= 1:
        return _execute_tier1(failed_node, error, state, iteration)

    if tier == 2:
        return await _execute_tier2(failed_node, error, state, iteration)

    if tier == 3:
        return await _execute_tier3(failed_node, error, state, iteration)

    # Unknown tier - escalate
    return {
        "recovery_result": "escalate",
        "recovery_actions": [
            {
                "action": "execute",
                "iteration": iteration,
                "result": f"Unknown tier {tier}, escalating",
            }
        ],
    }


async def verify_fix_node(state: RecoveryState) -> dict:
    """
    Verify whether the fix resolved the issue.

    For tier 1-2: Check if resolution_updates were produced.
    For tier 3: Check agent output for success indicators.

    Persists outcome to HarnessStore memory.
    """
    from harness.dag.error_resolution import persist_recovery_memory
    from harness.dag.langgraph_state import get_module_db

    iteration = state.get("recovery_iteration", 1)
    failed_node = state.get("failed_node", "unknown")
    role_name = state.get("role_name", "unknown")
    error_message = state.get("error_message", "")
    tier = state.get("recovery_tier", 1)
    result = state.get("recovery_result")

    # Already decided (escalate or resolved from earlier step)
    if result in ("escalate", "resolved"):
        return {}

    resolution_updates = state.get("resolution_updates")
    agent_output = state.get("agent_output")

    # Determine if fix was applied successfully
    fix_applied = False
    fix_description = "none"

    if tier <= 1 and resolution_updates:
        fix_applied = True
        fix_description = str(list(resolution_updates.keys()))
    elif tier == 2 and resolution_updates:
        fix_applied = True
        fix_description = state.get("recovery_plan", "tier2_fix")
    elif tier == 3 and agent_output:
        # Agent ran - check if it reported success
        output_lower = (agent_output or "").lower()
        fix_applied = any(
            indicator in output_lower
            for indicator in ["fixed", "resolved", "passing", "success", "tests pass"]
        )
        fix_description = "agent_fix"

    # Persist to store
    store = None
    db = get_module_db()
    if db is not None:
        try:
            from harness.dag.store import create_harness_store

            store = create_harness_store(db)
        except Exception:
            pass

    error_pattern = error_message[:100].strip().lower()
    persist_recovery_memory(
        store=store,
        node_name=failed_node,
        role_name=role_name,
        error_pattern=error_pattern,
        fix_applied=fix_description,
        success=fix_applied,
        iterations=iteration,
    )

    if fix_applied:
        return {
            "recovery_result": "resolved",
            "recovery_actions": [
                {
                    "action": "verify",
                    "iteration": iteration,
                    "result": "resolved",
                    "fix": fix_description,
                }
            ],
        }

    # Check if budget exhausted
    budget = state.get("recovery_budget", 3)
    if iteration >= budget:
        return {
            "recovery_result": "escalate",
            "recovery_actions": [
                {
                    "action": "verify",
                    "iteration": iteration,
                    "result": "budget_exhausted",
                }
            ],
        }

    # Continue looping
    return {
        "recovery_result": "continue",
        "recovery_actions": [
            {
                "action": "verify",
                "iteration": iteration,
                "result": "continue",
            }
        ],
    }


# =============================================================================
# ROUTING
# =============================================================================


def recovery_router(state: RecoveryState) -> str:
    """Route after verify_fix: loop back, finish resolved, or escalate."""
    result = state.get("recovery_result")
    if result == "resolved":
        return END
    if result == "escalate":
        return END
    # "continue" -> loop back to analyze
    return "analyze_failure"


# =============================================================================
# SUBGRAPH FACTORY
# =============================================================================


def create_recovery_subgraph() -> StateGraph:
    """
    Create the recovery subgraph with analyze -> plan -> execute -> verify loop.

    Returns:
        StateGraph ready to compile.
    """
    graph = StateGraph(RecoveryState)

    graph.add_node("analyze_failure", analyze_failure_node)
    graph.add_node("plan_fix", plan_fix_node)
    graph.add_node("execute_fix", execute_fix_node)
    graph.add_node("verify_fix", verify_fix_node)

    graph.set_entry_point("analyze_failure")
    graph.add_edge("analyze_failure", "plan_fix")
    graph.add_edge("plan_fix", "execute_fix")
    graph.add_edge("execute_fix", "verify_fix")
    graph.add_conditional_edges(
        "verify_fix",
        recovery_router,
        ["analyze_failure", END],
    )

    return graph


# =============================================================================
# ESCALATION
# =============================================================================


async def escalate_to_human(
    state: RecoveryState,
    notification_service: Any | None = None,
) -> dict:
    """
    Escalate to human when recovery budget is exhausted.

    Persists full recovery history and sends notification via
    Discord/email if notification_service is provided.

    Args:
        state: Current recovery state.
        notification_service: Optional NotificationService instance.

    Returns:
        State update with escalation markers.
    """
    failed_node = state.get("failed_node", "unknown")
    role_name = state.get("role_name", "unknown")
    iteration = state.get("recovery_iteration", 0)
    actions = state.get("recovery_actions", [])

    summary = _format_recovery_summary(state)

    if notification_service is not None:
        try:
            from harness.notifications import Notification, NotificationPriority, NotificationType

            notification = Notification(
                type=NotificationType.HUMAN_NEEDED,
                title=f"Recovery exhausted: {failed_node} for {role_name}",
                message=summary,
                priority=NotificationPriority.HIGH,
                data={
                    "failed_node": failed_node,
                    "role_name": role_name,
                    "iterations": iteration,
                    "actions": actions,
                },
            )
            await notification_service.send(notification)
        except Exception as e:
            logger.warning(f"Failed to send escalation notification: {e}")

    return {
        "recovery_result": "escalate",
        "awaiting_human_input": True,
        "errors": [
            f"Recovery exhausted after {iteration} iterations for {failed_node}"
        ],
    }


# =============================================================================
# TIER EXECUTION HELPERS
# =============================================================================


def _execute_tier1(
    failed_node: str,
    error: Any,
    state: RecoveryState,
    iteration: int,
) -> dict:
    """Execute a tier 1 inline fix."""
    from harness.dag.error_resolution import attempt_resolution

    resolution = attempt_resolution(
        error.resolution_hint,
        error.context,
        dict(state),
    )

    if resolution:
        return {
            "resolution_updates": resolution,
            "recovery_actions": [
                {
                    "action": "execute",
                    "iteration": iteration,
                    "tier": 1,
                    "hint": error.resolution_hint,
                    "resolution_keys": list(resolution.keys()),
                }
            ],
        }

    return {
        "resolution_updates": None,
        "recovery_actions": [
            {
                "action": "execute",
                "iteration": iteration,
                "tier": 1,
                "hint": error.resolution_hint,
                "result": "no_resolution_available",
            }
        ],
    }


def _build_tier2_plan(failed_node: str, error: Any, state: RecoveryState) -> str:
    """Build a tier 2 recovery plan (subprocess commands)."""
    if failed_node == "validate_deploy":
        worktree_path = state.get("worktree_path", ".")
        return (
            f"Tier 2: Search for site.yml in {worktree_path}, "
            f"check ansible/ subdirectory structure, "
            f"verify role path exists"
        )
    if failed_node in ("run_molecule", "run_pytest"):
        return (
            f"Tier 2: Re-run failing test with verbose output, "
            f"check for missing dependencies or configuration"
        )
    return f"Tier 2: Investigate {failed_node} failure via subprocess commands"


async def _execute_tier2(
    failed_node: str,
    error: Any,
    state: RecoveryState,
    iteration: int,
) -> dict:
    """Execute a tier 2 fix (subprocess-based investigation and fix)."""
    worktree_path = state.get("worktree_path", ".")
    role_name = state.get("role_name", "unknown")

    resolution = {}

    if failed_node == "validate_deploy":
        # Try to find site.yml
        ansible_path = Path(worktree_path) / "ansible"
        if ansible_path.exists() and (ansible_path / "site.yml").exists():
            resolution["ansible_cwd"] = str(ansible_path)
        else:
            root_site = Path(worktree_path) / "site.yml"
            if root_site.exists():
                resolution["ansible_cwd"] = str(worktree_path)

    if resolution:
        return {
            "resolution_updates": resolution,
            "recovery_actions": [
                {
                    "action": "execute",
                    "iteration": iteration,
                    "tier": 2,
                    "resolution_keys": list(resolution.keys()),
                }
            ],
        }

    return {
        "resolution_updates": None,
        "recovery_actions": [
            {
                "action": "execute",
                "iteration": iteration,
                "tier": 2,
                "result": "no_fix_found",
            }
        ],
    }


def _build_tier3_plan(failed_node: str, error: Any, state: RecoveryState) -> str:
    """Build a tier 3 recovery plan (Claude SDK agent)."""
    role_name = state.get("role_name", "unknown")
    return (
        f"Tier 3: Spawn Claude SDK agent to fix failing {failed_node} "
        f"for role {role_name}. Agent will read test output, "
        f"identify the issue, edit code, and re-run tests."
    )


async def _execute_tier3(
    failed_node: str,
    error: Any,
    state: RecoveryState,
    iteration: int,
) -> dict:
    """Execute a tier 3 fix (Claude SDK agent)."""
    from harness.hotl.claude_sdk_integration import sdk_available

    if not sdk_available():
        logger.warning("Claude SDK not available for tier 3 recovery")
        return {
            "resolution_updates": None,
            "recovery_actions": [
                {
                    "action": "execute",
                    "iteration": iteration,
                    "tier": 3,
                    "result": "sdk_not_available",
                }
            ],
        }

    role_name = state.get("role_name", "unknown")
    worktree_path = state.get("worktree_path", ".")
    error_message = state.get("error_message", "")
    budget_usd = state.get("agent_budget_usd", 0.50)

    try:
        from harness.dag.langgraph_state import get_module_config, get_module_db
        from harness.hotl.claude_sdk_integration import (
            SDKAgentConfig,
            SDKClaudeIntegration,
            PermissionMode,
        )

        db = get_module_db()
        config = get_module_config()

        if db is None:
            return {
                "resolution_updates": None,
                "recovery_actions": [
                    {
                        "action": "execute",
                        "iteration": iteration,
                        "tier": 3,
                        "result": "no_db_available",
                    }
                ],
            }

        system_prompt = _build_agent_system_prompt(failed_node, role_name, error_message)
        task = f"Fix failing {failed_node} for role {role_name}:\n\n{error_message[:2000]}"

        from harness.dag.recovery_config import get_recovery_config

        rc = get_recovery_config(failed_node)
        allowed_tools = list(rc.allowed_tools) if rc.allowed_tools else [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep"
        ]

        agent_config = SDKAgentConfig(
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            permission_mode=PermissionMode.ACCEPT_EDITS,
            max_turns=20,
            max_budget_usd=budget_usd,
            cwd=Path(worktree_path),
        )

        sdk = SDKClaudeIntegration(config=agent_config, session_manager=None, db=db)
        session = await sdk.spawn_agent_async(task=task, working_dir=Path(worktree_path))

        agent_output = getattr(session, "output", None) or ""
        session_id = getattr(session, "session_id", None)

        return {
            "agent_session_id": session_id,
            "agent_output": agent_output[:5000] if agent_output else "",
            "recovery_actions": [
                {
                    "action": "execute",
                    "iteration": iteration,
                    "tier": 3,
                    "agent_session_id": session_id,
                    "result": "agent_completed",
                }
            ],
        }

    except Exception as e:
        logger.error(f"Tier 3 agent execution failed: {e}")
        return {
            "resolution_updates": None,
            "recovery_actions": [
                {
                    "action": "execute",
                    "iteration": iteration,
                    "tier": 3,
                    "result": f"agent_error: {e}",
                }
            ],
        }


# =============================================================================
# HELPERS
# =============================================================================


def _build_agent_system_prompt(
    failed_node: str,
    role_name: str,
    error_message: str,
) -> str:
    """Build a system prompt for the Claude SDK recovery agent."""
    return f"""You are a recovery agent for an Ansible role workflow.
Your task is to fix a failing step in the workflow.

Failed node: {failed_node}
Role name: {role_name}

The error message is:
{error_message[:3000]}

Instructions:
1. Read the relevant files to understand the error
2. Make minimal, targeted fixes
3. Re-run the failing command to verify the fix
4. Report whether the fix was successful

Be concise and focused. Only fix what is broken.
"""


def _format_recovery_summary(state: RecoveryState) -> str:
    """Format recovery history into a human-readable summary."""
    failed_node = state.get("failed_node", "unknown")
    role_name = state.get("role_name", "unknown")
    iteration = state.get("recovery_iteration", 0)
    actions = state.get("recovery_actions", [])
    error_message = state.get("error_message", "")

    lines = [
        f"Recovery exhausted for {failed_node} (role: {role_name})",
        f"Iterations: {iteration}",
        f"Error: {error_message[:500]}",
        "",
        "Recovery actions:",
    ]

    for action in actions[-10:]:  # Last 10 actions
        lines.append(f"  - [{action.get('iteration', '?')}] {action.get('action', '?')}: "
                      f"{action.get('result', action.get('strategy', 'N/A'))}")

    return "\n".join(lines)


__all__ = [
    "RecoveryState",
    "create_recovery_subgraph",
    "analyze_failure_node",
    "plan_fix_node",
    "execute_fix_node",
    "verify_fix_node",
    "recovery_router",
    "escalate_to_human",
]
