"""Hypothesis strategies for property-based testing."""

from hypothesis import strategies as st

from harness.db.models import (
    Credential,
    DependencyType,
    Role,
    TestRun,
    TestStatus,
    TestType,
    Worktree,
    WorktreeStatus,
)

# Base strategies
wave_strategy = st.integers(min_value=0, max_value=4)

role_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=50,
).filter(lambda x: len(x) > 0 and x[0].isalpha() and not x.startswith("_"))


# Model strategies
@st.composite
def role_strategy(draw):
    """Generate valid Role instances."""
    name = draw(role_name_strategy)
    return Role(
        name=name,
        wave=draw(wave_strategy),
        wave_name=draw(st.one_of(st.none(), st.text(min_size=1, max_size=30))),
        description=draw(st.one_of(st.none(), st.text(max_size=200))),
        has_molecule_tests=draw(st.booleans()),
    )


@st.composite
def dependency_type_strategy(draw):
    """Generate valid DependencyType."""
    return draw(st.sampled_from(list(DependencyType)))


@st.composite
def worktree_status_strategy(draw):
    """Generate valid WorktreeStatus."""
    return draw(st.sampled_from(list(WorktreeStatus)))


@st.composite
def test_type_strategy(draw):
    """Generate valid TestType."""
    return draw(st.sampled_from(list(TestType)))


@st.composite
def test_status_strategy(draw):
    """Generate valid TestStatus."""
    return draw(st.sampled_from(list(TestStatus)))


@st.composite
def credential_strategy(draw, role_id: int = 1):
    """Generate valid Credential instances."""
    return Credential(
        role_id=role_id,
        entry_name=draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll", "Nd"), whitelist_characters="_-"
                ),
                min_size=1,
                max_size=50,
            ).filter(lambda x: len(x) > 0 and x[0].isalpha())
        ),
        purpose=draw(st.one_of(st.none(), st.text(max_size=100))),
        is_base58=draw(st.booleans()),
        attribute=draw(st.one_of(st.none(), st.text(min_size=1, max_size=30))),
    )


@st.composite
def worktree_strategy(draw, role_id: int = 1):
    """Generate valid Worktree instances."""
    role_name = draw(role_name_strategy)
    return Worktree(
        role_id=role_id,
        path=f"../.worktrees/sid-{role_name}",
        branch=f"sid/{role_name}",
        commits_ahead=draw(st.integers(min_value=0, max_value=100)),
        commits_behind=draw(st.integers(min_value=0, max_value=100)),
        uncommitted_changes=draw(st.integers(min_value=0, max_value=50)),
        status=draw(worktree_status_strategy()),
    )


# Graph strategies for cycle detection testing
@st.composite
def dag_graph_strategy(draw, min_nodes: int = 2, max_nodes: int = 10):
    """
    Generate a valid DAG (no cycles).

    Creates nodes and edges where edges only go from lower to higher indices,
    which guarantees a DAG structure.
    """
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = [f"role_{i}" for i in range(n_nodes)]
    edges = []

    # Only allow edges from lower to higher indices (guarantees DAG)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if draw(st.booleans()):
                edges.append((nodes[i], nodes[j]))

    return nodes, edges


@st.composite
def cyclic_graph_strategy(draw, min_nodes: int = 3, max_nodes: int = 8):
    """
    Generate a graph with at least one cycle.

    Creates a guaranteed cycle of at least 2 nodes, plus optional additional edges.
    """
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = [f"role_{i}" for i in range(n_nodes)]

    # Create a cycle (at least length 2)
    cycle_length = draw(st.integers(min_value=2, max_value=min(n_nodes, 5)))
    edges = [(nodes[i], nodes[(i + 1) % cycle_length]) for i in range(cycle_length)]

    # Add some random non-cycle edges
    extra_edges = draw(st.integers(min_value=0, max_value=5))
    for _ in range(extra_edges):
        i = draw(st.integers(0, n_nodes - 1))
        j = draw(st.integers(0, n_nodes - 1))
        if i != j:
            edge = (nodes[i], nodes[j])
            if edge not in edges:
                edges.append(edge)

    return nodes, edges


@st.composite
def self_loop_graph_strategy(draw, min_nodes: int = 1, max_nodes: int = 5):
    """Generate a graph with at least one self-loop."""
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = [f"role_{i}" for i in range(n_nodes)]
    edges = []

    # Add a self-loop
    loop_node = draw(st.integers(0, n_nodes - 1))
    edges.append((nodes[loop_node], nodes[loop_node]))

    # Add some random edges
    extra_edges = draw(st.integers(min_value=0, max_value=3))
    for _ in range(extra_edges):
        i = draw(st.integers(0, n_nodes - 1))
        j = draw(st.integers(0, n_nodes - 1))
        edge = (nodes[i], nodes[j])
        if edge not in edges:
            edges.append(edge)

    return nodes, edges


# Wave-based strategies for testing wave assignment
@st.composite
def roles_by_wave_strategy(draw):
    """Generate roles distributed across waves for dependency testing."""
    waves = {
        0: ["common", "windows_prerequisites"],
        1: ["iis_config", "ems_registry_urls"],
        2: ["sql_server_2022", "sql_management_studio", "database_clone"],
        3: ["ems_web_app", "ems_platform_services"],
        4: ["grafana_alloy", "monitoring"],
    }

    # Select subset of roles from each wave
    selected = []
    for wave, role_names in waves.items():
        count = draw(st.integers(min_value=0, max_value=len(role_names)))
        selected_names = draw(
            st.sampled_from([list(combo) for combo in _combinations(role_names, count)])
            if count > 0
            else st.just([])
        )
        for name in selected_names:
            selected.append(Role(name=name, wave=wave, has_molecule_tests=True))

    return selected


def _combinations(items, r):
    """Generate all combinations of r items."""
    if r == 0:
        return [()]
    if not items:
        return []
    first, rest = items[0], items[1:]
    with_first = [(first,) + combo for combo in _combinations(rest, r - 1)]
    without_first = _combinations(rest, r)
    return with_first + without_first


# Test data strategies
@st.composite
def test_run_strategy(draw, role_id: int = 1):
    """Generate valid TestRun instances."""
    return TestRun(
        role_id=role_id,
        test_type=draw(test_type_strategy()),
        status=draw(test_status_strategy()),
        duration_seconds=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=3600))),
        commit_sha=draw(
            st.one_of(st.none(), st.text(alphabet="0123456789abcdef", min_size=40, max_size=40))
        ),
    )


# Execution context strategies
@st.composite
def session_id_strategy(draw):
    """Generate valid session IDs."""
    import uuid

    return str(uuid.uuid4())


@st.composite
def capability_strategy(draw):
    """Generate valid capability strings."""
    actions = ["read", "write", "execute", "admin"]
    resources = ["roles", "worktrees", "tests", "workflows", "credentials", "context"]
    action = draw(st.sampled_from(actions))
    resource = draw(st.sampled_from(resources))
    return f"{action}:{resource}"


@st.composite
def capabilities_list_strategy(draw, max_capabilities: int = 5):
    """Generate a list of unique capabilities."""
    count = draw(st.integers(min_value=0, max_value=max_capabilities))
    capabilities = set()
    for _ in range(count):
        capabilities.add(draw(capability_strategy()))
    return list(capabilities)


# =============================================================================
# WORKFLOW STATE MACHINE STRATEGIES
# =============================================================================


@st.composite
def workflow_state_transition_strategy(draw):
    """
    Generate valid state transitions for workflow state machine.

    The workflow follows this state machine:
    PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED
    RUNNING -> PAUSED -> RUNNING
    """
    from harness.db.models import WorkflowStatus

    # Define valid transitions
    valid_transitions = {
        WorkflowStatus.PENDING: [WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED],
        WorkflowStatus.RUNNING: [
            WorkflowStatus.PAUSED,
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        ],
        WorkflowStatus.PAUSED: [WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED],
        WorkflowStatus.COMPLETED: [],  # Terminal state
        WorkflowStatus.FAILED: [],  # Terminal state
        WorkflowStatus.CANCELLED: [],  # Terminal state
    }

    # Start from PENDING
    current_state = WorkflowStatus.PENDING
    transitions = []
    max_transitions = draw(st.integers(min_value=1, max_value=10))

    for _ in range(max_transitions):
        next_states = valid_transitions[current_state]
        if not next_states:
            break  # Terminal state reached

        next_state = draw(st.sampled_from(next_states))
        transitions.append((current_state, next_state))
        current_state = next_state

    return transitions


@st.composite
def invalid_workflow_state_transition_strategy(draw):
    """Generate an invalid state transition for testing rejection."""
    from harness.db.models import WorkflowStatus

    # Define invalid transitions (e.g., going back from terminal states)
    invalid_transitions = [
        (WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING),
        (WorkflowStatus.COMPLETED, WorkflowStatus.PENDING),
        (WorkflowStatus.FAILED, WorkflowStatus.RUNNING),
        (WorkflowStatus.FAILED, WorkflowStatus.PENDING),
        (WorkflowStatus.CANCELLED, WorkflowStatus.RUNNING),
        (WorkflowStatus.PENDING, WorkflowStatus.COMPLETED),  # Skip RUNNING
        (WorkflowStatus.PENDING, WorkflowStatus.FAILED),  # Skip RUNNING
        (WorkflowStatus.PAUSED, WorkflowStatus.COMPLETED),  # Skip RUNNING
        (WorkflowStatus.PAUSED, WorkflowStatus.PENDING),  # Can't go back
    ]

    return draw(st.sampled_from(invalid_transitions))


@st.composite
def state_transition_sequence_strategy(draw, max_length: int = 15):
    """
    Generate a sequence of workflow state transitions.

    Returns a list of (from_state, to_state, is_valid) tuples.
    """
    from harness.db.models import WorkflowStatus

    # Valid transitions map
    valid_transitions = {
        WorkflowStatus.PENDING: {WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED},
        WorkflowStatus.RUNNING: {
            WorkflowStatus.PAUSED,
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        },
        WorkflowStatus.PAUSED: {WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED},
        WorkflowStatus.COMPLETED: set(),
        WorkflowStatus.FAILED: set(),
        WorkflowStatus.CANCELLED: set(),
    }

    all_states = list(WorkflowStatus)
    sequence = []
    current_state = WorkflowStatus.PENDING
    length = draw(st.integers(min_value=1, max_value=max_length))

    for _ in range(length):
        # Randomly choose to make a valid or invalid transition
        make_valid = draw(st.booleans())
        valid_next = valid_transitions[current_state]

        if make_valid and valid_next:
            # Make a valid transition
            next_state = draw(st.sampled_from(list(valid_next)))
            sequence.append((current_state, next_state, True))
            current_state = next_state
        elif not make_valid or not valid_next:
            # Make an invalid transition
            invalid_states = set(all_states) - valid_next - {current_state}
            if invalid_states:
                next_state = draw(st.sampled_from(list(invalid_states)))
                sequence.append((current_state, next_state, False))
                # Don't update current_state for invalid transitions
            else:
                # No invalid transition possible, skip
                pass

    return sequence


# =============================================================================
# CONCURRENT DB OPERATION STRATEGIES
# =============================================================================


@st.composite
def concurrent_db_operation_strategy(draw, max_ops: int = 10):
    """
    Generate a sequence of concurrent database operations.

    Returns a list of operations with their parameters.
    """
    operations = []
    num_ops = draw(st.integers(min_value=1, max_value=max_ops))

    op_types = [
        "upsert_role",
        "add_dependency",
        "add_credential",
        "upsert_worktree",
        "update_execution_status",
        "checkpoint_execution",
    ]

    for _ in range(num_ops):
        op_type = draw(st.sampled_from(op_types))

        if op_type == "upsert_role":
            operations.append(
                {
                    "type": op_type,
                    "role_name": draw(role_name_strategy),
                    "wave": draw(wave_strategy),
                }
            )
        elif op_type == "add_dependency":
            operations.append(
                {
                    "type": op_type,
                    "from_role": draw(role_name_strategy),
                    "to_role": draw(role_name_strategy),
                }
            )
        elif op_type == "add_credential":
            operations.append(
                {
                    "type": op_type,
                    "role_name": draw(role_name_strategy),
                    "entry_name": draw(
                        st.text(
                            alphabet=st.characters(
                                whitelist_categories=("Ll", "Nd"), whitelist_characters="_-"
                            ),
                            min_size=1,
                            max_size=30,
                        ).filter(lambda x: len(x) > 0 and x[0].isalpha())
                    ),
                }
            )
        elif op_type == "upsert_worktree":
            operations.append(
                {
                    "type": op_type,
                    "role_name": draw(role_name_strategy),
                    "branch": f"sid/{draw(role_name_strategy)}",
                }
            )
        elif op_type == "update_execution_status":
            from harness.db.models import WorkflowStatus

            operations.append(
                {
                    "type": op_type,
                    "execution_id": draw(st.integers(min_value=1, max_value=100)),
                    "status": draw(st.sampled_from(list(WorkflowStatus))),
                }
            )
        elif op_type == "checkpoint_execution":
            operations.append(
                {
                    "type": op_type,
                    "execution_id": draw(st.integers(min_value=1, max_value=100)),
                    "data": draw(checkpoint_metadata_strategy()),
                }
            )

    return operations


# =============================================================================
# ROLE DEPENDENCY CHAIN STRATEGIES
# =============================================================================


@st.composite
def role_dependency_chain_strategy(draw, min_length: int = 2, max_length: int = 8):
    """
    Generate a valid dependency chain: A -> B -> C -> ...

    Each role depends on the next in the chain (A depends on B, B on C, etc.).
    This creates a valid DAG with no cycles.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))

    # Generate unique role names
    roles = []
    for i in range(length):
        # Use numbered names to ensure uniqueness
        roles.append(f"chain_role_{i}")

    # Each role depends on the next one (index i depends on index i+1)
    dependencies = []
    for i in range(length - 1):
        dependencies.append((roles[i], roles[i + 1]))

    return roles, dependencies


@st.composite
def role_diamond_dependency_strategy(draw):
    """
    Generate a diamond dependency pattern: A -> B, A -> C, B -> D, C -> D

    This is a valid DAG pattern that tests multiple paths to the same node.
    """
    # Generate 4 unique roles
    top = draw(role_name_strategy)
    left = draw(role_name_strategy.filter(lambda x: x != top))
    right = draw(role_name_strategy.filter(lambda x: x not in [top, left]))
    bottom = draw(role_name_strategy.filter(lambda x: x not in [top, left, right]))

    roles = [top, left, right, bottom]

    # Diamond pattern: top -> left -> bottom, top -> right -> bottom
    dependencies = [
        (top, left),  # top depends on left
        (top, right),  # top depends on right
        (left, bottom),  # left depends on bottom
        (right, bottom),  # right depends on bottom
    ]

    return roles, dependencies


@st.composite
def deployment_order_strategy(draw):
    """
    Generate roles with dependencies and their expected deployment order.

    Returns (roles, dependencies, expected_valid_orders) where any valid
    topological order should have all dependencies come before dependents.
    """
    # Use the chain strategy as a base
    roles, deps = draw(role_dependency_chain_strategy())

    # For a chain A -> B -> C, the only valid order is [C, B, A]
    # (last depends on nothing, first depends on all before it)
    # In our chain, role i depends on role i+1, so role i+1 must come first
    expected_order = list(reversed(roles))

    return roles, deps, expected_order


# =============================================================================
# CHECKPOINT METADATA STRATEGIES
# =============================================================================


@st.composite
def checkpoint_metadata_strategy(draw):
    """
    Generate checkpoint metadata dicts for testing save/load cycles.
    """
    return {
        "version": draw(st.integers(min_value=1, max_value=5)),
        "timestamp": draw(
            st.datetimes(
                min_value=datetime(2024, 1, 1),
                max_value=datetime(2026, 12, 31),
            )
        ).isoformat(),
        "step": draw(st.integers(min_value=0, max_value=100)),
        "source": draw(st.sampled_from(["input", "loop", "condition", "parallel"])),
        "writes": draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        "state_summary": {
            "role_name": draw(role_name_strategy),
            "current_node": draw(
                st.sampled_from(
                    ["start", "validate", "analyze", "test", "commit", "mr", "end"]
                )
            ),
            "completed_nodes": draw(
                st.lists(
                    st.sampled_from(
                        ["start", "validate", "analyze", "test", "commit", "mr"]
                    ),
                    max_size=6,
                    unique=True,
                )
            ),
            "errors": draw(st.lists(st.text(min_size=1, max_size=50), max_size=3)),
            "molecule_passed": draw(st.one_of(st.none(), st.booleans())),
            "pytest_passed": draw(st.one_of(st.none(), st.booleans())),
        },
    }


@st.composite
def checkpoint_with_config_strategy(draw):
    """
    Generate a checkpoint with its associated config for roundtrip testing.
    """
    execution_id = draw(st.integers(min_value=1, max_value=10000))

    config = {
        "configurable": {
            "thread_id": str(execution_id),
            "checkpoint_ns": "",
        }
    }

    metadata = draw(checkpoint_metadata_strategy())

    # Checkpoint structure matching LangGraph format
    checkpoint = {
        "v": 1,
        "id": f"checkpoint-{draw(st.text(alphabet='0123456789abcdef', min_size=8, max_size=8))}",
        "ts": metadata["timestamp"],
        "channel_values": metadata["state_summary"],
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }

    return config, checkpoint, metadata


# Import datetime for checkpoint_metadata_strategy
from datetime import datetime
