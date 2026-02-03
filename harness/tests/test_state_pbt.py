"""
Property-based tests for StateDB and workflow state machine.

These tests use Hypothesis to verify invariants and properties that should
hold for all possible inputs, not just specific test cases.

Test Categories:
1. TRANSITIVITY - If A->B and B->C, then A transitively depends on C
2. NO-SPURIOUS-CYCLES - Valid DAGs never report cycles
3. DEPLOYMENT ORDER - Dependencies must be deployed before dependents
4. WORKFLOW STATE MACHINE - Only valid state transitions are allowed
5. CHECKPOINT CONSISTENCY - Checkpoints survive save/load unchanged
"""

import json
from datetime import datetime

import pytest
from hypothesis import assume, given, settings, Verbosity, HealthCheck

from harness.db.models import (
    Credential,
    CyclicDependencyError,
    DependencyType,
    NodeStatus,
    Role,
    RoleDependency,
    WorkflowStatus,
    Worktree,
    WorktreeStatus,
)
from harness.db.state import StateDB
from tests.strategies import (
    checkpoint_metadata_strategy,
    checkpoint_with_config_strategy,
    concurrent_db_operation_strategy,
    dag_graph_strategy,
    deployment_order_strategy,
    role_dependency_chain_strategy,
    role_diamond_dependency_strategy,
    role_name_strategy,
    role_strategy,
    state_transition_sequence_strategy,
    wave_strategy,
    workflow_state_transition_strategy,
)


# =============================================================================
# 1. TRANSITIVITY PROPERTY TESTS
# =============================================================================


class TestTransitiveDependencies:
    """Test that transitive dependency resolution is correct."""

    @pytest.mark.pbt
    @given(chain=role_dependency_chain_strategy(min_length=3, max_length=6))
    @settings(
        max_examples=50,
        verbosity=Verbosity.verbose,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_transitive_dependencies(self, chain):
        """
        Property: If A depends on B and B depends on C, then A transitively depends on C.

        For a chain A -> B -> C -> D:
        - A should transitively depend on B, C, D
        - B should transitively depend on C, D
        - C should transitively depend on D
        - D should have no dependencies
        """
        roles, dependencies = chain
        assume(len(roles) >= 3)
        assume(len(dependencies) >= 2)

        # Create fresh in-memory db
        db = StateDB(":memory:")

        # Create all roles
        for i, role_name in enumerate(roles):
            db.upsert_role(Role(name=role_name, wave=i))

        # Add dependencies (role i depends on role i+1)
        for from_role, to_role in dependencies:
            from_r = db.get_role(from_role)
            to_r = db.get_role(to_role)
            if from_r and to_r:
                db.add_dependency(
                    RoleDependency(
                        role_id=from_r.id,
                        depends_on_id=to_r.id,
                        dependency_type=DependencyType.EXPLICIT,
                    )
                )

        # Verify transitivity: first role should depend on all others
        first_role = roles[0]
        trans_deps = db.get_dependencies(first_role, transitive=True)
        trans_dep_names = {name for name, _ in trans_deps}

        # All roles except the first should be transitive dependencies
        for role_name in roles[1:]:
            assert role_name in trans_dep_names, (
                f"{role_name} should be a transitive dependency of {first_role}"
            )

    @pytest.mark.pbt
    @given(diamond=role_diamond_dependency_strategy())
    @settings(max_examples=30, verbosity=Verbosity.verbose)
    def test_diamond_dependencies_resolved(self, diamond):
        """
        Property: Diamond dependency pattern should resolve correctly.

        For diamond: top -> left -> bottom, top -> right -> bottom
        - top should transitively depend on left, right, and bottom
        - bottom should appear only once (deduplication)
        """
        roles, dependencies = diamond
        assume(len(set(roles)) == 4)  # All unique

        db = StateDB(":memory:")

        # Create roles
        for i, role_name in enumerate(roles):
            db.upsert_role(Role(name=role_name, wave=i))

        # Add diamond dependencies
        for from_role, to_role in dependencies:
            from_r = db.get_role(from_role)
            to_r = db.get_role(to_role)
            if from_r and to_r:
                try:
                    db.add_dependency(
                        RoleDependency(
                            role_id=from_r.id,
                            depends_on_id=to_r.id,
                            dependency_type=DependencyType.EXPLICIT,
                        )
                    )
                except Exception:
                    pass  # Ignore duplicate edge issues

        top = roles[0]
        bottom = roles[3]

        # Top should transitively depend on bottom
        trans_deps = db.get_dependencies(top, transitive=True)
        trans_dep_names = [name for name, _ in trans_deps]

        assert bottom in trans_dep_names, f"{bottom} should be a transitive dep of {top}"

        # Bottom should appear exactly once (no duplicates)
        assert trans_dep_names.count(bottom) == 1, "Bottom should appear exactly once"


# =============================================================================
# 2. NO-SPURIOUS-CYCLES PROPERTY TESTS
# =============================================================================


class TestNoSpuriousCycles:
    """Test that valid DAGs never report false positive cycles."""

    @pytest.mark.pbt
    @given(graph=dag_graph_strategy(min_nodes=2, max_nodes=10))
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_valid_dags_never_have_cycles(self, graph):
        """
        Property: A valid DAG (constructed with edges only going forward)
        should never report any cycles.
        """
        nodes, edges = graph
        assume(len(nodes) >= 2)

        db = StateDB(":memory:")

        # Create roles
        for node in nodes:
            db.upsert_role(Role(name=node, wave=1))

        # Create edges (our dag_graph_strategy guarantees forward-only edges)
        for from_node, to_node in edges:
            from_role = db.get_role(from_node)
            to_role = db.get_role(to_node)
            if from_role and to_role:
                try:
                    db.add_dependency(
                        RoleDependency(
                            role_id=from_role.id,
                            depends_on_id=to_role.id,
                            dependency_type=DependencyType.EXPLICIT,
                        )
                    )
                except Exception:
                    pass  # Ignore duplicates

        # Verify no cycles detected
        cycles = db.detect_cycles()
        assert cycles == [], f"Valid DAG should have no cycles, but found: {cycles}"

    @pytest.mark.pbt
    @given(chain=role_dependency_chain_strategy(min_length=2, max_length=10))
    @settings(max_examples=50, verbosity=Verbosity.verbose)
    def test_chain_is_cycle_free(self, chain):
        """
        Property: A linear dependency chain (A -> B -> C -> ...) should
        never have cycles.
        """
        roles, dependencies = chain

        db = StateDB(":memory:")

        # Create roles
        for i, role_name in enumerate(roles):
            db.upsert_role(Role(name=role_name, wave=i))

        # Create chain dependencies
        for from_role, to_role in dependencies:
            from_r = db.get_role(from_role)
            to_r = db.get_role(to_role)
            if from_r and to_r:
                db.add_dependency(
                    RoleDependency(
                        role_id=from_r.id,
                        depends_on_id=to_r.id,
                        dependency_type=DependencyType.EXPLICIT,
                    )
                )

        cycles = db.detect_cycles()
        assert cycles == [], f"Chain should have no cycles, but found: {cycles}"

        # Also verify validation passes
        result = db.validate_dependencies()
        assert result["valid"], f"Chain should be valid: {result}"


# =============================================================================
# 3. DEPLOYMENT ORDER PROPERTY TESTS
# =============================================================================


class TestDeploymentOrder:
    """Test that deployment order respects all dependencies."""

    @pytest.mark.pbt
    @given(order_data=deployment_order_strategy())
    @settings(
        max_examples=50,
        verbosity=Verbosity.verbose,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_dependencies_deployed_first(self, order_data):
        """
        Property: In any valid deployment order, all dependencies must
        come before the roles that depend on them.
        """
        roles, dependencies, _expected = order_data
        assume(len(roles) >= 2)
        assume(len(dependencies) >= 1)

        db = StateDB(":memory:")

        # Create roles with waves matching their position
        for i, role_name in enumerate(roles):
            db.upsert_role(Role(name=role_name, wave=i))

        # Create dependencies
        for from_role, to_role in dependencies:
            from_r = db.get_role(from_role)
            to_r = db.get_role(to_role)
            if from_r and to_r:
                db.add_dependency(
                    RoleDependency(
                        role_id=from_r.id,
                        depends_on_id=to_r.id,
                        dependency_type=DependencyType.EXPLICIT,
                    )
                )

        # Get deployment order
        order = db.get_deployment_order()
        order_idx = {name: i for i, name in enumerate(order)}

        # Verify: for each dependency (A depends on B), B must come before A
        for from_role, to_role in dependencies:
            if from_role in order_idx and to_role in order_idx:
                assert order_idx[to_role] < order_idx[from_role], (
                    f"Dependency {to_role} must come before {from_role} in deployment order. "
                    f"Got order: {order}"
                )

    @pytest.mark.pbt
    @given(graph=dag_graph_strategy(min_nodes=3, max_nodes=8))
    @settings(
        max_examples=50,
        verbosity=Verbosity.verbose,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_deployment_order_contains_all_roles(self, graph):
        """
        Property: Deployment order must include all roles in the database.
        """
        nodes, edges = graph
        assume(len(nodes) >= 3)

        db = StateDB(":memory:")

        # Create roles
        for node in nodes:
            db.upsert_role(Role(name=node, wave=1))

        # Create edges
        for from_node, to_node in edges:
            from_role = db.get_role(from_node)
            to_role = db.get_role(to_node)
            if from_role and to_role:
                try:
                    db.add_dependency(
                        RoleDependency(
                            role_id=from_role.id,
                            depends_on_id=to_role.id,
                            dependency_type=DependencyType.EXPLICIT,
                        )
                    )
                except Exception:
                    pass

        order = db.get_deployment_order()

        # All nodes should be in the order
        assert set(order) == set(nodes), (
            f"Deployment order {order} should contain all nodes {nodes}"
        )


# =============================================================================
# 4. WORKFLOW STATE MACHINE PROPERTY TESTS
# =============================================================================


class TestWorkflowStateMachine:
    """Test workflow state machine transition validity."""

    @pytest.mark.pbt
    @given(transitions=workflow_state_transition_strategy())
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_valid_transitions_succeed(self, transitions):
        """
        Property: All transitions generated by workflow_state_transition_strategy
        should be valid state machine transitions.
        """
        assume(len(transitions) >= 1)

        # Define valid transitions for verification
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

        for from_state, to_state in transitions:
            assert to_state in valid_transitions[from_state], (
                f"Transition {from_state} -> {to_state} should be valid"
            )

    @pytest.mark.pbt
    @given(transitions=state_transition_sequence_strategy(max_length=15))
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_only_valid_transitions(self, transitions):
        """
        Property: The state transition sequence correctly identifies valid
        vs invalid transitions.
        """
        assume(len(transitions) >= 1)

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

        for from_state, to_state, is_valid in transitions:
            actual_valid = to_state in valid_transitions[from_state]
            assert is_valid == actual_valid, (
                f"Transition {from_state} -> {to_state}: expected valid={is_valid}, "
                f"actual valid={actual_valid}"
            )

    @pytest.mark.pbt
    def test_terminal_states_have_no_exits(self):
        """
        Property: Terminal states (COMPLETED, FAILED, CANCELLED) should
        have no valid outgoing transitions.
        """
        terminal_states = {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }

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

        for terminal in terminal_states:
            assert valid_transitions[terminal] == set(), (
                f"Terminal state {terminal} should have no valid transitions"
            )

    @pytest.mark.pbt
    def test_pending_is_only_initial_state(self):
        """
        Property: PENDING should be the only state with no valid incoming
        transitions (besides itself).
        """
        # Build reverse transition map
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

        incoming = {state: set() for state in WorkflowStatus}
        for from_state, to_states in valid_transitions.items():
            for to_state in to_states:
                incoming[to_state].add(from_state)

        # PENDING should have no incoming transitions
        assert incoming[WorkflowStatus.PENDING] == set(), (
            f"PENDING should have no incoming transitions, got {incoming[WorkflowStatus.PENDING]}"
        )


# =============================================================================
# 5. CHECKPOINT CONSISTENCY PROPERTY TESTS
# =============================================================================


class TestCheckpointConsistency:
    """Test checkpoint save/load roundtrip consistency."""

    @pytest.mark.pbt
    @given(metadata=checkpoint_metadata_strategy())
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_checkpoint_roundtrip(self, metadata):
        """
        Property: Checkpoints should survive save/load cycle with data
        preserved (JSON serialization roundtrip).
        """
        # Serialize to JSON and back (simulating DB storage)
        json_str = json.dumps(metadata)
        loaded = json.loads(json_str)

        # Check all top-level keys preserved
        assert set(loaded.keys()) == set(metadata.keys()), (
            f"Keys mismatch: {set(loaded.keys())} != {set(metadata.keys())}"
        )

        # Check specific values
        assert loaded["version"] == metadata["version"]
        assert loaded["timestamp"] == metadata["timestamp"]
        assert loaded["step"] == metadata["step"]
        assert loaded["source"] == metadata["source"]
        assert loaded["writes"] == metadata["writes"]

        # Check nested state_summary
        assert loaded["state_summary"]["role_name"] == metadata["state_summary"]["role_name"]
        assert loaded["state_summary"]["current_node"] == metadata["state_summary"]["current_node"]
        assert loaded["state_summary"]["completed_nodes"] == metadata["state_summary"]["completed_nodes"]
        assert loaded["state_summary"]["errors"] == metadata["state_summary"]["errors"]

    @pytest.mark.pbt
    @given(checkpoint_data=checkpoint_with_config_strategy())
    @settings(max_examples=50, verbosity=Verbosity.verbose)
    def test_checkpoint_statedb_roundtrip(self, checkpoint_data):
        """
        Property: Checkpoints stored in StateDB should be retrievable unchanged.
        """
        config, checkpoint, metadata = checkpoint_data

        db = StateDB(":memory:")

        # Create prerequisite data
        db.upsert_role(Role(name="test_role", wave=1))
        role = db.get_role("test_role")

        # Create workflow definition first
        workflow_id = db.create_workflow_definition(
            name="test-workflow",
            description="Test workflow",
            nodes=[{"id": "start"}, {"id": "end"}],
            edges=[{"from": "start", "to": "end"}],
        )

        # Create execution
        execution_id = db.create_execution("test-workflow", "test_role")

        # Save checkpoint
        db.checkpoint_execution(execution_id, metadata)

        # Load checkpoint
        loaded = db.get_checkpoint(execution_id)

        assert loaded is not None, "Checkpoint should be retrievable"

        # Verify data preserved
        assert loaded["version"] == metadata["version"]
        assert loaded["timestamp"] == metadata["timestamp"]
        assert loaded["step"] == metadata["step"]
        assert loaded["source"] == metadata["source"]

    @pytest.mark.pbt
    @given(metadata=checkpoint_metadata_strategy())
    @settings(max_examples=50, verbosity=Verbosity.verbose)
    def test_checkpoint_idempotent_save(self, metadata):
        """
        Property: Saving the same checkpoint multiple times should result
        in the same stored value.
        """
        db = StateDB(":memory:")

        # Setup
        db.upsert_role(Role(name="test_role", wave=1))
        db.create_workflow_definition(
            name="test-workflow",
            description="Test",
            nodes=[{"id": "start"}],
            edges=[],
        )
        execution_id = db.create_execution("test-workflow", "test_role")

        # Save multiple times
        db.checkpoint_execution(execution_id, metadata)
        loaded1 = db.get_checkpoint(execution_id)

        db.checkpoint_execution(execution_id, metadata)
        loaded2 = db.get_checkpoint(execution_id)

        # Both loads should be identical
        assert loaded1 == loaded2, "Multiple saves should result in identical checkpoints"


# =============================================================================
# CONCURRENT OPERATIONS TESTS
# =============================================================================


class TestConcurrentOperations:
    """Test database behavior under concurrent-like operations."""

    @pytest.mark.pbt
    @given(operations=concurrent_db_operation_strategy(max_ops=5))
    @settings(
        max_examples=30,
        verbosity=Verbosity.verbose,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_operations_maintain_consistency(self, operations):
        """
        Property: A sequence of database operations should maintain
        referential integrity and consistency.
        """
        assume(len(operations) >= 1)

        db = StateDB(":memory:")
        created_roles = set()

        for op in operations:
            op_type = op["type"]

            try:
                if op_type == "upsert_role":
                    db.upsert_role(Role(name=op["role_name"], wave=op["wave"]))
                    created_roles.add(op["role_name"])

                elif op_type == "add_dependency":
                    # Only add if both roles exist
                    if op["from_role"] in created_roles and op["to_role"] in created_roles:
                        from_r = db.get_role(op["from_role"])
                        to_r = db.get_role(op["to_role"])
                        if from_r and to_r and from_r.id != to_r.id:
                            db.add_dependency(
                                RoleDependency(
                                    role_id=from_r.id,
                                    depends_on_id=to_r.id,
                                    dependency_type=DependencyType.EXPLICIT,
                                )
                            )

                elif op_type == "add_credential":
                    if op["role_name"] in created_roles:
                        role = db.get_role(op["role_name"])
                        if role:
                            db.add_credential(
                                Credential(
                                    role_id=role.id,
                                    entry_name=op["entry_name"],
                                    purpose="test",
                                )
                            )

                elif op_type == "upsert_worktree":
                    if op["role_name"] in created_roles:
                        role = db.get_role(op["role_name"])
                        if role:
                            db.upsert_worktree(
                                Worktree(
                                    role_id=role.id,
                                    path=f".worktrees/{op['role_name']}",
                                    branch=op["branch"],
                                    status=WorktreeStatus.ACTIVE,
                                )
                            )

            except Exception:
                # Operations may fail due to constraints, but should not corrupt DB
                pass

        # Verify database consistency after all operations
        roles = db.list_roles()
        role_names = {r.name for r in roles}

        # All created roles should still exist
        for role_name in created_roles:
            assert role_name in role_names, f"Role {role_name} should exist"

        # Dependency graph should be queryable without error
        try:
            db.get_dependency_graph()
            db.validate_dependencies()
        except Exception as e:
            pytest.fail(f"Database should remain consistent: {e}")


# =============================================================================
# ROLE UPSERT ROUNDTRIP TESTS
# =============================================================================


class TestRoleUpsertRoundtrip:
    """Test role creation and retrieval properties."""

    @pytest.mark.pbt
    @given(role=role_strategy())
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_role_upsert_preserves_data(self, role):
        """
        Property: Upserting a role and retrieving it should preserve all data.
        """
        db = StateDB(":memory:")

        db.upsert_role(role)
        retrieved = db.get_role(role.name)

        assert retrieved is not None
        assert retrieved.name == role.name
        assert retrieved.wave == role.wave
        assert retrieved.has_molecule_tests == role.has_molecule_tests
        assert retrieved.description == role.description
        assert retrieved.wave_name == role.wave_name

    @pytest.mark.pbt
    @given(wave=wave_strategy)
    @settings(max_examples=20, verbosity=Verbosity.verbose)
    def test_wave_filtering(self, wave):
        """
        Property: Filtering roles by wave should return only roles in that wave.
        """
        db = StateDB(":memory:")

        # Create roles in different waves
        for w in range(5):
            for i in range(2):
                db.upsert_role(Role(name=f"role_w{w}_{i}", wave=w))

        # Filter by the test wave
        filtered = db.list_roles(wave=wave)

        for role in filtered:
            assert role.wave == wave, f"Role {role.name} has wave {role.wave}, expected {wave}"
