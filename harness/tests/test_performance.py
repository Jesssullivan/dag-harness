"""
Performance benchmark tests for dag-harness.

Uses time.perf_counter() for high-resolution timing.
All assertions use generous margins to accommodate CI environments
where performance may vary due to shared resources.
"""

import json
import time
from pathlib import Path

import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    create_box_up_role_graph,
    create_initial_state,
)
from harness.dag.store import HarnessStore
from harness.db.models import DependencyType, Role, RoleDependency
from harness.db.state import StateDB


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def db():
    """Create an in-memory StateDB."""
    return StateDB(":memory:")


@pytest.fixture
def db_with_roles(db):
    """Database pre-populated with test roles and dependencies."""
    roles = [
        Role(name="common", wave=0, has_molecule_tests=True),
        Role(name="windows_prerequisites", wave=0, has_molecule_tests=True),
        Role(name="ems_registry_urls", wave=1, has_molecule_tests=False),
        Role(name="iis_config", wave=1, has_molecule_tests=True),
        Role(name="sql_server_2022", wave=2, has_molecule_tests=True),
        Role(name="sql_management_studio", wave=2, has_molecule_tests=False),
        Role(name="ems_web_app", wave=2, has_molecule_tests=True),
        Role(name="ems_platform_services", wave=3, has_molecule_tests=True),
        Role(name="ems_desktop_deploy", wave=3, has_molecule_tests=False),
        Role(name="grafana_alloy_windows", wave=4, has_molecule_tests=True),
    ]
    for role in roles:
        db.upsert_role(role)

    # Build dependency graph
    common = db.get_role("common")
    win_prereq = db.get_role("windows_prerequisites")
    ems_reg = db.get_role("ems_registry_urls")
    iis = db.get_role("iis_config")
    sql = db.get_role("sql_server_2022")
    sql_mgmt = db.get_role("sql_management_studio")
    ems_web = db.get_role("ems_web_app")
    ems_platform = db.get_role("ems_platform_services")

    deps = [
        (win_prereq.id, common.id),
        (ems_reg.id, common.id),
        (iis.id, common.id),
        (iis.id, win_prereq.id),
        (sql.id, common.id),
        (sql_mgmt.id, sql.id),
        (sql_mgmt.id, common.id),
        (ems_web.id, common.id),
        (ems_web.id, iis.id),
        (ems_platform.id, ems_web.id),
        (ems_platform.id, common.id),
    ]

    for role_id, depends_on_id in deps:
        db.add_dependency(
            RoleDependency(
                role_id=role_id,
                depends_on_id=depends_on_id,
                dependency_type=DependencyType.EXPLICIT,
            )
        )

    return db


@pytest.fixture
def store(db):
    """Create a HarnessStore with in-memory database."""
    return HarnessStore(db)


@pytest.fixture
def store_with_data(store):
    """HarnessStore pre-populated with data for search benchmarks."""
    for i in range(100):
        store.put(
            ("roles", f"role_{i}"),
            "status",
            {"passed": i % 3 != 0, "wave": i % 5, "duration": i * 1.5},
        )
    return store


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================


@pytest.mark.unit
class TestPerformanceBenchmarks:
    """Performance benchmarks with generous CI-safe margins."""

    def test_statedb_query_under_10ms(self, db_with_roles):
        """Database queries should complete under 10ms on average."""
        # Warm up
        db_with_roles.get_role("common")

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            db_with_roles.get_role("common")
        elapsed = (time.perf_counter() - start) / iterations

        assert elapsed < 0.010, f"Average query time {elapsed*1000:.2f}ms exceeds 10ms"

    def test_checkpoint_restore_under_100ms(self, db_with_roles):
        """Checkpoint save and restore should complete under 100ms."""
        db = db_with_roles

        # Create workflow definition and execution
        db.create_workflow_definition(
            name="box-up-role",
            description="Test workflow",
            nodes=[{"id": "start"}, {"id": "test"}, {"id": "end"}],
            edges=[{"from": "start", "to": "test"}, {"from": "test", "to": "end"}],
        )
        exec_id = db.create_execution("box-up-role", "common")

        checkpoint_data = {
            "role_name": "common",
            "current_node": "run_molecule",
            "completed_nodes": ["validate_role", "analyze_deps", "create_worktree"],
            "errors": [],
            "molecule_passed": True,
            "wave": 0,
            "explicit_deps": ["common"],
            "timing_data": {str(i): i * 0.1 for i in range(50)},
        }

        # Benchmark checkpoint save
        start = time.perf_counter()
        db.checkpoint_execution(exec_id, checkpoint_data)
        save_elapsed = time.perf_counter() - start

        # Benchmark checkpoint restore
        start = time.perf_counter()
        restored = db.get_checkpoint(exec_id)
        restore_elapsed = time.perf_counter() - start

        total_elapsed = save_elapsed + restore_elapsed

        assert restored is not None
        assert restored["role_name"] == "common"
        assert total_elapsed < 0.100, (
            f"Checkpoint save+restore {total_elapsed*1000:.2f}ms exceeds 100ms"
        )

    def test_store_put_get_under_5ms(self, store):
        """HarnessStore put/get should complete under 5ms."""
        value = {"status": "passed", "duration": 42.5, "tags": ["unit", "fast"]}

        # Warm up
        store.put(("warmup",), "key", {"warmup": True})
        store.get(("warmup",), "key")

        iterations = 100
        start = time.perf_counter()
        for i in range(iterations):
            store.put(("roles", "common"), f"run_{i}", value)
            store.get(("roles", "common"), f"run_{i}")
        elapsed = (time.perf_counter() - start) / iterations

        assert elapsed < 0.005, (
            f"Average put+get time {elapsed*1000:.2f}ms exceeds 5ms"
        )

    def test_state_serialization_under_1ms(self):
        """BoxUpRoleState serialization should complete under 1ms."""
        state = create_initial_state("common")
        state["explicit_deps"] = ["role_a", "role_b", "role_c"]
        state["completed_nodes"] = ["validate_role", "analyze_deps"]
        state["errors"] = []

        # Warm up
        json.dumps(dict(state))
        json.loads(json.dumps(dict(state)))

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            serialized = json.dumps(dict(state))
            json.loads(serialized)
        elapsed = (time.perf_counter() - start) / iterations

        assert elapsed < 0.001, (
            f"Average serialization time {elapsed*1000:.3f}ms exceeds 1ms"
        )

    def test_namespace_search_under_50ms(self, store):
        """Store namespace search should complete under 50ms even with 1000 entries."""
        # Populate store with 1000 entries
        for i in range(1000):
            store.put(
                ("roles", f"role_{i}"),
                "status",
                {"passed": i % 2 == 0, "wave": i % 5},
            )

        # Warm up
        store.search(("roles",))

        start = time.perf_counter()
        results = store.search(("roles",))
        elapsed = time.perf_counter() - start

        assert len(results) > 0
        assert elapsed < 0.050, (
            f"Search time {elapsed*1000:.2f}ms exceeds 50ms for 1000 entries"
        )

    def test_batch_operations_faster_than_individual(self, store):
        """Batch operations should be faster than individual calls."""
        from langgraph.store.base import GetOp, PutOp

        # Populate data
        for i in range(50):
            store.put(("roles", f"role_{i}"), "status", {"wave": i % 5})

        # Individual gets
        start = time.perf_counter()
        for i in range(50):
            store.get(("roles", f"role_{i}"), "status")
        individual_elapsed = time.perf_counter() - start

        # Batch gets
        ops = [GetOp(namespace=("roles", f"role_{i}"), key="status") for i in range(50)]
        start = time.perf_counter()
        store.batch(ops)
        batch_elapsed = time.perf_counter() - start

        # Batch should be faster (or at least not significantly slower)
        # Use a generous margin: batch should be at most 2x slower (accounting for overhead)
        assert batch_elapsed < individual_elapsed * 2.0, (
            f"Batch ({batch_elapsed*1000:.2f}ms) should not be much slower than "
            f"individual ({individual_elapsed*1000:.2f}ms)"
        )

    def test_graph_compilation_under_500ms(self):
        """Graph compilation should complete under 500ms."""
        # Warm up
        create_box_up_role_graph(
            db_path=":memory:",
            parallel_tests=True,
            enable_breakpoints=False,
        )

        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            graph, breakpoints = create_box_up_role_graph(
                db_path=":memory:",
                parallel_tests=True,
                enable_breakpoints=False,
            )
        elapsed = (time.perf_counter() - start) / iterations

        assert graph is not None
        assert elapsed < 0.500, (
            f"Graph compilation {elapsed*1000:.2f}ms exceeds 500ms"
        )

    def test_dependency_resolution_under_100ms(self, db_with_roles):
        """Full dependency resolution for all roles under 100ms."""
        db = db_with_roles
        roles = db.list_roles()

        # Warm up
        for role in roles:
            db.get_dependencies(role.name, transitive=True)

        start = time.perf_counter()
        for role in roles:
            deps = db.get_dependencies(role.name, transitive=True)
            reverse_deps = db.get_reverse_dependencies(role.name, transitive=False)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.100, (
            f"Full dependency resolution {elapsed*1000:.2f}ms exceeds 100ms "
            f"for {len(roles)} roles"
        )

    def test_role_listing_scales_linearly(self, db):
        """Role listing should scale approximately linearly with count."""
        # Insert 100 roles
        for i in range(100):
            db.upsert_role(Role(name=f"role_{i:03d}", wave=i % 5))

        # Warm up
        db.list_roles()

        # Time listing all roles
        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            roles = db.list_roles()
        elapsed_all = (time.perf_counter() - start) / iterations

        assert len(roles) == 100

        # Time listing filtered roles (should be faster)
        start = time.perf_counter()
        for _ in range(iterations):
            wave_roles = db.list_roles(wave=0)
        elapsed_filtered = (time.perf_counter() - start) / iterations

        # Filtered should be faster or comparable
        assert elapsed_filtered <= elapsed_all * 1.5, (
            f"Filtered listing ({elapsed_filtered*1000:.2f}ms) should not be "
            f"much slower than full listing ({elapsed_all*1000:.2f}ms)"
        )

    def test_store_namespace_listing_under_20ms(self, store_with_data):
        """Store namespace listing should complete under 20ms."""
        store = store_with_data

        # Warm up
        store.list_namespaces(prefix=("roles",))

        start = time.perf_counter()
        namespaces = store.list_namespaces(prefix=("roles",))
        elapsed = time.perf_counter() - start

        assert len(namespaces) > 0
        assert elapsed < 0.020, (
            f"Namespace listing {elapsed*1000:.2f}ms exceeds 20ms"
        )

    def test_initial_state_creation_under_1ms(self):
        """create_initial_state should complete under 1ms."""
        # Warm up
        create_initial_state("warmup")

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            state = create_initial_state("common")
        elapsed = (time.perf_counter() - start) / iterations

        assert state["role_name"] == "common"
        assert elapsed < 0.001, (
            f"Initial state creation {elapsed*1000:.3f}ms exceeds 1ms"
        )
