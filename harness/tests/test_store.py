"""
Tests for HarnessStore (LangGraph Store interface implementation).

Tests cover:
- Basic CRUD operations (put, get, delete)
- Search functionality with filtering
- Namespace operations
- Batch operations
- Async operations
"""

import pytest
from datetime import datetime, UTC

from harness.dag.store import HarnessStore, create_harness_store
from harness.db.state import StateDB


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def db():
    """Create an in-memory StateDB."""
    return StateDB(":memory:")


@pytest.fixture
def store(db):
    """Create a HarnessStore with in-memory database."""
    return HarnessStore(db)


# =============================================================================
# BASIC CRUD TESTS
# =============================================================================


class TestBasicOperations:
    """Test basic CRUD operations."""

    @pytest.mark.unit
    def test_put_and_get(self, store):
        """Should store and retrieve an item."""
        store.put(("roles", "common"), "status", {"passed": True})

        result = store.get(("roles", "common"), "status")

        assert result is not None
        assert result.value == {"passed": True}
        assert result.key == "status"
        assert result.namespace == ("roles", "common")

    @pytest.mark.unit
    def test_get_nonexistent(self, store):
        """Should return None for nonexistent key."""
        result = store.get(("roles", "common"), "nonexistent")
        assert result is None

    @pytest.mark.unit
    def test_put_overwrites(self, store):
        """Should overwrite existing value."""
        store.put(("roles", "common"), "status", {"version": 1})
        store.put(("roles", "common"), "status", {"version": 2})

        result = store.get(("roles", "common"), "status")
        assert result.value == {"version": 2}

    @pytest.mark.unit
    def test_delete(self, store):
        """Should delete an item."""
        store.put(("roles", "common"), "status", {"passed": True})
        store.delete(("roles", "common"), "status")

        result = store.get(("roles", "common"), "status")
        assert result is None

    @pytest.mark.unit
    def test_delete_nonexistent(self, store):
        """Deleting nonexistent key should not raise."""
        # Should not raise
        store.delete(("roles", "common"), "nonexistent")

    @pytest.mark.unit
    def test_complex_value(self, store):
        """Should handle complex nested values."""
        value = {
            "name": "common",
            "wave": 0,
            "tests": ["test1", "test2"],
            "metadata": {
                "author": "jsullivan2",
                "tags": ["foundation", "infrastructure"],
            },
        }

        store.put(("roles", "common"), "config", value)
        result = store.get(("roles", "common"), "config")

        assert result.value == value


# =============================================================================
# NAMESPACE TESTS
# =============================================================================


class TestNamespaces:
    """Test namespace handling."""

    @pytest.mark.unit
    def test_different_namespaces(self, store):
        """Items in different namespaces are independent."""
        store.put(("roles", "common"), "key", {"source": "roles"})
        store.put(("waves", "0"), "key", {"source": "waves"})

        result1 = store.get(("roles", "common"), "key")
        result2 = store.get(("waves", "0"), "key")

        assert result1.value["source"] == "roles"
        assert result2.value["source"] == "waves"

    @pytest.mark.unit
    def test_deep_namespace(self, store):
        """Should handle deep namespace hierarchies."""
        namespace = ("workflows", "box-up-role", "executions", "123", "nodes")
        store.put(namespace, "validate_role", {"status": "completed"})

        result = store.get(namespace, "validate_role")
        assert result.value == {"status": "completed"}
        assert result.namespace == namespace

    @pytest.mark.unit
    def test_list_namespaces(self, store):
        """Should list all namespaces."""
        store.put(("roles", "common"), "k1", {"v": 1})
        store.put(("roles", "sql_server"), "k1", {"v": 2})
        store.put(("waves", "0"), "k1", {"v": 3})

        namespaces = store.list_namespaces()

        assert len(namespaces) == 3
        assert ("roles", "common") in namespaces
        assert ("roles", "sql_server") in namespaces
        assert ("waves", "0") in namespaces

    @pytest.mark.unit
    def test_list_namespaces_with_prefix(self, store):
        """Should filter namespaces by prefix."""
        store.put(("roles", "common"), "k1", {"v": 1})
        store.put(("roles", "sql_server"), "k1", {"v": 2})
        store.put(("waves", "0"), "k1", {"v": 3})

        namespaces = store.list_namespaces(prefix=("roles",))

        assert len(namespaces) == 2
        assert ("roles", "common") in namespaces
        assert ("roles", "sql_server") in namespaces
        assert ("waves", "0") not in namespaces

    @pytest.mark.unit
    def test_list_namespaces_with_max_depth(self, store):
        """Should truncate namespaces to max_depth."""
        store.put(("a", "b", "c"), "k1", {"v": 1})
        store.put(("a", "b", "d"), "k1", {"v": 2})

        namespaces = store.list_namespaces(max_depth=2)

        # Both should be truncated to ("a", "b") and deduplicated
        assert namespaces == [("a", "b")]


# =============================================================================
# SEARCH TESTS
# =============================================================================


class TestSearch:
    """Test search functionality."""

    @pytest.mark.unit
    def test_search_by_namespace_prefix(self, store):
        """Should find items with namespace prefix."""
        store.put(("roles", "common"), "status", {"passed": True})
        store.put(("roles", "sql_server"), "status", {"passed": False})
        store.put(("waves", "0"), "status", {"passed": True})

        results = store.search(("roles",))

        assert len(results) == 2
        keys = {r.key for r in results}
        assert "status" in keys

    @pytest.mark.unit
    def test_search_with_filter(self, store):
        """Should filter results by value."""
        store.put(("roles", "common"), "status", {"passed": True, "wave": 0})
        store.put(("roles", "sql_server"), "status", {"passed": False, "wave": 2})
        store.put(("roles", "ems_web_app"), "status", {"passed": True, "wave": 3})

        results = store.search(("roles",), filter={"passed": True})

        assert len(results) == 2
        for r in results:
            assert r.value["passed"] is True

    @pytest.mark.unit
    def test_search_with_limit(self, store):
        """Should respect limit parameter."""
        for i in range(10):
            store.put(("roles", f"role_{i}"), "status", {"index": i})

        results = store.search(("roles",), limit=5)
        assert len(results) == 5

    @pytest.mark.unit
    def test_search_with_offset(self, store):
        """Should respect offset parameter."""
        for i in range(10):
            store.put(("roles", f"role_{i:02d}"), "status", {"index": i})

        all_results = store.search(("roles",), limit=100)
        offset_results = store.search(("roles",), limit=100, offset=5)

        assert len(offset_results) == len(all_results) - 5

    @pytest.mark.unit
    def test_search_empty_results(self, store):
        """Should return empty list when no matches."""
        store.put(("roles", "common"), "status", {"passed": True})

        results = store.search(("nonexistent",))
        assert results == []


# =============================================================================
# BATCH OPERATION TESTS
# =============================================================================


class TestBatchOperations:
    """Test batch operations."""

    @pytest.mark.unit
    def test_batch_multiple_puts(self, store):
        """Should execute multiple puts in one batch."""
        from langgraph.store.base import PutOp

        ops = [
            PutOp(("roles", "r1"), "k1", {"v": 1}),
            PutOp(("roles", "r2"), "k2", {"v": 2}),
            PutOp(("roles", "r3"), "k3", {"v": 3}),
        ]

        results = store.batch(ops)

        assert len(results) == 3
        assert store.get(("roles", "r1"), "k1").value == {"v": 1}
        assert store.get(("roles", "r2"), "k2").value == {"v": 2}
        assert store.get(("roles", "r3"), "k3").value == {"v": 3}

    @pytest.mark.unit
    def test_batch_mixed_operations(self, store):
        """Should handle mixed operation types."""
        from langgraph.store.base import GetOp, PutOp

        # Pre-populate
        store.put(("roles", "existing"), "key", {"original": True})

        ops = [
            PutOp(("roles", "new"), "key", {"new": True}),
            GetOp(("roles", "existing"), "key"),
            GetOp(("roles", "nonexistent"), "key"),
        ]

        results = store.batch(ops)

        assert len(results) == 3
        assert results[0] is None  # PutOp returns None
        assert results[1].value == {"original": True}  # GetOp returns Item
        assert results[2] is None  # GetOp for missing returns None


# =============================================================================
# ASYNC OPERATION TESTS
# =============================================================================


class TestAsyncOperations:
    """Test async operations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_aget(self, store):
        """Should get item asynchronously."""
        store.put(("roles", "common"), "status", {"passed": True})

        result = await store.aget(("roles", "common"), "status")

        assert result.value == {"passed": True}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_aput(self, store):
        """Should put item asynchronously."""
        await store.aput(("roles", "common"), "status", {"passed": True})

        result = store.get(("roles", "common"), "status")
        assert result.value == {"passed": True}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_adelete(self, store):
        """Should delete item asynchronously."""
        store.put(("roles", "common"), "status", {"passed": True})

        await store.adelete(("roles", "common"), "status")

        result = store.get(("roles", "common"), "status")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_asearch(self, store):
        """Should search asynchronously."""
        store.put(("roles", "r1"), "status", {"passed": True})
        store.put(("roles", "r2"), "status", {"passed": False})

        results = await store.asearch(("roles",), filter={"passed": True})

        assert len(results) == 1
        assert results[0].value["passed"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_alist_namespaces(self, store):
        """Should list namespaces asynchronously."""
        store.put(("roles", "r1"), "k", {"v": 1})
        store.put(("waves", "0"), "k", {"v": 2})

        namespaces = await store.alist_namespaces()

        assert len(namespaces) == 2


# =============================================================================
# TIMESTAMPS TESTS
# =============================================================================


class TestTimestamps:
    """Test timestamp handling."""

    @pytest.mark.unit
    def test_created_at_set(self, store):
        """Should set created_at timestamp."""
        store.put(("roles", "common"), "status", {"v": 1})

        result = store.get(("roles", "common"), "status")

        assert isinstance(result.created_at, datetime)

    @pytest.mark.unit
    def test_updated_at_changes(self, store):
        """Should update updated_at on modification."""
        store.put(("roles", "common"), "status", {"v": 1})
        first_result = store.get(("roles", "common"), "status")

        # Update the value
        store.put(("roles", "common"), "status", {"v": 2})
        second_result = store.get(("roles", "common"), "status")

        # updated_at should be >= first (could be same if fast)
        assert second_result.updated_at >= first_result.updated_at


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunction:
    """Test factory function."""

    @pytest.mark.unit
    def test_create_harness_store(self, db):
        """Should create HarnessStore via factory."""
        store = create_harness_store(db)

        assert isinstance(store, HarnessStore)

        # Verify it works
        store.put(("test",), "key", {"value": 1})
        assert store.get(("test",), "key").value == {"value": 1}
