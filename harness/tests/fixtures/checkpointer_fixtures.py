"""Checkpointing test fixtures for harness tests.

Provides comprehensive mocking for:
- LangGraph BaseCheckpointSaver
- Checkpoint data at various workflow stages
- Thread configuration dicts
- Async checkpointer operations
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# MOCK CHECKPOINTER
# =============================================================================


@pytest.fixture
def mock_checkpointer():
    """
    Mock BaseCheckpointSaver for testing without database.

    The mock provides:
    - put() / aput() - Store checkpoint
    - get() / aget() - Retrieve checkpoint
    - list() / alist() - List checkpoints for thread
    - get_tuple() / aget_tuple() - Get checkpoint with metadata

    The mock maintains an in-memory store for realistic behavior.

    Usage:
        def test_checkpoint_storage(mock_checkpointer):
            checkpointer = mock_checkpointer
            checkpointer.put(config, checkpoint, metadata)
            result = checkpointer.get(config)
    """
    store: dict[str, dict[str, Any]] = {}

    mock = MagicMock()

    def _make_key(config: dict) -> str:
        """Create storage key from config."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        return f"{thread_id}:{checkpoint_ns}"

    # Sync methods
    def put(config: dict, checkpoint: dict, metadata: dict | None = None, new_versions: dict | None = None) -> dict:
        key = _make_key(config)
        checkpoint_id = f"checkpoint_{len(store)}"
        entry = {
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint,
            "metadata": metadata or {},
            "parent_checkpoint_id": store.get(key, {}).get("checkpoint_id"),
            "created_at": datetime.utcnow().isoformat(),
        }
        store[key] = entry
        return {"configurable": {"checkpoint_id": checkpoint_id}}

    def get(config: dict) -> dict | None:
        key = _make_key(config)
        entry = store.get(key)
        if entry:
            return entry["checkpoint"]
        return None

    def list_checkpoints(config: dict, *, limit: int | None = None, before: dict | None = None) -> list[dict]:
        key = _make_key(config)
        entry = store.get(key)
        if entry:
            return [entry]
        return []

    def get_tuple(config: dict):
        key = _make_key(config)
        entry = store.get(key)
        if entry:
            # Return a namedtuple-like object
            result = MagicMock()
            result.checkpoint = entry["checkpoint"]
            result.metadata = entry["metadata"]
            result.config = config
            result.parent_config = None
            return result
        return None

    mock.put.side_effect = put
    mock.get.side_effect = get
    mock.list.side_effect = list_checkpoints
    mock.get_tuple.side_effect = get_tuple

    # Async methods (wrapping sync methods)
    async def aput(config: dict, checkpoint: dict, metadata: dict | None = None, new_versions: dict | None = None) -> dict:
        return put(config, checkpoint, metadata, new_versions)

    async def aget(config: dict) -> dict | None:
        return get(config)

    async def alist(config: dict, *, limit: int | None = None, before: dict | None = None):
        for item in list_checkpoints(config, limit=limit, before=before):
            yield item

    async def aget_tuple(config: dict):
        return get_tuple(config)

    mock.aput = AsyncMock(side_effect=aput)
    mock.aget = AsyncMock(side_effect=aget)
    mock.alist = alist
    mock.aget_tuple = AsyncMock(side_effect=aget_tuple)

    # Expose store for testing
    mock._store = store

    return mock


@pytest.fixture
def mock_async_checkpointer():
    """
    Pure async mock checkpointer for async-only testing.

    Similar to mock_checkpointer but only provides async methods.

    Usage:
        @pytest.mark.asyncio
        async def test_async_checkpoint(mock_async_checkpointer):
            await mock_async_checkpointer.aput(config, checkpoint)
    """
    store: dict[str, dict[str, Any]] = {}

    mock = AsyncMock()

    def _make_key(config: dict) -> str:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        return f"{thread_id}:{checkpoint_ns}"

    async def aput(config: dict, checkpoint: dict, metadata: dict | None = None, new_versions: dict | None = None) -> dict:
        key = _make_key(config)
        checkpoint_id = f"checkpoint_{len(store)}"
        entry = {
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint,
            "metadata": metadata or {},
            "parent_checkpoint_id": store.get(key, {}).get("checkpoint_id"),
        }
        store[key] = entry
        return {"configurable": {"checkpoint_id": checkpoint_id}}

    async def aget(config: dict) -> dict | None:
        key = _make_key(config)
        entry = store.get(key)
        if entry:
            return entry["checkpoint"]
        return None

    async def alist(config: dict, *, limit: int | None = None, before: dict | None = None):
        key = _make_key(config)
        entry = store.get(key)
        if entry:
            yield entry

    async def aget_tuple(config: dict):
        key = _make_key(config)
        entry = store.get(key)
        if entry:
            result = MagicMock()
            result.checkpoint = entry["checkpoint"]
            result.metadata = entry["metadata"]
            result.config = config
            return result
        return None

    mock.aput.side_effect = aput
    mock.aget.side_effect = aget
    mock.alist = alist
    mock.aget_tuple.side_effect = aget_tuple
    mock._store = store

    return mock


# =============================================================================
# CHECKPOINT DATA FIXTURES
# =============================================================================


@pytest.fixture
def checkpoint_data() -> dict[str, dict[str, Any]]:
    """
    Sample checkpoint dicts at various workflow stages.

    Provides realistic checkpoint data for testing checkpoint
    operations without running actual workflows.

    Returns:
        Dict mapping stage names to checkpoint data.

    Usage:
        def test_checkpoint_restore(checkpoint_data):
            initial = checkpoint_data["initial"]
            # Use checkpoint data in tests
    """
    return {
        "initial": {
            "v": 1,
            "ts": "2026-02-03T10:00:00Z",
            "id": "checkpoint_initial_001",
            "channel_values": {
                "role_name": "common",
                "execution_id": 42,
                "current_node": "validate_role",
                "completed_nodes": [],
                "errors": [],
            },
            "channel_versions": {
                "__start__": 1,
            },
            "versions_seen": {
                "__start__": {"__start__": 1},
            },
            "pending_sends": [],
        },
        "after_validation": {
            "v": 1,
            "ts": "2026-02-03T10:01:00Z",
            "id": "checkpoint_validation_002",
            "channel_values": {
                "role_name": "common",
                "execution_id": 42,
                "role_path": "/path/to/ansible/roles/common",
                "has_molecule_tests": True,
                "has_meta": True,
                "wave": 0,
                "wave_name": "wave_0_infrastructure",
                "explicit_deps": [],
                "reverse_deps": ["sql_server_2022", "ems_web_app"],
                "blocking_deps": [],
                "current_node": "create_worktree",
                "completed_nodes": ["validate_role", "analyze_deps", "check_reverse_deps"],
                "errors": [],
            },
            "channel_versions": {
                "__start__": 1,
                "validate_role": 2,
                "analyze_deps": 3,
                "check_reverse_deps": 4,
            },
            "versions_seen": {
                "__start__": {"__start__": 1},
                "validate_role": {"__start__": 1},
                "analyze_deps": {"validate_role": 2},
                "check_reverse_deps": {"analyze_deps": 3},
            },
            "pending_sends": [],
        },
        "after_tests": {
            "v": 1,
            "ts": "2026-02-03T10:05:00Z",
            "id": "checkpoint_tests_003",
            "channel_values": {
                "role_name": "common",
                "execution_id": 42,
                "role_path": "/path/to/ansible/roles/common",
                "has_molecule_tests": True,
                "molecule_passed": True,
                "molecule_duration": 245,
                "pytest_passed": True,
                "pytest_duration": 30,
                "deploy_passed": True,
                "all_tests_passed": True,
                "parallel_tests_completed": ["run_molecule", "run_pytest"],
                "worktree_path": "/path/to/.worktrees/sid-common",
                "branch": "sid/common",
                "current_node": "create_commit",
                "completed_nodes": [
                    "validate_role", "analyze_deps", "check_reverse_deps",
                    "create_worktree", "run_molecule", "run_pytest", "validate_deploy"
                ],
                "errors": [],
            },
            "channel_versions": {
                "__start__": 1,
                "validate_role": 2,
                "analyze_deps": 3,
                "check_reverse_deps": 4,
                "create_worktree": 5,
                "run_molecule": 6,
                "run_pytest": 7,
                "validate_deploy": 8,
            },
            "versions_seen": {},
            "pending_sends": [],
        },
        "awaiting_approval": {
            "v": 1,
            "ts": "2026-02-03T10:07:00Z",
            "id": "checkpoint_approval_004",
            "channel_values": {
                "role_name": "common",
                "execution_id": 42,
                "molecule_passed": True,
                "pytest_passed": True,
                "deploy_passed": True,
                "all_tests_passed": True,
                "pushed": True,
                "commit_sha": "abc123def456789",
                "issue_url": "https://gitlab.example.com/project/-/issues/123",
                "issue_iid": 123,
                "mr_url": "https://gitlab.example.com/project/-/merge_requests/456",
                "mr_iid": 456,
                "current_node": "human_approval",
                "completed_nodes": [
                    "validate_role", "analyze_deps", "check_reverse_deps",
                    "create_worktree", "run_molecule", "run_pytest", "validate_deploy",
                    "create_commit", "push_branch", "create_issue", "create_mr"
                ],
                "awaiting_human_input": True,
                "human_approved": None,
                "errors": [],
            },
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            # LangGraph interrupt data
            "__interrupt__": {
                "value": {
                    "message": "MR ready for review",
                    "mr_url": "https://gitlab.example.com/project/-/merge_requests/456",
                    "options": ["approve", "reject"],
                },
                "resumable": True,
            },
        },
        "completed": {
            "v": 1,
            "ts": "2026-02-03T10:10:00Z",
            "id": "checkpoint_completed_005",
            "channel_values": {
                "role_name": "common",
                "execution_id": 42,
                "all_tests_passed": True,
                "pushed": True,
                "mr_iid": 456,
                "merge_train_status": "added",
                "human_approved": True,
                "current_node": "report_summary",
                "completed_nodes": [
                    "validate_role", "analyze_deps", "check_reverse_deps",
                    "create_worktree", "run_molecule", "run_pytest", "validate_deploy",
                    "create_commit", "push_branch", "create_issue", "create_mr",
                    "human_approval", "add_to_merge_train", "report_summary"
                ],
                "errors": [],
                "summary": {
                    "role_name": "common",
                    "status": "success",
                    "mr_url": "https://gitlab.example.com/project/-/merge_requests/456",
                },
            },
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        },
        "failed": {
            "v": 1,
            "ts": "2026-02-03T10:03:00Z",
            "id": "checkpoint_failed_006",
            "channel_values": {
                "role_name": "sql_server_2022",
                "execution_id": 44,
                "molecule_passed": False,
                "molecule_output": "TASK [sql_server_2022] FAILED",
                "all_tests_passed": False,
                "current_node": "notify_failure",
                "completed_nodes": [
                    "validate_role", "analyze_deps", "check_reverse_deps",
                    "create_worktree", "run_molecule"
                ],
                "errors": ["Molecule tests failed: TASK [sql_server_2022] FAILED"],
            },
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        },
    }


@pytest.fixture
def checkpoint_metadata() -> dict[str, dict[str, Any]]:
    """
    Sample checkpoint metadata for various stages.

    LangGraph checkpoints include metadata about the checkpoint itself.
    """
    return {
        "initial": {
            "source": "loop",
            "step": 0,
            "writes": None,
            "parents": {},
        },
        "after_validation": {
            "source": "loop",
            "step": 3,
            "writes": {"check_reverse_deps": {"blocking_deps": []}},
            "parents": {"": "checkpoint_initial_001"},
        },
        "awaiting_approval": {
            "source": "loop",
            "step": 11,
            "writes": {"create_mr": {"mr_iid": 456}},
            "parents": {"": "checkpoint_tests_003"},
        },
        "completed": {
            "source": "loop",
            "step": 14,
            "writes": {"report_summary": {"summary": {"status": "success"}}},
            "parents": {"": "checkpoint_approval_004"},
        },
    }


# =============================================================================
# THREAD CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def thread_configs() -> dict[str, dict[str, Any]]:
    """
    Various LangGraph config dicts for testing.

    LangGraph uses config dicts with 'configurable' key for
    thread identification and checkpointing.

    Returns:
        Dict mapping config names to LangGraph config dicts.

    Usage:
        def test_with_config(thread_configs):
            config = thread_configs["basic"]
            result = await graph.ainvoke(state, config)
    """
    return {
        "basic": {
            "configurable": {
                "thread_id": "thread_basic_001",
            }
        },
        "with_namespace": {
            "configurable": {
                "thread_id": "thread_ns_002",
                "checkpoint_ns": "box-up-common",
            }
        },
        "with_checkpoint_id": {
            "configurable": {
                "thread_id": "thread_resume_003",
                "checkpoint_id": "checkpoint_approval_004",
            }
        },
        "role_common": {
            "configurable": {
                "thread_id": "box-up-common-42",
                "checkpoint_ns": "box-up-role",
            }
        },
        "role_sql_server": {
            "configurable": {
                "thread_id": "box-up-sql_server_2022-44",
                "checkpoint_ns": "box-up-role",
            }
        },
        "hotl_session": {
            "configurable": {
                "thread_id": "hotl-session-abc123",
                "checkpoint_ns": "hotl",
            }
        },
        "with_recursion_limit": {
            "configurable": {
                "thread_id": "thread_limited_004",
            },
            "recursion_limit": 50,
        },
        "with_callbacks": {
            "configurable": {
                "thread_id": "thread_callbacks_005",
            },
            "callbacks": [],  # Would contain callback handlers
        },
        "subgraph": {
            "configurable": {
                "thread_id": "main_workflow_006",
                "checkpoint_ns": "validation_subgraph",
            }
        },
    }


@pytest.fixture
def make_thread_config():
    """
    Factory fixture for creating custom thread configs.

    Usage:
        def test_custom_config(make_thread_config):
            config = make_thread_config(thread_id="custom-123", checkpoint_ns="test")
    """

    def _make_config(
        thread_id: str = "test-thread",
        checkpoint_ns: str = "",
        checkpoint_id: str | None = None,
        **extra_configurable,
    ) -> dict[str, Any]:
        configurable = {
            "thread_id": thread_id,
        }
        if checkpoint_ns:
            configurable["checkpoint_ns"] = checkpoint_ns
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        configurable.update(extra_configurable)

        return {"configurable": configurable}

    return _make_config


# =============================================================================
# SQLITE CHECKPOINTER FIXTURES
# =============================================================================


@pytest.fixture
def temp_checkpoint_db(tmp_path):
    """
    Temporary SQLite database path for checkpointer tests.

    Usage:
        def test_sqlite_checkpointer(temp_checkpoint_db):
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string(str(temp_checkpoint_db))
    """
    return tmp_path / "checkpoints.db"


@pytest.fixture
async def async_sqlite_checkpointer(temp_checkpoint_db):
    """
    Real AsyncSqliteSaver for integration tests.

    This creates a real async SQLite checkpointer for testing
    actual checkpoint operations.

    Usage:
        @pytest.mark.asyncio
        async def test_real_checkpointer(async_sqlite_checkpointer):
            async with async_sqlite_checkpointer as checkpointer:
                await checkpointer.aput(config, checkpoint)
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver.from_conn_string(str(temp_checkpoint_db))


# =============================================================================
# EXPORTED FIXTURES
# =============================================================================

__all__ = [
    "mock_checkpointer",
    "mock_async_checkpointer",
    "checkpoint_data",
    "checkpoint_metadata",
    "thread_configs",
    "make_thread_config",
    "temp_checkpoint_db",
    "async_sqlite_checkpointer",
]
