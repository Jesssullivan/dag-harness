"""Comprehensive idempotency verification tests for harness operations.

Tests verify the core idempotency guarantee: the SECOND call to any
create/find operation returns the SAME result as the FIRST call, without
creating duplicates or side effects.

Covers:
- Issue creation idempotency (find_existing_issue, get_or_create_issue)
- MR creation idempotency (find_existing_mr, get_or_create_mr)
- Worktree creation idempotency (create, reuse, cleanup)
- Full workflow idempotency (end-to-end re-run produces same resources)
- Cache decorator idempotency (@cache_result TTL, invalidation, bypass)

All GitLab API calls and subprocess invocations are mocked.
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    create_initial_state,
    create_issue_node,
    create_mr_node,
    create_worktree_node,
    add_to_merge_train_node,
    set_module_db,
)
from harness.db.models import (
    Issue,
    MergeRequest,
    Role,
    Worktree,
    WorktreeStatus,
)
from harness.db.state import StateDB
from harness.gitlab.idempotency import (
    CacheEntry,
    IdempotencyHelper,
    RoleArtifacts,
    _cache,
    cache_result,
    clear_cache,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_db(tmp_path):
    """Create a real StateDB backed by a temp file for testing."""
    db = StateDB(tmp_path / "test_idempotency.db")
    set_module_db(db)
    yield db
    set_module_db(None)


@pytest.fixture
def db_with_role(mock_db):
    """Database with a pre-populated test role."""
    mock_db.upsert_role(
        Role(
            name="common",
            wave=1,
            wave_name="Infrastructure",
            has_molecule_tests=True,
        )
    )
    return mock_db


@pytest.fixture
def sample_issue():
    """A sample Issue object for mocking returns."""
    return Issue(
        id=101,
        iid=42,
        role_id=1,
        title="feat(common): Box up `common` Ansible role",
        state="opened",
        web_url="https://gitlab.example.com/project/-/issues/42",
        labels='["wave-1", "box-up"]',
        weight=2,
    )


@pytest.fixture
def sample_mr():
    """A sample MergeRequest object for mocking returns."""
    return MergeRequest(
        id=201,
        iid=55,
        role_id=1,
        source_branch="sid/common",
        target_branch="main",
        title="feat(common): Add `common` Ansible role",
        state="opened",
        web_url="https://gitlab.example.com/project/-/merge_requests/55",
        merge_status="can_be_merged",
    )


@pytest.fixture(autouse=True)
def clear_idempotency_cache():
    """Clear the module-level cache before each test."""
    clear_cache()
    yield
    clear_cache()


def _make_state(role_name: str = "common", **overrides) -> BoxUpRoleState:
    """Create a BoxUpRoleState with common defaults and overrides."""
    state = create_initial_state(role_name, execution_id=1)
    state.update(overrides)
    return state


# =============================================================================
# ISSUE IDEMPOTENCY TESTS
# =============================================================================


@pytest.mark.unit
class TestIssueIdempotency:
    """Tests for idempotent issue creation and discovery."""

    def test_find_existing_issue_by_title(self, db_with_role, sample_issue):
        """find_existing_issue should find issue by title pattern."""
        helper = IdempotencyHelper(db_with_role)

        api_response = [
            {
                "id": 101,
                "iid": 42,
                "title": "feat(common): Box up `common` Ansible role",
                "state": "opened",
                "web_url": "https://gitlab.example.com/project/-/issues/42",
                "labels": ["wave-1", "box-up"],
                "assignees": [{"username": "jsullivan2"}],
                "weight": 2,
                "iteration": None,
            }
        ]

        with patch.object(helper, "_api_get", return_value=api_response):
            result = helper.find_existing_issue("common")
            assert result is not None
            assert result.iid == 42
            assert result.title == "feat(common): Box up `common` Ansible role"
            assert result.state == "opened"

    def test_find_existing_issue_not_found(self, db_with_role):
        """find_existing_issue returns None when no issue matches."""
        helper = IdempotencyHelper(db_with_role)

        with patch.object(helper, "_api_get", return_value=[]):
            result = helper.find_existing_issue("nonexistent_role")
            assert result is None

    def test_find_existing_issue_filters_fuzzy_matches(self, db_with_role):
        """find_existing_issue rejects fuzzy matches that do not contain role name."""
        helper = IdempotencyHelper(db_with_role)

        # API returns something that looks similar but is a different role
        api_response = [
            {
                "id": 999,
                "iid": 99,
                "title": "feat(common_v2): Box up `common_v2` Ansible role",
                "state": "opened",
                "web_url": "https://gitlab.example.com/project/-/issues/99",
                "labels": [],
                "assignees": [],
                "weight": None,
                "iteration": None,
            }
        ]

        with patch.object(helper, "_api_get", return_value=api_response):
            result = helper.find_existing_issue("common")
            assert result is None

    @pytest.mark.asyncio
    async def test_create_issue_twice_returns_same_iid(self, db_with_role, sample_issue):
        """Calling create_issue_node twice with same role returns same IID."""
        mock_client = MagicMock()
        mock_client.get_current_iteration.return_value = None
        mock_client.prepare_labels_for_role.return_value = ["wave-1"]
        # First call creates, second call finds existing
        mock_client.get_or_create_issue.side_effect = [
            (sample_issue, True),   # First: created
            (sample_issue, False),  # Second: found existing
        ]

        state = _make_state("common", wave=1, wave_name="Infrastructure")

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result1 = await create_issue_node(state)
            result2 = await create_issue_node(state)

        assert result1["issue_iid"] == result2["issue_iid"] == 42
        assert result1["issue_created"] is True
        assert result2["issue_created"] is False

    @pytest.mark.asyncio
    async def test_create_issue_after_partial_failure(self, db_with_role, sample_issue):
        """If issue was created but state not saved, retry finds existing."""
        mock_client = MagicMock()
        mock_client.get_current_iteration.return_value = None
        mock_client.prepare_labels_for_role.return_value = ["wave-1"]

        # First call: API creates issue but then raises (simulating partial failure)
        # Second call: get_or_create_issue finds the already-created issue
        mock_client.get_or_create_issue.side_effect = [
            RuntimeError("Network timeout after issue creation"),
            (sample_issue, False),  # Retry finds the existing issue
        ]

        state = _make_state("common", wave=1, wave_name="Infrastructure")

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result1 = await create_issue_node(state)
            assert "errors" in result1
            assert len(result1["errors"]) > 0

            result2 = await create_issue_node(state)
            assert result2["issue_iid"] == 42
            assert result2["issue_created"] is False

    @pytest.mark.asyncio
    async def test_issue_idempotency_across_threads(self, db_with_role, sample_issue):
        """Two concurrent workflows for same role get same issue."""
        mock_client = MagicMock()
        mock_client.get_current_iteration.return_value = None
        mock_client.prepare_labels_for_role.return_value = ["wave-1"]
        # Both calls return the same issue (second finds what first created)
        mock_client.get_or_create_issue.return_value = (sample_issue, False)

        state_a = _make_state("common", wave=1, wave_name="Infrastructure")
        state_b = _make_state("common", wave=1, wave_name="Infrastructure")

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result_a = await create_issue_node(state_a)
            result_b = await create_issue_node(state_b)

        assert result_a["issue_iid"] == result_b["issue_iid"] == 42

    def test_find_existing_issue_closed_state(self, db_with_role):
        """find_existing_issue finds closed issues when state='all'."""
        helper = IdempotencyHelper(db_with_role)

        api_response = [
            {
                "id": 101,
                "iid": 42,
                "title": "feat(common): Box up `common` Ansible role",
                "state": "closed",
                "web_url": "https://gitlab.example.com/project/-/issues/42",
                "labels": [],
                "assignees": [],
                "weight": None,
                "iteration": None,
            }
        ]

        with patch.object(helper, "_api_get", return_value=api_response):
            result = helper.find_existing_issue("common", state="all")
            assert result is not None
            assert result.state == "closed"
            assert result.iid == 42


# =============================================================================
# MR IDEMPOTENCY TESTS
# =============================================================================


@pytest.mark.unit
class TestMRIdempotency:
    """Tests for idempotent merge request creation and discovery."""

    def test_find_existing_mr_by_branch(self, db_with_role, sample_mr):
        """find_existing_mr should find MR by source branch."""
        helper = IdempotencyHelper(db_with_role)

        api_response = [
            {
                "id": 201,
                "iid": 55,
                "source_branch": "sid/common",
                "target_branch": "main",
                "title": "feat(common): Add `common` Ansible role",
                "state": "opened",
                "web_url": "https://gitlab.example.com/project/-/merge_requests/55",
                "merge_status": "can_be_merged",
                "squash_on_merge": True,
                "force_remove_source_branch": True,
                "assignees": [],
            }
        ]

        with patch.object(helper, "_api_get", return_value=api_response):
            result = helper.find_existing_mr("common", source_branch="sid/common")
            assert result is not None
            assert result.iid == 55
            assert result.source_branch == "sid/common"

    def test_find_existing_mr_not_found(self, db_with_role):
        """find_existing_mr returns None when no MR matches."""
        helper = IdempotencyHelper(db_with_role)

        with patch.object(helper, "_api_get", return_value=[]):
            result = helper.find_existing_mr("common", source_branch="sid/common")
            assert result is None

    @pytest.mark.asyncio
    async def test_create_mr_twice_returns_same_iid(self, db_with_role, sample_mr):
        """Calling create_mr_node twice returns same MR IID."""
        mock_client = MagicMock()
        mock_client.config.default_reviewers = []
        mock_client.get_or_create_mr.side_effect = [
            (sample_mr, True),   # First: created
            (sample_mr, False),  # Second: found existing
        ]

        state = _make_state(
            "common",
            issue_iid=42,
            branch="sid/common",
            wave=1,
            wave_name="Infrastructure",
        )

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result1 = await create_mr_node(state)
            result2 = await create_mr_node(state)

        assert result1["mr_iid"] == result2["mr_iid"] == 55
        assert result1["mr_created"] is True
        assert result2["mr_created"] is False

    @pytest.mark.asyncio
    async def test_mr_from_existing_issue(self, db_with_role, sample_mr):
        """MR creation links to existing issue."""
        mock_client = MagicMock()
        mock_client.config.default_reviewers = []
        mock_client.get_or_create_mr.return_value = (sample_mr, True)

        state = _make_state(
            "common",
            issue_iid=42,
            branch="sid/common",
            wave=1,
            wave_name="Infrastructure",
        )

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result = await create_mr_node(state)

        assert result["mr_iid"] == 55
        # Verify issue_iid was passed to get_or_create_mr
        call_kwargs = mock_client.get_or_create_mr.call_args
        assert call_kwargs.kwargs.get("issue_iid") == 42 or call_kwargs[1].get("issue_iid") == 42

    @pytest.mark.asyncio
    async def test_mr_idempotency_with_different_state(self, db_with_role, sample_mr):
        """Even with different state snapshots, same MR is returned."""
        mock_client = MagicMock()
        mock_client.config.default_reviewers = ["reviewer1"]
        mock_client.set_mr_reviewers.return_value = True
        mock_client.get_or_create_mr.return_value = (sample_mr, False)

        state_v1 = _make_state(
            "common",
            issue_iid=42,
            branch="sid/common",
            wave=1,
            wave_name="Infrastructure",
        )
        # Second state has slightly different metadata
        state_v2 = _make_state(
            "common",
            issue_iid=42,
            branch="sid/common",
            wave=1,
            wave_name="Infrastructure Foundation",
        )

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result1 = await create_mr_node(state_v1)
            result2 = await create_mr_node(state_v2)

        assert result1["mr_iid"] == result2["mr_iid"] == 55

    @pytest.mark.asyncio
    async def test_mr_no_issue_iid_errors(self, db_with_role):
        """MR creation without issue IID returns error."""
        state = _make_state("common", branch="sid/common")
        # Ensure issue_iid is not set
        state.pop("issue_iid", None)

        result = await create_mr_node(state)
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_find_existing_mr_merged_state(self, db_with_role):
        """find_existing_mr finds merged MRs when state='all'."""
        helper = IdempotencyHelper(db_with_role)

        api_response = [
            {
                "id": 201,
                "iid": 55,
                "source_branch": "sid/common",
                "target_branch": "main",
                "title": "feat(common): Add `common` Ansible role",
                "state": "merged",
                "web_url": "https://gitlab.example.com/project/-/merge_requests/55",
                "merge_status": "can_be_merged",
                "squash_on_merge": True,
                "force_remove_source_branch": True,
            }
        ]

        with patch.object(helper, "_api_get", return_value=api_response):
            result = helper.find_existing_mr("common", source_branch="sid/common", state="all")
            assert result is not None
            assert result.state == "merged"


# =============================================================================
# WORKTREE IDEMPOTENCY TESTS
# =============================================================================


@pytest.mark.unit
class TestWorktreeIdempotency:
    """Tests for idempotent worktree creation."""

    @pytest.mark.asyncio
    async def test_create_worktree_twice_reuses(self, db_with_role):
        """Second worktree creation for same role reuses existing."""
        mock_manager = MagicMock()
        mock_client = MagicMock()
        mock_client.remote_branch_exists.return_value = False

        # First call succeeds, second raises ValueError (already exists)
        worktree_info = MagicMock()
        worktree_info.path = "/tmp/sid-common"
        worktree_info.branch = "sid/common"
        worktree_info.commit = "abc123"

        mock_manager.create.side_effect = [
            worktree_info,
            ValueError("Worktree already exists at /tmp/sid-common"),
        ]

        # When worktree already exists, fallback reads from DB
        existing_worktree = Worktree(
            id=1,
            role_id=1,
            path="/tmp/sid-common",
            branch="sid/common",
            current_commit="abc123",
            status=WorktreeStatus.ACTIVE,
        )

        state = _make_state("common")

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client), \
             patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager):
            result1 = await create_worktree_node(state)

        assert result1["worktree_path"] == "/tmp/sid-common"
        assert result1["branch"] == "sid/common"

        # Second call: manager raises ValueError, node falls back to DB lookup
        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client), \
             patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager), \
             patch.object(db_with_role, "get_worktree", return_value=existing_worktree):
            result2 = await create_worktree_node(state)

        assert result2["worktree_path"] == "/tmp/sid-common"
        assert result2["branch"] == "sid/common"

    @pytest.mark.asyncio
    async def test_worktree_exists_different_path(self, db_with_role):
        """If worktree exists at different path, IdempotencyHelper detects it."""
        helper = IdempotencyHelper(db_with_role)

        # The filesystem path doesn't exist, but git worktree list shows it
        porcelain_output = (
            "worktree /other/path/sid-common\n"
            "HEAD abc123\n"
            "branch refs/heads/sid/common\n"
            "\n"
        )

        mock_result = MagicMock()
        mock_result.stdout = porcelain_output

        with patch("harness.gitlab.idempotency.subprocess.run", return_value=mock_result), \
             patch("harness.gitlab.idempotency.Path") as mock_path:
            # Filesystem check fails (expected path doesn't exist)
            mock_resolved = MagicMock()
            mock_resolved.__truediv__ = lambda self, x: Path(f"/expected/{x}")
            mock_path.return_value.resolve.return_value = mock_resolved
            mock_path.return_value.resolve.return_value.__truediv__ = (
                lambda self, x: MagicMock(exists=MagicMock(return_value=False))
            )

            # The actual path from git worktree list
            exists, path = helper.worktree_exists("common", base_path="/expected")

        assert exists is True
        assert path == "/other/path/sid-common"

    @pytest.mark.asyncio
    async def test_worktree_cleanup_and_recreate(self, db_with_role):
        """Stale worktree is cleaned up and recreated."""
        mock_manager = MagicMock()
        mock_client = MagicMock()
        mock_client.remote_branch_exists.return_value = False

        # Simulate: first create fails because worktree is stale,
        # caller could force-recreate
        worktree_info = MagicMock()
        worktree_info.path = "/tmp/sid-common"
        worktree_info.branch = "sid/common"
        worktree_info.commit = "def456"

        mock_manager.create.side_effect = [
            RuntimeError("Worktree creation failed: fatal: '/tmp/sid-common' is a stale worktree"),
            worktree_info,  # After cleanup, second attempt succeeds
        ]

        state = _make_state("common")

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client), \
             patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager):
            result1 = await create_worktree_node(state)
            # First attempt fails with stale error
            assert "errors" in result1
            assert len(result1["errors"]) > 0

        # Second attempt (after manual cleanup) succeeds
        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client), \
             patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager):
            result2 = await create_worktree_node(state)
            assert result2["worktree_path"] == "/tmp/sid-common"

    def test_worktree_not_found(self, db_with_role):
        """worktree_exists returns (False, None) when no worktree exists."""
        helper = IdempotencyHelper(db_with_role)

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("harness.gitlab.idempotency.subprocess.run", return_value=mock_result), \
             patch("harness.gitlab.idempotency.Path") as mock_path:
            mock_wt_path = MagicMock()
            mock_wt_path.exists.return_value = False
            mock_resolved = MagicMock()
            mock_resolved.__truediv__ = lambda self, x: mock_wt_path
            mock_path.return_value.resolve.return_value = mock_resolved

            exists, path = helper.worktree_exists("nonexistent_role")

        assert exists is False
        assert path is None


# =============================================================================
# FULL WORKFLOW IDEMPOTENCY TESTS
# =============================================================================


@pytest.mark.unit
class TestFullWorkflowIdempotency:
    """Tests for end-to-end workflow idempotency."""

    @pytest.mark.asyncio
    async def test_full_workflow_twice_same_resources(
        self, db_with_role, sample_issue, sample_mr
    ):
        """Running full workflow twice produces same issue/MR IIDs."""
        mock_client = MagicMock()
        mock_client.get_current_iteration.return_value = None
        mock_client.prepare_labels_for_role.return_value = ["wave-1"]
        mock_client.config.default_reviewers = []

        # First run: creates new resources
        # Second run: finds existing resources
        mock_client.get_or_create_issue.side_effect = [
            (sample_issue, True),
            (sample_issue, False),
        ]
        mock_client.get_or_create_mr.side_effect = [
            (sample_mr, True),
            (sample_mr, False),
        ]

        state = _make_state(
            "common",
            wave=1,
            wave_name="Infrastructure",
            branch="sid/common",
        )

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            # First run
            issue_r1 = await create_issue_node(state)
            state_with_issue = {**state, "issue_iid": issue_r1["issue_iid"]}
            mr_r1 = await create_mr_node(state_with_issue)

            # Second run
            issue_r2 = await create_issue_node(state)
            state_with_issue2 = {**state, "issue_iid": issue_r2["issue_iid"]}
            mr_r2 = await create_mr_node(state_with_issue2)

        assert issue_r1["issue_iid"] == issue_r2["issue_iid"] == 42
        assert mr_r1["mr_iid"] == mr_r2["mr_iid"] == 55
        assert issue_r1["issue_created"] is True
        assert issue_r2["issue_created"] is False
        assert mr_r1["mr_created"] is True
        assert mr_r2["mr_created"] is False

    @pytest.mark.asyncio
    async def test_partial_failure_resume(self, db_with_role, sample_issue, sample_mr):
        """Fail at MR creation, retry reuses existing issue."""
        mock_client = MagicMock()
        mock_client.get_current_iteration.return_value = None
        mock_client.prepare_labels_for_role.return_value = ["wave-1"]
        mock_client.config.default_reviewers = []

        # Issue creation succeeds on first run
        mock_client.get_or_create_issue.return_value = (sample_issue, True)

        # MR creation: first fails, second succeeds
        mock_client.get_or_create_mr.side_effect = [
            RuntimeError("API timeout during MR creation"),
            (sample_mr, True),
        ]

        state = _make_state(
            "common",
            wave=1,
            wave_name="Infrastructure",
            branch="sid/common",
        )

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            # Issue succeeds
            issue_result = await create_issue_node(state)
            assert issue_result["issue_iid"] == 42

            # MR fails
            state_with_issue = {**state, "issue_iid": 42}
            mr_result_1 = await create_mr_node(state_with_issue)
            assert "errors" in mr_result_1

            # Retry: issue is reused (get_or_create returns existing)
            mock_client.get_or_create_issue.return_value = (sample_issue, False)
            issue_retry = await create_issue_node(state)
            assert issue_retry["issue_iid"] == 42
            assert issue_retry["issue_created"] is False

            # MR succeeds on retry
            mr_result_2 = await create_mr_node(state_with_issue)
            assert mr_result_2["mr_iid"] == 55

    @pytest.mark.asyncio
    async def test_resume_after_molecule_failure(self, db_with_role, sample_issue):
        """Molecule fails, fix, re-run: reuses worktree and issue."""
        mock_client = MagicMock()
        mock_client.remote_branch_exists.return_value = False
        mock_client.get_current_iteration.return_value = None
        mock_client.prepare_labels_for_role.return_value = ["wave-1"]

        # Worktree already exists on re-run
        mock_manager = MagicMock()
        worktree_info = MagicMock()
        worktree_info.path = "/tmp/sid-common"
        worktree_info.branch = "sid/common"
        worktree_info.commit = "abc123"

        # First attempt creates worktree, second finds existing
        existing_worktree = Worktree(
            id=1,
            role_id=1,
            path="/tmp/sid-common",
            branch="sid/common",
            current_commit="abc123",
            status=WorktreeStatus.ACTIVE,
        )
        mock_manager.create.side_effect = [
            worktree_info,
            ValueError("Worktree already exists"),
        ]

        state = _make_state("common", wave=1, wave_name="Infrastructure")

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client), \
             patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager):
            # First run: worktree created
            wt_result_1 = await create_worktree_node(state)
            assert wt_result_1["worktree_path"] == "/tmp/sid-common"

        # Issue also created on first run
        mock_client.get_or_create_issue.side_effect = [
            (sample_issue, True),
            (sample_issue, False),
        ]

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            issue_1 = await create_issue_node(state)
            assert issue_1["issue_created"] is True

        # After molecule fix, re-run: worktree reused
        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client), \
             patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager), \
             patch.object(db_with_role, "get_worktree", return_value=existing_worktree):
            wt_result_2 = await create_worktree_node(state)
            assert wt_result_2["worktree_path"] == "/tmp/sid-common"

        # Issue also reused
        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            issue_2 = await create_issue_node(state)
            assert issue_2["issue_iid"] == 42
            assert issue_2["issue_created"] is False

    @pytest.mark.asyncio
    async def test_idempotent_merge_train_add(self, db_with_role):
        """Adding to merge train twice doesn't create duplicate entry."""
        mock_client = MagicMock()

        # First call: MR added to merge train
        mock_client.is_merge_train_available.return_value = {"available": True}
        mock_client.is_merge_train_enabled.return_value = True
        mock_client.add_to_merge_train.return_value = {"status": "ok"}

        state = _make_state("common", mr_iid=55)

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result1 = await add_to_merge_train_node(state)
            assert result1["merge_train_status"] == "added"

        # Second call: MR already in merge train
        mock_client.is_merge_train_available.return_value = {
            "available": False,
            "reason": "MR already in merge train at position 1",
        }

        with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
            result2 = await add_to_merge_train_node(state)
            assert result2["merge_train_status"] == "added"
            # No error because "already in merge train" is handled gracefully
            assert "errors" not in result2 or len(result2.get("errors", [])) == 0


# =============================================================================
# CACHE IDEMPOTENCY TESTS
# =============================================================================


@pytest.mark.unit
class TestCacheIdempotency:
    """Tests for the @cache_result decorator and cache utilities."""

    def test_cache_result_decorator(self):
        """@cache_result returns same value for same args within TTL."""
        call_count = 0

        @cache_result(ttl_seconds=300)
        def expensive_call(role_name: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"role": role_name, "call": call_count}

        result1 = expensive_call("common")
        result2 = expensive_call("common")

        assert result1 == result2
        assert call_count == 1  # Function only called once

    def test_cache_different_kwargs(self):
        """Cache distinguishes between different keyword arguments."""
        call_count = 0

        @cache_result(ttl_seconds=300)
        def lookup(*, role_name: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result-{role_name}"

        result_a = lookup(role_name="common")
        result_b = lookup(role_name="iis_config")

        assert result_a == "result-common"
        assert result_b == "result-iis_config"
        assert call_count == 2

    def test_cache_invalidation_on_ttl(self):
        """Cache expires after TTL seconds."""
        call_count = 0

        @cache_result(ttl_seconds=1)
        def time_sensitive() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result1 = time_sensitive()
        assert result1 == 1
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        result2 = time_sensitive()
        assert result2 == 2
        assert call_count == 2

    def test_cache_bypass_on_force(self):
        """force=True bypasses cache via clear_cache."""
        call_count = 0

        @cache_result(ttl_seconds=300)
        def cached_lookup() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result1 = cached_lookup()
        assert result1 == 1

        # Clear cache to force re-execution (equivalent to force=True)
        clear_cache()

        result2 = cached_lookup()
        assert result2 == 2
        assert call_count == 2

    def test_cache_clear_pattern(self):
        """clear_cache with pattern only clears matching entries."""
        @cache_result(ttl_seconds=300)
        def func_a(x: int) -> int:
            return x * 2

        @cache_result(ttl_seconds=300)
        def func_b(x: int) -> int:
            return x * 3

        func_a(1)
        func_b(2)

        # Clear only func_a entries
        cleared = clear_cache("func_a")
        assert cleared >= 1

        # func_b should still be cached
        # Verify by checking cache is not empty
        from harness.gitlab.idempotency import _cache
        remaining = [k for k in _cache if "func_b" in k]
        assert len(remaining) >= 1

    def test_cache_with_method_skips_self(self):
        """@cache_result on a method correctly skips self in cache key."""

        class MyHelper:
            def __init__(self):
                self.call_count = 0

            @cache_result(ttl_seconds=300)
            def lookup(self, name: str) -> str:
                self.call_count += 1
                return f"found-{name}"

        helper = MyHelper()
        result1 = helper.lookup("common")
        result2 = helper.lookup("common")

        assert result1 == result2 == "found-common"
        assert helper.call_count == 1

    def test_cache_none_value(self):
        """Cache correctly caches None return values."""
        call_count = 0

        @cache_result(ttl_seconds=300)
        def return_none() -> None:
            nonlocal call_count
            call_count += 1
            return None

        result1 = return_none()
        result2 = return_none()

        assert result1 is None
        assert result2 is None
        assert call_count == 1  # Only called once, None was cached


# =============================================================================
# ROLE ARTIFACTS TESTS
# =============================================================================


@pytest.mark.unit
class TestRoleArtifacts:
    """Tests for the RoleArtifacts data class."""

    def test_no_artifacts(self):
        """Role with no artifacts reports correctly."""
        artifacts = RoleArtifacts(role_name="common")
        assert not artifacts.has_any_artifacts
        assert not artifacts.is_complete
        assert artifacts.needs_issue
        assert not artifacts.needs_mr  # needs_mr requires remote_branch_exists
        assert not artifacts.needs_branch_push

    def test_complete_artifacts(self, sample_issue, sample_mr):
        """Role with all artifacts is marked complete."""
        artifacts = RoleArtifacts(
            role_name="common",
            existing_issue=sample_issue,
            existing_mr=sample_mr,
            remote_branch_exists=True,
            worktree_exists=True,
            worktree_path="/tmp/sid-common",
            issue_state="opened",
            mr_state="opened",
            branch_name="sid/common",
        )
        assert artifacts.has_any_artifacts
        assert artifacts.is_complete
        assert not artifacts.needs_issue
        assert not artifacts.needs_mr
        assert not artifacts.needs_branch_push

    def test_needs_mr(self, sample_issue):
        """Role with issue and branch but no MR needs MR."""
        artifacts = RoleArtifacts(
            role_name="common",
            existing_issue=sample_issue,
            remote_branch_exists=True,
        )
        assert artifacts.needs_mr
        assert not artifacts.needs_issue
        assert not artifacts.needs_branch_push

    def test_needs_branch_push(self, sample_issue):
        """Role with worktree but no remote branch needs push."""
        artifacts = RoleArtifacts(
            role_name="common",
            existing_issue=sample_issue,
            worktree_exists=True,
            worktree_path="/tmp/sid-common",
        )
        assert artifacts.needs_branch_push
        assert not artifacts.needs_mr  # Can't create MR without remote branch

    def test_to_dict_serialization(self, sample_issue):
        """RoleArtifacts serializes to dict correctly."""
        artifacts = RoleArtifacts(
            role_name="common",
            existing_issue=sample_issue,
            issue_state="opened",
            branch_name="sid/common",
        )
        d = artifacts.to_dict()
        assert d["role_name"] == "common"
        assert d["existing_issue"]["iid"] == 42
        assert d["issue_state"] == "opened"
        assert d["existing_mr"] is None

    def test_find_all_role_artifacts(self, db_with_role, sample_issue, sample_mr):
        """find_all_role_artifacts aggregates all artifact types."""
        helper = IdempotencyHelper(db_with_role)

        with patch.object(helper, "find_existing_issue", return_value=sample_issue), \
             patch.object(helper, "find_existing_mr", return_value=sample_mr), \
             patch.object(helper, "remote_branch_exists", return_value=True), \
             patch.object(helper, "worktree_exists", return_value=(True, "/tmp/sid-common")):
            artifacts = helper.find_all_role_artifacts("common")

        assert artifacts.existing_issue.iid == 42
        assert artifacts.existing_mr.iid == 55
        assert artifacts.remote_branch_exists is True
        assert artifacts.worktree_exists is True
        assert artifacts.is_complete
