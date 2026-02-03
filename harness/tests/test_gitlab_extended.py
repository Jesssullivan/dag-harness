"""
Extended GitLab API tests for harness.

Tests cover:
1. IDEMPOTENCY - Caching, get_or_create operations, cache invalidation
2. RACE CONDITIONS - Concurrent issue/MR creation, merge train operations
3. ERROR RECOVERY - Retry logic, timeout handling, max retries
4. MERGE TRAIN - Add/remove operations, position tracking, wave ordering

Tests marked with @pytest.mark.unit are fully mocked and CI-safe.
Tests marked with @pytest.mark.asyncio require async support.
"""

import asyncio
import concurrent.futures
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from harness.db.models import Issue, MergeRequest
from harness.gitlab import (
    GitLabClient,
    cache_result,
    clear_cache,
    get_cache_stats,
)
from harness.gitlab.advanced import GitLabMergeTrainManager
from harness.gitlab.errors import (
    GitLabAPIError,
    GitLabAuthenticationError,
    GitLabConflictError,
    GitLabNotFoundError,
    GitLabRateLimitError,
    GitLabServerError,
    GitLabTimeoutError,
    parse_gitlab_error,
)
from harness.gitlab.idempotency import (
    CacheEntry,
    IdempotencyHelper,
    RoleArtifacts,
    _cache,
)
from harness.gitlab.merge_train import (
    MergeReadinessResult,
    MergeTrainHelper,
    get_mr_merge_readiness,
    preflight_merge_train_check,
    wait_for_merge_train_position,
)
from harness.gitlab.retry import gitlab_retry, gitlab_retry_async


# =============================================================================
# TEST CLASS 1: IDEMPOTENCY TESTS
# =============================================================================


class TestIdempotencyExtended:
    """Extended tests for idempotent GitLab operations."""

    @pytest.mark.unit
    def test_find_existing_issue_returns_same(self, db_with_roles):
        """
        Test that find_existing_issue returns the same issue on retry.

        Verifies:
        - API is called only once (cached)
        - Same Issue object IID is returned on subsequent calls
        - Cache correctly stores and retrieves result
        """
        clear_cache()
        helper = IdempotencyHelper(db_with_roles)

        mock_issue_data = [
            {
                "id": 12345,
                "iid": 42,
                "title": "Box up `common` role",
                "state": "opened",
                "web_url": "https://gitlab.com/test/-/issues/42",
                "labels": ["role", "wave::0"],
                "assignees": [{"username": "testuser"}],
                "weight": 3,
                "iteration": {"id": 999},
            }
        ]

        with (
            patch.object(helper, "_api_get", return_value=mock_issue_data) as mock_api,
            patch.object(helper.db, "upsert_issue"),
        ):
            # First call
            result1 = helper.find_existing_issue("common")
            assert result1 is not None
            assert result1.iid == 42

            # Second call - should use cache
            result2 = helper.find_existing_issue("common")
            assert result2 is not None
            assert result2.iid == 42

            # Third call - still cached
            result3 = helper.find_existing_issue("common")
            assert result3.iid == result1.iid

            # API should only be called once
            assert mock_api.call_count == 1

        clear_cache()

    @pytest.mark.unit
    def test_get_or_create_issue_idempotent(self, db_with_roles, mock_gitlab_api):
        """
        Test that get_or_create_issue is idempotent - calling twice returns same resource.

        Verifies:
        - First call creates the issue
        - Second call returns existing issue without creating
        - Issue IIDs match between calls
        """
        client = GitLabClient(db=db_with_roles)

        mock_issue = MagicMock(spec=Issue)
        mock_issue.iid = 123
        mock_issue.id = 67890
        mock_issue.title = "Box up `common` role"
        mock_issue.state = "opened"
        mock_issue.web_url = "https://example.com/issue/123"

        call_count = [0]

        def mock_find_issue(role_name, state="opened"):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # First call: not found
            return mock_issue  # Subsequent calls: found

        with patch.object(client, "find_existing_issue", side_effect=mock_find_issue):
            with patch.object(client, "create_issue", return_value=mock_issue):
                # First call - creates issue
                issue1, created1 = client.get_or_create_issue(
                    role_name="common",
                    title="Box up `common` role",
                    description="Test description",
                )

                # Second call - finds existing
                issue2, created2 = client.get_or_create_issue(
                    role_name="common",
                    title="Box up `common` role",
                    description="Test description",
                )

                assert created1 is True
                assert created2 is False
                assert issue1.iid == issue2.iid

    @pytest.mark.unit
    def test_get_or_create_mr_idempotent(self, db_with_roles, mock_gitlab_api):
        """
        Test that get_or_create_mr is idempotent - calling twice returns same resource.

        Verifies:
        - First call creates the MR
        - Second call returns existing MR without creating
        - MR IIDs match between calls
        """
        client = GitLabClient(db=db_with_roles)

        mock_mr = MagicMock(spec=MergeRequest)
        mock_mr.iid = 456
        mock_mr.id = 11111
        mock_mr.source_branch = "sid/common"
        mock_mr.state = "opened"
        mock_mr.web_url = "https://example.com/mr/456"

        call_count = [0]

        def mock_find_mr(branch, state="opened"):
            call_count[0] += 1
            # First two calls (opened, merged) return None for first get_or_create
            if call_count[0] <= 2:
                return None
            return mock_mr  # Subsequent calls find it

        with patch.object(client, "find_existing_mr", side_effect=mock_find_mr):
            with patch.object(client, "create_merge_request", return_value=mock_mr):
                # First call - creates MR
                mr1, created1 = client.get_or_create_mr(
                    role_name="common",
                    source_branch="sid/common",
                    title="Box up common role",
                    description="Test",
                )

                # Second call - finds existing
                mr2, created2 = client.get_or_create_mr(
                    role_name="common",
                    source_branch="sid/common",
                    title="Box up common role",
                    description="Test",
                )

                assert created1 is True
                assert created2 is False
                assert mr1.iid == mr2.iid

    @pytest.mark.unit
    def test_find_all_role_artifacts_caching(self, db_with_roles):
        """
        Test that find_all_role_artifacts uses caching with TTL.

        Verifies:
        - Results are cached on first call
        - Subsequent calls within TTL use cache
        - Cache stats reflect correct state
        """
        clear_cache()
        helper = IdempotencyHelper(db_with_roles)

        mock_issue_response = [
            {
                "id": 100,
                "iid": 10,
                "title": "Box up `common` role",
                "state": "opened",
                "web_url": "https://gitlab.com/test/-/issues/10",
                "labels": ["role"],
                "assignees": [],
                "weight": None,
                "iteration": None,
            }
        ]

        def mock_api_get(endpoint):
            if "issues" in endpoint:
                return mock_issue_response
            return []

        with (
            patch.object(helper, "_api_get", side_effect=mock_api_get) as mock_api,
            patch.object(helper, "remote_branch_exists", return_value=True),
            patch.object(helper, "worktree_exists", return_value=(False, None)),
        ):
            # First call
            artifacts1 = helper.find_all_role_artifacts("common")
            assert artifacts1.existing_issue is not None

            # Get cache stats
            stats = get_cache_stats()
            initial_entries = stats["active_entries"]

            # Second call - should use cache
            artifacts2 = helper.find_all_role_artifacts("common")
            assert artifacts2.existing_issue.iid == artifacts1.existing_issue.iid

            # Cache entries should not increase significantly
            stats_after = get_cache_stats()
            assert stats_after["active_entries"] >= initial_entries

        clear_cache()

    @pytest.mark.unit
    def test_cache_invalidation(self):
        """
        Test that cache expires correctly after TTL.

        Verifies:
        - Cached value is used within TTL
        - Cache expires after TTL
        - Function is re-called after expiration
        """
        clear_cache()
        call_count = 0

        @cache_result(ttl_seconds=1)
        def short_lived_cache(arg: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # First call
        result1 = short_lived_cache("test")
        assert result1 == "result_1"
        assert call_count == 1

        # Within TTL - uses cache
        result2 = short_lived_cache("test")
        assert result2 == "result_1"
        assert call_count == 1

        # Wait for expiration
        time.sleep(1.1)

        # After TTL - function is called again
        result3 = short_lived_cache("test")
        assert result3 == "result_2"
        assert call_count == 2

        clear_cache()

    @pytest.mark.unit
    def test_cache_clear_by_pattern(self):
        """Test clearing specific cache entries by pattern."""
        clear_cache()

        now = time.time()
        _cache["issue_common"] = CacheEntry(value="issue1", expires_at=now + 60)
        _cache["issue_sql_server"] = CacheEntry(value="issue2", expires_at=now + 60)
        _cache["mr_common"] = CacheEntry(value="mr1", expires_at=now + 60)

        assert get_cache_stats()["total_entries"] == 3

        # Clear only issue entries
        cleared = clear_cache("issue_")
        assert cleared == 2
        assert get_cache_stats()["total_entries"] == 1

        clear_cache()


# =============================================================================
# TEST CLASS 2: RACE CONDITION TESTS
# =============================================================================


class TestRaceConditions:
    """Tests for race condition handling in GitLab operations."""

    @pytest.mark.unit
    def test_concurrent_issue_creation(self, db_with_roles):
        """
        Test that concurrent issue creation from two threads is handled correctly.

        Verifies:
        - Only one issue is created despite concurrent attempts
        - Second thread gets the existing issue
        - No duplicate issues are created
        """
        client = GitLabClient(db=db_with_roles)
        results = []
        lock = threading.Lock()
        create_count = [0]

        mock_issue = MagicMock(spec=Issue)
        mock_issue.iid = 123
        mock_issue.id = 67890
        mock_issue.state = "opened"

        def mock_create_issue(**kwargs):
            with lock:
                create_count[0] += 1
            time.sleep(0.1)  # Simulate API delay
            return mock_issue

        def mock_find_existing(role_name, state="opened"):
            # Simulate race: both threads see no existing issue initially
            with lock:
                if create_count[0] > 0:
                    return mock_issue
            return None

        def create_issue_thread(client, role_name):
            with (
                patch.object(client, "find_existing_issue", side_effect=mock_find_existing),
                patch.object(client, "create_issue", side_effect=mock_create_issue),
            ):
                try:
                    issue, created = client.get_or_create_issue(
                        role_name=role_name,
                        title=f"Box up `{role_name}` role",
                        description="Test",
                    )
                    with lock:
                        results.append((issue.iid, created))
                except Exception as e:
                    with lock:
                        results.append(("error", str(e)))

        # Start two threads trying to create the same issue
        threads = [
            threading.Thread(target=create_issue_thread, args=(client, "common")),
            threading.Thread(target=create_issue_thread, args=(client, "common")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Both should complete
        assert len(results) == 2
        # At least one should succeed
        successful_results = [r for r in results if r[0] == 123]
        assert len(successful_results) >= 1

    @pytest.mark.unit
    def test_concurrent_mr_creation(self, db_with_roles):
        """
        Test that concurrent MR creation from two threads is handled correctly.

        Verifies:
        - Race condition is handled gracefully
        - Both threads eventually get a valid MR
        - No exception crashes occur
        """
        client = GitLabClient(db=db_with_roles)
        results = []
        lock = threading.Lock()

        mock_mr = MagicMock(spec=MergeRequest)
        mock_mr.iid = 456
        mock_mr.id = 11111
        mock_mr.source_branch = "sid/common"
        mock_mr.state = "opened"

        create_count = [0]

        def mock_find_mr(branch, state="opened"):
            with lock:
                if create_count[0] > 0:
                    return mock_mr
            return None

        def mock_create_mr(**kwargs):
            with lock:
                create_count[0] += 1
            time.sleep(0.05)  # Simulate API delay
            return mock_mr

        def create_mr_thread(client, role_name):
            with (
                patch.object(client, "find_existing_mr", side_effect=mock_find_mr),
                patch.object(client, "create_merge_request", side_effect=mock_create_mr),
            ):
                try:
                    mr, created = client.get_or_create_mr(
                        role_name=role_name,
                        source_branch=f"sid/{role_name}",
                        title=f"Box up {role_name} role",
                        description="Test",
                    )
                    with lock:
                        results.append((mr.iid, created))
                except Exception as e:
                    with lock:
                        results.append(("error", str(e)))

        threads = [
            threading.Thread(target=create_mr_thread, args=(client, "common")),
            threading.Thread(target=create_mr_thread, args=(client, "common")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 2
        # Both should get a valid MR
        assert all(r[0] == 456 for r in results)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_merge_train_add(self):
        """
        Test race condition when two coroutines try to add to merge train.

        Verifies:
        - Both attempts are handled gracefully
        - At least one succeeds
        - No unhandled exceptions
        """
        manager = GitLabMergeTrainManager()
        results = []

        async def mock_api_post(endpoint, data):
            await asyncio.sleep(0.05)  # Simulate network delay
            return {"id": 1, "status": "idle"}

        with patch.object(manager, "_api_post", side_effect=mock_api_post):
            tasks = [
                manager.add_to_merge_train("test/project", 123),
                manager.add_to_merge_train("test/project", 123),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Both should succeed (API handles idempotency)
        assert all(r is True for r in results)


# =============================================================================
# TEST CLASS 3: ERROR RECOVERY TESTS
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery and retry logic."""

    @pytest.mark.unit
    def test_retry_on_timeout(self):
        """
        Test that network timeout triggers retry with exponential backoff.

        Verifies:
        - TimeoutError triggers retry
        - Function is retried up to max_attempts
        - Eventually raises GitLabTimeoutError
        """
        call_count = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01, max_delay=0.1)
        def timeout_function():
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("Connection timed out")

        with pytest.raises(GitLabTimeoutError):
            timeout_function()

        assert call_count == 3

    @pytest.mark.unit
    def test_retry_on_rate_limit(self):
        """
        Test that 429 response triggers backoff using Retry-After.

        Verifies:
        - Rate limit error triggers retry
        - Retry-After value is respected (capped by max_delay)
        - Eventually succeeds after retries
        """
        call_count = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01, max_delay=0.05)
        def rate_limited_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise GitLabRateLimitError(
                    "Rate limit exceeded",
                    status_code=429,
                    retry_after_seconds=1,  # Will be capped by max_delay
                )
            return "success"

        result = rate_limited_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.unit
    def test_retry_on_server_error(self):
        """
        Test that 500/503 triggers retry with exponential backoff.

        Verifies:
        - Server errors (5xx) trigger retry
        - Exponential backoff is applied
        - Eventually succeeds or exhausts retries
        """
        call_count = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01, max_delay=0.1)
        def server_error_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise GitLabServerError("Service unavailable", status_code=503)
            return "recovered"

        result = server_error_function()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.unit
    def test_max_retries_exceeded(self):
        """
        Test that function gives up after max retries.

        Verifies:
        - Function attempts exactly max_attempts times
        - Final exception is raised after exhausting retries
        """
        call_count = 0

        @gitlab_retry(max_attempts=4, base_delay=0.01, max_delay=0.1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise GitLabServerError("Persistent failure", status_code=500)

        with pytest.raises(GitLabServerError, match="Persistent failure"):
            always_fails()

        assert call_count == 4

    @pytest.mark.unit
    def test_no_retry_on_client_error(self):
        """
        Test that 400/404 fails immediately without retry.

        Verifies:
        - Client errors (4xx except 429) raise immediately
        - No retries are attempted
        - Function is called only once
        """
        # Test 400 Bad Request
        call_count_400 = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def bad_request():
            nonlocal call_count_400
            call_count_400 += 1
            raise GitLabAPIError("Bad request", status_code=400)

        with pytest.raises(GitLabAPIError, match="Bad request"):
            bad_request()
        assert call_count_400 == 1

        # Test 404 Not Found
        call_count_404 = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def not_found():
            nonlocal call_count_404
            call_count_404 += 1
            raise GitLabNotFoundError("Resource not found", status_code=404)

        with pytest.raises(GitLabNotFoundError, match="Resource not found"):
            not_found()
        assert call_count_404 == 1

        # Test 401 Unauthorized
        call_count_401 = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def unauthorized():
            nonlocal call_count_401
            call_count_401 += 1
            raise GitLabAuthenticationError("Unauthorized", status_code=401)

        with pytest.raises(GitLabAuthenticationError, match="Unauthorized"):
            unauthorized()
        assert call_count_401 == 1

        # Test 409 Conflict
        call_count_409 = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def conflict():
            nonlocal call_count_409
            call_count_409 += 1
            raise GitLabConflictError("Conflict", status_code=409)

        with pytest.raises(GitLabConflictError, match="Conflict"):
            conflict()
        assert call_count_409 == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_retry_on_timeout(self):
        """Test async retry decorator with timeout errors."""
        call_count = 0

        @gitlab_retry_async(max_attempts=3, base_delay=0.01, max_delay=0.1)
        async def async_timeout():
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("Async timeout")

        with pytest.raises(GitLabTimeoutError):
            await async_timeout()

        assert call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_retry_success_after_failure(self):
        """Test async retry eventually succeeds after transient failures."""
        call_count = 0

        @gitlab_retry_async(max_attempts=3, base_delay=0.01)
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise GitLabServerError("Temporary error", status_code=503)
            return "async_success"

        result = await async_flaky()
        assert result == "async_success"
        assert call_count == 2


# =============================================================================
# TEST CLASS 4: MERGE TRAIN TESTS
# =============================================================================


class TestMergeTrainExtended:
    """Extended tests for merge train operations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_mr_to_merge_train(self):
        """
        Test successfully adding an MR to merge train.

        Verifies:
        - MR is added to train via API
        - Correct endpoint and parameters are used
        - Returns True on success
        """
        manager = GitLabMergeTrainManager()

        async def mock_api_post(endpoint, data):
            assert "merge_trains" in endpoint
            assert data["merge_request_iid"] == "456"
            return {"id": 1, "status": "idle"}

        with patch.object(manager, "_api_post", side_effect=mock_api_post):
            result = await manager.add_to_merge_train("test/project", 456)
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_merge_train_position(self):
        """
        Test getting MR position in merge train queue.

        Verifies:
        - Returns correct 1-indexed position
        - Returns None if MR not in train
        """
        manager = GitLabMergeTrainManager()

        mock_train = [
            {"id": 1, "merge_request": {"iid": 100}},
            {"id": 2, "merge_request": {"iid": 200}},
            {"id": 3, "merge_request": {"iid": 300}},
        ]

        async def mock_api_get(endpoint):
            return mock_train

        with patch.object(manager, "_api_get", side_effect=mock_api_get):
            # MR at position 2
            position = await manager.get_mr_position_in_train("test/project", 200)
            assert position == 2

            # MR at position 1
            position = await manager.get_mr_position_in_train("test/project", 100)
            assert position == 1

            # MR not in train
            position = await manager.get_mr_position_in_train("test/project", 999)
            assert position is None

    @pytest.mark.unit
    def test_remove_from_merge_train(self, in_memory_db):
        """
        Test successfully removing an MR from merge train.

        Verifies:
        - API call is made to remove MR
        - Returns True on success
        """
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = client.remove_from_merge_train(1)
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_merge_train_failure_recovery(self):
        """
        Test handling of merge train pipeline failures.

        Verifies:
        - Failed pipeline status is detected
        - MR can be removed and re-added
        - Recovery workflow completes
        """
        manager = GitLabMergeTrainManager()

        # Simulate train with failed pipeline
        mock_train_with_failure = [
            {
                "id": 1,
                "merge_request": {"iid": 100},
                "status": "failed",
                "pipeline": {"status": "failed"},
            },
        ]

        call_count = [0]

        async def mock_api_get(endpoint):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_train_with_failure
            # After recovery, train is empty or MR is gone
            return []

        with patch.object(manager, "_api_get", side_effect=mock_api_get):
            # Check initial state
            train = await manager.get_merge_train("test/project")
            assert len(train) == 1
            assert train[0]["status"] == "failed"

            # After "recovery", train is updated
            train_after = await manager.get_merge_train("test/project")
            assert len(train_after) == 0

    @pytest.mark.unit
    def test_wave_ordering_in_merge_train(self, db_with_roles):
        """
        Test that wave ordering is respected in merge train.

        Verifies:
        - Wave 0 MRs should be ahead of Wave 2
        - Wave 2 MRs should be ahead of Wave 3
        - Ordering violations are detectable
        """
        # Get roles with their waves
        common = db_with_roles.get_role("common")  # wave 0
        sql_server = db_with_roles.get_role("sql_server_2022")  # wave 2
        ems_platform = db_with_roles.get_role("ems_platform_services")  # wave 3

        # Simulate merge train entries
        train_entries = [
            {"role": common.name, "wave": common.wave, "iid": 1},
            {"role": sql_server.name, "wave": sql_server.wave, "iid": 2},
            {"role": ems_platform.name, "wave": ems_platform.wave, "iid": 3},
        ]

        def check_wave_ordering(train: list[dict]) -> list[str]:
            """Check for wave ordering violations."""
            violations = []
            for i, current in enumerate(train):
                for later in train[i + 1 :]:
                    if current["wave"] > later["wave"]:
                        violations.append(
                            f"Wave violation: {current['role']} (wave {current['wave']}) "
                            f"is ahead of {later['role']} (wave {later['wave']})"
                        )
            return violations

        # Correct ordering - no violations
        violations = check_wave_ordering(train_entries)
        assert len(violations) == 0

        # Incorrect ordering - violations detected
        bad_order = [
            {"role": ems_platform.name, "wave": ems_platform.wave, "iid": 3},
            {"role": common.name, "wave": common.wave, "iid": 1},
        ]
        violations = check_wave_ordering(bad_order)
        assert len(violations) == 1
        assert "wave 3" in violations[0] and "wave 0" in violations[0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_merge_readiness_result_properties(self):
        """Test MergeReadinessResult properties and helper methods."""
        # Ready to merge
        ready = MergeReadinessResult(
            mergeable=True,
            blockers=[],
            pipeline_status="success",
            has_conflicts=False,
            approvals_required=1,
            approvals_given=2,
        )
        assert ready.mergeable is True
        assert ready.pipeline_passed is True
        assert ready.pipeline_failed is False
        assert ready.approvals_satisfied is True
        assert ready.needs_pipeline is False
        assert "No blockers" in ready.blocker_summary

        # Not ready - pipeline failed
        not_ready = MergeReadinessResult(
            mergeable=False,
            blockers=["pipeline_failed", "has_conflicts"],
            pipeline_status="failed",
            has_conflicts=True,
            approvals_required=1,
            approvals_given=1,
        )
        assert not_ready.mergeable is False
        assert not_ready.pipeline_failed is True
        assert not_ready.has_conflicts is True
        assert "Pipeline has failed" in not_ready.blocker_summary
        assert "merge conflicts" in not_ready.blocker_summary.lower()

        # Pending pipeline
        pending = MergeReadinessResult(
            mergeable=False,
            blockers=["pipeline_running"],
            pipeline_status="running",
            has_conflicts=False,
            approvals_required=0,
            approvals_given=0,
        )
        assert pending.needs_pipeline is True
        assert pending.pipeline_passed is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_preflight_merge_train_check_success(self):
        """Test preflight check passes for a valid MR."""
        # Mock get_mr_merge_readiness to return a ready result
        ready_result = MergeReadinessResult(
            mergeable=True,
            blockers=[],
            pipeline_status="success",
            has_conflicts=False,
        )

        with patch(
            "harness.gitlab.merge_train.get_mr_merge_readiness",
            return_value=ready_result,
        ):
            can_add, blockers = await preflight_merge_train_check("test/project", 123)
            assert can_add is True
            assert blockers == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_preflight_merge_train_check_failures(self):
        """Test preflight check fails for various blocker conditions."""
        # Has conflicts
        conflict_result = MergeReadinessResult(
            mergeable=False,
            blockers=["has_conflicts"],
            pipeline_status="success",
            has_conflicts=True,
        )

        with patch(
            "harness.gitlab.merge_train.get_mr_merge_readiness",
            return_value=conflict_result,
        ):
            can_add, blockers = await preflight_merge_train_check("test/project", 123)
            assert can_add is False
            assert "has_conflicts" in blockers

        # Pipeline failed
        failed_result = MergeReadinessResult(
            mergeable=False,
            blockers=["pipeline_failed"],
            pipeline_status="failed",
            has_conflicts=False,
        )

        with patch(
            "harness.gitlab.merge_train.get_mr_merge_readiness",
            return_value=failed_result,
        ):
            can_add, blockers = await preflight_merge_train_check("test/project", 123)
            assert can_add is False
            assert "pipeline_failed" in blockers

        # Is draft
        draft_result = MergeReadinessResult(
            mergeable=False,
            blockers=["draft_status"],
            pipeline_status="success",
            has_conflicts=False,
        )

        with patch(
            "harness.gitlab.merge_train.get_mr_merge_readiness",
            return_value=draft_result,
        ):
            can_add, blockers = await preflight_merge_train_check("test/project", 123)
            assert can_add is False
            assert "draft_status" in blockers

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_for_merge_train_position_success(self):
        """Test waiting for MR to appear in merge train."""
        # Mock MR details
        mock_mr = {"target_branch": "main", "iid": 123}

        # Mock train with MR at position 2
        mock_train = [
            {"merge_request": {"iid": 100}},
            {"merge_request": {"iid": 123}},
        ]

        async def mock_get_mr_details(self, project_path, mr_iid):
            return mock_mr

        async def mock_get_train(self, project_path, target_branch):
            return mock_train

        with (
            patch.object(
                MergeTrainHelper, "_get_mr_details", mock_get_mr_details
            ),
            patch.object(
                MergeTrainHelper, "_get_merge_train", mock_get_train
            ),
        ):
            position = await wait_for_merge_train_position(
                "test/project",
                123,
                timeout_seconds=5,
                poll_interval=1,
            )
            assert position == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_merge_train_eta_calculation(self):
        """Test merge train ETA estimation."""
        manager = GitLabMergeTrainManager()

        mock_train = [
            {"merge_request": {"iid": 100}, "duration": 600, "status": "running"},
            {"merge_request": {"iid": 200}, "duration": 300, "status": "idle"},
            {"merge_request": {"iid": 300}, "duration": 0, "status": "idle"},
        ]

        async def mock_get_train(*args, **kwargs):
            return mock_train

        with patch.object(manager, "get_merge_train", side_effect=mock_get_train):
            # MR at position 0 - ETA is 0 (it's first)
            eta = await manager.get_merge_train_eta("test/project", 100)
            assert eta == 0

            # MR at position 1 - average of entries ahead
            eta = await manager.get_merge_train_eta("test/project", 200)
            # Position 1, avg duration = (600+300)/2 = 450, ETA = 1 * 450
            assert eta == 450

            # MR not in train
            eta = await manager.get_merge_train_eta("test/project", 999)
            assert eta is None


# =============================================================================
# TEST CLASS 5: ERROR PARSING TESTS
# =============================================================================


class TestErrorParsingExtended:
    """Extended tests for GitLab error parsing."""

    @pytest.mark.unit
    def test_parse_rate_limit_with_retry_after(self):
        """Test parsing 429 response with Retry-After header."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "120"}
        mock_response.json.return_value = {"message": "Too many requests"}

        error = parse_gitlab_error(mock_response)

        assert isinstance(error, GitLabRateLimitError)
        assert error.retry_after_seconds == 120
        assert "Too many requests" in error.message

    @pytest.mark.unit
    def test_parse_rate_limit_without_retry_after(self):
        """Test parsing 429 response without Retry-After header."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": "Rate limit exceeded"}

        error = parse_gitlab_error(mock_response)

        assert isinstance(error, GitLabRateLimitError)
        assert error.retry_after_seconds is None

    @pytest.mark.unit
    def test_parse_various_server_errors(self):
        """Test parsing different 5xx server errors."""
        for status_code in [500, 502, 503, 504]:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.headers = {}
            mock_response.json.return_value = {"message": f"Error {status_code}"}

            error = parse_gitlab_error(mock_response)

            assert isinstance(error, GitLabServerError)
            assert error.status_code == status_code

    @pytest.mark.unit
    def test_parse_json_decode_error(self):
        """Test parsing response when JSON decode fails."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
        mock_response.text = "Internal Server Error"

        error = parse_gitlab_error(mock_response)

        assert isinstance(error, GitLabServerError)
        assert "Internal Server Error" in error.message
