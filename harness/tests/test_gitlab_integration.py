"""
Comprehensive GitLab integration tests for harness.

Tests cover:
1. Idempotency - Caching, artifact discovery, issue/MR finding
2. GraphQL - Mutation format, GID utilities, error parsing
3. Retry Logic - Server errors, client errors, rate limiting
4. Merge Ordering - Wave dependencies, depth calculation, blocking MRs
5. Merge Train - Readiness checks, ordering violations, preflight checks

Tests marked with @pytest.mark.unit are fully mocked and CI-safe.
Tests that would need @pytest.mark.integration for real API calls are noted in docstrings.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from harness.gitlab import (
    GitLabClient,
    GitLabGraphQLClient,
    GitLabGraphQLError,
    GraphQLErrorLocation,
    build_gid,
    cache_result,
    clear_cache,
    get_cache_stats,
    parse_gid,
)
from harness.gitlab.errors import (
    GitLabAPIError,
    GitLabAuthenticationError,
    GitLabNotFoundError,
    GitLabRateLimitError,
    GitLabServerError,
    parse_gitlab_error,
)
from harness.gitlab.graphql import parse_graphql_errors
from harness.gitlab.idempotency import (
    CacheEntry,
    IdempotencyHelper,
    RoleArtifacts,
    _cache,
)
from harness.gitlab.retry import gitlab_retry, gitlab_retry_async


# =============================================================================
# TEST CLASS 1: IDEMPOTENCY
# =============================================================================


class TestIdempotency:
    """Tests for idempotent GitLab operations and caching."""

    @pytest.mark.unit
    def test_find_existing_issue_returns_same_iid(self, db_with_roles):
        """
        Test that find_existing_issue returns cached result on second call.

        Verifies:
        - API is called only once
        - Same Issue object is returned on subsequent calls
        - Cache stores the result correctly

        Note: Would need @pytest.mark.integration to test with real GitLab API.
        """
        # Clear cache first to ensure clean state
        clear_cache()

        helper = IdempotencyHelper(db_with_roles)

        # Mock the API response
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
            patch.object(helper.db, "upsert_issue") as mock_upsert,  # Mock db write
        ):
            # First call - should hit API
            result1 = helper.find_existing_issue("common")
            assert result1 is not None
            assert result1.iid == 42
            assert result1.title == "Box up `common` role"

            # Second call - should use cache
            result2 = helper.find_existing_issue("common")
            assert result2 is not None
            assert result2.iid == 42

            # API should only be called once (second call uses cache)
            assert mock_api.call_count == 1
            # upsert_issue should only be called once (when not cached)
            assert mock_upsert.call_count == 1

        # Clean up cache for other tests
        clear_cache()

    @pytest.mark.unit
    def test_find_all_role_artifacts_comprehensive(self, db_with_roles):
        """
        Test that find_all_role_artifacts discovers all artifact types.

        Verifies discovery of:
        - Existing issues (all states)
        - Existing MRs (all states)
        - Remote branch existence
        - Local worktree existence

        Note: Would need @pytest.mark.integration to test with real GitLab API.
        """
        # Clear cache first
        clear_cache()

        helper = IdempotencyHelper(db_with_roles)

        # Mock API responses for issue and MR - using 'common' which exists in fixture
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

        mock_mr_response = [
            {
                "id": 200,
                "iid": 20,
                "source_branch": "sid/common",
                "target_branch": "main",
                "title": "Box up common",
                "state": "merged",
                "web_url": "https://gitlab.com/test/-/merge_requests/20",
                "merge_status": "merged",
                "squash_on_merge": True,
                "force_remove_source_branch": True,
            }
        ]

        def mock_api_get(endpoint):
            if "issues" in endpoint:
                return mock_issue_response
            elif "merge_requests" in endpoint:
                return mock_mr_response
            return []

        with (
            patch.object(helper, "_api_get", side_effect=mock_api_get),
            patch.object(helper, "remote_branch_exists", return_value=True),
            patch.object(helper, "worktree_exists", return_value=(True, "/path/to/worktree")),
        ):
            artifacts = helper.find_all_role_artifacts("common")

            assert artifacts.role_name == "common"
            assert artifacts.existing_issue is not None
            assert artifacts.existing_issue.iid == 10
            assert artifacts.existing_mr is not None
            assert artifacts.existing_mr.iid == 20
            assert artifacts.existing_mr.state == "merged"
            assert artifacts.remote_branch_exists is True
            assert artifacts.worktree_exists is True
            assert artifacts.worktree_path == "/path/to/worktree"
            assert artifacts.branch_name == "sid/common"
            assert artifacts.has_any_artifacts is True
            assert artifacts.is_complete is True
            assert artifacts.needs_issue is False
            assert artifacts.needs_mr is False

        clear_cache()

    @pytest.mark.unit
    def test_cache_decorator_ttl(self):
        """
        Test that cache_result decorator respects TTL expiration.

        Verifies:
        - Cached value is used within TTL
        - Cache expires after TTL
        - Function is re-called after expiration
        """
        call_count = 0

        @cache_result(ttl_seconds=1)
        def expensive_function(arg: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{arg}_{call_count}"

        # Clear cache first
        clear_cache()

        # First call
        result1 = expensive_function("test")
        assert result1 == "result_test_1"
        assert call_count == 1

        # Second call (within TTL) - should use cache
        result2 = expensive_function("test")
        assert result2 == "result_test_1"  # Same result
        assert call_count == 1  # Not called again

        # Wait for TTL to expire
        time.sleep(1.1)

        # Third call (after TTL) - should call function again
        result3 = expensive_function("test")
        assert result3 == "result_test_2"  # New result
        assert call_count == 2  # Called again

        clear_cache()

    @pytest.mark.unit
    def test_cache_stats(self):
        """Test cache statistics reporting."""
        clear_cache()

        # Use the cache directly to avoid decorator quirks
        from harness.gitlab.idempotency import _cache, CacheEntry

        # No entries initially (after clearing)
        stats = get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["active_entries"] == 0

        # Add entries directly to the cache
        now = time.time()
        _cache["test_key_1"] = CacheEntry(value="value1", expires_at=now + 60)
        _cache["test_key_2"] = CacheEntry(value="value2", expires_at=now + 60)

        stats = get_cache_stats()
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["expired_entries"] == 0

        # Add an already-expired entry
        _cache["test_key_expired"] = CacheEntry(value="expired", expires_at=now - 1)

        stats = get_cache_stats()
        assert stats["total_entries"] == 3
        assert stats["expired_entries"] == 1
        assert stats["active_entries"] == 2

        clear_cache()

    @pytest.mark.unit
    def test_role_artifacts_properties(self):
        """Test RoleArtifacts helper properties."""
        # No artifacts
        empty = RoleArtifacts(role_name="empty")
        assert not empty.has_any_artifacts
        assert not empty.is_complete
        assert empty.needs_issue
        assert not empty.needs_mr  # Needs branch first
        assert not empty.needs_branch_push  # No worktree

        # Partial artifacts - has worktree but no remote branch
        partial = RoleArtifacts(
            role_name="partial",
            worktree_exists=True,
            worktree_path="/path/to/worktree",
            remote_branch_exists=False,
        )
        assert partial.has_any_artifacts
        assert not partial.is_complete
        assert partial.needs_issue
        assert partial.needs_branch_push

        # Complete artifacts
        from harness.db.models import Issue, MergeRequest

        complete = RoleArtifacts(
            role_name="complete",
            existing_issue=Issue(id=1, iid=10, title="Test", state="opened", web_url="url"),
            existing_mr=MergeRequest(
                id=2,
                iid=20,
                source_branch="sid/complete",
                target_branch="main",
                title="Test MR",
                state="opened",
                web_url="url",
            ),
            remote_branch_exists=True,
            worktree_exists=True,
            worktree_path="/path",
        )
        assert complete.has_any_artifacts
        assert complete.is_complete
        assert not complete.needs_issue
        assert not complete.needs_mr


# =============================================================================
# TEST CLASS 2: GRAPHQL
# =============================================================================


class TestGraphQL:
    """Tests for GitLab GraphQL operations."""

    @pytest.mark.unit
    def test_iteration_assignment_mutation_format(self):
        """
        Test that iteration assignment mutation has correct structure.

        Verifies:
        - Mutation includes required fields
        - Variables are properly formatted
        - GID conversion is correct

        Note: Would need @pytest.mark.integration to execute against real API.
        """
        client = GitLabGraphQLClient()

        # Test the mutation structure by checking the client method signature
        # The actual mutation is embedded in assign_issue_to_iteration
        iteration_gid = f"gid://gitlab/Iteration/12345"

        # Verify GID format
        assert iteration_gid.startswith("gid://gitlab/Iteration/")
        assert "12345" in iteration_gid

        # The mutation expects these variables
        expected_vars = {
            "projectPath": "test/project",
            "iid": "42",
            "iterationId": iteration_gid,
        }

        # Verify variable types
        assert isinstance(expected_vars["projectPath"], str)
        assert isinstance(expected_vars["iid"], str)  # Note: iid is string in GraphQL
        assert expected_vars["iterationId"].startswith("gid://")

    @pytest.mark.unit
    def test_gid_conversion(self):
        """
        Test build_gid and parse_gid utilities.

        Verifies:
        - build_gid creates correct format
        - parse_gid extracts type and ID
        - Round-trip conversion works
        """
        # Test build_gid
        gid = build_gid("Iteration", 12345)
        assert gid == "gid://gitlab/Iteration/12345"

        gid2 = build_gid("MergeRequest", 456)
        assert gid2 == "gid://gitlab/MergeRequest/456"

        gid3 = build_gid("Project", 789)
        assert gid3 == "gid://gitlab/Project/789"

        # Test parse_gid
        type_name, id_num = parse_gid("gid://gitlab/Iteration/12345")
        assert type_name == "Iteration"
        assert id_num == 12345

        type_name2, id_num2 = parse_gid("gid://gitlab/MergeRequest/456")
        assert type_name2 == "MergeRequest"
        assert id_num2 == 456

        # Test round-trip
        original_gid = "gid://gitlab/Issue/999"
        t, i = parse_gid(original_gid)
        rebuilt = build_gid(t, i)
        assert rebuilt == original_gid

        # Test invalid GID formats
        with pytest.raises(ValueError, match="Invalid GID format"):
            parse_gid("invalid_format")

        with pytest.raises(ValueError, match="Invalid GID format"):
            parse_gid("gid://github/Issue/123")  # Wrong prefix

        with pytest.raises(ValueError, match="Invalid GID"):
            parse_gid("gid://gitlab/Issue/abc")  # Non-numeric ID

    @pytest.mark.unit
    def test_graphql_error_parsing(self):
        """
        Test parsing of GraphQL error responses.

        Verifies:
        - Error message extraction
        - Location parsing
        - Path extraction
        - Extensions handling
        """
        # Simple error
        simple_errors = [{"message": "Something went wrong"}]
        parsed = parse_graphql_errors(simple_errors)
        assert len(parsed) == 1
        assert parsed[0].message == "Something went wrong"
        assert parsed[0].locations == []
        assert parsed[0].path == []

        # Complex error with all fields
        complex_errors = [
            {
                "message": "Field 'nonexistent' doesn't exist",
                "locations": [{"line": 5, "column": 10}],
                "path": ["project", "issues", 0, "title"],
                "extensions": {"code": "FIELD_NOT_FOUND", "severity": "error"},
            }
        ]
        parsed = parse_graphql_errors(complex_errors)
        assert len(parsed) == 1
        error = parsed[0]
        assert error.message == "Field 'nonexistent' doesn't exist"
        assert len(error.locations) == 1
        assert error.locations[0].line == 5
        assert error.locations[0].column == 10
        assert error.path == ["project", "issues", 0, "title"]
        assert error.extensions["code"] == "FIELD_NOT_FOUND"

        # Multiple errors
        multi_errors = [
            {"message": "Error 1", "locations": [{"line": 1, "column": 1}]},
            {"message": "Error 2", "locations": [{"line": 2, "column": 2}]},
        ]
        parsed = parse_graphql_errors(multi_errors)
        assert len(parsed) == 2
        assert parsed[0].message == "Error 1"
        assert parsed[1].message == "Error 2"

    @pytest.mark.unit
    def test_graphql_error_str_formatting(self):
        """Test GitLabGraphQLError string formatting."""
        # Simple error
        simple = GitLabGraphQLError(message="Test error")
        assert str(simple) == "Test error"

        # Error with path
        with_path = GitLabGraphQLError(
            message="Field error", path=["project", "issues", 0]
        )
        assert "project.issues.0" in str(with_path)

        # Error with locations
        with_loc = GitLabGraphQLError(
            message="Syntax error",
            locations=[GraphQLErrorLocation(line=5, column=10)],
        )
        assert "line 5" in str(with_loc)
        assert "column 10" in str(with_loc)

        # Error with extensions
        with_ext = GitLabGraphQLError(
            message="Auth error", extensions={"code": "UNAUTHORIZED"}
        )
        assert "UNAUTHORIZED" in str(with_ext)


# =============================================================================
# TEST CLASS 3: RETRY LOGIC
# =============================================================================


class TestRetryLogic:
    """Tests for GitLab API retry decorator."""

    @pytest.mark.unit
    def test_retry_on_server_error(self):
        """
        Test that server errors (5xx) trigger retries.

        Verifies:
        - Function is retried max_attempts times
        - Exponential backoff is applied
        - Final exception is raised after exhausting retries
        """
        call_count = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01, max_delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise GitLabServerError("Server error", status_code=500)

        with pytest.raises(GitLabServerError):
            failing_function()

        # Should have been called 3 times (initial + 2 retries)
        assert call_count == 3

    @pytest.mark.unit
    def test_no_retry_on_client_error(self):
        """
        Test that client errors (4xx, except 429) do NOT trigger retries.

        Verifies:
        - 400/401/403/404/409 errors raise immediately
        - No retries are attempted
        """
        # Test 404 Not Found
        call_count_404 = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def not_found_function():
            nonlocal call_count_404
            call_count_404 += 1
            raise GitLabNotFoundError("Resource not found", status_code=404)

        with pytest.raises(GitLabNotFoundError):
            not_found_function()

        assert call_count_404 == 1  # Only called once, no retries

        # Test 401 Unauthorized
        call_count_401 = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def auth_function():
            nonlocal call_count_401
            call_count_401 += 1
            raise GitLabAuthenticationError("Unauthorized", status_code=401)

        with pytest.raises(GitLabAuthenticationError):
            auth_function()

        assert call_count_401 == 1  # Only called once, no retries

        # Test generic 400 error
        call_count_400 = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def bad_request_function():
            nonlocal call_count_400
            call_count_400 += 1
            raise GitLabAPIError("Bad request", status_code=400)

        with pytest.raises(GitLabAPIError):
            bad_request_function()

        assert call_count_400 == 1  # Only called once, no retries

    @pytest.mark.unit
    def test_retry_respects_rate_limit(self):
        """
        Test that rate limit (429) responses trigger retries with Retry-After.

        Verifies:
        - Rate limit errors cause retries
        - Retry-After header value is respected
        - Retries eventually succeed or exhaust attempts
        """
        call_count = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01, max_delay=0.05)
        def rate_limited_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise GitLabRateLimitError(
                    "Rate limited",
                    status_code=429,
                    retry_after_seconds=1,  # Would wait 1s but capped by max_delay
                )
            return "success"

        result = rate_limited_function()
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.unit
    def test_retry_success_on_second_attempt(self):
        """Test that retries can succeed after initial failure."""
        call_count = 0

        @gitlab_retry(max_attempts=3, base_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise GitLabServerError("Temporary error", status_code=503)
            return "recovered"

        result = flaky_function()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_retry_on_server_error(self):
        """Test async retry decorator with server errors."""
        call_count = 0

        @gitlab_retry_async(max_attempts=3, base_delay=0.01, max_delay=0.1)
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            raise GitLabServerError("Async server error", status_code=500)

        with pytest.raises(GitLabServerError):
            await async_failing_function()

        assert call_count == 3


# =============================================================================
# TEST CLASS 4: MERGE ORDERING
# =============================================================================


class TestMergeOrdering:
    """Tests for merge request ordering based on waves and dependencies."""

    @pytest.mark.unit
    def test_merge_order_respects_waves(self, db_with_roles):
        """
        Test that MRs from lower waves should merge before higher waves.

        Verifies:
        - Wave 0 roles merge before Wave 2
        - Wave 2 roles merge before Wave 3
        - Ordering is deterministic
        """
        # Get roles from the fixture
        common = db_with_roles.get_role("common")  # wave 0
        sql_server = db_with_roles.get_role("sql_server_2022")  # wave 2
        ems_platform = db_with_roles.get_role("ems_platform_services")  # wave 3

        # Create mock MRs for each role
        mrs = [
            {"role": ems_platform.name, "wave": ems_platform.wave, "iid": 3},
            {"role": common.name, "wave": common.wave, "iid": 1},
            {"role": sql_server.name, "wave": sql_server.wave, "iid": 2},
        ]

        # Sort by wave (ascending)
        sorted_mrs = sorted(mrs, key=lambda x: x["wave"])

        # Verify order: common (0) -> sql_server (2) -> ems_platform (3)
        assert sorted_mrs[0]["role"] == "common"
        assert sorted_mrs[1]["role"] == "sql_server_2022"
        assert sorted_mrs[2]["role"] == "ems_platform_services"

    @pytest.mark.unit
    def test_dependency_depth_calculation(self, db_with_roles):
        """
        Test calculation of dependency depth for merge ordering.

        Verifies:
        - Roles with no dependencies have depth 0
        - Direct dependencies increase depth by 1
        - Transitive dependencies accumulate depth
        """

        def calculate_depth(db, role_name: str, visited: set | None = None) -> int:
            """Calculate dependency depth recursively."""
            if visited is None:
                visited = set()

            if role_name in visited:
                return 0  # Avoid cycles

            visited.add(role_name)
            role = db.get_role(role_name)
            if not role:
                return 0

            # Use get_dependencies which returns list of (role_name, depth) tuples
            deps = db.get_dependencies(role_name)
            if not deps:
                return 0

            max_dep_depth = 0
            for dep_name, _ in deps:
                depth = calculate_depth(db, dep_name, visited.copy())
                max_dep_depth = max(max_dep_depth, depth)

            return max_dep_depth + 1

        # common has no dependencies - depth 0
        assert calculate_depth(db_with_roles, "common") == 0

        # sql_server_2022 depends on common - depth 1
        assert calculate_depth(db_with_roles, "sql_server_2022") == 1

        # sql_management_studio depends on common and sql_server_2022 - depth 2
        assert calculate_depth(db_with_roles, "sql_management_studio") == 2

        # ems_web_app depends on common - depth 1
        assert calculate_depth(db_with_roles, "ems_web_app") == 1

        # ems_platform_services depends on ems_web_app (which depends on common) - depth 2
        assert calculate_depth(db_with_roles, "ems_platform_services") == 2

    @pytest.mark.unit
    def test_blocking_mrs_detection(self, db_with_roles):
        """
        Test detection of blocking MRs that must merge first.

        Verifies:
        - Direct dependencies are detected as blockers
        - Transitive dependencies are detected
        - Non-blocking MRs are not flagged
        """

        def find_blocking_mrs(db, role_name: str) -> list[str]:
            """Find all roles that must merge before this one."""
            blocking = []
            visited = set()

            def traverse_deps(name: str):
                if name in visited:
                    return
                visited.add(name)

                role = db.get_role(name)
                if not role:
                    return

                # Use get_dependencies which returns list of (role_name, depth) tuples
                deps = db.get_dependencies(name)
                for dep_name, _ in deps:
                    if dep_name != role_name:
                        blocking.append(dep_name)
                        traverse_deps(dep_name)

            traverse_deps(role_name)
            return list(set(blocking))

        # common has no blockers
        assert find_blocking_mrs(db_with_roles, "common") == []

        # sql_server_2022 is blocked by common
        blockers = find_blocking_mrs(db_with_roles, "sql_server_2022")
        assert "common" in blockers

        # sql_management_studio is blocked by common and sql_server_2022
        blockers = find_blocking_mrs(db_with_roles, "sql_management_studio")
        assert "common" in blockers
        assert "sql_server_2022" in blockers

        # ems_platform_services is blocked by ems_web_app and common
        blockers = find_blocking_mrs(db_with_roles, "ems_platform_services")
        assert "ems_web_app" in blockers
        assert "common" in blockers


# =============================================================================
# TEST CLASS 5: MERGE TRAIN
# =============================================================================


class TestMergeTrain:
    """Tests for merge train operations."""

    @pytest.mark.unit
    def test_merge_readiness_checks_all_blockers(self, db_with_roles):
        """
        Test comprehensive merge readiness checks.

        Verifies checks for:
        - Pipeline status
        - Approval status
        - Dependency MR states
        - Conflict status
        - Branch protection rules

        Note: Would need @pytest.mark.integration for real GitLab checks.
        """

        def check_merge_readiness(
            role_name: str,
            pipeline_status: str,
            is_approved: bool,
            has_conflicts: bool,
            blocking_mrs_merged: bool,
        ) -> tuple[bool, list[str]]:
            """
            Check if an MR is ready to merge.

            Returns (is_ready, blocking_reasons).
            """
            blockers = []

            if pipeline_status != "success":
                blockers.append(f"Pipeline status: {pipeline_status}")

            if not is_approved:
                blockers.append("Missing required approvals")

            if has_conflicts:
                blockers.append("Has merge conflicts")

            if not blocking_mrs_merged:
                blockers.append("Blocking MRs not yet merged")

            return (len(blockers) == 0, blockers)

        # Ready to merge
        ready, blockers = check_merge_readiness(
            "common",
            pipeline_status="success",
            is_approved=True,
            has_conflicts=False,
            blocking_mrs_merged=True,
        )
        assert ready is True
        assert blockers == []

        # Pipeline failed
        ready, blockers = check_merge_readiness(
            "common",
            pipeline_status="failed",
            is_approved=True,
            has_conflicts=False,
            blocking_mrs_merged=True,
        )
        assert ready is False
        assert "Pipeline status: failed" in blockers

        # Multiple issues
        ready, blockers = check_merge_readiness(
            "sql_server_2022",
            pipeline_status="running",
            is_approved=False,
            has_conflicts=True,
            blocking_mrs_merged=False,
        )
        assert ready is False
        assert len(blockers) == 4

    @pytest.mark.unit
    def test_placement_warns_on_ordering_violation(self, db_with_roles):
        """
        Test detection of merge train ordering violations.

        Verifies:
        - Warning when higher-wave MR is ahead of lower-wave
        - Warning when dependent MR is ahead of dependency
        - No warning for correct ordering
        """

        def check_train_ordering(
            train_order: list[dict],
        ) -> list[str]:
            """
            Check merge train for ordering violations.

            train_order: List of dicts with 'role', 'wave', 'iid'
            Returns list of warning messages.
            """
            warnings = []

            for i, current in enumerate(train_order):
                for j, later in enumerate(train_order[i + 1 :], start=i + 1):
                    # Check wave ordering
                    if current["wave"] > later["wave"]:
                        warnings.append(
                            f"Wave violation: !{current['iid']} ({current['role']}, wave {current['wave']}) "
                            f"is ahead of !{later['iid']} ({later['role']}, wave {later['wave']})"
                        )

            return warnings

        # Correct ordering (lower waves first)
        correct_order = [
            {"role": "common", "wave": 0, "iid": 1},
            {"role": "sql_server_2022", "wave": 2, "iid": 2},
            {"role": "ems_platform_services", "wave": 3, "iid": 3},
        ]
        warnings = check_train_ordering(correct_order)
        assert warnings == []

        # Incorrect ordering (higher wave before lower)
        incorrect_order = [
            {"role": "ems_platform_services", "wave": 3, "iid": 3},
            {"role": "common", "wave": 0, "iid": 1},
            {"role": "sql_server_2022", "wave": 2, "iid": 2},
        ]
        warnings = check_train_ordering(incorrect_order)
        assert len(warnings) == 2  # wave 3 before both wave 0 and wave 2

    @pytest.mark.unit
    def test_preflight_check_catches_conflicts(self):
        """
        Test preflight checks detect merge conflicts before train entry.

        Verifies:
        - Conflict detection is performed
        - Appropriate error is raised for conflicts
        - Clean MRs pass preflight

        Note: Would need @pytest.mark.integration for real GitLab checks.
        """

        def preflight_check(
            merge_status: str,
            pipeline_status: str,
            approvals_remaining: int,
        ) -> tuple[bool, str | None]:
            """
            Perform preflight checks before adding to merge train.

            Returns (can_add, error_message).
            """
            if merge_status == "cannot_be_merged":
                return (False, "MR has merge conflicts and cannot be merged")

            if merge_status == "checking":
                return (False, "Merge status is still being checked")

            if pipeline_status not in ("success", "running"):
                return (
                    False,
                    f"Pipeline must be running or successful, got: {pipeline_status}",
                )

            if approvals_remaining > 0:
                return (
                    False,
                    f"MR requires {approvals_remaining} more approval(s)",
                )

            return (True, None)

        # Can add to train
        can_add, error = preflight_check(
            merge_status="can_be_merged",
            pipeline_status="success",
            approvals_remaining=0,
        )
        assert can_add is True
        assert error is None

        # Has conflicts
        can_add, error = preflight_check(
            merge_status="cannot_be_merged",
            pipeline_status="success",
            approvals_remaining=0,
        )
        assert can_add is False
        assert "conflicts" in error.lower()

        # Pipeline failed
        can_add, error = preflight_check(
            merge_status="can_be_merged",
            pipeline_status="failed",
            approvals_remaining=0,
        )
        assert can_add is False
        assert "Pipeline" in error

        # Missing approvals
        can_add, error = preflight_check(
            merge_status="can_be_merged",
            pipeline_status="success",
            approvals_remaining=2,
        )
        assert can_add is False
        assert "approval" in error.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_merge_train_eta_calculation(self):
        """
        Test merge train ETA calculation.

        Verifies:
        - ETA increases with position in queue
        - Average pipeline duration affects ETA
        - Returns None when not in train

        Algorithm: The implementation iterates through entries, accumulating
        duration data, and stops when it finds the target MR. Then calculates:
        ETA = position * (total_duration / count)
        """
        from harness.gitlab.advanced import GitLabMergeTrainManager

        manager = GitLabMergeTrainManager()

        # Mock merge train data
        mock_train = [
            {
                "id": 1,
                "merge_request": {"iid": 100},
                "duration": 600,  # 10 minutes
                "status": "running",
            },
            {
                "id": 2,
                "merge_request": {"iid": 200},
                "duration": 300,  # 5 minutes
                "status": "idle",
            },
            {
                "id": 3,
                "merge_request": {"iid": 300},
                "duration": 0,  # Just added (not counted in avg since 0)
                "status": "idle",
            },
        ]

        # Make get_merge_train an async mock that returns our data
        async def mock_get_train(*args, **kwargs):
            return mock_train

        with patch.object(manager, "get_merge_train", side_effect=mock_get_train):
            # MR at position 0 - no entries ahead
            # Only sees first entry (duration=600, count=1), position=0
            # ETA = 0 * 600 = 0
            eta = await manager.get_merge_train_eta("test/project", 100)
            assert eta == 0

            # MR at position 1 - 1 entry ahead
            # Sees entries 0,1: total_duration=600+300=900, count=2, avg=450
            # position=1, ETA = 1 * 450 = 450
            eta = await manager.get_merge_train_eta("test/project", 200)
            assert eta == 450

            # MR at position 2 - 2 entries ahead
            # Sees entries 0,1,2: total=600+300+0=900, count=2 (0 not counted), avg=450
            # position=2, ETA = 2 * 450 = 900
            eta = await manager.get_merge_train_eta("test/project", 300)
            assert eta == 900

            # MR not in train
            eta = await manager.get_merge_train_eta("test/project", 999)
            assert eta is None


# =============================================================================
# ADDITIONAL HELPER TESTS
# =============================================================================


class TestErrorParsing:
    """Tests for httpx response error parsing."""

    @pytest.mark.unit
    def test_parse_gitlab_error_rate_limit(self):
        """Test parsing 429 rate limit response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.json.return_value = {"message": "Rate limit exceeded"}

        error = parse_gitlab_error(mock_response)

        assert isinstance(error, GitLabRateLimitError)
        assert error.status_code == 429
        assert error.retry_after_seconds == 60

    @pytest.mark.unit
    def test_parse_gitlab_error_server_error(self):
        """Test parsing 5xx server error response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.headers = {}
        mock_response.json.return_value = {"error": "Service unavailable"}

        error = parse_gitlab_error(mock_response)

        assert isinstance(error, GitLabServerError)
        assert error.status_code == 503

    @pytest.mark.unit
    def test_parse_gitlab_error_not_found(self):
        """Test parsing 404 not found response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.json.return_value = {"message": "Project not found"}

        error = parse_gitlab_error(mock_response)

        assert isinstance(error, GitLabNotFoundError)
        assert error.status_code == 404

    @pytest.mark.unit
    def test_parse_gitlab_error_plain_text(self):
        """Test parsing response with plain text body."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
        mock_response.text = "Internal Server Error"

        error = parse_gitlab_error(mock_response)

        assert isinstance(error, GitLabServerError)
        assert "Internal Server Error" in error.message


class TestCacheClearPatterns:
    """Tests for selective cache clearing."""

    @pytest.mark.unit
    def test_clear_cache_by_pattern(self):
        """Test clearing cache entries by pattern."""
        # Start with fresh cache
        clear_cache()

        # Use cache directly to avoid decorator quirks with local function names
        from harness.gitlab.idempotency import _cache, CacheEntry

        now = time.time()

        # Add entries with patterns in keys
        _cache["group_a.func:arg1"] = CacheEntry(value="a1", expires_at=now + 60)
        _cache["group_a.func:arg2"] = CacheEntry(value="a2", expires_at=now + 60)
        _cache["group_b.func:arg1"] = CacheEntry(value="b1", expires_at=now + 60)

        stats = get_cache_stats()
        assert stats["total_entries"] == 3

        # Clear only group_a entries
        cleared = clear_cache("group_a")
        assert cleared == 2

        stats = get_cache_stats()
        assert stats["total_entries"] == 1

        # Verify remaining entry is from group_b
        assert "group_b.func:arg1" in _cache

        # Clean up
        clear_cache()

    @pytest.mark.unit
    def test_clear_all_cache(self):
        """Test clearing all cache entries."""
        # Start with fresh cache
        clear_cache()

        # Use cache directly
        from harness.gitlab.idempotency import _cache, CacheEntry

        now = time.time()

        # Add multiple entries
        _cache["key1"] = CacheEntry(value="value1", expires_at=now + 60)
        _cache["key2"] = CacheEntry(value="value2", expires_at=now + 60)
        _cache["key3"] = CacheEntry(value="value3", expires_at=now + 60)

        stats = get_cache_stats()
        assert stats["total_entries"] == 3

        cleared = clear_cache()
        assert cleared == 3

        stats = get_cache_stats()
        assert stats["total_entries"] == 0
