"""
Tests for error_resolution module (v0.5.0 self-correction loops + v0.6.0 tiers).

Tests cover:
- Error classification logic
- Resolution strategy execution
- Recovery attempt tracking
- State update generation
- Tier selection (v0.6.0)
- Recovery memory functions (v0.6.0)
- should_attempt_recovery_v2 with RecoveryConfig (v0.6.0)
"""

import pytest

from harness.dag.error_resolution import (
    ClassifiedError,
    ErrorType,
    attempt_resolution,
    classify_error,
    create_recovery_state_update,
    get_recovery_attempt_count,
    select_recovery_tier,
    should_attempt_recovery,
    should_attempt_recovery_v2,
)


class TestErrorClassification:
    """Tests for classify_error function."""

    def test_worktree_exists_is_recoverable(self):
        """Worktree already exists should be recoverable."""
        state = {"role_name": "common", "worktree_path": "/tmp/worktree"}
        error = classify_error("Worktree already exists at /tmp/worktree", "create_worktree", state)

        assert error.error_type == ErrorType.RECOVERABLE
        assert error.resolution_hint == "remove_existing_worktree"
        assert error.context["role_name"] == "common"

    def test_branch_already_checked_out_is_recoverable(self):
        """Branch already checked out should be recoverable."""
        state = {"role_name": "common"}
        error = classify_error(
            "fatal: 'sid/common' is already checked out at '/tmp/other'",
            "create_worktree",
            state,
        )

        assert error.error_type == ErrorType.RECOVERABLE
        assert error.resolution_hint == "remove_existing_worktree"

    def test_branch_exists_is_recoverable(self):
        """Branch already exists should be recoverable."""
        state = {"role_name": "common"}
        error = classify_error(
            "fatal: A branch named 'sid/common' already exists",
            "create_worktree",
            state,
        )

        assert error.error_type == ErrorType.RECOVERABLE
        assert error.resolution_hint == "delete_existing_branch"

    def test_site_yml_not_found_is_recoverable(self):
        """Site.yml not found should be recoverable (check ansible/ subdir)."""
        state = {"role_name": "common", "worktree_path": "/tmp/worktree"}
        error = classify_error(
            "ERROR! the playbook: site.yml could not be found",
            "validate_deploy",
            state,
        )

        assert error.error_type == ErrorType.RECOVERABLE
        assert error.resolution_hint == "check_ansible_subdirectory"

    def test_syntax_error_is_user_fixable(self):
        """Actual syntax errors (not path issues) are user-fixable."""
        state = {"role_name": "common", "worktree_path": "/tmp/worktree"}
        error = classify_error(
            "Syntax check failed: YAML syntax error on line 42",
            "validate_deploy",
            state,
        )

        assert error.error_type == ErrorType.USER_FIXABLE
        assert error.resolution_hint == "fix_ansible_syntax"

    def test_test_failure_is_user_fixable(self):
        """Test failures should be user-fixable."""
        state = {"role_name": "common"}
        error = classify_error(
            "Molecule tests failed: assertion error in test_default.py",
            "run_molecule",
            state,
        )

        assert error.error_type == ErrorType.USER_FIXABLE
        assert error.resolution_hint == "fix_failing_tests"

    def test_timeout_is_transient(self):
        """Timeout errors should be transient."""
        state = {"role_name": "common"}
        error = classify_error("Connection timeout after 30 seconds", "any_node", state)

        assert error.error_type == ErrorType.TRANSIENT
        assert error.resolution_hint == "retry_with_backoff"

    def test_rate_limit_is_transient(self):
        """Rate limit errors should be transient."""
        state = {"role_name": "common"}
        error = classify_error("API rate limit exceeded (429)", "create_issue", state)

        assert error.error_type == ErrorType.TRANSIENT

    def test_unknown_error_is_unexpected(self):
        """Unknown errors should be classified as unexpected."""
        state = {"role_name": "common"}
        error = classify_error("Something completely unexpected happened", "any_node", state)

        assert error.error_type == ErrorType.UNEXPECTED
        assert error.resolution_hint is None


class TestAttemptResolution:
    """Tests for attempt_resolution function."""

    def test_check_ansible_subdirectory_success(self, tmp_path):
        """Resolution should find site.yml in ansible/ subdirectory."""
        ansible_dir = tmp_path / "ansible"
        ansible_dir.mkdir()
        (ansible_dir / "site.yml").touch()

        state = {"worktree_path": str(tmp_path)}
        result = attempt_resolution("check_ansible_subdirectory", {}, state)

        assert result is not None
        assert result["ansible_cwd"] == str(ansible_dir)

    def test_check_ansible_subdirectory_at_root(self, tmp_path):
        """Resolution should find site.yml at worktree root as fallback."""
        (tmp_path / "site.yml").touch()

        state = {"worktree_path": str(tmp_path)}
        result = attempt_resolution("check_ansible_subdirectory", {}, state)

        assert result is not None
        assert result["ansible_cwd"] == str(tmp_path)

    def test_check_ansible_subdirectory_not_found(self, tmp_path):
        """Resolution should return None if site.yml not found anywhere."""
        state = {"worktree_path": str(tmp_path)}
        result = attempt_resolution("check_ansible_subdirectory", {}, state)

        assert result is None

    def test_remove_existing_worktree_returns_flag(self):
        """Resolution should signal force recreate."""
        state = {"role_name": "common"}
        result = attempt_resolution("remove_existing_worktree", {"role_name": "common"}, state)

        assert result is not None
        assert result["worktree_force_recreate"] is True

    def test_delete_existing_branch_returns_flag(self):
        """Resolution should signal branch force recreate."""
        state = {"role_name": "common"}
        result = attempt_resolution("delete_existing_branch", {"branch": "sid/common"}, state)

        assert result is not None
        assert result["branch_force_recreate"] is True

    def test_unknown_hint_returns_none(self):
        """Unknown resolution hints should return None."""
        state = {"role_name": "common"}
        result = attempt_resolution("unknown_hint", {}, state)

        assert result is None


class TestRecoveryAttemptTracking:
    """Tests for recovery attempt counting and limiting."""

    def test_get_recovery_attempt_count_empty(self):
        """Should return 0 for no recovery attempts."""
        state = {"recovery_attempts": []}
        count = get_recovery_attempt_count(state, "create_worktree")

        assert count == 0

    def test_get_recovery_attempt_count_with_attempts(self):
        """Should count attempts for specific node."""
        state = {
            "recovery_attempts": [
                {"node": "create_worktree", "hint": "remove_existing_worktree"},
                {"node": "validate_deploy", "hint": "check_ansible_subdirectory"},
                {"node": "create_worktree", "hint": "remove_existing_worktree"},
            ]
        }
        count = get_recovery_attempt_count(state, "create_worktree")

        assert count == 2

    def test_get_recovery_attempt_count_different_node(self):
        """Should not count attempts for different node."""
        state = {
            "recovery_attempts": [
                {"node": "create_worktree", "hint": "remove_existing_worktree"},
            ]
        }
        count = get_recovery_attempt_count(state, "validate_deploy")

        assert count == 0

    def test_should_attempt_recovery_below_max(self):
        """Should return True when below max attempts."""
        state = {"recovery_attempts": [{"node": "create_worktree", "hint": "test"}]}
        result = should_attempt_recovery(state, "create_worktree", max_attempts=2)

        assert result is True

    def test_should_attempt_recovery_at_max(self):
        """Should return False when at max attempts."""
        state = {
            "recovery_attempts": [
                {"node": "create_worktree", "hint": "test"},
                {"node": "create_worktree", "hint": "test"},
            ]
        }
        result = should_attempt_recovery(state, "create_worktree", max_attempts=2)

        assert result is False

    def test_should_attempt_recovery_above_max(self):
        """Should return False when above max attempts."""
        state = {
            "recovery_attempts": [
                {"node": "create_worktree", "hint": "test"},
                {"node": "create_worktree", "hint": "test"},
                {"node": "create_worktree", "hint": "test"},
            ]
        }
        result = should_attempt_recovery(state, "create_worktree", max_attempts=2)

        assert result is False


class TestCreateRecoveryStateUpdate:
    """Tests for create_recovery_state_update function."""

    def test_creates_basic_update(self):
        """Should create basic recovery state update."""
        error = ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message="Worktree exists",
            resolution_hint="remove_existing_worktree",
            context={"role_name": "common"},
        )

        update = create_recovery_state_update("create_worktree", error, None)

        assert update["recovery_context"]["node"] == "create_worktree"
        assert update["recovery_context"]["error"] == "Worktree exists"
        assert update["recovery_context"]["strategy"] == "remove_existing_worktree"
        assert update["last_error_type"] == "recoverable"
        assert update["last_error_message"] == "Worktree exists"
        assert update["last_error_node"] == "create_worktree"
        assert len(update["recovery_attempts"]) == 1
        assert update["recovery_attempts"][0]["node"] == "create_worktree"

    def test_includes_resolution_updates(self):
        """Should include resolution state updates."""
        error = ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message="Worktree exists",
            resolution_hint="remove_existing_worktree",
        )
        resolution = {"worktree_force_recreate": True}

        update = create_recovery_state_update("create_worktree", error, resolution)

        assert update["worktree_force_recreate"] is True

    def test_handles_none_resolution(self):
        """Should handle None resolution gracefully."""
        error = ClassifiedError(
            error_type=ErrorType.UNEXPECTED,
            message="Unknown error",
        )

        update = create_recovery_state_update("any_node", error, None)

        assert "worktree_force_recreate" not in update
        assert update["last_error_type"] == "unexpected"


class TestErrorTypeEnum:
    """Tests for ErrorType enum values."""

    def test_all_types_have_string_values(self):
        """All error types should have string values."""
        assert ErrorType.TRANSIENT.value == "transient"
        assert ErrorType.RECOVERABLE.value == "recoverable"
        assert ErrorType.USER_FIXABLE.value == "user_fixable"
        assert ErrorType.UNEXPECTED.value == "unexpected"

    def test_types_are_distinct(self):
        """All error types should be distinct."""
        types = [ErrorType.TRANSIENT, ErrorType.RECOVERABLE, ErrorType.USER_FIXABLE, ErrorType.UNEXPECTED]
        values = [t.value for t in types]
        assert len(set(values)) == len(values)


# =============================================================================
# v0.6.0 TIER SELECTION TESTS
# =============================================================================


class TestSelectRecoveryTier:
    """Tests for select_recovery_tier function (v0.6.0)."""

    def test_transient_returns_tier0(self):
        """Transient errors should always return tier 0."""
        error = ClassifiedError(
            error_type=ErrorType.TRANSIENT,
            message="timeout",
            resolution_hint="retry_with_backoff",
        )
        assert select_recovery_tier(error, "any_node") == 0

    def test_unexpected_returns_tier4(self):
        """Unexpected errors should always return tier 4."""
        error = ClassifiedError(
            error_type=ErrorType.UNEXPECTED,
            message="crash",
        )
        assert select_recovery_tier(error, "any_node") == 4

    def test_recoverable_simple_returns_tier1(self):
        """Simple recoverable hints should return tier 1."""
        error = ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message="exists",
            resolution_hint="remove_existing_worktree",
        )
        assert select_recovery_tier(error, "create_worktree") == 1

    def test_recoverable_capped_by_max_tier(self):
        """Tier should not exceed node's max_tier."""
        error = ClassifiedError(
            error_type=ErrorType.RECOVERABLE,
            message="complex issue",
            resolution_hint="unknown_complex_fix",
        )
        # create_worktree has max_tier=1
        assert select_recovery_tier(error, "create_worktree") <= 1

    def test_user_fixable_gets_tier3_for_test_nodes(self):
        """USER_FIXABLE should get tier 3 for test nodes with high max_tier."""
        error = ClassifiedError(
            error_type=ErrorType.USER_FIXABLE,
            message="test failed",
            resolution_hint="fix_failing_tests",
        )
        # run_molecule has max_tier=3
        assert select_recovery_tier(error, "run_molecule") == 3


class TestShouldAttemptRecoveryV2:
    """Tests for should_attempt_recovery_v2 with per-node configs (v0.6.0)."""

    def test_respects_node_config_max_iterations(self):
        """Should use node's max_iterations from RecoveryConfig."""
        from harness.dag.recovery_config import RecoveryConfig

        config = RecoveryConfig(max_iterations=5)
        state = {
            "recovery_attempts": [{"node": "test", "hint": "fix"} for _ in range(4)]
        }
        assert should_attempt_recovery_v2(state, "test", config=config) is True

        state["recovery_attempts"].append({"node": "test", "hint": "fix"})
        assert should_attempt_recovery_v2(state, "test", config=config) is False

    def test_uses_registry_when_no_config(self):
        """Should look up config from registry when none provided."""
        # run_molecule has max_iterations=12
        state = {
            "recovery_attempts": [{"node": "run_molecule", "hint": "fix"} for _ in range(10)]
        }
        assert should_attempt_recovery_v2(state, "run_molecule") is True

    def test_backward_compat_with_v050(self):
        """v0.5.0 should_attempt_recovery should still work."""
        state = {
            "recovery_attempts": [{"node": "test", "hint": "fix"}]
        }
        # Original function with explicit max_attempts
        assert should_attempt_recovery(state, "test", max_attempts=2) is True
        state["recovery_attempts"].append({"node": "test", "hint": "fix"})
        assert should_attempt_recovery(state, "test", max_attempts=2) is False
