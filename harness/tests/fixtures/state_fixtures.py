"""BoxUpRoleState test fixtures for harness tests.

Provides BoxUpRoleState instances at various workflow stages:
- Initial state (fresh)
- After validation
- After molecule tests
- After MR creation
- Awaiting approval
- Completed
- Failed
"""

from typing import Any

import pytest

from harness.dag.langgraph_engine import BoxUpRoleState


# =============================================================================
# INITIAL STATE FIXTURE
# =============================================================================


@pytest.fixture
def initial_state() -> BoxUpRoleState:
    """
    Fresh state with only role_name set.

    This is the state at the very beginning of a workflow,
    before any nodes have executed.

    Usage:
        def test_validate_node(initial_state):
            result = await validate_role_node(initial_state)
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=None,
        has_molecule_tests=False,
        has_meta=False,
        wave=0,
        wave_name="",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=[],
        credentials=[],
        tags=[],
        blocking_deps=[],
        pushed=False,
        molecule_skipped=False,
        pytest_skipped=False,
        deploy_skipped=False,
        all_tests_passed=None,
        parallel_tests_completed=[],
        test_phase_start_time=None,
        test_phase_duration=None,
        parallel_execution_enabled=True,
        iteration_assigned=False,
        issue_created=False,
        mr_created=False,
        reviewers_set=False,
        branch_existed=False,
        current_node="validate_role",
        completed_nodes=[],
        errors=[],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


@pytest.fixture
def initial_state_with_execution_id() -> BoxUpRoleState:
    """
    Initial state with an execution_id for database tracking.

    Use this when testing workflows that need to record
    progress in the database.
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=42,
        has_molecule_tests=False,
        has_meta=False,
        wave=0,
        wave_name="",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=[],
        credentials=[],
        tags=[],
        blocking_deps=[],
        pushed=False,
        molecule_skipped=False,
        pytest_skipped=False,
        deploy_skipped=False,
        all_tests_passed=None,
        parallel_tests_completed=[],
        test_phase_start_time=None,
        test_phase_duration=None,
        parallel_execution_enabled=True,
        iteration_assigned=False,
        issue_created=False,
        mr_created=False,
        reviewers_set=False,
        branch_existed=False,
        current_node="validate_role",
        completed_nodes=[],
        errors=[],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


# =============================================================================
# AFTER VALIDATION STATE
# =============================================================================


@pytest.fixture
def after_validate_state() -> BoxUpRoleState:
    """
    State after validate_role_node completes successfully.

    The role has been validated and metadata extracted.
    Dependencies have been analyzed.
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=42,
        role_path="/path/to/ansible/roles/common",
        has_molecule_tests=True,
        has_meta=True,
        wave=0,
        wave_name="wave_0_infrastructure",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=["sql_server_2022", "ems_web_app", "iis_config"],
        credentials=[
            {"entry_name": "ansible-self", "purpose": "WinRM auth", "is_base58": True}
        ],
        tags=["infrastructure", "common"],
        blocking_deps=[],
        worktree_path="",
        branch="",
        commit_sha=None,
        pushed=False,
        molecule_skipped=False,
        pytest_skipped=False,
        deploy_skipped=False,
        all_tests_passed=None,
        parallel_tests_completed=[],
        test_phase_start_time=None,
        test_phase_duration=None,
        parallel_execution_enabled=True,
        iteration_assigned=False,
        issue_created=False,
        mr_created=False,
        reviewers_set=False,
        branch_existed=False,
        current_node="create_worktree",
        completed_nodes=["validate_role", "analyze_deps", "check_reverse_deps"],
        errors=[],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


@pytest.fixture
def after_validate_state_with_blocking_deps() -> BoxUpRoleState:
    """
    State after validation with blocking dependencies.

    The role depends on other roles that haven't been boxed up yet.
    This should prevent the workflow from proceeding.
    """
    return BoxUpRoleState(
        role_name="ems_platform_services",
        execution_id=43,
        role_path="/path/to/ansible/roles/ems_platform_services",
        has_molecule_tests=True,
        has_meta=True,
        wave=3,
        wave_name="wave_3_platform",
        explicit_deps=["ems_web_app", "common"],
        implicit_deps=[],
        reverse_deps=[],
        credentials=[],
        tags=["platform"],
        blocking_deps=["ems_web_app"],  # ems_web_app not yet boxed up
        worktree_path="",
        branch="",
        pushed=False,
        molecule_skipped=False,
        pytest_skipped=False,
        deploy_skipped=False,
        all_tests_passed=None,
        parallel_tests_completed=[],
        parallel_execution_enabled=True,
        iteration_assigned=False,
        issue_created=False,
        mr_created=False,
        reviewers_set=False,
        branch_existed=False,
        current_node="check_reverse_deps",
        completed_nodes=["validate_role", "analyze_deps"],
        errors=["Blocking dependency: ems_web_app has not been boxed up"],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


# =============================================================================
# AFTER MOLECULE STATE
# =============================================================================


@pytest.fixture
def after_molecule_state() -> BoxUpRoleState:
    """
    State after run_molecule_node completes successfully.

    Molecule tests have passed. Worktree exists with changes.
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=42,
        role_path="/path/to/ansible/roles/common",
        has_molecule_tests=True,
        has_meta=True,
        wave=0,
        wave_name="wave_0_infrastructure",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=["sql_server_2022", "ems_web_app"],
        credentials=[
            {"entry_name": "ansible-self", "purpose": "WinRM auth", "is_base58": True}
        ],
        tags=["infrastructure", "common"],
        blocking_deps=[],
        worktree_path="/path/to/.worktrees/sid-common",
        branch="sid/common",
        commit_sha=None,
        pushed=False,
        molecule_passed=True,
        molecule_skipped=False,
        molecule_duration=245,
        molecule_output="PLAY [Converge] ***\n...all tests passed...",
        pytest_passed=None,
        pytest_skipped=False,
        deploy_passed=None,
        deploy_skipped=False,
        all_tests_passed=None,
        parallel_tests_completed=["run_molecule"],
        test_phase_start_time=1706968800.0,
        test_phase_duration=None,
        parallel_execution_enabled=True,
        iteration_assigned=False,
        issue_created=False,
        mr_created=False,
        reviewers_set=False,
        branch_existed=False,
        current_node="run_pytest",
        completed_nodes=[
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
        ],
        errors=[],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


@pytest.fixture
def after_molecule_state_failed() -> BoxUpRoleState:
    """
    State after run_molecule_node fails.

    Molecule tests have failed. Workflow should route to error handling.
    """
    return BoxUpRoleState(
        role_name="sql_server_2022",
        execution_id=44,
        role_path="/path/to/ansible/roles/sql_server_2022",
        has_molecule_tests=True,
        has_meta=True,
        wave=2,
        wave_name="wave_2_database",
        explicit_deps=["common"],
        implicit_deps=[],
        reverse_deps=["sql_management_studio"],
        credentials=[],
        tags=["database"],
        blocking_deps=[],
        worktree_path="/path/to/.worktrees/sid-sql_server_2022",
        branch="sid/sql_server_2022",
        commit_sha=None,
        pushed=False,
        molecule_passed=False,
        molecule_skipped=False,
        molecule_duration=180,
        molecule_output="TASK [sql_server_2022 : Install SQL Server] ***\nfatal: ...",
        pytest_passed=None,
        pytest_skipped=False,
        deploy_passed=None,
        deploy_skipped=False,
        all_tests_passed=None,
        parallel_tests_completed=["run_molecule"],
        test_phase_start_time=1706968800.0,
        test_phase_duration=None,
        parallel_execution_enabled=True,
        iteration_assigned=False,
        issue_created=False,
        mr_created=False,
        reviewers_set=False,
        branch_existed=False,
        current_node="run_molecule",
        completed_nodes=[
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
        ],
        errors=["Molecule tests failed: TASK [sql_server_2022 : Install SQL Server] fatal"],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


# =============================================================================
# AFTER CREATE MR STATE
# =============================================================================


@pytest.fixture
def after_create_mr_state() -> BoxUpRoleState:
    """
    State after create_mr_node completes successfully.

    All tests passed, commit pushed, issue and MR created.
    Awaiting human approval.
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=42,
        role_path="/path/to/ansible/roles/common",
        has_molecule_tests=True,
        has_meta=True,
        wave=0,
        wave_name="wave_0_infrastructure",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=["sql_server_2022", "ems_web_app"],
        credentials=[
            {"entry_name": "ansible-self", "purpose": "WinRM auth", "is_base58": True}
        ],
        tags=["infrastructure", "common"],
        blocking_deps=[],
        worktree_path="/path/to/.worktrees/sid-common",
        branch="sid/common",
        commit_sha="abc123def456789012345678901234567890abcd",
        commit_message="feat(common): box up common role\n\n- Molecule tests passing\n- Dependencies validated",
        pushed=True,
        molecule_passed=True,
        molecule_skipped=False,
        molecule_duration=245,
        pytest_passed=True,
        pytest_skipped=False,
        pytest_duration=30,
        deploy_passed=True,
        deploy_skipped=False,
        all_tests_passed=True,
        parallel_tests_completed=["run_molecule", "run_pytest"],
        test_phase_start_time=1706968800.0,
        test_phase_duration=275.0,
        parallel_execution_enabled=True,
        issue_url="https://gitlab.example.com/project/-/issues/123",
        issue_iid=123,
        issue_created=True,
        mr_url="https://gitlab.example.com/project/-/merge_requests/456",
        mr_iid=456,
        mr_created=True,
        reviewers_set=True,
        iteration_assigned=True,
        branch_existed=False,
        current_node="human_approval",
        completed_nodes=[
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
            "run_pytest",
            "validate_deploy",
            "create_commit",
            "push_branch",
            "create_issue",
            "create_mr",
        ],
        errors=[],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=True,
    )


# =============================================================================
# AWAITING APPROVAL STATE
# =============================================================================


@pytest.fixture
def awaiting_approval_state() -> BoxUpRoleState:
    """
    State at human_approval_node interrupt.

    The workflow has paused and is waiting for human approval.
    This is the state that would be checkpointed.
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=42,
        role_path="/path/to/ansible/roles/common",
        has_molecule_tests=True,
        has_meta=True,
        wave=0,
        wave_name="wave_0_infrastructure",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=["sql_server_2022", "ems_web_app"],
        credentials=[
            {"entry_name": "ansible-self", "purpose": "WinRM auth", "is_base58": True}
        ],
        tags=["infrastructure", "common"],
        blocking_deps=[],
        worktree_path="/path/to/.worktrees/sid-common",
        branch="sid/common",
        commit_sha="abc123def456789012345678901234567890abcd",
        commit_message="feat(common): box up common role",
        pushed=True,
        molecule_passed=True,
        molecule_skipped=False,
        molecule_duration=245,
        pytest_passed=True,
        pytest_skipped=False,
        pytest_duration=30,
        deploy_passed=True,
        deploy_skipped=False,
        all_tests_passed=True,
        parallel_tests_completed=["run_molecule", "run_pytest"],
        test_phase_start_time=1706968800.0,
        test_phase_duration=275.0,
        parallel_execution_enabled=True,
        issue_url="https://gitlab.example.com/project/-/issues/123",
        issue_iid=123,
        issue_created=True,
        mr_url="https://gitlab.example.com/project/-/merge_requests/456",
        mr_iid=456,
        mr_created=True,
        reviewers_set=True,
        iteration_assigned=True,
        branch_existed=False,
        current_node="human_approval",
        completed_nodes=[
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
            "run_pytest",
            "validate_deploy",
            "create_commit",
            "push_branch",
            "create_issue",
            "create_mr",
        ],
        errors=[],
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=True,
    )


# =============================================================================
# COMPLETED STATE
# =============================================================================


@pytest.fixture
def completed_state() -> BoxUpRoleState:
    """
    Final successful state after workflow completes.

    Human approved, MR added to merge train, summary generated.
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=42,
        role_path="/path/to/ansible/roles/common",
        has_molecule_tests=True,
        has_meta=True,
        wave=0,
        wave_name="wave_0_infrastructure",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=["sql_server_2022", "ems_web_app"],
        credentials=[
            {"entry_name": "ansible-self", "purpose": "WinRM auth", "is_base58": True}
        ],
        tags=["infrastructure", "common"],
        blocking_deps=[],
        worktree_path="/path/to/.worktrees/sid-common",
        branch="sid/common",
        commit_sha="abc123def456789012345678901234567890abcd",
        commit_message="feat(common): box up common role",
        pushed=True,
        molecule_passed=True,
        molecule_skipped=False,
        molecule_duration=245,
        pytest_passed=True,
        pytest_skipped=False,
        pytest_duration=30,
        deploy_passed=True,
        deploy_skipped=False,
        all_tests_passed=True,
        parallel_tests_completed=["run_molecule", "run_pytest"],
        test_phase_start_time=1706968800.0,
        test_phase_duration=275.0,
        parallel_execution_enabled=True,
        issue_url="https://gitlab.example.com/project/-/issues/123",
        issue_iid=123,
        issue_created=True,
        mr_url="https://gitlab.example.com/project/-/merge_requests/456",
        mr_iid=456,
        mr_created=True,
        reviewers_set=True,
        iteration_assigned=True,
        merge_train_status="added",
        branch_existed=False,
        current_node="report_summary",
        completed_nodes=[
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
            "run_pytest",
            "validate_deploy",
            "create_commit",
            "push_branch",
            "create_issue",
            "create_mr",
            "human_approval",
            "add_to_merge_train",
            "report_summary",
        ],
        errors=[],
        summary={
            "role_name": "common",
            "status": "success",
            "tests_passed": True,
            "mr_url": "https://gitlab.example.com/project/-/merge_requests/456",
            "merge_train_status": "added",
            "duration_seconds": 320,
        },
        human_approved=True,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


# =============================================================================
# FAILED STATE
# =============================================================================


@pytest.fixture
def failed_state() -> BoxUpRoleState:
    """
    State with errors representing a failed workflow.

    The workflow failed during molecule tests.
    """
    return BoxUpRoleState(
        role_name="sql_server_2022",
        execution_id=44,
        role_path="/path/to/ansible/roles/sql_server_2022",
        has_molecule_tests=True,
        has_meta=True,
        wave=2,
        wave_name="wave_2_database",
        explicit_deps=["common"],
        implicit_deps=[],
        reverse_deps=["sql_management_studio"],
        credentials=[],
        tags=["database"],
        blocking_deps=[],
        worktree_path="/path/to/.worktrees/sid-sql_server_2022",
        branch="sid/sql_server_2022",
        commit_sha=None,
        pushed=False,
        molecule_passed=False,
        molecule_skipped=False,
        molecule_duration=180,
        molecule_output="PLAY [Converge] ***\n\nTASK [sql_server_2022 : Install SQL Server] ***\nfatal: [vmnode876]: FAILED! => {\"msg\": \"Installation timed out\"}",
        pytest_passed=None,
        pytest_skipped=True,
        deploy_passed=None,
        deploy_skipped=True,
        all_tests_passed=False,
        parallel_tests_completed=["run_molecule"],
        test_phase_start_time=1706968800.0,
        test_phase_duration=180.0,
        parallel_execution_enabled=True,
        issue_url=None,
        issue_iid=None,
        issue_created=False,
        mr_url=None,
        mr_iid=None,
        mr_created=False,
        reviewers_set=False,
        iteration_assigned=False,
        branch_existed=False,
        current_node="notify_failure",
        completed_nodes=[
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
            "notify_failure",
        ],
        errors=[
            "Molecule tests failed for sql_server_2022",
            "TASK [sql_server_2022 : Install SQL Server] FAILED: Installation timed out",
        ],
        summary={
            "role_name": "sql_server_2022",
            "status": "failed",
            "tests_passed": False,
            "error": "Molecule tests failed",
            "duration_seconds": 200,
        },
        human_approved=None,
        human_rejection_reason=None,
        awaiting_human_input=False,
    )


@pytest.fixture
def failed_state_human_rejected() -> BoxUpRoleState:
    """
    State where human rejected the MR.

    All tests passed but human declined to approve.
    """
    return BoxUpRoleState(
        role_name="common",
        execution_id=42,
        role_path="/path/to/ansible/roles/common",
        has_molecule_tests=True,
        has_meta=True,
        wave=0,
        wave_name="wave_0_infrastructure",
        explicit_deps=[],
        implicit_deps=[],
        reverse_deps=["sql_server_2022", "ems_web_app"],
        credentials=[],
        tags=["infrastructure", "common"],
        blocking_deps=[],
        worktree_path="/path/to/.worktrees/sid-common",
        branch="sid/common",
        commit_sha="abc123def456789012345678901234567890abcd",
        commit_message="feat(common): box up common role",
        pushed=True,
        molecule_passed=True,
        molecule_skipped=False,
        molecule_duration=245,
        pytest_passed=True,
        pytest_skipped=False,
        pytest_duration=30,
        deploy_passed=True,
        deploy_skipped=False,
        all_tests_passed=True,
        parallel_tests_completed=["run_molecule", "run_pytest"],
        test_phase_start_time=1706968800.0,
        test_phase_duration=275.0,
        parallel_execution_enabled=True,
        issue_url="https://gitlab.example.com/project/-/issues/123",
        issue_iid=123,
        issue_created=True,
        mr_url="https://gitlab.example.com/project/-/merge_requests/456",
        mr_iid=456,
        mr_created=True,
        reviewers_set=True,
        iteration_assigned=True,
        merge_train_status=None,
        branch_existed=False,
        current_node="notify_failure",
        completed_nodes=[
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
            "run_pytest",
            "validate_deploy",
            "create_commit",
            "push_branch",
            "create_issue",
            "create_mr",
            "human_approval",
            "notify_failure",
        ],
        errors=["Human rejected MR: Needs additional review for credential handling"],
        summary={
            "role_name": "common",
            "status": "rejected",
            "tests_passed": True,
            "mr_url": "https://gitlab.example.com/project/-/merge_requests/456",
            "rejection_reason": "Needs additional review for credential handling",
        },
        human_approved=False,
        human_rejection_reason="Needs additional review for credential handling",
        awaiting_human_input=False,
    )


# =============================================================================
# PARAMETRIZED FIXTURES FOR BATCH TESTING
# =============================================================================


@pytest.fixture(params=[
    "initial_state",
    "after_validate_state",
    "after_molecule_state",
    "after_create_mr_state",
    "awaiting_approval_state",
    "completed_state",
    "failed_state",
])
def any_workflow_state(request) -> BoxUpRoleState:
    """
    Parametrized fixture that yields each workflow state in turn.

    Useful for testing invariants that should hold across all states.

    Usage:
        def test_state_has_role_name(any_workflow_state):
            assert any_workflow_state.get("role_name")
    """
    return request.getfixturevalue(request.param)


# =============================================================================
# FACTORY FIXTURES
# =============================================================================


@pytest.fixture
def make_state():
    """
    Factory fixture for creating custom BoxUpRoleState instances.

    Usage:
        def test_custom_state(make_state):
            state = make_state(role_name="custom_role", wave=2)
            assert state["role_name"] == "custom_role"
    """

    def _make_state(**overrides) -> BoxUpRoleState:
        defaults: dict[str, Any] = {
            "role_name": "test_role",
            "execution_id": None,
            "has_molecule_tests": False,
            "has_meta": False,
            "wave": 0,
            "wave_name": "",
            "explicit_deps": [],
            "implicit_deps": [],
            "reverse_deps": [],
            "credentials": [],
            "tags": [],
            "blocking_deps": [],
            "pushed": False,
            "molecule_skipped": False,
            "pytest_skipped": False,
            "deploy_skipped": False,
            "all_tests_passed": None,
            "parallel_tests_completed": [],
            "test_phase_start_time": None,
            "test_phase_duration": None,
            "parallel_execution_enabled": True,
            "iteration_assigned": False,
            "issue_created": False,
            "mr_created": False,
            "reviewers_set": False,
            "branch_existed": False,
            "current_node": "validate_role",
            "completed_nodes": [],
            "errors": [],
            "human_approved": None,
            "human_rejection_reason": None,
            "awaiting_human_input": False,
        }
        defaults.update(overrides)
        return BoxUpRoleState(**defaults)

    return _make_state


# =============================================================================
# EXPORTED FIXTURES
# =============================================================================

__all__ = [
    "initial_state",
    "initial_state_with_execution_id",
    "after_validate_state",
    "after_validate_state_with_blocking_deps",
    "after_molecule_state",
    "after_molecule_state_failed",
    "after_create_mr_state",
    "awaiting_approval_state",
    "completed_state",
    "failed_state",
    "failed_state_human_rejected",
    "any_workflow_state",
    "make_state",
]
