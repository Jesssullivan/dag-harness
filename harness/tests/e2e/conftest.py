"""E2E test fixtures for DAG Harness.

Required environment variables:
- GITLAB_E2E_PROJECT: GitLab project path (e.g., "tinyland/projects/dag-harness-tests")
- GITLAB_E2E_TOKEN: GitLab API token with write access

When GITLAB_E2E_TOKEN is not set, tests that require GitLab are skipped.
Tests that only need the LangGraph engine with mocked externals still run.
"""

import os
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.config import GitLabConfig, HarnessConfig
from harness.db.models import Role, RoleDependency, DependencyType
from harness.db.state import StateDB

from .utils import cleanup_test_artifacts


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Register E2E markers."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end (requires GitLab access)")


# ============================================================================
# SKIP GUARDS
# ============================================================================


HAS_GITLAB_TOKEN = bool(os.environ.get("GITLAB_E2E_TOKEN"))

requires_gitlab = pytest.mark.skipif(
    not HAS_GITLAB_TOKEN,
    reason="GITLAB_E2E_TOKEN not set -- skipping live GitLab tests",
)


@pytest.fixture(scope="session")
def skip_if_no_gitlab():
    """Skip E2E tests if GITLAB_E2E_TOKEN not set."""
    if not HAS_GITLAB_TOKEN:
        pytest.skip("GITLAB_E2E_TOKEN not set -- skipping E2E tests")


# ============================================================================
# GITLAB FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def gitlab_test_project() -> str:
    """Get GitLab test project from env, skip if not configured.

    Returns the project path string, e.g. "tinyland/projects/dag-harness-tests".
    """
    project = os.environ.get("GITLAB_E2E_PROJECT")
    if not project:
        pytest.skip("GITLAB_E2E_PROJECT not set -- skipping E2E tests")
    return project


# ============================================================================
# CLEANUP TRACKING
# ============================================================================


@dataclass
class ResourceTracker:
    """Tracks GitLab resources created during a test for cleanup."""

    project_path: str = ""
    issues: list[int] = field(default_factory=list)
    merge_requests: list[int] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)

    def track_issue(self, iid: int) -> None:
        """Register an issue IID for cleanup."""
        self.issues.append(iid)

    def track_mr(self, iid: int) -> None:
        """Register a merge request IID for cleanup."""
        self.merge_requests.append(iid)

    def track_branch(self, branch: str) -> None:
        """Register a branch name for cleanup."""
        self.branches.append(branch)


@pytest.fixture
def e2e_cleanup() -> Generator[ResourceTracker, None, None]:
    """Track created resources (issues, MRs, branches) for cleanup after test.

    Yields a ResourceTracker; cleans up all tracked resources in the finalizer.
    When GITLAB_E2E_PROJECT is not set, cleanup is a no-op.
    """
    project = os.environ.get("GITLAB_E2E_PROJECT", "")
    tracker = ResourceTracker(project_path=project)
    yield tracker

    # Finalizer: best-effort cleanup (only if project is configured)
    if tracker.project_path:
        cleanup_test_artifacts(
            project_path=tracker.project_path,
            tracker={
                "issues": tracker.issues,
                "merge_requests": tracker.merge_requests,
                "branches": tracker.branches,
            },
        )


# ============================================================================
# ISOLATION FIXTURES
# ============================================================================


@pytest.fixture
def unique_role_name() -> str:
    """Generate a UUID-based unique role name for test isolation."""
    short_id = uuid.uuid4().hex[:8]
    return f"e2e_test_role_{short_id}"


@pytest.fixture
def e2e_db(tmp_path: Path) -> Generator[StateDB, None, None]:
    """Create a temporary StateDB for E2E tests, pre-populated with roles and workflow defs."""
    db_path = tmp_path / "e2e_harness.db"
    database = StateDB(db_path)

    # Seed roles across waves for wave execution tests
    wave_defs = {
        0: ("Foundation", [("common", True)]),
        1: (
            "Infrastructure Foundation",
            [
                ("windows_prerequisites", True),
                ("ems_registry_urls", True),
                ("iis_config", True),
            ],
        ),
        2: (
            "Core Platform",
            [
                ("ems_platform_services", True),
                ("ems_download_artifacts", False),
                ("database_clone", True),
            ],
        ),
        3: (
            "Web Applications",
            [
                ("ems_web_app", True),
                ("ems_master_calendar", True),
                ("ems_campus_webservice", True),
            ],
        ),
    }
    for wave_num, (wave_name, roles) in wave_defs.items():
        for role_name, has_molecule in roles:
            database.upsert_role(
                Role(
                    name=role_name,
                    wave=wave_num,
                    wave_name=wave_name,
                    has_molecule_tests=has_molecule,
                )
            )

    # Add cross-wave dependencies for dependency tests
    common = database.get_role("common")
    win_prereqs = database.get_role("windows_prerequisites")
    ems_platform = database.get_role("ems_platform_services")

    if common and win_prereqs:
        database.add_dependency(
            RoleDependency(
                role_id=win_prereqs.id,
                depends_on_id=common.id,
                dependency_type=DependencyType.EXPLICIT,
            )
        )
    if common and ems_platform:
        database.add_dependency(
            RoleDependency(
                role_id=ems_platform.id,
                depends_on_id=common.id,
                dependency_type=DependencyType.EXPLICIT,
            )
        )

    # Create workflow definition matching LangGraphWorkflowRunner
    database.create_workflow_definition(
        name="box_up_role_langgraph",
        description="LangGraph-based box-up-role workflow",
        nodes=[
            {"name": n}
            for n in [
                "validate_role",
                "analyze_deps",
                "check_reverse_deps",
                "create_worktree",
                "run_molecule",
                "run_pytest",
                "merge_test_results",
                "validate_deploy",
                "create_commit",
                "push_branch",
                "create_issue",
                "create_mr",
                "human_approval",
                "add_to_merge_train",
                "report_summary",
                "notify_failure",
            ]
        ],
        edges=[],
    )

    yield database


@pytest.fixture
def e2e_db_with_unique_role(e2e_db: StateDB, unique_role_name: str) -> StateDB:
    """E2E database with the unique test role registered in wave 1."""
    e2e_db.upsert_role(
        Role(
            name=unique_role_name,
            wave=1,
            wave_name="E2E Test Wave",
            has_molecule_tests=True,
        )
    )
    return e2e_db


@pytest.fixture
def e2e_config(tmp_path: Path) -> HarnessConfig:
    """Create HarnessConfig for E2E tests.

    Uses the E2E GitLab project when available, otherwise defaults to a
    placeholder so tests can run with mocked GitLab operations.
    """
    project = os.environ.get("GITLAB_E2E_PROJECT", "test/e2e-harness-mock")
    db_path = tmp_path / "e2e_harness.db"
    return HarnessConfig(
        db_path=str(db_path),
        repo_root=str(tmp_path),
        gitlab=GitLabConfig(
            project_path=project,
            group_path=project.rsplit("/", 1)[0] if "/" in project else "",
            default_assignee="",
            default_labels=["e2e-test"],
            default_iteration="",
        ),
    )


# ============================================================================
# ROLE DIRECTORY FIXTURES
# ============================================================================


@pytest.fixture
def e2e_role_dir(tmp_path: Path, unique_role_name: str) -> Path:
    """Create a temporary Ansible role directory tree for E2E tests.

    Builds the role under ``ansible/roles/<name>`` so that
    ``validate_role_node`` finds it when the cwd is ``tmp_path``.
    """
    from .utils import create_test_role_structure

    roles_dir = tmp_path / "ansible" / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    role_dir = create_test_role_structure(roles_dir, unique_role_name)

    # Add molecule directory so has_molecule_tests is True
    mol_dir = role_dir / "molecule" / "default"
    mol_dir.mkdir(parents=True, exist_ok=True)
    (mol_dir / "converge.yml").write_text(
        f"---\n- hosts: all\n  roles:\n    - {unique_role_name}\n"
    )

    # Add a pytest test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / f"test_{unique_role_name}.py").write_text(
        f'"""E2E test for {unique_role_name}."""\n\ndef test_placeholder():\n    assert True\n'
    )

    return role_dir


# ============================================================================
# SUBPROCESS MOCKS
# ============================================================================


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run to simulate git, molecule, pytest, and ansible commands.

    Returns the mock so tests can inspect calls or customize behavior.
    """
    with patch("subprocess.run") as mock_run:

        def _side_effect(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)

            if "ls-remote" in cmd_str:
                result.stdout = ""  # no remote branch
            elif "status" in cmd_str and "porcelain" in cmd_str:
                result.stdout = "M ansible/roles/test/tasks/main.yml\n"
            elif "rev-parse" in cmd_str:
                result.stdout = "abc123def456"
            elif "push" in cmd_str:
                result.stdout = "Everything up-to-date"
            elif "commit" in cmd_str:
                result.stdout = "[sid/test abc123] feat: commit"
            elif "molecule" in cmd_str:
                result.stdout = "ok=5  changed=0  unreachable=0  failed=0"
            elif "pytest" in cmd_str:
                result.stdout = "1 passed"
            elif "syntax-check" in cmd_str:
                result.stdout = "playbook: site.yml"
            elif "auth" in cmd_str and "status" in cmd_str:
                result.stdout = "Token: glpat-test"
            return result

        mock_run.side_effect = _side_effect
        yield mock_run


# ============================================================================
# GITLAB CLIENT MOCKS
# ============================================================================


@pytest.fixture
def mock_gitlab_client():
    """Mock GitLabClient with realistic idempotent behaviour.

    Patches the import inside each node so all GitLab-facing nodes use
    the mock instead of making real API calls.
    """
    mock_client = MagicMock()

    # Issue
    mock_issue = MagicMock()
    mock_issue.iid = 42
    mock_issue.web_url = "https://gitlab.example.com/project/-/issues/42"
    mock_client.get_or_create_issue.return_value = (mock_issue, True)

    # MR
    mock_mr = MagicMock()
    mock_mr.iid = 99
    mock_mr.web_url = "https://gitlab.example.com/project/-/merge_requests/99"
    mock_client.get_or_create_mr.return_value = (mock_mr, True)

    # Merge train
    mock_client.is_merge_train_available.return_value = {"available": True}
    mock_client.is_merge_train_enabled.return_value = True
    mock_client.add_to_merge_train.return_value = None

    # Labels & iterations
    mock_client.prepare_labels_for_role.return_value = ["role", "wave-1"]
    mock_iter = MagicMock()
    mock_iter.id = 100
    mock_client.get_current_iteration.return_value = mock_iter

    # Reviewers
    mock_client.config = MagicMock()
    mock_client.config.default_reviewers = []
    mock_client.set_mr_reviewers.return_value = True

    # Remote branch
    mock_client.remote_branch_exists.return_value = False

    # Issue failure update
    mock_client.update_issue_on_failure.return_value = True

    with patch("harness.gitlab.api.GitLabClient", return_value=mock_client):
        yield mock_client


# ============================================================================
# WORKTREE MOCKS
# ============================================================================


@pytest.fixture
def mock_worktree_manager():
    """Mock WorktreeManager.create() to return a fake worktree info object."""
    mock_info = MagicMock()
    mock_info.path = "/tmp/e2e_worktree"
    mock_info.branch = "sid/e2e_test_role"
    mock_info.commit = "abc123"

    mock_manager = MagicMock()
    mock_manager.create.return_value = mock_info

    with patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager):
        yield mock_manager
