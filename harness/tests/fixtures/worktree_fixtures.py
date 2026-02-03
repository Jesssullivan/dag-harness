"""Git worktree mock fixtures for harness tests.

Provides comprehensive mocking for:
- WorktreeManager class
- Temporary git repositories
- WorktreeStatus enum test cases
- Git add/commit/push operations
"""

import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from harness.db.models import WorktreeStatus


# =============================================================================
# WORKTREE STATUS TEST CASES
# =============================================================================


@pytest.fixture
def worktree_states() -> dict[str, dict[str, Any]]:
    """
    Dictionary of WorktreeStatus enum test cases.

    Provides test data for each worktree status with expected behaviors.

    Returns:
        Dict mapping status names to their test configurations.

    Usage:
        def test_worktree_status(worktree_states):
            active_state = worktree_states["active"]
            assert active_state["status"] == WorktreeStatus.ACTIVE
    """
    return {
        "active": {
            "status": WorktreeStatus.ACTIVE,
            "commits_ahead": 3,
            "commits_behind": 0,
            "uncommitted_changes": 0,
            "description": "Clean worktree with commits ready to push",
            "can_push": True,
            "needs_rebase": False,
            "needs_commit": False,
        },
        "stale": {
            "status": WorktreeStatus.STALE,
            "commits_ahead": 1,
            "commits_behind": 15,
            "uncommitted_changes": 0,
            "description": "Worktree significantly behind origin/main",
            "can_push": False,
            "needs_rebase": True,
            "needs_commit": False,
        },
        "dirty": {
            "status": WorktreeStatus.DIRTY,
            "commits_ahead": 2,
            "commits_behind": 3,
            "uncommitted_changes": 5,
            "description": "Worktree with uncommitted changes",
            "can_push": False,
            "needs_rebase": False,
            "needs_commit": True,
        },
        "merged": {
            "status": WorktreeStatus.MERGED,
            "commits_ahead": 0,
            "commits_behind": 0,
            "uncommitted_changes": 0,
            "description": "Worktree branch has been merged to main",
            "can_push": False,
            "needs_rebase": False,
            "needs_commit": False,
        },
        "pruned": {
            "status": WorktreeStatus.PRUNED,
            "commits_ahead": 0,
            "commits_behind": 0,
            "uncommitted_changes": 0,
            "description": "Worktree has been removed/pruned",
            "can_push": False,
            "needs_rebase": False,
            "needs_commit": False,
        },
        "fresh": {
            "status": WorktreeStatus.ACTIVE,
            "commits_ahead": 0,
            "commits_behind": 0,
            "uncommitted_changes": 0,
            "description": "Newly created worktree, in sync with origin",
            "can_push": False,
            "needs_rebase": False,
            "needs_commit": False,
        },
        "diverged": {
            "status": WorktreeStatus.STALE,
            "commits_ahead": 5,
            "commits_behind": 8,
            "uncommitted_changes": 2,
            "description": "Worktree has diverged from origin with local changes",
            "can_push": False,
            "needs_rebase": True,
            "needs_commit": True,
        },
    }


# =============================================================================
# TEMPORARY GIT REPOSITORY
# =============================================================================


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """
    Creates a temporary git repository for testing.

    The repository includes:
    - Initialized git repo
    - Initial commit with README
    - 'main' branch as default
    - Sample role structure

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to the temporary git repository

    Usage:
        def test_git_operations(temp_git_repo):
            # temp_git_repo is a fully initialized git repo
            result = subprocess.run(["git", "status"], cwd=temp_git_repo)
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_path, capture_output=True)

    # Configure git user (required for commits)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
    )

    # Create README
    readme = repo_path / "README.md"
    readme.write_text("# Test Repository\n\nThis is a test repository for harness tests.\n")

    # Create sample role structure
    role_dir = repo_path / "ansible" / "roles" / "common"
    role_dir.mkdir(parents=True)

    # tasks/main.yml
    tasks_dir = role_dir / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "main.yml").write_text(
        """---
- name: Sample task
  ansible.builtin.debug:
    msg: "Hello from common role"
"""
    )

    # meta/main.yml
    meta_dir = role_dir / "meta"
    meta_dir.mkdir()
    (meta_dir / "main.yml").write_text(
        """---
galaxy_info:
  description: Common role for tests
dependencies: []
"""
    )

    # defaults/main.yml
    defaults_dir = role_dir / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "main.yml").write_text(
        """---
common_variable: "default_value"
"""
    )

    # Make initial commit
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
    )

    return repo_path


@pytest.fixture
def temp_git_repo_with_remote(temp_git_repo: Path, tmp_path: Path) -> tuple[Path, Path]:
    """
    Creates a git repo with a local 'remote' for testing push/fetch.

    Returns:
        Tuple of (repo_path, remote_path)

    Usage:
        def test_push(temp_git_repo_with_remote):
            repo, remote = temp_git_repo_with_remote
            subprocess.run(["git", "push", "origin", "main"], cwd=repo)
    """
    # Create bare remote
    remote_path = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "clone", "--bare", str(temp_git_repo), str(remote_path)],
        capture_output=True,
    )

    # Add remote to repo
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote_path)],
        cwd=temp_git_repo,
        capture_output=True,
    )

    return temp_git_repo, remote_path


# =============================================================================
# MOCK WORKTREE MANAGER
# =============================================================================


@pytest.fixture
def mock_worktree_manager():
    """
    Mock WorktreeManager class for testing without real git operations.

    The mock provides:
    - list_worktrees() -> Returns configurable worktree list
    - create() -> Creates mock worktree
    - remove() -> Removes mock worktree
    - sync_all() -> No-op
    - get_stale() -> Returns filtered stale worktrees
    - get_dirty() -> Returns filtered dirty worktrees

    Usage:
        def test_workflow(mock_worktree_manager):
            manager = mock_worktree_manager
            worktrees = manager.list_worktrees()
    """
    from harness.worktree.manager import WorktreeInfo

    # Default worktree data
    worktrees_data = [
        WorktreeInfo(
            path="/path/to/worktrees/sid-common",
            branch="sid/common",
            commit="abc123def456789",
            ahead=2,
            behind=0,
            uncommitted=0,
            status=WorktreeStatus.ACTIVE,
        ),
        WorktreeInfo(
            path="/path/to/worktrees/sid-sql_server_2022",
            branch="sid/sql_server_2022",
            commit="def456abc789012",
            ahead=0,
            behind=15,
            uncommitted=3,
            status=WorktreeStatus.DIRTY,
        ),
        WorktreeInfo(
            path="/path/to/worktrees/sid-ems_web_app",
            branch="sid/ems_web_app",
            commit="789012def456abc",
            ahead=5,
            behind=20,
            uncommitted=0,
            status=WorktreeStatus.STALE,
        ),
    ]

    mock_manager = MagicMock()

    # Configure list_worktrees
    mock_manager.list_worktrees.return_value = worktrees_data

    # Configure create
    def mock_create(role_name: str, force: bool = False) -> WorktreeInfo:
        return WorktreeInfo(
            path=f"/path/to/worktrees/sid-{role_name}",
            branch=f"sid/{role_name}",
            commit="newcommit123456",
            ahead=0,
            behind=0,
            uncommitted=0,
            status=WorktreeStatus.ACTIVE,
        )

    mock_manager.create.side_effect = mock_create

    # Configure remove
    mock_manager.remove.return_value = True

    # Configure sync_all
    mock_manager.sync_all.return_value = len(worktrees_data)

    # Configure get_stale
    mock_manager.get_stale.return_value = [
        wt for wt in worktrees_data if wt.behind > 10
    ]

    # Configure get_dirty
    mock_manager.get_dirty.return_value = [
        wt for wt in worktrees_data if wt.uncommitted > 0
    ]

    # Configure prune
    mock_manager.prune.return_value = 0

    # Store worktrees data for modification in tests
    mock_manager._worktrees_data = worktrees_data

    return mock_manager


# =============================================================================
# MOCK GIT OPERATIONS
# =============================================================================


@pytest.fixture
def mock_git_operations():
    """
    Mock git add/commit/push operations.

    Provides a patch for subprocess.run that handles common git commands.

    The mock tracks:
    - Which files were added
    - Commit messages used
    - Push destinations

    Usage:
        def test_commit_flow(mock_git_operations):
            mock_git = mock_git_operations
            # Run code that calls git commands
            ...
            # Check what was called
            assert "commit" in mock_git.calls
    """
    state = {
        "staged_files": [],
        "commits": [],
        "pushes": [],
        "calls": [],
        "current_branch": "sid/common",
        "current_commit": "abc123def456",
    }

    def _mock_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        cmd_str = " ".join(cmd)
        state["calls"].append(cmd_str)

        # Skip non-git commands
        if not cmd[0] == "git":
            return result

        subcmd = cmd[1] if len(cmd) > 1 else ""

        # git add
        if subcmd == "add":
            files = cmd[2:]
            state["staged_files"].extend(files)
            return result

        # git commit
        if subcmd == "commit":
            message = ""
            if "-m" in cmd:
                msg_idx = cmd.index("-m") + 1
                if msg_idx < len(cmd):
                    message = cmd[msg_idx]
            state["commits"].append({
                "message": message,
                "files": state["staged_files"].copy(),
            })
            state["staged_files"] = []
            state["current_commit"] = "new_commit_sha"
            result.stdout = f"[{state['current_branch']} new_commit_sha] {message}\n"
            return result

        # git push
        if subcmd == "push":
            remote = cmd[2] if len(cmd) > 2 else "origin"
            branch = cmd[3] if len(cmd) > 3 else state["current_branch"]
            state["pushes"].append({
                "remote": remote,
                "branch": branch,
            })
            result.stdout = f"To {remote}\n * [new branch] {branch} -> {branch}\n"
            return result

        # git status
        if subcmd == "status":
            if "--porcelain" in cmd:
                result.stdout = ""  # Clean working directory
            else:
                result.stdout = f"On branch {state['current_branch']}\nnothing to commit, working tree clean\n"
            return result

        # git rev-parse
        if subcmd == "rev-parse":
            if "HEAD" in cmd:
                result.stdout = state["current_commit"]
            elif "--abbrev-ref" in cmd:
                result.stdout = state["current_branch"]
            return result

        # git branch
        if subcmd == "branch":
            if "--show-current" in cmd:
                result.stdout = state["current_branch"]
            else:
                result.stdout = f"* {state['current_branch']}\n  main\n"
            return result

        # git fetch
        if subcmd == "fetch":
            return result

        # git log
        if subcmd == "log":
            result.stdout = f"{state['current_commit']} HEAD -> {state['current_branch']}\n"
            return result

        # git diff
        if subcmd == "diff":
            result.stdout = ""  # No diff
            return result

        # git rev-list (for ahead/behind counts)
        if subcmd == "rev-list":
            if "--count" in cmd:
                result.stdout = "0"
            return result

        # git worktree
        if subcmd == "worktree":
            if "list" in cmd:
                result.stdout = f"worktree /path/to/worktrees/sid-common\nHEAD {state['current_commit']}\nbranch refs/heads/{state['current_branch']}\n"
            return result

        return result

    with patch("subprocess.run", side_effect=_mock_run) as mock_run:
        mock_run.state = state
        yield mock_run


@pytest.fixture
def mock_git_operations_failure():
    """
    Mock git operations that fail.

    Useful for testing error handling in git operations.

    Usage:
        def test_git_failure(mock_git_operations_failure):
            # All git commands will fail
            ...
    """

    def _mock_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        result = MagicMock(spec=subprocess.CompletedProcess)

        if cmd[0] == "git":
            result.returncode = 1
            result.stdout = ""
            result.stderr = "fatal: not a git repository (or any of the parent directories): .git"
        else:
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

        return result

    with patch("subprocess.run", side_effect=_mock_run) as mock_run:
        yield mock_run


# =============================================================================
# ASYNC FIXTURES
# =============================================================================


@pytest.fixture
async def async_mock_worktree_manager(mock_worktree_manager):
    """
    Async version of mock_worktree_manager for async tests.

    Usage:
        @pytest.mark.asyncio
        async def test_async_workflow(async_mock_worktree_manager):
            manager = async_mock_worktree_manager
            ...
    """
    return mock_worktree_manager


# =============================================================================
# EXPORTED FIXTURES
# =============================================================================

__all__ = [
    "worktree_states",
    "temp_git_repo",
    "temp_git_repo_with_remote",
    "mock_worktree_manager",
    "mock_git_operations",
    "mock_git_operations_failure",
    "async_mock_worktree_manager",
]
