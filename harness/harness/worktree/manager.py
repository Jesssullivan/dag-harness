"""
Git worktree management for parallel role development.

Handles:
- Worktree creation from origin/main
- Status tracking (ahead/behind, uncommitted changes)
- Cleanup and pruning
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.config import HarnessConfig
from harness.db.models import Worktree, WorktreeStatus
from harness.db.state import StateDB


@dataclass
class WorktreeInfo:
    """Information about a git worktree."""

    path: str
    branch: str
    commit: str
    ahead: int = 0
    behind: int = 0
    uncommitted: int = 0
    status: WorktreeStatus = WorktreeStatus.ACTIVE


class WorktreeManager:
    """
    Manages git worktrees for role development.

    Each role gets its own worktree at ../sid-<role> with branch sid/<role>.
    """

    def __init__(self, db: StateDB, config: HarnessConfig | None = None):
        self.db = db
        self.config = config or HarnessConfig.load()

    def _run_git(self, *args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    def list_worktrees(self) -> list[WorktreeInfo]:
        """List all git worktrees with status."""
        result = self._run_git("worktree", "list", "--porcelain")
        if result.returncode != 0:
            return []

        worktrees = []
        current: dict[str, str] = {}

        for line in result.stdout.split("\n"):
            if line.startswith("worktree "):
                current["path"] = line.split(" ", 1)[1]
            elif line.startswith("HEAD "):
                current["commit"] = line.split(" ", 1)[1]
            elif line.startswith("branch "):
                current["branch"] = line.split(" ", 1)[1].replace("refs/heads/", "")
            elif line == "" and current.get("path"):
                # End of worktree entry
                if current.get("branch", "").startswith("sid/"):
                    info = self._get_worktree_info(current)
                    worktrees.append(info)
                current = {}

        return worktrees

    def _get_worktree_info(self, current: dict[str, str]) -> WorktreeInfo:
        """Get detailed info for a worktree."""
        path = current["path"]
        branch = current.get("branch", "")
        commit = current.get("commit", "")

        # Get ahead/behind counts
        ahead = behind = 0
        try:
            ahead_result = self._run_git("rev-list", "--count", "origin/main..HEAD", cwd=path)
            ahead = int(ahead_result.stdout.strip() or 0)

            behind_result = self._run_git("rev-list", "--count", "HEAD..origin/main", cwd=path)
            behind = int(behind_result.stdout.strip() or 0)
        except Exception:
            pass

        # Get uncommitted changes count
        uncommitted = 0
        status_result = self._run_git("status", "--porcelain", cwd=path)
        if status_result.stdout.strip():
            uncommitted = len(status_result.stdout.strip().split("\n"))

        # Determine status
        if uncommitted > 0:
            status = WorktreeStatus.DIRTY
        elif behind > 10:
            status = WorktreeStatus.STALE
        else:
            status = WorktreeStatus.ACTIVE

        return WorktreeInfo(
            path=path,
            branch=branch,
            commit=commit,
            ahead=ahead,
            behind=behind,
            uncommitted=uncommitted,
            status=status,
        )

    def create(
        self, role_name: str, force: bool = False, base_ref: str | None = None
    ) -> WorktreeInfo:
        """
        Create a worktree for a role.

        Args:
            role_name: Name of the role
            force: If True, remove existing worktree first
            base_ref: Git ref to create the worktree from. If None, uses origin/main.
                     For new roles not on main, pass HEAD or current branch.

        Returns:
            WorktreeInfo for the created worktree
        """
        branch = f"sid/{role_name}"
        # Resolve to absolute path to avoid issues with relative paths during workflow execution
        worktree_path = str((Path(self.config.worktree.base_path) / f"sid-{role_name}").resolve())

        # Check if already exists
        if Path(worktree_path).exists():
            if force:
                self.remove(role_name)
            else:
                raise ValueError(f"Worktree already exists at {worktree_path}")

        # Determine the base ref for the worktree
        if base_ref is None:
            # Default: fetch and use origin/main
            self._run_git("fetch", "origin", "main")
            base_ref = "origin/main"

        # Create worktree with new branch from the specified base
        result = self._run_git("worktree", "add", "-b", branch, worktree_path, base_ref)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        # Get worktree info
        info = WorktreeInfo(
            path=worktree_path, branch=branch, commit="", status=WorktreeStatus.ACTIVE
        )

        # Get commit SHA
        sha_result = self._run_git("rev-parse", "HEAD", cwd=worktree_path)
        if sha_result.returncode == 0:
            info.commit = sha_result.stdout.strip()

        # Copy essential files
        self._copy_essential_files(worktree_path)

        # Update database
        role = self.db.get_role(role_name)
        if role and role.id:
            worktree = Worktree(
                role_id=role.id,
                path=worktree_path,
                branch=branch,
                base_commit=info.commit,
                current_commit=info.commit,
                status=WorktreeStatus.ACTIVE,
            )
            self.db.upsert_worktree(worktree)

        return info

    def _copy_essential_files(self, worktree_path: str) -> None:
        """Copy essential files to worktree."""
        import shutil

        essential_files = ["ems.kdbx", ".env.local"]

        for filename in essential_files:
            src = Path(filename)
            if src.exists():
                dst = Path(worktree_path) / filename
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

    def remove(self, role_name: str, force: bool = True) -> bool:
        """
        Remove a worktree.

        Args:
            role_name: Name of the role
            force: If True, force removal even with uncommitted changes

        Returns:
            True if removed successfully
        """
        # Resolve to absolute path for consistency
        worktree_path = str((Path(self.config.worktree.base_path) / f"sid-{role_name}").resolve())

        args = ["worktree", "remove", worktree_path]
        if force:
            args.append("--force")

        result = self._run_git(*args)

        if result.returncode == 0:
            # Update database
            role = self.db.get_role(role_name)
            if role and role.id:
                with self.db.connection() as conn:
                    conn.execute(
                        "UPDATE worktrees SET status = ? WHERE role_id = ?",
                        (WorktreeStatus.PRUNED.value, role.id),
                    )
            return True

        return False

    def prune(self) -> int:
        """
        Prune stale worktree references.

        Returns:
            Number of entries pruned
        """
        result = self._run_git("worktree", "prune", "--verbose")
        if result.returncode != 0:
            return 0

        # Count pruned entries from output
        pruned = 0
        for line in result.stderr.split("\n"):
            if "Removing" in line or "pruning" in line.lower():
                pruned += 1

        return pruned

    def sync_all(self) -> int:
        """
        Sync all worktree statuses to database.

        Returns:
            Number of worktrees synced
        """
        worktrees = self.list_worktrees()
        synced = 0

        for info in worktrees:
            role_name = info.branch.replace("sid/", "")
            role = self.db.get_role(role_name)
            if not role or not role.id:
                continue

            worktree = Worktree(
                role_id=role.id,
                path=info.path,
                branch=info.branch,
                current_commit=info.commit,
                commits_ahead=info.ahead,
                commits_behind=info.behind,
                uncommitted_changes=info.uncommitted,
                status=info.status,
            )
            self.db.upsert_worktree(worktree)
            synced += 1

        return synced

    def get_stale(self, threshold: int = 10) -> list[WorktreeInfo]:
        """
        Get worktrees that are significantly behind origin/main.

        Args:
            threshold: Number of commits behind to consider stale

        Returns:
            List of stale worktrees
        """
        return [wt for wt in self.list_worktrees() if wt.behind > threshold]

    def get_dirty(self) -> list[WorktreeInfo]:
        """
        Get worktrees with uncommitted changes.

        Returns:
            List of dirty worktrees
        """
        return [wt for wt in self.list_worktrees() if wt.uncommitted > 0]
