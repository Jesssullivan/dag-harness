"""
Node definitions for DAG workflow execution.

Follows LangGraph patterns:
- Nodes are functions that take state and return updates
- Edges define transitions between nodes
- Conditional edges allow branching based on state
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeResult(str, Enum):
    """Possible outcomes from node execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIP = "skip"
    RETRY = "retry"
    HUMAN_NEEDED = "human_needed"


@dataclass
class NodeContext:
    """
    Context passed to each node during execution.

    Contains:
    - Current state (mutable, accumulated across nodes)
    - Role being processed
    - Execution metadata
    - Access to state database
    - Repository configuration (repo_root, repo_python)
    """

    role_name: str
    execution_id: int
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    repo_root: Path | None = None
    repo_python: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self.state[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple state values."""
        self.state.update(updates)


class Node(ABC):
    """
    Abstract base class for workflow nodes.

    Each node represents a discrete step in the workflow.
    Nodes are:
    - Idempotent where possible
    - Checkpointable (state can be saved/restored)
    - Observable (emit events for monitoring)
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.retries = 3
        self.timeout_seconds = 300

    @abstractmethod
    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        """
        Execute the node logic.

        Args:
            ctx: The node context with current state

        Returns:
            Tuple of (result status, state updates)
        """
        pass

    async def rollback(self, ctx: NodeContext) -> None:
        """
        Rollback any changes made by this node.

        Called when the workflow fails after this node completed.
        Override in subclasses where rollback is possible.
        """
        pass

    def can_skip(self, ctx: NodeContext) -> bool:
        """
        Check if this node can be skipped based on current state.

        Override to implement skip logic (e.g., already completed).
        """
        return False


class FunctionNode(Node):
    """Node that wraps a simple function."""

    def __init__(
        self,
        name: str,
        func: Callable[[NodeContext], tuple[NodeResult, dict[str, Any]]],
        description: str = "",
    ):
        super().__init__(name, description)
        self._func = func

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(ctx)
        return self._func(ctx)


class ConditionalEdge:
    """
    Conditional edge that routes to different nodes based on state.

    Example:
        ConditionalEdge(
            condition=lambda ctx: ctx.get("tests_passed"),
            if_true="create_commit",
            if_false="fix_tests"
        )
    """

    def __init__(self, condition: Callable[[NodeContext], bool], if_true: str, if_false: str):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def evaluate(self, ctx: NodeContext) -> str:
        """Evaluate condition and return next node name."""
        return self.if_true if self.condition(ctx) else self.if_false


class RouterEdge:
    """
    Router edge that can route to multiple possible nodes.

    Example:
        RouterEdge(
            router=lambda ctx: "deploy" if ctx.get("ready") else "wait",
            possible_targets=["deploy", "wait", "abort"]
        )
    """

    def __init__(self, router: Callable[[NodeContext], str], possible_targets: list[str]):
        self.router = router
        self.possible_targets = possible_targets

    def evaluate(self, ctx: NodeContext) -> str:
        """Evaluate router and return next node name."""
        target = self.router(ctx)
        if target not in self.possible_targets:
            raise ValueError(f"Router returned invalid target: {target}")
        return target


# Type alias for edge definitions
Edge = str | ConditionalEdge | RouterEdge


@dataclass
class NodeDefinition:
    """Definition of a node in the workflow graph."""

    name: str
    node: Node
    edges: dict[NodeResult, Edge] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage."""
        edges_dict = {}
        for result, edge in self.edges.items():
            if isinstance(edge, str):
                edges_dict[result.value] = {"type": "direct", "target": edge}
            elif isinstance(edge, ConditionalEdge):
                edges_dict[result.value] = {
                    "type": "conditional",
                    "if_true": edge.if_true,
                    "if_false": edge.if_false,
                }
            elif isinstance(edge, RouterEdge):
                edges_dict[result.value] = {"type": "router", "targets": edge.possible_targets}
        return {
            "name": self.name,
            "description": self.node.description,
            "retries": self.node.retries,
            "timeout_seconds": self.node.timeout_seconds,
            "edges": edges_dict,
        }


# ============================================================================
# BUILT-IN NODES FOR BOX-UP-ROLE WORKFLOW
# ============================================================================


class ValidateRoleNode(Node):
    """Validate that the role exists and extract metadata."""

    def __init__(self):
        super().__init__("validate_role", "Validate role exists and extract metadata")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        # Use repo_root from context if available
        repo_root = ctx.repo_root or Path.cwd()
        role_path = repo_root / "ansible" / "roles" / ctx.role_name

        if not role_path.exists():
            return NodeResult.FAILURE, {"error": f"Role not found: {ctx.role_name}"}

        # Check for molecule tests
        has_molecule = (role_path / "molecule").exists()

        # Check for meta/main.yml
        meta_path = role_path / "meta" / "main.yml"
        has_meta = meta_path.exists()

        return NodeResult.SUCCESS, {
            "role_path": str(role_path),
            "has_molecule_tests": has_molecule,
            "has_meta": has_meta,
        }


class AnalyzeDependenciesNode(Node):
    """Analyze role dependencies using analyze-role-deps.py."""

    def __init__(self):
        super().__init__("analyze_dependencies", "Analyze role dependencies and credentials")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import json
        import subprocess
        import sys

        # Determine Python interpreter and script path
        repo_root = ctx.repo_root or Path.cwd()

        # Use repo's venv python if available, otherwise fallback
        if ctx.repo_python:
            python_path = ctx.repo_python
        else:
            # Try to find venv python
            venv_python = repo_root / ".venv" / "bin" / "python"
            if venv_python.exists():
                python_path = str(venv_python)
            else:
                python_path = sys.executable

        script_path = repo_root / "scripts" / "analyze-role-deps.py"

        try:
            result = subprocess.run(
                [python_path, str(script_path), ctx.role_name, "--json"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(repo_root),
            )

            # Exit code 2 = reverse deps exist (warning, not error)
            if result.returncode not in (0, 2):
                return NodeResult.FAILURE, {"error": result.stderr or result.stdout}

            analysis = json.loads(result.stdout)

            # Determine if this is a foundation role
            wave = analysis.get("wave", 0)
            explicit_deps = analysis.get("explicit_deps", [])
            is_foundation = wave == 0 and len(explicit_deps) == 0

            return NodeResult.SUCCESS, {
                "wave": wave,
                "wave_name": analysis.get("wave_name", ""),
                "explicit_deps": explicit_deps,
                "implicit_deps": analysis.get("implicit_deps", []),
                "credentials": analysis.get("credentials", []),
                "reverse_deps": analysis.get("reverse_deps", []),
                "tags": analysis.get("tags", []),
                "is_foundation": is_foundation,
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "Analysis timed out"}
        except json.JSONDecodeError as e:
            return NodeResult.FAILURE, {"error": f"Invalid JSON output: {e}"}
        except FileNotFoundError:
            return NodeResult.FAILURE, {"error": f"Script not found: {script_path}"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class CheckDependenciesNode(Node):
    """
    Check if UPSTREAM dependencies are already boxed up.

    This node verifies that roles THIS role depends on have already
    been processed. This is the correct direction for dependency checking.

    Foundation roles (Wave 0 with no dependencies) automatically pass.
    """

    def __init__(self):
        super().__init__("check_dependencies", "Verify upstream dependencies are boxed up first")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        # Foundation roles have no dependencies - always pass
        if ctx.get("is_foundation", False):
            return NodeResult.SUCCESS, {
                "blocking_deps": [],
                "foundation_role": True,
                "info": "Foundation role - no upstream dependencies to check",
            }

        # Check DEPENDENCIES (roles this role needs), not reverse deps
        dependencies = ctx.get("explicit_deps", []) + ctx.get("implicit_deps", [])
        if not dependencies:
            return NodeResult.SUCCESS, {"blocking_deps": []}

        blocking = []
        for dep in dependencies:
            # Check if branch exists on origin (already boxed up)
            result = subprocess.run(
                ["git", "ls-remote", "--heads", "origin", f"sid/{dep}"],
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                blocking.append(dep)

        if blocking:
            return NodeResult.FAILURE, {
                "blocking_deps": blocking,
                "error": f"Upstream dependencies must be boxed up first: {', '.join(blocking)}",
            }

        return NodeResult.SUCCESS, {"blocking_deps": []}


class WarnReverseDepsNode(Node):
    """
    Warn about downstream dependencies (informational only).

    This node provides information about which roles depend on this role
    and may need updates after this role is merged. It does NOT block.
    """

    def __init__(self):
        super().__init__("warn_reverse_deps", "Note downstream roles that may need updates")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        reverse_deps = ctx.get("reverse_deps", [])
        if not reverse_deps:
            return NodeResult.SUCCESS, {"reverse_deps_warning": None, "pending_downstream": []}

        # Check which reverse deps haven't been boxed up yet (informational)
        not_boxed = []
        for dep in reverse_deps:
            result = subprocess.run(
                ["git", "ls-remote", "--heads", "origin", f"sid/{dep}"],
                capture_output=True,
                text=True,
            )
            if not result.stdout.strip():
                not_boxed.append(dep)

        if not_boxed:
            # SUCCESS with informational warning - don't block
            return NodeResult.SUCCESS, {
                "reverse_deps_warning": f"After merging, consider updating: {', '.join(not_boxed)}",
                "pending_downstream": not_boxed,
            }

        return NodeResult.SUCCESS, {"reverse_deps_warning": None, "pending_downstream": []}


# Keep old name as alias for backwards compatibility
CheckReverseDepsNode = CheckDependenciesNode


class CreateWorktreeNode(Node):
    """Create git worktree for isolated development."""

    def __init__(self):
        super().__init__("create_worktree", "Create isolated git worktree")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        repo_root = ctx.repo_root or Path.cwd()

        try:
            result = subprocess.run(
                ["scripts/create-role-worktree.sh", ctx.role_name],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(repo_root),
            )

            # Exit code 2 = warnings (ok to proceed)
            if result.returncode not in (0, 2):
                return NodeResult.FAILURE, {"error": result.stderr or result.stdout}

            worktree_path = f"../sid-{ctx.role_name}"
            return NodeResult.SUCCESS, {
                "worktree_path": worktree_path,
                "branch": f"sid/{ctx.role_name}",
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "Worktree creation timed out"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}

    async def rollback(self, ctx: NodeContext) -> None:
        import subprocess

        worktree_path = ctx.get("worktree_path")
        if worktree_path:
            subprocess.run(
                ["git", "worktree", "remove", worktree_path, "--force"], capture_output=True
            )


class RunMoleculeTestsNode(Node):
    """Run molecule tests for the role."""

    def __init__(self):
        super().__init__("run_molecule_tests", "Execute molecule tests (blocking)")
        self.timeout_seconds = 1200  # 20 minutes

    def _load_env_file(self, repo_root: Path) -> dict[str, str]:
        """Load environment variables from .env file."""
        import os

        env = os.environ.copy()
        env_file = repo_root / ".env"

        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Handle 'export VAR=value' and 'VAR=value'
                    if line.startswith("export "):
                        line = line[7:]
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            env[key] = value
        return env

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess
        import time

        if not ctx.get("has_molecule_tests", False):
            return NodeResult.SKIP, {"molecule_skipped": True, "reason": "No molecule tests"}

        repo_root = ctx.repo_root or Path.cwd()
        worktree_path = ctx.get("worktree_path")
        cwd = worktree_path if worktree_path else str(repo_root)

        # Load .env file from repo root for credentials
        env = self._load_env_file(repo_root)

        # Determine molecule command - use repo's venv
        molecule_cmd = str(repo_root / ".venv" / "bin" / "molecule")
        role_dir = repo_root / "ansible" / "roles" / ctx.role_name

        start_time = time.time()
        try:
            result = subprocess.run(
                [molecule_cmd, "test"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(role_dir),
                env=env,
            )

            duration = int(time.time() - start_time)

            if result.returncode != 0:
                return NodeResult.FAILURE, {
                    "molecule_passed": False,
                    "molecule_duration": duration,
                    "molecule_stdout": result.stdout[-5000:] if result.stdout else "",
                    "molecule_stderr": result.stderr[-5000:] if result.stderr else "",
                    "returncode": result.returncode,
                    "error": "Molecule tests failed",
                }

            return NodeResult.SUCCESS, {"molecule_passed": True, "molecule_duration": duration}

        except subprocess.TimeoutExpired:
            return NodeResult.FAILURE, {
                "molecule_passed": False,
                "error": "Molecule tests timed out",
            }


class CreateCommitNode(Node):
    """Create commit with semantic message."""

    def __init__(self):
        super().__init__("create_commit", "Create signed commit")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        worktree_path = ctx.get("worktree_path", ".")
        wave = ctx.get("wave", 0)
        wave_name = ctx.get("wave_name", "")

        commit_msg = f"""feat({ctx.role_name}): Add {ctx.role_name} Ansible role

Wave {wave}: {wave_name}

- Molecule tests passing
- Ready for merge train
"""

        try:
            # Stage all changes
            subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True)

            # Check if there are changes to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=worktree_path, capture_output=True, text=True
            )

            if not status.stdout.strip():
                return NodeResult.SKIP, {"commit_skipped": True, "reason": "No changes to commit"}

            # Create commit
            result = subprocess.run(
                [
                    "git",
                    "commit",
                    "--author=Jess Sullivan <jsullivan2@bates.edu>",
                    "-m",
                    commit_msg,
                ],
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )

            if result.returncode not in (0, 2):
                return NodeResult.FAILURE, {"error": result.stderr}

            # Get commit SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=worktree_path, capture_output=True, text=True
            )

            return NodeResult.SUCCESS, {
                "commit_sha": sha_result.stdout.strip(),
                "commit_message": commit_msg,
            }

        except subprocess.CalledProcessError as e:
            return NodeResult.FAILURE, {"error": str(e)}


class PushBranchNode(Node):
    """Push branch to origin."""

    def __init__(self):
        super().__init__("push_branch", "Push branch to origin with tracking")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import subprocess

        worktree_path = ctx.get("worktree_path", ".")
        branch = ctx.get("branch", f"sid/{ctx.role_name}")

        try:
            result = subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=worktree_path,
                capture_output=True,
                text=True,
            )

            if result.returncode not in (0, 2):
                return NodeResult.FAILURE, {"error": result.stderr}

            return NodeResult.SUCCESS, {"pushed": True}

        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class CreateGitLabIssueNode(Node):
    """Create GitLab issue with iteration assignment."""

    def __init__(self):
        super().__init__("create_gitlab_issue", "Create GitLab issue")

    def _get_env_with_venv(self, repo_root: Path) -> dict[str, str]:
        """Get environment with venv's bin directory in PATH."""
        import os

        env = os.environ.copy()

        # Load .env file
        env_file = repo_root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:]
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            env[key] = value

        # Prepend venv's bin to PATH
        venv_bin = repo_root / ".venv" / "bin"
        if venv_bin.exists():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        return env

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import json
        import subprocess

        repo_root = ctx.repo_root or Path.cwd()
        env = self._get_env_with_venv(repo_root)

        # Use wave info from context (already computed by analyze_dependencies)
        wave = ctx.get("wave", 0)
        wave_name = ctx.get("wave_name", "Unassigned")

        # Build issue title and description
        title = f"Deploy `{ctx.role_name}` role (Wave {wave})"
        description = f"""## Role: {ctx.role_name}

**Wave**: {wave} - {wave_name}

### Checklist
- [ ] Molecule tests passing
- [ ] Code review complete
- [ ] Ready for merge train

---
*Created by dag-harness*
"""

        try:
            # Use glab CLI directly
            result = subprocess.run(
                [
                    "glab", "issue", "create",
                    "--title", title,
                    "--description", description,
                    "--label", "role,ansible,molecule",
                    "--assignee", "jsullivan2",
                    "--yes",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(repo_root),
                env=env,
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {
                    "error": f"glab issue create failed: {result.stderr}",
                    "stdout": result.stdout,
                }

            # Parse issue URL from glab output (format: "https://gitlab.com/.../issues/123")
            import re
            url_match = re.search(r'(https://[^\s]+/issues/\d+)', result.stdout)
            if url_match:
                issue_url = url_match.group(1)
                iid_match = re.search(r'/issues/(\d+)', issue_url)
                issue_iid = iid_match.group(1) if iid_match else None
            else:
                # Try stderr too (glab sometimes outputs URL there)
                url_match = re.search(r'(https://[^\s]+/issues/\d+)', result.stderr)
                if url_match:
                    issue_url = url_match.group(1)
                    iid_match = re.search(r'/issues/(\d+)', issue_url)
                    issue_iid = iid_match.group(1) if iid_match else None
                else:
                    return NodeResult.FAILURE, {
                        "error": "Could not parse issue URL from glab output",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }

            return NodeResult.SUCCESS, {
                "issue_url": issue_url,
                "issue_iid": issue_iid,
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "Issue creation timed out"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class CreateMergeRequestNode(Node):
    """Create GitLab merge request."""

    def __init__(self):
        super().__init__("create_merge_request", "Create GitLab merge request")

    def _get_env_with_venv(self, repo_root: Path) -> dict[str, str]:
        """Get environment with venv's bin directory in PATH."""
        import os

        env = os.environ.copy()

        # Load .env file
        env_file = repo_root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:]
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            env[key] = value

        # Prepend venv's bin to PATH
        venv_bin = repo_root / ".venv" / "bin"
        if venv_bin.exists():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        return env

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        import re
        import subprocess

        repo_root = ctx.repo_root or Path.cwd()
        worktree_path = ctx.get("worktree_path", str(repo_root))
        env = self._get_env_with_venv(repo_root)
        issue_iid = ctx.get("issue_iid")
        issue_url = ctx.get("issue_url", "")

        # Get wave info from context
        wave = ctx.get("wave", 0)
        wave_name = ctx.get("wave_name", "Unassigned")

        # Build MR title and description
        title = f"feat({ctx.role_name}): Deploy {ctx.role_name} role (Wave {wave})"
        description = f"""## Summary
Deploy `{ctx.role_name}` Ansible role.

**Wave**: {wave} - {wave_name}

Closes #{issue_iid}

### Checklist
- [x] Molecule tests passing
- [ ] Code review complete
- [ ] Ready for merge

---
*Created by dag-harness*
"""

        try:
            # Use glab CLI directly from the worktree
            result = subprocess.run(
                [
                    "glab", "mr", "create",
                    "--title", title,
                    "--description", description,
                    "--assignee", "jsullivan2",
                    "--yes",
                    "--push",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=worktree_path,
                env=env,
            )

            if result.returncode != 0:
                return NodeResult.FAILURE, {
                    "error": f"glab mr create failed: {result.stderr}",
                    "stdout": result.stdout,
                }

            # Parse MR URL from glab output
            url_match = re.search(r'(https://[^\s]+/merge_requests/\d+)', result.stdout + result.stderr)
            if url_match:
                mr_url = url_match.group(1)
                iid_match = re.search(r'/merge_requests/(\d+)', mr_url)
                mr_iid = iid_match.group(1) if iid_match else None
            else:
                return NodeResult.FAILURE, {
                    "error": "Could not parse MR URL from glab output",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

            return NodeResult.SUCCESS, {
                "mr_url": mr_url,
                "mr_iid": mr_iid,
            }

        except subprocess.TimeoutExpired:
            return NodeResult.RETRY, {"error": "MR creation timed out"}
        except Exception as e:
            return NodeResult.FAILURE, {"error": str(e)}


class ReportSummaryNode(Node):
    """Generate and report final summary."""

    def __init__(self):
        super().__init__("report_summary", "Generate workflow summary")

    async def execute(self, ctx: NodeContext) -> tuple[NodeResult, dict[str, Any]]:
        summary = {
            "role": ctx.role_name,
            "wave": ctx.get("wave"),
            "wave_name": ctx.get("wave_name"),
            "is_foundation": ctx.get("is_foundation", False),
            "worktree_path": ctx.get("worktree_path"),
            "branch": ctx.get("branch"),
            "commit_sha": ctx.get("commit_sha"),
            "issue_url": ctx.get("issue_url"),
            "mr_url": ctx.get("mr_url"),
            "molecule_passed": ctx.get("molecule_passed"),
            "credentials": ctx.get("credentials", []),
            "dependencies": ctx.get("explicit_deps", []),
            "reverse_deps_warning": ctx.get("reverse_deps_warning"),
            "pending_downstream": ctx.get("pending_downstream", []),
        }

        return NodeResult.SUCCESS, {"summary": summary}
