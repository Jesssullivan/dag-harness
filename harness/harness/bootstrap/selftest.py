"""Post-install verification tests for harness bootstrap.

This module runs self-tests after bootstrap to verify:
- Database connection and schema
- GitLab API authentication
- MCP server functionality
- Hook script executability
- Git worktree support
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TestStatus(Enum):
    """Status of a self-test."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


@dataclass
class TestResult:
    """Result of a single self-test."""
    name: str
    status: TestStatus
    message: str
    details: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class SelfTestResult:
    """Result of all self-tests."""
    all_passed: bool
    tests: list[TestResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.PASS)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.FAIL)

    @property
    def warn_count(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.WARN)


class SelfTester:
    """Runs post-installation self-tests.

    Tests verify that all components are properly installed and functional.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize self-tester.

        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.harness_dir = self.project_root / "harness"

    def run_all(self, quick: bool = False) -> SelfTestResult:
        """Run all self-tests.

        Args:
            quick: Skip slow tests (network calls, etc.)

        Returns:
            SelfTestResult with all test outcomes
        """
        tests = []

        # Database tests
        tests.append(self._test_database_connection())
        tests.append(self._test_database_schema())

        # Git tests
        tests.append(self._test_git_available())
        tests.append(self._test_git_worktree_support())

        # MCP client integration tests
        tests.append(self._test_hooks_executable())
        tests.append(self._test_settings_valid())

        # MCP server tests
        tests.append(self._test_mcp_importable())

        # Network tests (skip if quick mode)
        if not quick:
            tests.append(self._test_gitlab_auth())

        all_passed = all(
            t.status in (TestStatus.PASS, TestStatus.SKIP, TestStatus.WARN)
            for t in tests
        )

        return SelfTestResult(all_passed=all_passed, tests=tests)

    def _test_database_connection(self) -> TestResult:
        """Test database can be opened."""
        import time
        start = time.time()

        try:
            from harness.db.state import StateDB

            db_path = os.environ.get(
                "HARNESS_DB_PATH",
                str(self.harness_dir / "harness.db")
            )

            db = StateDB(db_path)
            # Quick query to verify connection
            with db.connection() as conn:
                conn.execute("SELECT 1").fetchone()

            duration = (time.time() - start) * 1000
            return TestResult(
                name="database_connection",
                status=TestStatus.PASS,
                message="Database connection successful",
                details=f"Path: {db_path}",
                duration_ms=duration
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="database_connection",
                status=TestStatus.FAIL,
                message="Database connection failed",
                details=str(e),
                duration_ms=duration
            )

    def _test_database_schema(self) -> TestResult:
        """Test database schema is valid."""
        import time
        start = time.time()

        try:
            from harness.db.state import StateDB

            db_path = os.environ.get(
                "HARNESS_DB_PATH",
                str(self.harness_dir / "harness.db")
            )

            db = StateDB(db_path)
            result = db.validate_schema()

            duration = (time.time() - start) * 1000

            if result.get("valid"):
                return TestResult(
                    name="database_schema",
                    status=TestStatus.PASS,
                    message="Database schema is valid",
                    details=f"Tables: {result.get('table_count', 0)}, Indexes: {result.get('index_count', 0)}",
                    duration_ms=duration
                )
            else:
                return TestResult(
                    name="database_schema",
                    status=TestStatus.WARN,
                    message="Database schema has issues",
                    details=str(result.get("missing_tables", [])),
                    duration_ms=duration
                )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="database_schema",
                status=TestStatus.FAIL,
                message="Schema validation failed",
                details=str(e),
                duration_ms=duration
            )

    def _test_git_available(self) -> TestResult:
        """Test git is available."""
        import time
        start = time.time()

        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            duration = (time.time() - start) * 1000

            if result.returncode == 0:
                version = result.stdout.strip()
                return TestResult(
                    name="git_available",
                    status=TestStatus.PASS,
                    message="Git is available",
                    details=version,
                    duration_ms=duration
                )
            else:
                return TestResult(
                    name="git_available",
                    status=TestStatus.FAIL,
                    message="Git command failed",
                    details=result.stderr,
                    duration_ms=duration
                )

        except FileNotFoundError:
            return TestResult(
                name="git_available",
                status=TestStatus.FAIL,
                message="Git not found",
                details="git command not in PATH"
            )
        except Exception as e:
            return TestResult(
                name="git_available",
                status=TestStatus.FAIL,
                message="Git check failed",
                details=str(e)
            )

    def _test_git_worktree_support(self) -> TestResult:
        """Test git worktree support."""
        import time
        start = time.time()

        try:
            result = subprocess.run(
                ["git", "worktree", "list"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(self.project_root)
            )

            duration = (time.time() - start) * 1000

            if result.returncode == 0:
                lines = [l for l in result.stdout.strip().split("\n") if l]
                return TestResult(
                    name="git_worktree",
                    status=TestStatus.PASS,
                    message="Git worktree support available",
                    details=f"{len(lines)} worktree(s) found",
                    duration_ms=duration
                )
            else:
                return TestResult(
                    name="git_worktree",
                    status=TestStatus.WARN,
                    message="Git worktree command failed",
                    details=result.stderr,
                    duration_ms=duration
                )

        except Exception as e:
            return TestResult(
                name="git_worktree",
                status=TestStatus.WARN,
                message="Git worktree check failed",
                details=str(e)
            )

    def _test_hooks_executable(self) -> TestResult:
        """Test hook scripts are executable."""
        import time
        start = time.time()

        hooks_dir = self.project_root / ".claude" / "hooks"

        if not hooks_dir.exists():
            return TestResult(
                name="hooks_executable",
                status=TestStatus.SKIP,
                message="Hooks directory not found",
                details=str(hooks_dir)
            )

        issues = []
        checked = 0

        for hook_file in hooks_dir.iterdir():
            if hook_file.suffix in (".sh", ".py"):
                checked += 1
                if not os.access(hook_file, os.X_OK):
                    issues.append(f"{hook_file.name} not executable")

        duration = (time.time() - start) * 1000

        if issues:
            return TestResult(
                name="hooks_executable",
                status=TestStatus.WARN,
                message=f"{len(issues)} hook(s) not executable",
                details="; ".join(issues),
                duration_ms=duration
            )
        else:
            return TestResult(
                name="hooks_executable",
                status=TestStatus.PASS,
                message=f"All {checked} hooks are executable",
                duration_ms=duration
            )

    def _test_settings_valid(self) -> TestResult:
        """Test settings.json is valid JSON."""
        import time
        import json
        start = time.time()

        settings_path = self.project_root / ".claude" / "settings.json"

        if not settings_path.exists():
            return TestResult(
                name="settings_valid",
                status=TestStatus.SKIP,
                message="settings.json not found",
                details=str(settings_path)
            )

        try:
            with open(settings_path) as f:
                settings = json.load(f)

            duration = (time.time() - start) * 1000

            # Check for MCP server config
            has_mcp = "mcpServers" in settings and "dag-harness" in settings.get("mcpServers", {})

            if has_mcp:
                return TestResult(
                    name="settings_valid",
                    status=TestStatus.PASS,
                    message="settings.json is valid with MCP config",
                    duration_ms=duration
                )
            else:
                return TestResult(
                    name="settings_valid",
                    status=TestStatus.WARN,
                    message="settings.json valid but missing MCP config",
                    details="dag-harness MCP server not configured",
                    duration_ms=duration
                )

        except json.JSONDecodeError as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="settings_valid",
                status=TestStatus.FAIL,
                message="settings.json is invalid JSON",
                details=str(e),
                duration_ms=duration
            )

    def _test_mcp_importable(self) -> TestResult:
        """Test MCP server module can be imported."""
        import time
        import importlib
        start = time.time()

        try:
            # Use importlib to avoid any path issues
            # The harness package should already be importable since we're running from it
            mcp_server = importlib.import_module("harness.mcp.server")

            # Verify the create_mcp_server function exists
            if not hasattr(mcp_server, "create_mcp_server"):
                duration = (time.time() - start) * 1000
                return TestResult(
                    name="mcp_importable",
                    status=TestStatus.FAIL,
                    message="MCP server missing create_mcp_server",
                    duration_ms=duration
                )

            duration = (time.time() - start) * 1000
            return TestResult(
                name="mcp_importable",
                status=TestStatus.PASS,
                message="MCP server module importable",
                duration_ms=duration
            )

        except ImportError as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="mcp_importable",
                status=TestStatus.FAIL,
                message="MCP server import failed",
                details=str(e),
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="mcp_importable",
                status=TestStatus.WARN,
                message="MCP server import issue",
                details=str(e),
                duration_ms=duration
            )

    def _test_gitlab_auth(self) -> TestResult:
        """Test GitLab authentication."""
        import time
        start = time.time()

        gitlab_token = os.environ.get("GITLAB_TOKEN") or os.environ.get("GL_TOKEN")

        if not gitlab_token:
            return TestResult(
                name="gitlab_auth",
                status=TestStatus.SKIP,
                message="GITLAB_TOKEN not set",
                details="Skipping GitLab auth test"
            )

        try:
            import urllib.request
            import urllib.error

            gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com")
            req = urllib.request.Request(
                f"{gitlab_url}/api/v4/user",
                headers={"PRIVATE-TOKEN": gitlab_token}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                duration = (time.time() - start) * 1000

                if response.status == 200:
                    import json
                    data = json.loads(response.read().decode())
                    username = data.get("username", "unknown")
                    return TestResult(
                        name="gitlab_auth",
                        status=TestStatus.PASS,
                        message="GitLab authentication successful",
                        details=f"Authenticated as @{username}",
                        duration_ms=duration
                    )

        except urllib.error.HTTPError as e:
            duration = (time.time() - start) * 1000
            if e.code == 401:
                return TestResult(
                    name="gitlab_auth",
                    status=TestStatus.FAIL,
                    message="GitLab authentication failed",
                    details="Invalid token (401 Unauthorized)",
                    duration_ms=duration
                )
            return TestResult(
                name="gitlab_auth",
                status=TestStatus.WARN,
                message="GitLab API error",
                details=f"HTTP {e.code}",
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="gitlab_auth",
                status=TestStatus.WARN,
                message="GitLab auth check failed",
                details=str(e),
                duration_ms=duration
            )

        return TestResult(
            name="gitlab_auth",
            status=TestStatus.WARN,
            message="GitLab auth check inconclusive"
        )
