"""Interactive setup wizard for harness bootstrap.

This module provides an interactive wizard that guides users through:
- Environment detection
- Credential configuration
- Path resolution
- Database initialization
- MCP client integration installation
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from harness.bootstrap.credentials import CredentialDiscovery, CredentialStatus
from harness.bootstrap.paths import PathResolver, PathStatus
from harness.bootstrap.selftest import SelfTester, TestStatus


class WizardStep(Enum):
    """Steps in the bootstrap wizard."""
    DETECT_ENV = "detect_env"
    CHECK_CREDENTIALS = "check_credentials"
    CHECK_PATHS = "check_paths"
    INIT_DATABASE = "init_database"
    INSTALL_CLAUDE = "install_claude"
    RUN_SELFTESTS = "run_selftests"
    COMPLETE = "complete"


@dataclass
class WizardState:
    """State maintained across wizard steps."""
    current_step: WizardStep = WizardStep.DETECT_ENV
    project_root: Optional[Path] = None
    credentials_ok: bool = False
    paths_ok: bool = False
    database_ok: bool = False
    claude_ok: bool = False
    selftests_ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def can_continue(self) -> bool:
        """Check if wizard can continue to next step."""
        return len(self.errors) == 0


@dataclass
class WizardResult:
    """Final result of running the wizard."""
    success: bool
    state: WizardState
    message: str


class BootstrapWizard:
    """Interactive bootstrap wizard.

    Guides users through the complete setup process with prompts
    and validation at each step.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        console: Optional[Console] = None,
        interactive: bool = True
    ):
        """Initialize the wizard.

        Args:
            project_root: Override for project root detection
            console: Rich console for output
            interactive: Whether to prompt for user input
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.console = console or Console()
        self.interactive = interactive

        # Component helpers
        self.credential_discovery = CredentialDiscovery(self.project_root)
        self.path_resolver = PathResolver(self.project_root)
        self.self_tester = SelfTester(self.project_root)

        # Wizard state
        self.state = WizardState(project_root=self.project_root)

    def run(self, check_only: bool = False) -> WizardResult:
        """Run the bootstrap wizard.

        Args:
            check_only: Only check current state, don't make changes

        Returns:
            WizardResult with success status and details
        """
        self._print_header()

        if check_only:
            return self._run_checks()

        try:
            # Step 1: Detect environment
            self._step_detect_env()
            if not self.state.can_continue:
                return self._make_result(False, "Environment detection failed")

            # Step 2: Check credentials
            self._step_check_credentials()

            # Step 3: Check paths
            self._step_check_paths()
            if not self.state.can_continue:
                return self._make_result(False, "Path validation failed")

            # Step 4: Initialize database
            self._step_init_database()
            if not self.state.can_continue:
                return self._make_result(False, "Database initialization failed")

            # Step 5: Install MCP client integration
            self._step_install_claude()
            if not self.state.can_continue:
                return self._make_result(False, "MCP client installation failed")

            # Step 6: Run self-tests
            self._step_run_selftests()

            # Complete
            self.state.current_step = WizardStep.COMPLETE
            return self._make_result(True, "Bootstrap completed successfully")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Bootstrap cancelled by user[/yellow]")
            return self._make_result(False, "Cancelled by user")
        except Exception as e:
            self.state.errors.append(str(e))
            return self._make_result(False, f"Bootstrap failed: {e}")

    def _run_checks(self) -> WizardResult:
        """Run checks without making changes."""
        self.console.print("\n[bold]Running bootstrap checks...[/bold]\n")

        all_ok = True

        # Check credentials
        self.console.print("[bold]Credentials:[/bold]")
        cred_result = self.credential_discovery.check_all(validate=False)
        for cred in cred_result.credentials:
            status_str = self._status_icon(cred.status == CredentialStatus.FOUND)
            required = "[red](required)[/red]" if cred.required else "[dim](optional)[/dim]"
            self.console.print(f"  {status_str} {cred.name} {required}")
            if cred.source:
                self.console.print(f"      [dim]Source: {cred.source}[/dim]")
            if cred.error:
                self.console.print(f"      [yellow]{cred.error}[/yellow]")

        if not cred_result.all_required_present:
            all_ok = False

        # Check paths
        self.console.print("\n[bold]Paths:[/bold]")
        path_result = self.path_resolver.check_all()
        for path in path_result.paths:
            status_str = self._status_icon(
                path.status in (PathStatus.VALID, PathStatus.WRITABLE)
            )
            self.console.print(f"  {status_str} {path.name}")
            if path.path:
                self.console.print(f"      [dim]{path.path}[/dim]")
            if path.error:
                self.console.print(f"      [yellow]{path.error}[/yellow]")

        if not path_result.all_valid:
            all_ok = False

        # Check installation
        self.console.print("\n[bold]MCP client Integration:[/bold]")
        from harness.install import MCPInstaller, InstallStatus
        installer = MCPInstaller(self.project_root)
        install_status, components = installer.check()

        status_str = self._status_icon(install_status == InstallStatus.COMPLETE)
        self.console.print(f"  {status_str} Installation: {install_status.value}")

        for comp in components:
            comp_status = self._status_icon(comp.installed)
            self.console.print(f"    {comp_status} {comp.name}")

        if install_status not in (InstallStatus.COMPLETE, InstallStatus.OUTDATED):
            all_ok = False

        # Run quick self-tests
        self.console.print("\n[bold]Self-Tests:[/bold]")
        test_result = self.self_tester.run_all(quick=True)
        for test in test_result.tests:
            status_str = self._status_icon(test.status == TestStatus.PASS)
            if test.status == TestStatus.SKIP:
                status_str = "[dim]○[/dim]"
            elif test.status == TestStatus.WARN:
                status_str = "[yellow]⚠[/yellow]"
            self.console.print(f"  {status_str} {test.name}: {test.message}")

        if not test_result.all_passed:
            all_ok = False

        # Summary
        self.console.print()
        if all_ok:
            self.console.print(Panel(
                "[green]All checks passed![/green]\n\nHarness is properly configured.",
                title="Bootstrap Status"
            ))
        else:
            self.console.print(Panel(
                "[yellow]Some checks failed.[/yellow]\n\nRun 'harness bootstrap' to fix issues.",
                title="Bootstrap Status"
            ))

        self.state.credentials_ok = cred_result.all_required_present
        self.state.paths_ok = path_result.all_valid
        self.state.claude_ok = install_status == InstallStatus.COMPLETE
        self.state.selftests_ok = test_result.all_passed

        return WizardResult(
            success=all_ok,
            state=self.state,
            message="Check complete" if all_ok else "Some checks failed"
        )

    def _print_header(self):
        """Print wizard header."""
        self.console.print(Panel(
            "[bold blue]DAG Harness Bootstrap Wizard[/bold blue]\n\n"
            "This wizard will guide you through setting up the harness.",
            title="Welcome"
        ))

    def _step_detect_env(self):
        """Step 1: Detect environment."""
        self.state.current_step = WizardStep.DETECT_ENV
        self.console.print("\n[bold]Step 1: Detecting Environment[/bold]\n")

        # Check Python version
        py_version = sys.version_info
        self.console.print(f"  Python: {py_version.major}.{py_version.minor}.{py_version.micro}")

        if py_version < (3, 10):
            self.state.warnings.append("Python 3.10+ recommended")

        # Check if in git repo
        git_root = self.path_resolver._find_git_root()
        if git_root:
            self.console.print(f"  Git root: {git_root}")
        else:
            self.state.warnings.append("Not in a git repository")

        # Check for harness package
        # First try repo root structure (harness/harness)
        harness_pkg = self.project_root / "harness" / "harness"
        # Fall back to being run from within harness/ directory (harness/)
        if not harness_pkg.exists():
            harness_pkg = self.project_root / "harness"
        if harness_pkg.exists() and (harness_pkg / "__init__.py").exists():
            self.console.print(f"  Harness package: {harness_pkg}")
        else:
            self.state.errors.append(f"Harness package not found at {harness_pkg}")

        self.console.print()

    def _step_check_credentials(self):
        """Step 2: Check and configure credentials."""
        self.state.current_step = WizardStep.CHECK_CREDENTIALS
        self.console.print("\n[bold]Step 2: Checking Credentials[/bold]\n")

        result = self.credential_discovery.check_all(validate=True)

        table = Table(show_header=True, header_style="bold")
        table.add_column("Credential")
        table.add_column("Status")
        table.add_column("Source")

        for cred in result.credentials:
            if cred.status == CredentialStatus.FOUND:
                status = "[green]Found[/green]"
            elif cred.status == CredentialStatus.OPTIONAL_MISSING:
                status = "[dim]Not set (optional)[/dim]"
            elif cred.status == CredentialStatus.INVALID:
                status = "[red]Invalid[/red]"
            else:
                status = "[yellow]Missing[/yellow]"

            source = cred.source or "-"
            table.add_row(cred.name, status, source)

        self.console.print(table)

        if result.all_required_present:
            self.console.print("\n[green]All required credentials found.[/green]")
            self.state.credentials_ok = True
        else:
            missing = result.missing_required
            self.console.print(f"\n[yellow]Missing required credentials: {', '.join(missing)}[/yellow]")

            if self.interactive:
                self.console.print("\n[dim]Set these environment variables and re-run bootstrap.[/dim]")
                self.console.print("[dim]Example: export GITLAB_TOKEN=glpat-xxx[/dim]")

            self.state.warnings.append(f"Missing credentials: {', '.join(missing)}")

    def _step_check_paths(self):
        """Step 3: Check and configure paths."""
        self.state.current_step = WizardStep.CHECK_PATHS
        self.console.print("\n[bold]Step 3: Checking Paths[/bold]\n")

        result = self.path_resolver.check_all()

        for path_result in result.paths:
            if path_result.status in (PathStatus.VALID, PathStatus.WRITABLE):
                status = "[green]OK[/green]"
            elif path_result.status == PathStatus.NOT_FOUND:
                status = "[yellow]Not found[/yellow]"
            else:
                status = "[red]Invalid[/red]"

            self.console.print(f"  {status} {path_result.name}")
            if path_result.path:
                self.console.print(f"      [dim]{path_result.path}[/dim]")
            if path_result.error:
                self.console.print(f"      [yellow]{path_result.error}[/yellow]")

        if result.all_valid:
            self.console.print("\n[green]All paths are valid.[/green]")
            self.state.paths_ok = True
        else:
            for name in result.missing:
                self.state.errors.append(f"Invalid path: {name}")

    def _step_init_database(self):
        """Step 4: Initialize database."""
        self.state.current_step = WizardStep.INIT_DATABASE
        self.console.print("\n[bold]Step 4: Initializing Database[/bold]\n")

        try:
            from harness.db.state import StateDB

            db_path = os.environ.get(
                "HARNESS_DB_PATH",
                str(self.project_root / "harness" / "harness.db")
            )

            self.console.print(f"  Database path: {db_path}")

            db = StateDB(db_path)

            # Validate schema
            schema_result = db.validate_schema()
            if schema_result.get("valid"):
                self.console.print(f"  Tables: {schema_result.get('table_count', 0)}")
                self.console.print(f"  Indexes: {schema_result.get('index_count', 0)}")
                self.console.print("\n[green]Database initialized successfully.[/green]")
                self.state.database_ok = True
            else:
                self.console.print("\n[yellow]Database schema has issues.[/yellow]")
                self.state.warnings.append("Database schema incomplete")
                self.state.database_ok = True  # Can still continue

        except Exception as e:
            self.console.print(f"\n[red]Database initialization failed: {e}[/red]")
            self.state.errors.append(f"Database init failed: {e}")

    def _step_install_claude(self):
        """Step 5: Install MCP client integration."""
        self.state.current_step = WizardStep.INSTALL_CLAUDE
        self.console.print("\n[bold]Step 5: Installing MCP client Integration[/bold]\n")

        try:
            from harness.install import MCPInstaller, InstallStatus

            installer = MCPInstaller(self.project_root)

            # Check current status
            status, _ = installer.check()

            if status == InstallStatus.COMPLETE:
                self.console.print("[green]MCP client integration already installed.[/green]")
                self.state.claude_ok = True
                return

            if status == InstallStatus.OUTDATED:
                self.console.print("[yellow]Upgrading existing installation...[/yellow]")
                result = installer.upgrade()
            else:
                self.console.print("Installing MCP client integration...")
                result = installer.install(force=False)

            # Show component status
            for comp in result.components:
                status_str = "[green]OK[/green]" if comp.installed else "[red]FAIL[/red]"
                self.console.print(f"  {status_str} {comp.name}")

            if result.success:
                self.console.print("\n[green]MCP client integration installed.[/green]")
                self.state.claude_ok = True
            else:
                for error in result.errors:
                    self.state.errors.append(error)
                for warning in result.warnings:
                    self.state.warnings.append(warning)

        except Exception as e:
            self.console.print(f"\n[red]Installation failed: {e}[/red]")
            self.state.errors.append(f"Installation failed: {e}")

    def _step_run_selftests(self):
        """Step 6: Run self-tests."""
        self.state.current_step = WizardStep.RUN_SELFTESTS
        self.console.print("\n[bold]Step 6: Running Self-Tests[/bold]\n")

        result = self.self_tester.run_all(quick=False)

        for test in result.tests:
            if test.status == TestStatus.PASS:
                icon = "[green]✓[/green]"
            elif test.status == TestStatus.FAIL:
                icon = "[red]✗[/red]"
            elif test.status == TestStatus.SKIP:
                icon = "[dim]○[/dim]"
            else:
                icon = "[yellow]⚠[/yellow]"

            self.console.print(f"  {icon} {test.name}")
            if test.details and test.status != TestStatus.PASS:
                self.console.print(f"      [dim]{test.details}[/dim]")

        self.console.print()
        self.console.print(
            f"  Passed: {result.passed_count} | "
            f"Failed: {result.failed_count} | "
            f"Warnings: {result.warn_count}"
        )

        if result.all_passed:
            self.console.print("\n[green]All self-tests passed.[/green]")
            self.state.selftests_ok = True
        else:
            self.console.print("\n[yellow]Some tests had issues.[/yellow]")
            self.state.warnings.append(f"{result.failed_count} self-tests failed")
            # Tests with warnings still count as ok
            self.state.selftests_ok = result.failed_count == 0

    def _status_icon(self, ok: bool) -> str:
        """Get status icon."""
        return "[green]✓[/green]" if ok else "[red]✗[/red]"

    def _make_result(self, success: bool, message: str) -> WizardResult:
        """Create wizard result."""
        if success:
            self.console.print(Panel(
                f"[green]{message}[/green]\n\n"
                "The harness is ready to use.\n"
                "Try: harness status",
                title="Bootstrap Complete"
            ))
        else:
            error_list = "\n".join(f"  - {e}" for e in self.state.errors)
            self.console.print(Panel(
                f"[red]{message}[/red]\n\n"
                f"Errors:\n{error_list}",
                title="Bootstrap Failed"
            ))

        return WizardResult(
            success=success,
            state=self.state,
            message=message
        )
