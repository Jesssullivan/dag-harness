"""
CLI entry point for harness operations.

Usage:
    harness box-up-role <role-name>
    harness status [role-name]
    harness sync
    harness list-roles [--wave N]
    harness deps <role-name> [--reverse] [--transitive]
    harness worktrees
    harness resume <execution-id>
    harness mcp-server
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from harness.config import HarnessConfig
from harness.db.state import StateDB
from harness.dag.graph import create_box_up_role_graph
from harness.gitlab.api import GitLabClient

app = typer.Typer(
    name="harness",
    help="DAG-based orchestration harness for Ansible role deployment"
)
console = Console()


def get_db(config: Optional[HarnessConfig] = None) -> StateDB:
    """Get database instance."""
    cfg = config or HarnessConfig.load()
    return StateDB(cfg.db_path)


@app.command("box-up-role")
def box_up_role(
    role_name: str = typer.Argument(..., help="Name of the Ansible role"),
    breakpoints: Optional[str] = typer.Option(None, "--breakpoints", "-b", help="Comma-separated node names to pause before"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes")
):
    """Execute the box-up-role workflow for a role."""
    config = HarnessConfig.load()
    db = get_db(config)

    # Check role exists using configured repo_root
    repo_root = Path(config.repo_root)
    role_path = repo_root / "ansible" / "roles" / role_name
    if not role_path.exists():
        console.print(f"[red]Error:[/red] Role '{role_name}' not found at {role_path}")
        raise typer.Exit(1)

    if dry_run:
        console.print(f"[yellow]DRY RUN:[/yellow] Would execute box-up-role for '{role_name}'")
        console.print("\nWorkflow nodes:")
        graph = create_box_up_role_graph(db)
        for name, node_def in graph.nodes.items():
            console.print(f"  - {name}: {node_def.node.description}")
        return

    console.print(f"[blue]Starting box-up-role workflow for:[/blue] {role_name}")

    # Parse breakpoints
    bp_set = set(breakpoints.split(",")) if breakpoints else None

    # Execute workflow
    graph = create_box_up_role_graph(db)

    # Add event handler for progress output
    def event_handler(event):
        if event.event_type == "node_started":
            console.print(f"  [blue]→[/blue] {event.node_name}")
        elif event.event_type == "node_completed":
            console.print(f"  [green]✓[/green] {event.node_name}")
        elif event.event_type == "node_failed":
            console.print(f"  [red]✗[/red] {event.node_name}: {event.data.get('error')}")
        elif event.event_type == "node_skipped":
            console.print(f"  [yellow]○[/yellow] {event.node_name} (skipped)")

    graph.add_event_handler(event_handler)

    result = asyncio.run(graph.execute(role_name, breakpoints=bp_set))

    # Output result
    if result["status"] == "completed":
        console.print("\n[green]Workflow completed successfully![/green]")
        summary = result.get("summary", {})
        if summary:
            console.print(f"\n  Issue URL: {summary.get('issue_url', 'N/A')}")
            console.print(f"  MR URL: {summary.get('mr_url', 'N/A')}")
            console.print(f"  Worktree: {summary.get('worktree_path', 'N/A')}")
    elif result["status"] == "paused":
        console.print(f"\n[yellow]Workflow paused at:[/yellow] {result.get('paused_at')}")
        console.print(f"Resume with: harness resume {result['execution_id']}")
    elif result["status"] == "human_needed":
        console.print(f"\n[yellow]Human input needed at:[/yellow] {result.get('paused_at')}")
        console.print(f"Resume with: harness resume {result['execution_id']}")
    else:
        console.print(f"\n[red]Workflow failed:[/red] {result.get('error')}")
        console.print(f"Failed at node: {result.get('failed_at')}")
        raise typer.Exit(1)


@app.command("status")
def status(
    role_name: Optional[str] = typer.Argument(None, help="Role name (optional, shows all if not specified)")
):
    """Show status of roles and their deployments."""
    db = get_db()

    if role_name:
        status = db.get_role_status(role_name)
        if not status:
            console.print(f"[red]Role not found:[/red] {role_name}")
            raise typer.Exit(1)

        console.print(f"\n[bold]{status.name}[/bold] (Wave {status.wave}: {status.wave_name or ''})")
        console.print(f"  Worktree: {status.worktree_status or 'None'}")
        if status.commits_ahead or status.commits_behind:
            console.print(f"    {status.commits_ahead} ahead, {status.commits_behind} behind")
        console.print(f"  Issue: {status.issue_state or 'None'} - {status.issue_url or 'N/A'}")
        console.print(f"  MR: {status.mr_state or 'None'} - {status.mr_url or 'N/A'}")
        console.print(f"  Tests: {status.passed_tests} passed, {status.failed_tests} failed")
    else:
        statuses = db.list_role_statuses()
        if not statuses:
            console.print("[yellow]No roles in database. Run 'harness sync' first.[/yellow]")
            return

        table = Table(title="Role Status")
        table.add_column("Role", style="cyan")
        table.add_column("Wave")
        table.add_column("Worktree")
        table.add_column("Issue")
        table.add_column("MR")
        table.add_column("Tests")

        for s in statuses:
            wt_status = s.worktree_status or "-"
            if s.worktree_status == "dirty":
                wt_status = "[yellow]dirty[/yellow]"
            elif s.worktree_status == "stale":
                wt_status = "[red]stale[/red]"
            elif s.worktree_status == "active":
                wt_status = "[green]active[/green]"

            table.add_row(
                s.name,
                str(s.wave),
                wt_status,
                s.issue_state or "-",
                s.mr_state or "-",
                f"{s.passed_tests}/{s.passed_tests + s.failed_tests}"
            )

        console.print(table)


@app.command("sync")
def sync(
    roles: bool = typer.Option(True, "--roles/--no-roles", help="Sync roles from filesystem"),
    worktrees: bool = typer.Option(True, "--worktrees/--no-worktrees", help="Sync worktrees from git"),
    gitlab: bool = typer.Option(False, "--gitlab", help="Sync issues/MRs from GitLab")
):
    """Sync state from filesystem and GitLab."""
    db = get_db()

    if roles:
        console.print("[blue]Syncing roles from filesystem...[/blue]")
        import yaml
        from harness.config import HarnessConfig

        # Load config to get repo_root
        config = HarnessConfig.load()
        repo_root = Path(config.repo_root)
        roles_path = repo_root / "ansible" / "roles"

        if not roles_path.exists():
            console.print(f"[red]Roles path not found: {roles_path}[/red]")
            console.print("[yellow]Check repo_root in harness.yml[/yellow]")
            raise typer.Exit(1)
        added = updated = 0
        for role_dir in roles_path.iterdir():
            if not role_dir.is_dir() or role_dir.name.startswith("_"):
                continue

            from harness.db.models import Role
            wave_num, _ = config.get_wave_for_role(role_dir.name)
            role = Role(
                name=role_dir.name,
                wave=wave_num,
                molecule_path=str(role_dir / "molecule") if (role_dir / "molecule").exists() else None,
                has_molecule_tests=(role_dir / "molecule").exists()
            )

            meta_path = role_dir / "meta" / "main.yml"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = yaml.safe_load(f)
                        if meta and "galaxy_info" in meta:
                            role.description = meta["galaxy_info"].get("description", "")
                except Exception:
                    pass

            existing = db.get_role(role.name)
            db.upsert_role(role)
            if existing:
                updated += 1
            else:
                added += 1

        console.print(f"  Added {added}, updated {updated} roles")

    if worktrees:
        console.print("[blue]Syncing worktrees from git...[/blue]")
        import subprocess
        from harness.db.models import Worktree, WorktreeStatus

        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True
        )

        synced = 0
        wt_path = None
        for line in result.stdout.split("\n"):
            if line.startswith("worktree "):
                wt_path = line.split(" ", 1)[1]
            elif line.startswith("branch ") and wt_path:
                branch = line.split(" ", 1)[1].replace("refs/heads/", "")
                if branch.startswith("sid/"):
                    role_name = branch.replace("sid/", "")
                    role = db.get_role(role_name)
                    if role and role.id:
                        worktree = Worktree(
                            role_id=role.id,
                            path=wt_path,
                            branch=branch,
                            status=WorktreeStatus.ACTIVE
                        )
                        db.upsert_worktree(worktree)
                        synced += 1
                wt_path = None

        console.print(f"  Synced {synced} worktrees")

    if gitlab:
        console.print("[blue]Syncing from GitLab...[/blue]")
        client = GitLabClient(db)
        issues = client.sync_issues()
        mrs = client.sync_merge_requests()
        console.print(f"  Synced {issues} issues, {mrs} merge requests")


@app.command("list-roles")
def list_roles(
    wave: Optional[int] = typer.Option(None, "--wave", "-w", help="Filter by wave number")
):
    """List all Ansible roles."""
    db = get_db()
    roles = db.list_roles(wave=wave)

    if not roles:
        console.print("[yellow]No roles found. Run 'harness sync' first.[/yellow]")
        return

    table = Table(title=f"Ansible Roles{f' (Wave {wave})' if wave is not None else ''}")
    table.add_column("Name", style="cyan")
    table.add_column("Wave")
    table.add_column("Has Molecule")
    table.add_column("Description")

    for role in roles:
        table.add_row(
            role.name,
            str(role.wave),
            "[green]✓[/green]" if role.has_molecule_tests else "[red]✗[/red]",
            (role.description or "")[:50]
        )

    console.print(table)


@app.command("deps")
def deps(
    role_name: str = typer.Argument(..., help="Name of the role"),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Show reverse dependencies"),
    transitive: bool = typer.Option(False, "--transitive", "-t", help="Include transitive dependencies")
):
    """Show dependencies for a role."""
    db = get_db()

    if reverse:
        deps_list = db.get_reverse_dependencies(role_name, transitive=transitive)
        title = f"Roles that depend on {role_name}"
    else:
        deps_list = db.get_dependencies(role_name, transitive=transitive)
        title = f"Dependencies of {role_name}"

    if not deps_list:
        console.print(f"[yellow]No {'reverse ' if reverse else ''}dependencies found for {role_name}[/yellow]")
        return

    tree = Tree(f"[bold]{title}[/bold]")
    current_depth = 0
    current_branch = tree

    for name, depth in deps_list:
        if transitive:
            # Build tree structure
            prefix = "  " * (depth - 1)
            console.print(f"{prefix}└─ {name} (depth {depth})")
        else:
            console.print(f"  - {name}")


@app.command("worktrees")
def worktrees(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """List all git worktrees."""
    db = get_db()
    wts = db.list_worktrees()

    if json_output:
        print(json.dumps([w.model_dump() for w in wts], default=str, indent=2))
        return

    if not wts:
        console.print("[yellow]No worktrees found.[/yellow]")
        return

    table = Table(title="Git Worktrees")
    table.add_column("Branch", style="cyan")
    table.add_column("Path")
    table.add_column("Ahead")
    table.add_column("Behind")
    table.add_column("Changes")
    table.add_column("Status")

    for wt in wts:
        status = wt.status.value
        if wt.status.value == "dirty":
            status = "[yellow]dirty[/yellow]"
        elif wt.status.value == "stale":
            status = "[red]stale[/red]"
        elif wt.status.value == "active":
            status = "[green]active[/green]"

        table.add_row(
            wt.branch,
            wt.path,
            str(wt.commits_ahead),
            str(wt.commits_behind),
            str(wt.uncommitted_changes),
            status
        )

    console.print(table)


@app.command("resume")
def resume(
    execution_id: int = typer.Argument(..., help="Execution ID to resume"),
    breakpoints: Optional[str] = typer.Option(None, "--breakpoints", "-b", help="Comma-separated node names to pause before")
):
    """Resume a paused workflow execution."""
    config = HarnessConfig.load()
    db = get_db(config)

    # Get execution info
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT we.*, r.name as role_name
            FROM workflow_executions we
            JOIN roles r ON we.role_id = r.id
            WHERE we.id = ?
            """,
            (execution_id,)
        ).fetchone()

        if not row:
            console.print(f"[red]Execution {execution_id} not found[/red]")
            raise typer.Exit(1)

        if row["status"] not in ("paused", "failed"):
            console.print(f"[yellow]Execution is {row['status']}, not resumable[/yellow]")
            raise typer.Exit(1)

    console.print(f"[blue]Resuming execution {execution_id} for {row['role_name']}[/blue]")
    console.print(f"  Last node: {row['current_node']}")

    graph = create_box_up_role_graph(db)

    def event_handler(event):
        if event.event_type == "node_started":
            console.print(f"  [blue]→[/blue] {event.node_name}")
        elif event.event_type == "node_completed":
            console.print(f"  [green]✓[/green] {event.node_name}")
        elif event.event_type == "node_failed":
            console.print(f"  [red]✗[/red] {event.node_name}: {event.data.get('error')}")

    graph.add_event_handler(event_handler)

    bp_set = set(breakpoints.split(",")) if breakpoints else None
    result = asyncio.run(graph.execute(row["role_name"], resume_from=execution_id, breakpoints=bp_set))

    if result["status"] == "completed":
        console.print("\n[green]Workflow completed successfully![/green]")
    elif result["status"] == "paused":
        console.print(f"\n[yellow]Workflow paused at:[/yellow] {result.get('paused_at')}")
    else:
        console.print(f"\n[red]Workflow failed:[/red] {result.get('error')}")
        raise typer.Exit(1)


@app.command("mcp-server")
def mcp_server():
    """Start the MCP server for MCP client integration."""
    from harness.mcp.server import create_mcp_server

    console.print("[blue]Starting harness MCP server...[/blue]")
    mcp = create_mcp_server()
    mcp.run()


@app.command("init")
def init(
    config_path: str = typer.Option("harness.yml", "--config", "-c", help="Path to save config")
):
    """Initialize harness configuration."""
    config = HarnessConfig()

    # Check if we're in the right directory
    if not Path("ansible/roles").exists():
        console.print("[yellow]Warning: ansible/roles not found. Are you in the EMS repo root?[/yellow]")

    config.save(config_path)
    console.print(f"[green]Configuration saved to {config_path}[/green]")

    # Initialize database
    db = StateDB(config.db_path)
    console.print(f"[green]Database initialized at {config.db_path}[/green]")


@app.command("graph")
def show_graph(
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, mermaid")
):
    """Show the box-up-role workflow graph."""
    db = get_db()
    graph = create_box_up_role_graph(db)

    if format == "json":
        print(json.dumps(graph.to_dict(), indent=2))
    elif format == "mermaid":
        console.print("```mermaid")
        console.print("graph TD")
        for name, node_def in graph.nodes.items():
            for result, edge in node_def.edges.items():
                if isinstance(edge, str):
                    console.print(f"    {name} -->|{result.value}| {edge}")
        console.print("```")
    else:
        console.print("[bold]Box-Up-Role Workflow[/bold]\n")
        for name, node_def in graph.nodes.items():
            prefix = "[green]▶[/green]" if name == graph.entry_point else " "
            suffix = "[red]■[/red]" if name in graph.terminal_nodes else ""
            console.print(f"{prefix} {name} {suffix}")
            console.print(f"    {node_def.node.description}")
            for result, edge in node_def.edges.items():
                if isinstance(edge, str):
                    console.print(f"    └─ {result.value} → {edge}")


# =========================================================================
# SELF-CHECK COMMANDS
# =========================================================================

@app.command("check")
def check(
    schema: bool = typer.Option(True, "--schema/--no-schema", help="Validate schema"),
    data: bool = typer.Option(True, "--data/--no-data", help="Validate data integrity"),
    graph: bool = typer.Option(True, "--graph/--no-graph", help="Validate dependency graph"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """Run self-checks on the harness database."""
    db = get_db()
    results = {"passed": True, "checks": {}}

    if schema:
        schema_result = db.validate_schema()
        results["checks"]["schema"] = schema_result
        if not schema_result["valid"]:
            results["passed"] = False

    if data:
        data_result = db.validate_data_integrity()
        results["checks"]["data"] = data_result
        if not data_result["valid"]:
            results["passed"] = False

    if graph:
        graph_result = db.validate_dependencies()
        results["checks"]["graph"] = graph_result
        if not graph_result["valid"]:
            results["passed"] = False

    if json_output:
        print(json.dumps(results, indent=2, default=str))
        return

    # Pretty output
    for check_name, check_result in results["checks"].items():
        status = "[green]✓[/green]" if check_result.get("valid", True) else "[red]✗[/red]"
        console.print(f"{status} {check_name.capitalize()} check")

        if not check_result.get("valid"):
            for key, value in check_result.items():
                if key != "valid" and value:
                    console.print(f"    {key}: {value}")

    if results["passed"]:
        console.print("\n[green]All checks passed![/green]")
    else:
        console.print("\n[red]Some checks failed![/red]")
        raise typer.Exit(1)


# =========================================================================
# DATABASE MANAGEMENT COMMANDS
# =========================================================================

db_app = typer.Typer(name="db", help="Database management commands")
app.add_typer(db_app)


@db_app.command("stats")
def db_stats(json_output: bool = typer.Option(False, "--json", help="Output as JSON")):
    """Show database statistics."""
    db = get_db()
    stats = db.get_statistics()

    if json_output:
        print(json.dumps(stats, indent=2))
        return

    table = Table(title="Database Statistics")
    table.add_column("Table/Metric")
    table.add_column("Value", justify="right")

    for key, value in stats.items():
        if key == "db_size_bytes":
            display_value = f"{value / 1024:.1f} KB"
        elif isinstance(value, int) and value < 0:
            display_value = "[red]error[/red]"
        else:
            display_value = str(value)
        table.add_row(key, display_value)

    console.print(table)


@db_app.command("reset")
def db_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm reset without prompting")
):
    """Reset database to initial state (DESTRUCTIVE)."""
    if not yes:
        confirm = typer.confirm("This will DELETE ALL DATA. Are you sure?")
        if not confirm:
            raise typer.Abort()

    db = get_db()
    db.reset_database(confirm=True)
    console.print("[green]Database reset complete.[/green]")


@db_app.command("clear")
def db_clear(
    table: str = typer.Argument(..., help="Table to clear (audit_log, test_runs, etc.)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm without prompting")
):
    """Clear a specific table (audit_log, test_runs, workflow_executions, etc.)."""
    if not yes:
        confirm = typer.confirm(f"Clear all data from {table}?")
        if not confirm:
            raise typer.Abort()

    db = get_db()
    try:
        count = db.clear_table(table, confirm=True)
        console.print(f"[green]Cleared {count} rows from {table}.[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@db_app.command("vacuum")
def db_vacuum():
    """Vacuum database to reclaim space."""
    db = get_db()
    saved = db.vacuum()
    if saved > 0:
        console.print(f"[green]Vacuum complete. Saved {saved / 1024:.1f} KB.[/green]")
    else:
        console.print("[green]Vacuum complete.[/green]")


@db_app.command("backup")
def db_backup(
    output: str = typer.Argument(..., help="Backup file path")
):
    """Create a backup of the database."""
    db = get_db()
    if db.backup(output):
        console.print(f"[green]Database backed up to {output}[/green]")
    else:
        console.print("[red]Backup failed (in-memory database?)[/red]")
        raise typer.Exit(1)


@db_app.command("info")
def db_info():
    """Show database file information."""
    config = HarnessConfig.load()
    db_path = Path(config.db_path)

    console.print(f"[bold]Database Path:[/bold] {db_path}")
    console.print(f"[bold]Exists:[/bold] {db_path.exists()}")

    if db_path.exists():
        stats = db_path.stat()
        console.print(f"[bold]Size:[/bold] {stats.st_size / 1024:.1f} KB")
        console.print(f"[bold]Modified:[/bold] {stats.st_mtime}")

    # Schema validation
    db = get_db(config)
    schema = db.validate_schema()
    console.print(f"[bold]Tables:[/bold] {schema['table_count']}")
    console.print(f"[bold]Indexes:[/bold] {schema['index_count']}")
    console.print(f"[bold]Schema Valid:[/bold] {'[green]Yes[/green]' if schema['valid'] else '[red]No[/red]'}")


# =========================================================================
# METRICS COMMANDS
# =========================================================================

metrics_app = typer.Typer(name="metrics", help="Golden metrics management")
app.add_typer(metrics_app)


@metrics_app.command("status")
def metrics_status(json_output: bool = typer.Option(False, "--json", help="Output as JSON")):
    """Show current status of all golden metrics."""
    from harness.metrics.golden import GoldenMetricsTracker

    db = get_db()
    tracker = GoldenMetricsTracker(db)
    health = tracker.get_health()

    if json_output:
        print(json.dumps(health, indent=2))
        return

    # Overall health header
    overall = health["overall"]
    if overall == "healthy":
        console.print(f"\n[bold green]Overall Health: HEALTHY[/bold green]")
    elif overall == "warning":
        console.print(f"\n[bold yellow]Overall Health: WARNING[/bold yellow]")
    elif overall == "critical":
        console.print(f"\n[bold red]Overall Health: CRITICAL[/bold red]")
    else:
        console.print(f"\n[bold dim]Overall Health: UNKNOWN[/bold dim]")

    console.print(f"  OK: {health['ok']} | Warning: {health['warning']} | Critical: {health['critical']} | Unknown: {health['unknown']}\n")

    # Individual metrics table
    table = Table(title="Golden Metrics Status")
    table.add_column("Metric")
    table.add_column("Status")
    table.add_column("Description")

    metrics = tracker.list_metrics()
    for metric in sorted(metrics, key=lambda m: m.name):
        status = health["metrics"].get(metric.name, "unknown")

        if status == "ok":
            status_str = "[green]OK[/green]"
        elif status == "warning":
            status_str = "[yellow]WARNING[/yellow]"
        elif status == "critical":
            status_str = "[red]CRITICAL[/red]"
        else:
            status_str = "[dim]unknown[/dim]"

        table.add_row(metric.name, status_str, metric.description or "")

    console.print(table)


@metrics_app.command("record")
def metrics_record(
    name: str = typer.Argument(..., help="Metric name"),
    value: float = typer.Argument(..., help="Metric value"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="JSON context metadata")
):
    """Record a metric value."""
    from harness.metrics.golden import GoldenMetricsTracker

    db = get_db()
    tracker = GoldenMetricsTracker(db)

    ctx = json.loads(context) if context else None

    try:
        status = tracker.record(name, value, context=ctx)

        if status == "ok":
            console.print(f"[green]✓[/green] {name} = {value} (OK)")
        elif status == "warning":
            console.print(f"[yellow]⚠[/yellow] {name} = {value} (WARNING)")
        else:
            console.print(f"[red]✗[/red] {name} = {value} (CRITICAL)")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@metrics_app.command("history")
def metrics_history(
    name: str = typer.Argument(..., help="Metric name"),
    hours: int = typer.Option(24, "--hours", "-h", help="Hours to look back"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum entries to show")
):
    """Show recent history for a metric."""
    from harness.metrics.golden import GoldenMetricsTracker

    db = get_db()
    tracker = GoldenMetricsTracker(db)
    history = tracker.get_recent(name, hours=hours)

    if not history:
        console.print(f"[yellow]No data for {name} in last {hours} hours[/yellow]")
        return

    table = Table(title=f"{name} - Last {hours} hours")
    table.add_column("Time")
    table.add_column("Value", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Status")

    for entry in history[:limit]:
        timestamp = entry.get("recorded_at", "")[:19]  # Trim to datetime
        value = entry.get("value", 0)
        baseline = entry.get("baseline")
        status = entry.get("status", "unknown")

        if status == "ok":
            status_str = "[green]ok[/green]"
        elif status == "warning":
            status_str = "[yellow]warning[/yellow]"
        elif status == "critical":
            status_str = "[red]critical[/red]"
        else:
            status_str = "[dim]unknown[/dim]"

        table.add_row(
            timestamp,
            f"{value:.3f}",
            f"{baseline:.3f}" if baseline is not None else "-",
            status_str
        )

    console.print(table)

    if len(history) > limit:
        console.print(f"[dim]... and {len(history) - limit} more entries[/dim]")


@metrics_app.command("trend")
def metrics_trend(
    name: str = typer.Argument(..., help="Metric name"),
    hours: int = typer.Option(24, "--hours", "-h", help="Hours to analyze")
):
    """Show trend analysis for a metric."""
    from harness.metrics.golden import GoldenMetricsTracker

    db = get_db()
    tracker = GoldenMetricsTracker(db)
    trend = tracker.get_trend(name, hours=hours)

    if trend["count"] == 0:
        console.print(f"[yellow]No data for {name} in last {hours} hours[/yellow]")
        return

    console.print(f"\n[bold]{name} Trend Analysis[/bold] ({hours}h window)\n")
    console.print(f"  Samples: {trend['count']}")
    console.print(f"  Latest:  {trend['latest']:.3f}")
    console.print(f"  Average: {trend['average']:.3f}")
    console.print(f"  Min:     {trend['min']:.3f}")
    console.print(f"  Max:     {trend['max']:.3f}")

    trend_str = trend["trend"]
    if trend_str == "increasing":
        console.print(f"  Trend:   [red]↑ Increasing[/red]")
    elif trend_str == "decreasing":
        console.print(f"  Trend:   [green]↓ Decreasing[/green]")
    elif trend_str == "stable":
        console.print(f"  Trend:   [blue]→ Stable[/blue]")
    else:
        console.print(f"  Trend:   [dim]? Unknown[/dim]")


@metrics_app.command("list")
def metrics_list():
    """List all registered golden metrics."""
    from harness.metrics.golden import GoldenMetricsTracker

    db = get_db()
    tracker = GoldenMetricsTracker(db)
    metrics = tracker.list_metrics()

    table = Table(title="Registered Golden Metrics")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Baseline", justify="right")
    table.add_column("Warning", justify="right")
    table.add_column("Critical", justify="right")
    table.add_column("Unit")

    for m in sorted(metrics, key=lambda x: x.name):
        table.add_row(
            m.name,
            m.metric_type.value,
            f"{m.baseline_value:.2f}",
            f"{m.warning_threshold:.2f}x",
            f"{m.critical_threshold:.2f}x",
            m.unit or "-"
        )

    console.print(table)


@metrics_app.command("purge")
def metrics_purge(
    days: int = typer.Option(30, "--days", "-d", help="Purge records older than N days"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm without prompting")
):
    """Purge old metric records."""
    from harness.metrics.golden import GoldenMetricsTracker

    if not yes:
        confirm = typer.confirm(f"Delete metric records older than {days} days?")
        if not confirm:
            raise typer.Abort()

    db = get_db()
    tracker = GoldenMetricsTracker(db)
    count = tracker.purge_old(days=days)
    console.print(f"[green]Purged {count} records older than {days} days.[/green]")


# =============================================================================
# HOTL (Human Out of The Loop) Commands
# =============================================================================

hotl_app = typer.Typer(name="hotl", help="HOTL autonomous operation commands")
app.add_typer(hotl_app)


@hotl_app.command("start")
def hotl_start(
    max_iterations: int = typer.Option(100, "--max-iterations", "-n", help="Maximum iterations"),
    notify_interval: int = typer.Option(300, "--notify-interval", "-i", help="Seconds between notifications"),
    discord_webhook: Optional[str] = typer.Option(None, "--discord", envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL"),
    email_to: Optional[str] = typer.Option(None, "--email", envvar="HOTL_EMAIL_TO", help="Email recipient"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background")
):
    """Start HOTL autonomous operation mode."""
    from harness.hotl.supervisor import HOTLSupervisor

    db = get_db()

    config = {}
    if discord_webhook:
        config["discord_webhook_url"] = discord_webhook
    if email_to:
        config["email_to"] = email_to

    supervisor = HOTLSupervisor(db, config=config)

    console.print(f"[bold blue]Starting HOTL Mode[/bold blue]")
    console.print(f"  Max iterations: {max_iterations}")
    console.print(f"  Notify interval: {notify_interval}s")
    console.print(f"  Discord: {'configured' if discord_webhook else 'not configured'}")
    console.print(f"  Email: {email_to or 'not configured'}")
    console.print()

    if background:
        console.print("[yellow]Background mode not yet implemented. Running in foreground.[/yellow]")

    try:
        # Run synchronously - supervisor.run() is now sync
        final_state = supervisor.run(
            max_iterations=max_iterations,
            notification_interval=notify_interval
        )

        console.print("\n[bold green]HOTL completed[/bold green]")
        console.print(f"  Final phase: {final_state.get('phase', 'unknown')}")
        console.print(f"  Iterations: {final_state.get('iteration_count', 0)}")
        console.print(f"  Completed tasks: {len(final_state.get('completed_tasks', []))}")
        console.print(f"  Failed tasks: {len(final_state.get('failed_tasks', []))}")
        console.print(f"  Errors: {len(final_state.get('errors', []))}")

    except KeyboardInterrupt:
        console.print("\n[yellow]HOTL interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]HOTL error: {e}[/red]")
        raise typer.Exit(1)


@hotl_app.command("status")
def hotl_status(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """Show current HOTL status from database."""
    db = get_db()

    # Check for recent HOTL sessions in checkpointer/database
    # For now, show database statistics as a proxy for HOTL activity
    stats = db.get_statistics()

    status_info = {
        "pending_executions": stats.get("pending_executions", 0),
        "running_executions": stats.get("running_executions", 0),
        "active_regressions": stats.get("active_regressions", 0),
        "roles_count": stats.get("roles", 0),
        "test_runs": stats.get("test_runs", 0),
    }

    if json_output:
        print(json.dumps(status_info, indent=2))
        return

    console.print("\n[bold]HOTL Status[/bold]")
    console.print(f"  Pending executions: {status_info['pending_executions']}")
    console.print(f"  Running executions: {status_info['running_executions']}")
    console.print(f"  Active regressions: {status_info['active_regressions']}")
    console.print(f"  Total roles: {status_info['roles_count']}")
    console.print(f"  Test runs recorded: {status_info['test_runs']}")

    if status_info['running_executions'] > 0:
        console.print("\n[yellow]Note: HOTL may be running. Use Ctrl+C to stop if in foreground.[/yellow]")
    else:
        console.print("\n[dim]No active HOTL session detected.[/dim]")


@hotl_app.command("stop")
def hotl_stop(
    force: bool = typer.Option(False, "--force", "-f", help="Force stop (cancel running executions)")
):
    """Request HOTL to stop gracefully."""
    db = get_db()

    if force:
        # Cancel any running executions in the database
        from harness.db.models import WorkflowStatus
        with db.connection() as conn:
            count = conn.execute(
                "UPDATE workflow_executions SET status = ? WHERE status = ?",
                (WorkflowStatus.CANCELLED.value, WorkflowStatus.RUNNING.value)
            ).rowcount
        if count > 0:
            console.print(f"[yellow]Cancelled {count} running execution(s)[/yellow]")
        else:
            console.print("[dim]No running executions to cancel[/dim]")
    else:
        console.print("[dim]HOTL stop works by pressing Ctrl+C in the running session.[/dim]")
        console.print("[dim]Use --force to cancel running executions in the database.[/dim]")


@hotl_app.command("resume")
def hotl_resume(
    thread_id: str = typer.Argument(..., help="Thread ID to resume from checkpoint"),
    max_iterations: int = typer.Option(100, "--max-iterations", "-n"),
    notify_interval: int = typer.Option(300, "--notify-interval", "-i")
):
    """Resume HOTL from a checkpoint."""
    from harness.hotl.supervisor import HOTLSupervisor

    db = get_db()
    supervisor = HOTLSupervisor(db)

    console.print(f"[bold blue]Resuming HOTL from checkpoint: {thread_id}[/bold blue]")

    try:
        # Run synchronously - supervisor.run() is now sync
        final_state = supervisor.run(
            max_iterations=max_iterations,
            notification_interval=notify_interval,
            resume_from=thread_id
        )

        console.print("\n[bold green]HOTL resumed and completed[/bold green]")
        console.print(f"  Final phase: {final_state.get('phase', 'unknown')}")
        console.print(f"  Iterations: {final_state.get('iteration_count', 0)}")

    except Exception as e:
        console.print(f"\n[red]HOTL resume error: {e}[/red]")
        raise typer.Exit(1)


# =============================================================================
# INSTALL Commands (MCP client Integration)
# =============================================================================

install_app = typer.Typer(name="install", help="MCP client installation management")
app.add_typer(install_app)


@install_app.command("run")
def install_run(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    skip_hooks: bool = typer.Option(False, "--skip-hooks", help="Don't install hook scripts"),
    skip_skills: bool = typer.Option(False, "--skip-skills", help="Don't install skill definitions"),
    skip_mcp: bool = typer.Option(False, "--skip-mcp", help="Don't configure MCP server")
):
    """Install harness into MCP client.

    This command sets up the complete MCP client integration including:
    - MCP server configuration
    - Hook scripts (PreToolUse, PostToolUse)
    - Skill definitions (/box-up-role, /hotl, /observability)
    """
    from harness.install import MCPInstaller, print_install_result

    installer = MCPInstaller()
    result = installer.install(
        force=force,
        skip_hooks=skip_hooks,
        skip_skills=skip_skills,
        skip_mcp=skip_mcp
    )

    print_install_result(result)

    if not result.success:
        raise typer.Exit(1)


@install_app.command("check")
def install_check():
    """Verify MCP client installation status.

    Checks that all required components are installed:
    - Directory structure (.claude/hooks/, .claude/skills/)
    - MCP server configuration
    - Required hook scripts
    - Skill definitions
    """
    from harness.install import MCPInstaller, InstallStatus, print_check_result

    installer = MCPInstaller()
    status, components = installer.check()

    print_check_result(status, components)

    if status == InstallStatus.NOT_INSTALLED:
        console.print("\n[dim]Run 'harness install run' to install.[/dim]")
        raise typer.Exit(1)
    elif status == InstallStatus.PARTIAL:
        console.print("\n[dim]Run 'harness install run --force' to repair installation.[/dim]")
        raise typer.Exit(1)
    elif status == InstallStatus.OUTDATED:
        console.print("\n[dim]Run 'harness install upgrade' to update to latest version.[/dim]")


@install_app.command("uninstall")
def install_uninstall(
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm without prompting"),
    remove_data: bool = typer.Option(False, "--remove-data", help="Also remove data files")
):
    """Remove harness from MCP client.

    This removes:
    - MCP server configuration
    - Hook scripts
    - Skill definitions

    Use --remove-data to also remove databases and logs.
    """
    from harness.install import MCPInstaller

    if not yes:
        confirm = typer.confirm("Remove harness from MCP client?")
        if not confirm:
            raise typer.Abort()

    installer = MCPInstaller()
    if installer.uninstall(remove_data=remove_data):
        console.print("[green]Harness uninstalled successfully.[/green]")
    else:
        console.print("[red]Uninstall failed.[/red]")
        raise typer.Exit(1)


@install_app.command("upgrade")
def install_upgrade():
    """Upgrade existing MCP client installation.

    Re-installs hooks and skills while preserving custom settings.
    """
    from harness.install import MCPInstaller, print_install_result

    installer = MCPInstaller()
    result = installer.upgrade()

    print_install_result(result)

    if result.success:
        console.print("[green]Upgrade complete![/green]")
    else:
        raise typer.Exit(1)


# =============================================================================
# CREDENTIAL Commands
# =============================================================================

@app.command("scan-tokens")
def scan_tokens_cmd(
    save: bool = typer.Option(False, "--save", "-s", help="Save first valid token to .env file"),
    env_file: Optional[str] = typer.Option(None, "--env-file", "-e", help="Target .env file path (default: project .env)")
):
    """Scan local .env files for GitLab tokens and test them.

    Searches multiple locations for GITLAB_TOKEN, GL_TOKEN, or GLAB_TOKEN:
    - Project .env files
    - Home directory .env files
    - Sibling project directories
    - Config directories

    Each found token is tested against the GitLab API to verify:
    - Token validity (authentication)
    - Required scopes (api)
    - Associated username

    Examples:
        harness scan-tokens              # Find and test all tokens
        harness scan-tokens --save       # Save first valid token to .env
        harness scan-tokens -s -e ~/.env # Save to specific file
    """
    from harness.bootstrap.credentials import CredentialDiscovery
    from harness.config import HarnessConfig

    # Use harness config to get proper project root
    try:
        config = HarnessConfig.load()
        project_root = Path(config.repo_root)
    except Exception:
        project_root = Path.cwd()

    discovery = CredentialDiscovery(project_root=project_root)
    console.print("[bold]Scanning for GitLab tokens...[/bold]\n")

    results = discovery.scan_and_test_gitlab_tokens()

    if not results:
        console.print("[yellow]No GitLab tokens found in any .env files.[/yellow]")
        console.print("\n[dim]Checked locations:[/dim]")
        for path in discovery._get_env_file_search_paths()[:10]:
            console.print(f"[dim]  - {path}[/dim]")
        raise typer.Exit(1)

    # Display results in a table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Variable")
    table.add_column("Source")
    table.add_column("Token")
    table.add_column("Valid")
    table.add_column("User")
    table.add_column("Scopes")

    first_valid = None
    for r in results:
        valid_str = "[green]✓[/green]" if r["valid"] else "[red]✗[/red]"
        scopes_str = ", ".join(r["scopes"]) if r["scopes"] else r.get("error", "-")
        user_str = r.get("username") or "-"
        var_name = r.get("var_name") or "-"

        # Truncate source path for display
        source = r["source"]
        if source.startswith("env:"):
            source = "[env]"
        elif len(source) > 30:
            source = "..." + source[-27:]

        table.add_row(
            var_name[:25] + "..." if len(var_name) > 25 else var_name,
            source,
            r["token_prefix"],
            valid_str,
            user_str,
            scopes_str[:20] + "..." if len(scopes_str) > 20 else scopes_str
        )

        if r["valid"] and first_valid is None:
            first_valid = r

    console.print(table)

    # Summary
    valid_count = sum(1 for r in results if r["valid"])
    console.print(f"\n[bold]Found {len(results)} tokens, {valid_count} valid.[/bold]")

    # Save if requested
    if save and first_valid:
        target_file = Path(env_file) if env_file else None

        # Need to get the actual token value (not just prefix)
        # Re-read from source
        actual_token = None
        var_to_find = first_valid.get("var_name")
        token_prefix = first_valid["token_prefix"].replace("...", "")  # Get actual prefix

        if first_valid["source"] == "[env]" or first_valid["source"].startswith("env:"):
            # From environment
            import os
            actual_token = os.environ.get(var_to_find)
        else:
            # Read from file - need to handle 'export VAR=' format
            import re
            try:
                with open(first_valid["source"]) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # Handle both 'VAR=value' and 'export VAR=value'
                        clean_line = line.replace("export ", "").strip()
                        if "=" in clean_line:
                            key, _, val = clean_line.partition("=")
                            key = key.strip()
                            val = val.strip().strip('"').strip("'")
                            # Match by variable name or token prefix
                            if (key == var_to_find or val.startswith(token_prefix)) and val:
                                actual_token = val
                                break
            except Exception as e:
                console.print(f"[dim]Error reading source file: {e}[/dim]")

        if actual_token:
            if discovery.save_credential_to_env("GITLAB_TOKEN", actual_token, target_file):
                target_path = target_file or (discovery.project_root / ".env")
                console.print(f"\n[green]Saved valid token to {target_path}[/green]")
                console.print(f"[dim]Token from: {first_valid['var_name']} ({first_valid['username']})[/dim]")
            else:
                console.print("\n[red]Failed to save token to .env file[/red]")
                raise typer.Exit(1)
        else:
            console.print("\n[red]Could not retrieve token value for saving[/red]")
            console.print(f"[dim]Looking for var '{var_to_find}' with prefix '{token_prefix}' in {first_valid['source']}[/dim]")
            raise typer.Exit(1)
    elif save and not first_valid:
        console.print("\n[yellow]No valid token found to save.[/yellow]")
        raise typer.Exit(1)


# =============================================================================
# BOOTSTRAP Command
# =============================================================================

@app.command("bootstrap")
def bootstrap_cmd(
    check_only: bool = typer.Option(False, "--check-only", "-c", help="Only check current state, don't make changes"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick check (skip network tests)")
):
    """Bootstrap the harness with interactive setup.

    This command runs an interactive wizard that:
    - Detects your environment (Python, git, paths)
    - Checks for required credentials (GITLAB_TOKEN, etc.)
    - Validates paths (database, worktrees, etc.)
    - Initializes the database schema
    - Installs MCP client integration (MCP server, hooks, skills)
    - Runs self-tests to verify the setup

    Use --check-only to verify current state without making changes.

    Examples:
        harness bootstrap              # Full interactive setup
        harness bootstrap --check-only # Just check current state
        harness bootstrap --quick      # Skip network tests
    """
    from harness.bootstrap import BootstrapRunner

    runner = BootstrapRunner(interactive=True)

    if quick and not check_only:
        # Quick check for whether bootstrap is needed
        if runner.quick_check():
            console.print("[green]Harness appears to be properly configured.[/green]")
            console.print("[dim]Run 'harness bootstrap --check-only' for detailed status.[/dim]")
            return
        else:
            console.print("[yellow]Bootstrap may be needed. Running full bootstrap...[/yellow]\n")

    result = runner.run(check_only=check_only)

    if not result.success:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
