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
from datetime import UTC
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from harness import __version__
from harness.config import HarnessConfig
from harness.dag.langgraph_builder import create_box_up_role_graph
from harness.dag.langgraph_runner import LangGraphWorkflowRunner
from harness.db.state import StateDB
from harness.gitlab.api import GitLabClient


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(f"dag-harness {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="harness", help="DAG-based orchestration harness for Ansible role deployment"
)
console = Console()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-V", callback=version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    """DAG-based orchestration harness for Ansible role deployment."""
    pass


def get_db(config: HarnessConfig | None = None) -> StateDB:
    """Get database instance."""
    cfg = config or HarnessConfig.load()
    return StateDB(cfg.db_path)


@app.command("box-up-role")
def box_up_role(
    role_name: str = typer.Argument(..., help="Name of the Ansible role"),
    breakpoints: str | None = typer.Option(
        None, "--breakpoints", "-b", help="Comma-separated node names to pause before"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without making changes"
    ),
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
        # Show workflow nodes from LangGraph builder
        graph, _ = create_box_up_role_graph(config.db_path)
        for node_name in graph.nodes:
            if node_name not in ("__start__", "__end__"):
                console.print(f"  - {node_name}: {_get_node_description(node_name)}")
        return

    console.print(f"[blue]Starting box-up-role workflow for:[/blue] {role_name}")

    # Execute workflow using LangGraphWorkflowRunner
    runner = LangGraphWorkflowRunner(db=db, db_path=config.db_path)

    async def run_workflow():
        """Run workflow with progress via runner."""
        # Use runner's execute method which handles DB operations correctly
        result = await runner.execute(role_name)

        # Print progress based on completed nodes
        state = result.get("state", {})
        for node in state.get("completed_nodes", []):
            console.print(f"  [green]✓[/green] {node}")

        return result

    result = asyncio.run(run_workflow())

    # Output result
    if result["status"] == "completed":
        console.print("\n[green]Workflow completed successfully![/green]")
        state = result.get("state", {})
        console.print(f"\n  Issue URL: {state.get('issue_url', 'N/A')}")
        console.print(f"  MR URL: {state.get('mr_url', 'N/A')}")
        console.print(f"  Worktree: {state.get('worktree_path', 'N/A')}")
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


def _get_node_description(node_name: str) -> str:
    """Get description for a node name."""
    descriptions = {
        "validate_role": "Validate role exists and extract metadata",
        "analyze_dependencies": "Analyze role dependencies and credentials",
        "check_dependencies": "Verify upstream dependencies are boxed up first",
        "warn_reverse_deps": "Note downstream roles that may need updates",
        "create_worktree": "Create isolated git worktree",
        "run_molecule_tests": "Execute molecule tests (blocking)",
        "run_pytest_tests": "Execute pytest tests",
        "validate_deploy": "Validate deployment target",
        "create_commit": "Create signed commit",
        "push_branch": "Push branch to origin with tracking",
        "create_gitlab_issue": "Create GitLab issue",
        "create_merge_request": "Create GitLab merge request",
        "human_approval": "Wait for human approval",
        "add_to_merge_train": "Add MR to merge train",
        "report_summary": "Generate workflow summary",
        "notify_failure": "Send failure notification",
    }
    return descriptions.get(node_name, node_name)


@app.command("status")
def status(
    role_name: str | None = typer.Argument(
        None, help="Role name (optional, shows all if not specified)"
    ),
):
    """Show status of roles and their deployments."""
    db = get_db()

    if role_name:
        status = db.get_role_status(role_name)
        if not status:
            console.print(f"[red]Role not found:[/red] {role_name}")
            raise typer.Exit(1)

        console.print(
            f"\n[bold]{status.name}[/bold] (Wave {status.wave}: {status.wave_name or ''})"
        )
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
                f"{s.passed_tests}/{s.passed_tests + s.failed_tests}",
            )

        console.print(table)


@app.command("sync")
def sync(
    roles: bool = typer.Option(True, "--roles/--no-roles", help="Sync roles from filesystem"),
    worktrees: bool = typer.Option(
        True, "--worktrees/--no-worktrees", help="Sync worktrees from git"
    ),
    gitlab: bool = typer.Option(False, "--gitlab", help="Sync issues/MRs from GitLab"),
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
                molecule_path=str(role_dir / "molecule")
                if (role_dir / "molecule").exists()
                else None,
                has_molecule_tests=(role_dir / "molecule").exists(),
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
            ["git", "worktree", "list", "--porcelain"], capture_output=True, text=True
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
                            status=WorktreeStatus.ACTIVE,
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
def list_roles(wave: int | None = typer.Option(None, "--wave", "-w", help="Filter by wave number")):
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
            (role.description or "")[:50],
        )

    console.print(table)


@app.command("deps")
def deps(
    role_name: str = typer.Argument(..., help="Name of the role"),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Show reverse dependencies"),
    transitive: bool = typer.Option(
        False, "--transitive", "-t", help="Include transitive dependencies"
    ),
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
        console.print(
            f"[yellow]No {'reverse ' if reverse else ''}dependencies found for {role_name}[/yellow]"
        )
        return

    Tree(f"[bold]{title}[/bold]")

    for name, depth in deps_list:
        if transitive:
            # Build tree structure
            prefix = "  " * (depth - 1)
            console.print(f"{prefix}└─ {name} (depth {depth})")
        else:
            console.print(f"  - {name}")


@app.command("worktrees")
def worktrees(json_output: bool = typer.Option(False, "--json", help="Output as JSON")):
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
            status,
        )

    console.print(table)


@app.command("resume")
def resume(
    execution_id: int = typer.Argument(..., help="Execution ID to resume"),
    approve: bool = typer.Option(
        False, "--approve", "-a", help="Approve and continue (for HITL interrupts)"
    ),
    reject: bool = typer.Option(
        False, "--reject", "-r", help="Reject and stop (for HITL interrupts)"
    ),
    reason: str | None = typer.Option(None, "--reason", help="Reason for rejection"),
    breakpoints: str | None = typer.Option(
        None, "--breakpoints", "-b", help="Comma-separated node names to pause before"
    ),
):
    """
    Resume a paused workflow execution.

    For workflows paused at human approval nodes, use --approve or --reject:

        harness resume 123 --approve          # Approve and continue to merge train
        harness resume 123 --reject           # Reject without reason
        harness resume 123 --reject --reason "Tests need more coverage"

    For workflows paused at breakpoints, just resume:

        harness resume 123                    # Continue from breakpoint
    """

    config = HarnessConfig.load()
    db = get_db(config)

    # Validate options
    if approve and reject:
        console.print("[red]Cannot use both --approve and --reject[/red]")
        raise typer.Exit(1)

    # Get execution info
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT we.*, r.name as role_name
            FROM workflow_executions we
            JOIN roles r ON we.role_id = r.id
            WHERE we.id = ?
            """,
            (execution_id,),
        ).fetchone()

        if not row:
            console.print(f"[red]Execution {execution_id} not found[/red]")
            raise typer.Exit(1)

        if row["status"] not in ("paused", "failed", "running"):
            console.print(f"[yellow]Execution is {row['status']}, not resumable[/yellow]")
            raise typer.Exit(1)

    console.print(f"[blue]Resuming execution {execution_id} for {row['role_name']}[/blue]")
    console.print(f"  Last node: {row['current_node']}")

    # Handle HITL approval/rejection
    if approve or reject:
        if row["current_node"] != "human_approval":
            console.print(
                f"[yellow]Warning: Current node is '{row['current_node']}', not 'human_approval'[/yellow]"
            )
            console.print("[dim]--approve/--reject are for human approval nodes[/dim]")

        # Build the resume command with approval decision
        resume_value = {"approved": approve, "reason": reason or ""}

        if approve:
            console.print("[green]Approving merge train...[/green]")
        else:
            console.print(f"[yellow]Rejecting: {reason or 'No reason given'}[/yellow]")

        # Resume with LangGraph Command
        asyncio.run(_resume_with_command(db, execution_id, row["role_name"], resume_value))
    else:
        # Standard resume without HITL response
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
        repo_root = Path(config.repo_root)
        result = asyncio.run(
            graph.execute(
                row["role_name"],
                resume_from=execution_id,
                breakpoints=bp_set,
                repo_root=repo_root,
                repo_python=config.repo_python,
            )
        )

        if result["status"] == "completed":
            console.print("\n[green]Workflow completed successfully![/green]")
        elif result["status"] == "paused":
            console.print(f"\n[yellow]Workflow paused at:[/yellow] {result.get('paused_at')}")
        else:
            console.print(f"\n[red]Workflow failed:[/red] {result.get('error')}")
            raise typer.Exit(1)


async def _resume_with_command(db, execution_id: int, role_name: str, resume_value: dict):
    """
    Resume a workflow using LangGraph Command(resume=...) pattern.

    This is used for human-in-the-loop approvals where the workflow
    is paused at an interrupt() call.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    from harness.dag.langgraph_engine import create_box_up_role_graph, set_module_db
    from harness.db.models import WorkflowStatus

    set_module_db(db)

    # Get the graph and compile with checkpointer
    graph = create_box_up_role_graph()

    # Get database path from config
    config = HarnessConfig.load()
    db_path = config.db_path

    # Create checkpointer and compile graph
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)

        # Create thread config for this execution
        thread_config = {"configurable": {"thread_id": str(execution_id)}}

        # Resume with Command containing the approval decision
        try:
            result = await compiled.ainvoke(Command(resume=resume_value), config=thread_config)

            # Update execution status based on result
            if result.get("errors"):
                error_msg = "; ".join(result.get("errors", []))
                db.update_execution_status(
                    execution_id, WorkflowStatus.FAILED, error_message=error_msg
                )
                console.print(f"\n[red]Workflow failed:[/red] {error_msg}")
            else:
                db.update_execution_status(execution_id, WorkflowStatus.COMPLETED)
                console.print("\n[green]Workflow completed successfully![/green]")
                summary = result.get("summary", {})
                if summary:
                    console.print(f"  MR URL: {summary.get('mr_url', 'N/A')}")
                    console.print(f"  Merge train: {summary.get('merge_train_status', 'N/A')}")

        except Exception as e:
            db.update_execution_status(execution_id, WorkflowStatus.FAILED, error_message=str(e))
            console.print(f"\n[red]Resume failed:[/red] {e}")
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
    repo_root: Path | None = typer.Option(None, "--repo-root", help="Repository root path"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration"),
    no_detect_roles: bool = typer.Option(
        False, "--no-detect-roles", help="Skip automatic role detection"
    ),
    config_path: str | None = typer.Option(
        None, "--config", "-c", help="Path to save config file"
    ),
    skip_claude: bool = typer.Option(
        False, "--skip-claude", help="Skip deploying .claude/ assets (hooks, skills, settings)"
    ),
):
    """Initialize harness in a repository.

    Detects the git repository root, creates .harness/ directory,
    initializes the database, generates harness.yml, detects Ansible
    roles, updates .gitignore, and deploys .claude/ assets.

    Examples:
        harness init                          # Auto-detect everything
        harness init --repo-root /path/to/repo
        harness init --force                  # Overwrite existing config
        harness init --no-detect-roles        # Skip role scanning
        harness init --skip-claude            # Skip .claude/ deployment
    """
    from harness.bootstrap.init import init_harness

    try:
        result = init_harness(
            repo_root=repo_root,
            force=force,
            no_detect_roles=no_detect_roles,
            config_path=config_path,
            skip_claude=skip_claude,
        )
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]Repository root:[/green] {result['repo_root']}")
    console.print(f"[green]Harness directory:[/green] {result['harness_dir']}")
    console.print(f"[green]Database:[/green] {result['db_path']}")

    if result["config_created"]:
        console.print(f"[green]Config created:[/green] {result['config_path']}")
    else:
        console.print(f"[dim]Config already exists:[/dim] {result['config_path']}")

    if result["roles_detected"]:
        console.print(
            f"[green]Roles detected:[/green] {len(result['roles_detected'])} "
            f"({', '.join(result['roles_detected'][:5])}"
            f"{'...' if len(result['roles_detected']) > 5 else ''})"
        )
    else:
        console.print("[yellow]No roles detected in ansible/roles/[/yellow]")

    if result["gitignore_updated"]:
        console.print("[green].gitignore updated with .harness/ entries[/green]")
    else:
        console.print("[dim].gitignore already has .harness/ entries[/dim]")

    # Report .claude/ deployment status
    claude_info = result.get("claude_deployed")
    if claude_info is not None:
        if claude_info["success"]:
            console.print(
                f"[green].claude/ deployed:[/green] "
                f"{claude_info['components_installed']}/{claude_info['components_total']} components"
            )
        else:
            console.print("[yellow].claude/ deployment had issues:[/yellow]")
            for err in claude_info.get("errors", []):
                console.print(f"  [red]{err}[/red]")
            for warn in claude_info.get("warnings", []):
                console.print(f"  [yellow]{warn}[/yellow]")
    elif skip_claude:
        console.print("[dim].claude/ deployment skipped (--skip-claude)[/dim]")

    if result.get("npm_scripts_patched"):
        console.print("[green]package.json patched with harness npm scripts[/green]")
    elif not skip_claude and (repo_root or Path.cwd()).joinpath("package.json").exists():
        console.print("[dim]package.json already has harness scripts[/dim]")


@app.command("config")
def config_cmd(
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate configuration"),
    schema: bool = typer.Option(False, "--schema", "-s", help="Output JSON Schema"),
    path: str | None = typer.Option(None, "--path", "-p", help="Config file path"),
):
    """Validate configuration or output JSON schema.

    Examples:
        harness config --validate                # Validate harness.yml
        harness config --validate --path my.yml  # Validate specific file
        harness config --schema                  # Output JSON Schema
    """
    from harness.config import HarnessConfigModel

    if schema:
        schema_dict = HarnessConfigModel.model_json_schema()
        print(json.dumps(schema_dict, indent=2))
        return

    if validate:
        config_path = path or "harness.yml"
        try:
            cfg = HarnessConfigModel.from_yaml(config_path)
            console.print(f"[green]Configuration valid:[/green] {config_path}")
            console.print(f"  Project: {cfg.project_name}")
            console.print(f"  GitLab: {cfg.gitlab.project_path}")
            console.print(f"  Waves: {len(cfg.waves)}")
            console.print(f"  DB: {cfg.db_path}")
            console.print(f"  Log level: {cfg.log_level}")
            if cfg.waves:
                total_roles = sum(len(w.roles) for w in cfg.waves)
                console.print(f"  Total roles: {total_roles}")
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Validation error:[/red] {e}")
            raise typer.Exit(1)
        return

    # Default: show current config summary
    console.print("[dim]Use --validate to validate config, --schema for JSON Schema[/dim]")


@app.command("graph")
def show_graph(
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, mermaid"),
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
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
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
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm reset without prompting"),
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
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm without prompting"),
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
def db_backup(output: str = typer.Argument(..., help="Backup file path")):
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
    console.print(
        f"[bold]Schema Valid:[/bold] {'[green]Yes[/green]' if schema['valid'] else '[red]No[/red]'}"
    )


@db_app.command("migrate")
def db_migrate(
    target: int | None = typer.Option(None, "--target", "-t", help="Target migration version"),
    show_status: bool = typer.Option(False, "--status", "-s", help="Show migration status"),
    rollback_to: int | None = typer.Option(
        None, "--rollback", "-r", help="Rollback to target version"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Run database schema migrations.

    Applies pending migrations to bring the database up to the latest version.
    Use --status to see current migration state, --target to migrate to a
    specific version, or --rollback to reverse migrations.

    Examples:
        harness db migrate              # Apply all pending migrations
        harness db migrate --status     # Show migration status
        harness db migrate --target 2   # Migrate to version 2
        harness db migrate --rollback 1 # Roll back to version 1
    """
    from harness.db.migrations import MigrationRunner

    config = HarnessConfig.load()
    runner = MigrationRunner(config.db_path)

    if show_status:
        status_info = runner.status()

        if json_output:
            print(json.dumps(status_info, indent=2, default=str))
            return

        console.print(f"\n[bold]Migration Status[/bold]")
        console.print(f"  Current version: {status_info['current_version']}")
        console.print(
            f"  Applied: {status_info['applied_count']}/{status_info['total_migrations']}"
        )
        console.print(f"  Pending: {status_info['pending_count']}")

        if status_info["applied"]:
            console.print("\n[bold]Applied migrations:[/bold]")
            for m in status_info["applied"]:
                console.print(
                    f"  [green]v{m['version']}[/green] {m['description']} ({m['applied_at']})"
                )

        if status_info["pending"]:
            console.print("\n[bold]Pending migrations:[/bold]")
            for m in status_info["pending"]:
                console.print(f"  [yellow]v{m['version']}[/yellow] {m['description']}")

        return

    if rollback_to is not None:
        current = runner.get_current_version()
        if rollback_to >= current:
            console.print(
                f"[yellow]Target version {rollback_to} is not below current version {current}[/yellow]"
            )
            return

        console.print(
            f"[blue]Rolling back from version {current} to {rollback_to}...[/blue]"
        )
        results = runner.rollback(rollback_to)

        if json_output:
            print(json.dumps(results, indent=2))
            return

        for msg in results:
            console.print(f"  [yellow]{msg}[/yellow]")

        console.print(
            f"\n[green]Rollback complete. Now at version {runner.get_current_version()}[/green]"
        )
        return

    # Apply migrations
    pending = runner.get_pending()
    if not pending:
        console.print("[green]Database is up to date. No pending migrations.[/green]")
        return

    console.print(f"[blue]Applying {len(pending)} pending migration(s)...[/blue]")
    results = runner.migrate(target=target)

    if json_output:
        print(json.dumps(results, indent=2))
        return

    for msg in results:
        console.print(f"  [green]{msg}[/green]")

    console.print(
        f"\n[green]Migration complete. Now at version {runner.get_current_version()}[/green]"
    )


@db_app.command("migrate-checkpoints")
def db_migrate_checkpoints(
    sqlite_path: str | None = typer.Option(
        None, "--sqlite", "-s", help="SQLite database path (default: harness.db)"
    ),
    postgres_url: str | None = typer.Option(
        None, "--postgres", "-p", envvar="POSTGRES_URL", help="PostgreSQL connection URL"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm migration without prompting"),
):
    """Migrate LangGraph checkpoints from SQLite to PostgreSQL.

    This command migrates all checkpoint data from a SQLite database to PostgreSQL.
    Use this when transitioning from development (SQLite) to production (PostgreSQL).

    The migration preserves:
    - All checkpoint data and state
    - Thread IDs and timestamps
    - Checkpoint parent relationships

    Examples:
        harness db migrate-checkpoints --postgres postgresql://user:pass@localhost/db
        harness db migrate-checkpoints -s ./dev.db -p $POSTGRES_URL
    """
    from harness.dag.checkpointer import is_postgres_available, migrate_sqlite_to_postgres

    config = HarnessConfig.load()
    source_path = sqlite_path or config.db_path

    if not postgres_url:
        console.print(
            "[red]Error:[/red] PostgreSQL URL required. Set POSTGRES_URL or use --postgres"
        )
        raise typer.Exit(1)

    if not is_postgres_available():
        console.print("[red]Error:[/red] PostgreSQL dependencies not installed.")
        console.print("[dim]Install with: pip install dag-harness[postgres][/dim]")
        raise typer.Exit(1)

    if not Path(source_path).exists():
        console.print(f"[red]Error:[/red] SQLite database not found: {source_path}")
        raise typer.Exit(1)

    if not yes:
        console.print("[bold]Migration Plan:[/bold]")
        console.print(f"  Source: {source_path} (SQLite)")
        console.print("  Target: PostgreSQL")
        console.print()
        confirm = typer.confirm("Proceed with migration?")
        if not confirm:
            raise typer.Abort()

    console.print(f"[blue]Migrating checkpoints from {source_path} to PostgreSQL...[/blue]")

    result = asyncio.run(migrate_sqlite_to_postgres(source_path, postgres_url))

    if result["success"]:
        console.print("\n[green]Migration completed successfully![/green]")
        console.print(f"  Checkpoints migrated: {result['checkpoints_migrated']}")
        console.print(f"  Writes migrated: {result['writes_migrated']}")
        console.print(f"  Duration: {result['duration_seconds']:.2f}s")
    else:
        console.print("\n[red]Migration failed![/red]")
        for error in result["errors"]:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@db_app.command("cleanup-checkpoints")
def db_cleanup_checkpoints(
    days: int = typer.Option(30, "--days", "-d", help="Delete checkpoints older than N days"),
    postgres_url: str | None = typer.Option(
        None, "--postgres", "-p", envvar="POSTGRES_URL", help="PostgreSQL connection URL"
    ),
    sqlite_path: str | None = typer.Option(None, "--sqlite", "-s", help="SQLite database path"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm cleanup without prompting"),
):
    """Clean up old LangGraph checkpoints.

    Removes checkpoints older than the specified number of days while preserving
    the most recent checkpoint for each thread (for resumability).

    Examples:
        harness db cleanup-checkpoints --days 7    # Delete checkpoints older than 7 days
        harness db cleanup-checkpoints --postgres $POSTGRES_URL --days 30
    """
    from harness.dag.checkpointer import (
        cleanup_old_checkpoints_by_url,
        is_postgres_available,
    )

    config = HarnessConfig.load()

    if not yes:
        confirm = typer.confirm(f"Delete checkpoints older than {days} days?")
        if not confirm:
            raise typer.Abort()

    # Determine which database to clean up
    if postgres_url:
        if not is_postgres_available():
            console.print("[red]Error:[/red] PostgreSQL dependencies not installed.")
            raise typer.Exit(1)

        console.print(f"[blue]Cleaning up PostgreSQL checkpoints older than {days} days...[/blue]")
        result = asyncio.run(cleanup_old_checkpoints_by_url(postgres_url, days))
    elif sqlite_path:
        # SQLite cleanup via direct SQL
        console.print(f"[blue]Cleaning up SQLite checkpoints older than {days} days...[/blue]")
        import sqlite3
        from datetime import datetime, timedelta

        cutoff = datetime.now(UTC) - timedelta(days=days)
        cutoff_ts = cutoff.isoformat()

        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_ts < ?
                AND thread_id NOT IN (
                    SELECT thread_id
                    FROM checkpoints
                    GROUP BY thread_id
                    HAVING MAX(thread_ts) = thread_ts
                )
                """,
                (cutoff_ts,),
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            result = {"deleted_count": deleted, "errors": []}
        except Exception as e:
            result = {"deleted_count": 0, "errors": [str(e)]}
    else:
        # Use default harness.db
        sqlite_path = config.db_path
        console.print(
            f"[blue]Cleaning up checkpoints in {sqlite_path} older than {days} days...[/blue]"
        )
        import sqlite3
        from datetime import datetime, timedelta

        cutoff = datetime.now(UTC) - timedelta(days=days)
        cutoff_ts = cutoff.isoformat()

        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_ts < ?
                AND thread_id NOT IN (
                    SELECT thread_id
                    FROM checkpoints
                    GROUP BY thread_id
                    HAVING MAX(thread_ts) = thread_ts
                )
                """,
                (cutoff_ts,),
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            result = {"deleted_count": deleted, "errors": []}
        except sqlite3.OperationalError:
            # Table might not exist
            result = {"deleted_count": 0, "errors": ["checkpoints table not found"]}
        except Exception as e:
            result = {"deleted_count": 0, "errors": [str(e)]}

    if result["errors"]:
        console.print("[yellow]Cleanup completed with warnings:[/yellow]")
        for error in result["errors"]:
            console.print(f"  - {error}")
    else:
        console.print(
            f"[green]Cleanup complete. Deleted {result['deleted_count']} old checkpoints.[/green]"
        )


@db_app.command("check-postgres")
def db_check_postgres(
    postgres_url: str | None = typer.Option(
        None, "--postgres", "-p", envvar="POSTGRES_URL", help="PostgreSQL connection URL"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Check PostgreSQL connection health for checkpointing.

    Verifies:
    - Connection to PostgreSQL server
    - Query latency
    - Checkpointer table existence
    - Server version

    Examples:
        harness db check-postgres
        harness db check-postgres --postgres postgresql://user:pass@localhost/db
    """
    from harness.dag.checkpointer import (
        check_postgres_health,
        get_postgres_url,
        is_postgres_available,
    )

    url = postgres_url or get_postgres_url()

    if not url:
        console.print("[yellow]No PostgreSQL URL configured.[/yellow]")
        console.print("[dim]Set POSTGRES_URL or DATABASE_URL environment variable,[/dim]")
        console.print("[dim]or use --postgres to specify a connection URL.[/dim]")
        if json_output:
            print(json.dumps({"configured": False, "connected": False}))
        raise typer.Exit(1)

    if not is_postgres_available():
        console.print("[red]PostgreSQL dependencies not installed.[/red]")
        console.print("[dim]Install with: pip install dag-harness[postgres][/dim]")
        if json_output:
            print(json.dumps({"configured": True, "installed": False, "connected": False}))
        raise typer.Exit(1)

    console.print("[blue]Checking PostgreSQL connection...[/blue]")

    result = asyncio.run(check_postgres_health(url))

    if json_output:
        print(json.dumps(result, indent=2))
        if not result["connected"]:
            raise typer.Exit(1)
        return

    if result["connected"]:
        console.print("\n[green]PostgreSQL connection healthy![/green]")
        console.print(f"  Latency: {result['latency_ms']:.2f}ms")
        if result.get("version"):
            # Truncate version string for display
            version = result["version"]
            if len(version) > 60:
                version = version[:57] + "..."
            console.print(f"  Version: {version}")
        console.print(f"  Checkpoints: {result.get('pool_size', 'N/A')}")
    else:
        console.print("\n[red]PostgreSQL connection failed![/red]")
        console.print(f"  Error: {result.get('error', 'Unknown error')}")
        console.print(f"  Latency: {result['latency_ms']:.2f}ms")
        raise typer.Exit(1)


# =========================================================================
# COSTS COMMANDS
# =========================================================================

costs_app = typer.Typer(name="costs", help="Cost tracking and reporting")
app.add_typer(costs_app)


@costs_app.command("report")
def costs_report(
    days: int = typer.Option(30, "--days", "-d", help="Days to look back"),
    by_model: bool = typer.Option(True, "--by-model/--no-by-model", help="Show breakdown by model"),
    by_session: bool = typer.Option(False, "--by-session", "-s", help="Show breakdown by session"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show cost breakdown by model and date.

    Displays token usage and costs from the token_usage table, with
    breakdowns by model and optionally by session.

    Examples:
        harness costs report              # Last 30 days by model
        harness costs report --days 7     # Last 7 days
        harness costs report --by-session # Include session breakdown
    """
    from datetime import datetime, timedelta

    db = get_db()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    summary = db.get_cost_summary(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    if json_output:
        print(json.dumps(summary, indent=2, default=str))
        return

    # Header
    console.print(f"\n[bold]Cost Report[/bold] ({days} days)")
    console.print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

    # Totals
    console.print(f"[bold]Total Cost:[/bold] ${summary['total_cost']:.4f}")
    console.print(f"[bold]Total Input Tokens:[/bold] {summary['total_input_tokens']:,}")
    console.print(f"[bold]Total Output Tokens:[/bold] {summary['total_output_tokens']:,}")
    console.print(f"[bold]API Calls:[/bold] {summary['record_count']:,}\n")

    # By model breakdown
    if by_model and summary.get("by_model"):
        table = Table(title="Cost by Model")
        table.add_column("Model")
        table.add_column("Cost", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Calls", justify="right")

        for model, data in sorted(
            summary["by_model"].items(),
            key=lambda x: x[1]["cost"],
            reverse=True,
        ):
            table.add_row(
                model,
                f"${data['cost']:.4f}",
                f"{data['input_tokens']:,}",
                f"{data['output_tokens']:,}",
                f"{data['count']:,}",
            )

        console.print(table)
        console.print()

    # By session breakdown
    if by_session and summary.get("by_session"):
        table = Table(title="Top Sessions by Cost")
        table.add_column("Session ID")
        table.add_column("Cost", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Calls", justify="right")

        for session_id, data in list(summary["by_session"].items())[:10]:
            # Truncate session ID for display
            display_id = session_id[:20] + "..." if len(session_id) > 20 else session_id
            table.add_row(
                display_id,
                f"${data['cost']:.4f}",
                f"{data['input_tokens']:,}",
                f"{data['output_tokens']:,}",
                f"{data['count']:,}",
            )

        console.print(table)


@costs_app.command("daily")
def costs_daily(
    days: int = typer.Option(14, "--days", "-d", help="Days to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show daily cost breakdown.

    Displays costs aggregated by day for trend analysis.
    """
    from harness.costs.tracker import TokenUsageTracker

    db = get_db()
    tracker = TokenUsageTracker(db)
    daily = tracker.get_daily_costs(days=days)

    if json_output:
        # Convert Decimal to float for JSON
        for d in daily:
            d["total_cost"] = float(d["total_cost"])
        print(json.dumps(daily, indent=2))
        return

    if not daily:
        console.print("[yellow]No cost data found.[/yellow]")
        return

    table = Table(title=f"Daily Costs (Last {days} Days)")
    table.add_column("Date")
    table.add_column("Cost", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Calls", justify="right")

    for day in daily:
        table.add_row(
            str(day["date"]),
            f"${float(day['total_cost']):.4f}",
            f"{day['total_input_tokens']:,}",
            f"{day['total_output_tokens']:,}",
            f"{day['record_count']:,}",
        )

    console.print(table)


@costs_app.command("session")
def costs_session(
    session_id: str = typer.Argument(..., help="Session ID to query"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show costs for a specific session."""
    db = get_db()
    summary = db.get_session_costs(session_id)

    if json_output:
        print(json.dumps(summary, indent=2, default=str))
        return

    if summary["record_count"] == 0:
        console.print(f"[yellow]No cost data found for session: {session_id}[/yellow]")
        return

    console.print(f"\n[bold]Session:[/bold] {session_id}")
    console.print(f"[bold]Total Cost:[/bold] ${summary['total_cost']:.4f}")
    console.print(f"[bold]Input Tokens:[/bold] {summary['total_input_tokens']:,}")
    console.print(f"[bold]Output Tokens:[/bold] {summary['total_output_tokens']:,}")
    console.print(f"[bold]API Calls:[/bold] {summary['record_count']:,}\n")

    if summary.get("by_model"):
        table = Table(title="Breakdown by Model")
        table.add_column("Model")
        table.add_column("Cost", justify="right")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")

        for model, data in summary["by_model"].items():
            table.add_row(
                model,
                f"${data['cost']:.4f}",
                f"{data['input_tokens']:,}",
                f"{data['output_tokens']:,}",
            )

        console.print(table)


@costs_app.command("pricing")
def costs_pricing():
    """Show current model pricing information."""
    from harness.costs.pricing import CLAUDE_PRICING

    table = Table(title="Claude Model Pricing (per 1M tokens)")
    table.add_column("Model ID")
    table.add_column("Display Name")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")

    for model_id, pricing in sorted(CLAUDE_PRICING.items()):
        table.add_row(
            model_id,
            pricing.display_name,
            f"${pricing.input_cost_per_mtok:.2f}",
            f"${pricing.output_cost_per_mtok:.2f}",
        )

    console.print(table)


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
        console.print("\n[bold green]Overall Health: HEALTHY[/bold green]")
    elif overall == "warning":
        console.print("\n[bold yellow]Overall Health: WARNING[/bold yellow]")
    elif overall == "critical":
        console.print("\n[bold red]Overall Health: CRITICAL[/bold red]")
    else:
        console.print("\n[bold dim]Overall Health: UNKNOWN[/bold dim]")

    console.print(
        f"  OK: {health['ok']} | Warning: {health['warning']} | Critical: {health['critical']} | Unknown: {health['unknown']}\n"
    )

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
    context: str | None = typer.Option(None, "--context", "-c", help="JSON context metadata"),
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
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum entries to show"),
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
            status_str,
        )

    console.print(table)

    if len(history) > limit:
        console.print(f"[dim]... and {len(history) - limit} more entries[/dim]")


@metrics_app.command("trend")
def metrics_trend(
    name: str = typer.Argument(..., help="Metric name"),
    hours: int = typer.Option(24, "--hours", "-h", help="Hours to analyze"),
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
        console.print("  Trend:   [red]↑ Increasing[/red]")
    elif trend_str == "decreasing":
        console.print("  Trend:   [green]↓ Decreasing[/green]")
    elif trend_str == "stable":
        console.print("  Trend:   [blue]→ Stable[/blue]")
    else:
        console.print("  Trend:   [dim]? Unknown[/dim]")


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
            m.unit or "-",
        )

    console.print(table)


@metrics_app.command("purge")
def metrics_purge(
    days: int = typer.Option(30, "--days", "-d", help="Purge records older than N days"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm without prompting"),
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
    notify_interval: int = typer.Option(
        300, "--notify-interval", "-i", help="Seconds between notifications"
    ),
    discord_webhook: str | None = typer.Option(
        None, "--discord", envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL"
    ),
    email_to: str | None = typer.Option(
        None, "--email", envvar="HOTL_EMAIL_TO", help="Email recipient"
    ),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
    waves: str | None = typer.Option(
        None, "--waves", "-w", help="Comma-separated wave numbers to process (e.g. '1,2,3')"
    ),
    max_concurrent: int = typer.Option(
        2, "--max-concurrent", "-j", help="Maximum concurrent role processing"
    ),
):
    """Start HOTL autonomous operation mode.

    Without --waves, runs the standard supervisor loop. With --waves,
    uses the wave-based orchestrator to process roles from the specified
    waves through box-up-role workflows.

    Examples:
        harness hotl start                         # Standard supervisor
        harness hotl start --waves 1               # Process wave 1 roles
        harness hotl start --waves 1,2 -j 3        # Waves 1-2, 3 concurrent
    """
    if waves:
        # Wave-based orchestrator mode
        from harness.hotl.orchestrator import HOTLOrchestrator
        from harness.hotl.queue import RoleQueue

        db = get_db()
        harness_config = HarnessConfig.load()
        orch_config = {
            "repo_root": harness_config.repo_root,
            "repo_python": harness_config.repo_python,
        }

        queue = RoleQueue(db, max_concurrent=max_concurrent)
        orchestrator = HOTLOrchestrator(db, config=orch_config, queue=queue)

        wave_list = [int(w.strip()) for w in waves.split(",")]

        console.print("[bold blue]Starting HOTL Orchestrator Mode[/bold blue]")
        console.print(f"  Waves: {wave_list}")
        console.print(f"  Max concurrent: {max_concurrent}")
        console.print()

        try:
            asyncio.run(orchestrator.start(waves=wave_list, max_concurrent=max_concurrent))

            console.print("\n[bold green]HOTL orchestrator completed[/bold green]")
            queue_status = queue.get_status()
            console.print(f"  Completed: {queue_status.get('completed', 0)}")
            console.print(f"  Failed: {queue_status.get('failed', 0)}")
            console.print(f"  Skipped: {queue_status.get('skipped', 0)}")

        except KeyboardInterrupt:
            console.print("\n[yellow]HOTL orchestrator interrupted by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            console.print(f"\n[red]HOTL orchestrator error: {e}[/red]")
            raise typer.Exit(1)

        return

    # Standard supervisor mode
    from harness.hotl.supervisor import HOTLSupervisor

    db = get_db()

    config = {}
    if discord_webhook:
        config["discord_webhook_url"] = discord_webhook
    if email_to:
        config["email_to"] = email_to

    supervisor = HOTLSupervisor(db, config=config)

    console.print("[bold blue]Starting HOTL Mode[/bold blue]")
    console.print(f"  Max iterations: {max_iterations}")
    console.print(f"  Notify interval: {notify_interval}s")
    console.print(f"  Discord: {'configured' if discord_webhook else 'not configured'}")
    console.print(f"  Email: {email_to or 'not configured'}")
    console.print()

    if background:
        console.print(
            "[yellow]Background mode not yet implemented. Running in foreground.[/yellow]"
        )

    try:
        # Run synchronously - supervisor.run() is now sync
        final_state = supervisor.run(
            max_iterations=max_iterations, notification_interval=notify_interval
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
def hotl_status(json_output: bool = typer.Option(False, "--json", help="Output as JSON")):
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

    if status_info["running_executions"] > 0:
        console.print(
            "\n[yellow]Note: HOTL may be running. Use Ctrl+C to stop if in foreground.[/yellow]"
        )
    else:
        console.print("\n[dim]No active HOTL session detected.[/dim]")


@hotl_app.command("stop")
def hotl_stop(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force stop (cancel running executions)"
    ),
):
    """Request HOTL to stop gracefully."""
    db = get_db()

    if force:
        # Cancel any running executions in the database
        from harness.db.models import WorkflowStatus

        with db.connection() as conn:
            count = conn.execute(
                "UPDATE workflow_executions SET status = ? WHERE status = ?",
                (WorkflowStatus.CANCELLED.value, WorkflowStatus.RUNNING.value),
            ).rowcount
        if count > 0:
            console.print(f"[yellow]Cancelled {count} running execution(s)[/yellow]")
        else:
            console.print("[dim]No running executions to cancel[/dim]")
    else:
        console.print("[dim]HOTL stop works by pressing Ctrl+C in the running session.[/dim]")
        console.print("[dim]Use --force to cancel running executions in the database.[/dim]")


@hotl_app.command("approve")
def hotl_approve(
    role_name: str = typer.Argument(..., help="Role name to approve"),
):
    """Approve a paused role workflow to continue processing.

    When a role workflow pauses for human approval, use this command
    to approve it and allow processing to continue.

    Examples:
        harness hotl approve common
        harness hotl approve ems_web_app
    """
    from harness.hotl.orchestrator import HOTLOrchestrator
    from harness.hotl.queue import QueueItemStatus, RoleQueue

    db = get_db()

    # Check for paused executions for this role
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT we.id, we.status, r.name
            FROM workflow_executions we
            JOIN roles r ON we.role_id = r.id
            WHERE r.name = ? AND we.status = 'paused'
            ORDER BY we.id DESC LIMIT 1
            """,
            (role_name,),
        ).fetchone()

    if row:
        # Resume via the standard resume mechanism
        from harness.db.models import WorkflowStatus

        db.update_execution_status(row["id"], WorkflowStatus.RUNNING)
        console.print(f"[green]Approved role '{role_name}' (execution {row['id']})[/green]")
    else:
        console.print(f"[yellow]No paused execution found for '{role_name}'[/yellow]")
        console.print("[dim]This command works with the HOTL orchestrator queue.[/dim]")


@hotl_app.command("reject")
def hotl_reject(
    role_name: str = typer.Argument(..., help="Role name to reject"),
    reason: str = typer.Option("", "--reason", "-r", help="Reason for rejection"),
):
    """Reject a paused role workflow and skip it.

    When a role workflow pauses for human approval, use this command
    to reject it. The role will be marked as skipped and the queue
    will continue with the next role.

    Examples:
        harness hotl reject common --reason "Tests need more coverage"
        harness hotl reject ems_web_app -r "Not ready"
    """
    db = get_db()

    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT we.id, we.status, r.name
            FROM workflow_executions we
            JOIN roles r ON we.role_id = r.id
            WHERE r.name = ? AND we.status = 'paused'
            ORDER BY we.id DESC LIMIT 1
            """,
            (role_name,),
        ).fetchone()

    if row:
        from harness.db.models import WorkflowStatus

        db.update_execution_status(
            row["id"],
            WorkflowStatus.CANCELLED,
            error_message=reason or "Rejected by user",
        )
        console.print(f"[yellow]Rejected role '{role_name}' (execution {row['id']})[/yellow]")
        if reason:
            console.print(f"  Reason: {reason}")
    else:
        console.print(f"[yellow]No paused execution found for '{role_name}'[/yellow]")
        console.print("[dim]This command works with the HOTL orchestrator queue.[/dim]")


@hotl_app.command("resume")
def hotl_resume(
    thread_id: str = typer.Argument(..., help="Thread ID to resume from checkpoint"),
    max_iterations: int = typer.Option(100, "--max-iterations", "-n"),
    notify_interval: int = typer.Option(300, "--notify-interval", "-i"),
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
            resume_from=thread_id,
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
    skip_skills: bool = typer.Option(
        False, "--skip-skills", help="Don't install skill definitions"
    ),
    skip_mcp: bool = typer.Option(False, "--skip-mcp", help="Don't configure MCP server"),
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
        force=force, skip_hooks=skip_hooks, skip_skills=skip_skills, skip_mcp=skip_mcp
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
    from harness.install import InstallStatus, MCPInstaller, print_check_result

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
    remove_data: bool = typer.Option(False, "--remove-data", help="Also remove data files"),
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
    env_file: str | None = typer.Option(
        None, "--env-file", "-e", help="Target .env file path (default: project .env)"
    ),
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
            scopes_str[:20] + "..." if len(scopes_str) > 20 else scopes_str,
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
                console.print(
                    f"[dim]Token from: {first_valid['var_name']} ({first_valid['username']})[/dim]"
                )
            else:
                console.print("\n[red]Failed to save token to .env file[/red]")
                raise typer.Exit(1)
        else:
            console.print("\n[red]Could not retrieve token value for saving[/red]")
            console.print(
                f"[dim]Looking for var '{var_to_find}' with prefix '{token_prefix}' in {first_valid['source']}[/dim]"
            )
            raise typer.Exit(1)
    elif save and not first_valid:
        console.print("\n[yellow]No valid token found to save.[/yellow]")
        raise typer.Exit(1)


# =============================================================================
# CREDENTIALS Command
# =============================================================================


@app.command("credentials")
def credentials_cmd(
    prompt: bool = typer.Option(False, "--prompt", "-p", help="Prompt for missing credentials"),
    save: bool = typer.Option(False, "--save", "-s", help="Save entered credentials to .env"),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate discovered credentials"
    ),
    env_file: str | None = typer.Option(
        None, "--env-file", "-e", help="Target .env file for saving"
    ),
):
    """Discover and validate all credentials.

    This command searches for credentials in multiple locations:
    - Environment variables
    - .env files (project, home, config directories)
    - glab CLI configuration
    - macOS Keychain

    It validates discovered credentials against their respective services
    (GitLab API, Anthropic, Discord, etc.) with configurable timeouts.

    Examples:
        harness credentials                # Discover and validate
        harness credentials --prompt       # Prompt for missing required
        harness credentials --save         # Save to .env
        harness credentials --no-validate  # Skip validation
    """
    import asyncio

    from harness.bootstrap.discovery import KeyDiscovery
    from harness.bootstrap.prompts import CredentialPrompts
    from harness.bootstrap.validation import CredentialValidator, ValidationStatus

    console.print("[bold]Credential Discovery[/bold]\n")

    # Discover credentials
    discovery = KeyDiscovery()
    keys = discovery.discover_all()

    # Display results
    prompts = CredentialPrompts(console)
    prompts.display_discovery_results(keys, show_values=True)

    # Validate if requested
    if validate:
        console.print("\n[bold]Validating credentials...[/bold]\n")

        validator = CredentialValidator()
        validation_results = asyncio.run(validator.validate_all(keys))
        prompts.display_validation_results(validation_results)

        # Check for invalid credentials
        invalid = [
            name
            for name, result in validation_results.items()
            if result.status == ValidationStatus.INVALID
        ]
        if invalid:
            console.print(f"\n[red]Invalid credentials: {', '.join(invalid)}[/red]")

    # Prompt for missing if requested
    if prompt:
        from harness.bootstrap.discovery import KeyStatus

        missing = {name: info for name, info in keys.items() if info.status != KeyStatus.FOUND}

        if missing:
            console.print("\n[bold]Configure missing credentials:[/bold]")
            entered = prompts.prompt_for_missing(keys, required_only=not save)

            if entered and save:
                target = Path(env_file) if env_file else Path.cwd() / ".env"
                if prompts.confirm_save(entered, target):
                    prompts.save_to_env(entered, target)
        else:
            console.print("\n[green]All credentials found![/green]")

    # Summary
    from harness.bootstrap.discovery import KeyStatus

    found = sum(1 for info in keys.values() if info.status == KeyStatus.FOUND)
    total = len(keys)
    console.print(f"\n[dim]Found {found}/{total} credentials[/dim]")


# =============================================================================
# UPGRADE Command
# =============================================================================


@app.command("upgrade")
def upgrade_cmd(
    check: bool = typer.Option(False, "--check", "-c", help="Check only, don't install"),
    force: bool = typer.Option(False, "--force", "-f", help="Force upgrade even if up to date"),
):
    """Check for and install dag-harness updates.

    Checks PyPI and GitHub releases for newer versions and upgrades
    using the same method that was used for initial installation
    (uv, pip, pipx, or binary).

    Examples:
        harness upgrade         # Check and install if available
        harness upgrade --check # Check only
        harness upgrade --force # Force reinstall
    """
    from harness import __version__
    from harness.bootstrap.upgrade import (
        UpgradeStatus,
        check_for_upgrade,
        upgrade,
    )

    console.print("[bold]dag-harness upgrade[/bold]")
    console.print(f"Current version: {__version__}\n")

    # Check for updates
    version_info = check_for_upgrade()

    if version_info.upgrade_available:
        console.print(f"[green]New version available: {version_info.latest}[/green]")
        console.print(f"Source: {version_info.source}")

        if check:
            console.print("\n[dim]Run 'harness upgrade' to install.[/dim]")
            return

        console.print("\n[bold]Upgrading...[/bold]")
        result = upgrade(check_only=False)

        if result.status == UpgradeStatus.UPGRADED:
            console.print(f"\n[green]Successfully upgraded to {result.new_version}[/green]")
        else:
            console.print(f"\n[red]Upgrade failed: {result.message}[/red]")
            if result.error:
                console.print(f"[dim]{result.error}[/dim]")
            raise typer.Exit(1)

    elif force:
        console.print("[yellow]Forcing reinstall...[/yellow]")
        result = upgrade(check_only=False)

        if result.status == UpgradeStatus.UPGRADED:
            console.print(f"\n[green]Reinstalled version {result.new_version}[/green]")
        else:
            console.print(f"\n[red]Reinstall failed: {result.message}[/red]")
            raise typer.Exit(1)

    else:
        console.print(f"[green]Already at latest version ({version_info.current})[/green]")
        console.print(f"[dim]Source checked: {version_info.source}[/dim]")


# =============================================================================
# BOOTSTRAP Command
# =============================================================================


@app.command("bootstrap")
def bootstrap_cmd(
    check_only: bool = typer.Option(
        False, "--check-only", "-c", help="Only check current state, don't make changes"
    ),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick check (skip network tests)"),
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
