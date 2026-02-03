# API Reference

Complete reference for the dag-harness CLI, Python API, and configuration models.

## CLI Commands

The `harness` CLI is built with [Typer](https://typer.tiangolo.com/) and
[Rich](https://rich.readthedocs.io/) for formatted output.

```bash
harness [OPTIONS] COMMAND [ARGS]
```

### Global Options

| Option | Description |
|--------|-------------|
| `--version`, `-V` | Show version and exit |
| `--help` | Show help and exit |

---

### harness init

Initialize harness configuration in the current directory.

```bash
harness init [--config PATH]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config`, `-c` | `harness.yml` | Path to save configuration file |

Creates a `harness.yml` template with default values. Edit this file to
configure your project before running `harness bootstrap`.

---

### harness bootstrap

Run the interactive bootstrap wizard.

```bash
harness bootstrap [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--check-only`, `-c` | Only check current state, do not modify anything |
| `--quick`, `-q` | Quick check (skip network tests) |

The wizard detects your environment, validates credentials, initializes the
database, and installs Claude Code integration.

---

### harness box-up-role

Execute the box-up-role DAG workflow for a role.

```bash
harness box-up-role ROLE_NAME [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `ROLE_NAME` | Name of the Ansible role (must exist in `ansible/roles/`) |

| Option | Description |
|--------|-------------|
| `--breakpoints`, `-b` | Comma-separated node names to pause before |
| `--dry-run` | Show what would happen without executing |

Workflow nodes: `validate_role` -> `analyze_deps` -> `check_reverse_deps` ->
`create_worktree` -> `run_molecule` -> `run_pytest` -> `validate_deploy` ->
`create_commit` -> `push_branch` -> `create_issue` -> `create_mr` ->
`add_to_merge_train` -> `report_summary`

---

### harness status

Show the status of roles and their deployments.

```bash
harness status [ROLE_NAME]
```

Without `ROLE_NAME`, displays a table of all roles with wave, worktree,
issue, MR, and test status. With a role name, shows detailed status for
that specific role.

---

### harness sync

Sync state from the filesystem and optionally GitLab.

```bash
harness sync [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--roles/--no-roles` | `--roles` | Scan `ansible/roles/` for role metadata |
| `--worktrees/--no-worktrees` | `--worktrees` | Sync git worktrees |
| `--gitlab` | off | Sync issues and MRs from GitLab API |

---

### harness list-roles

List all Ansible roles in the database.

```bash
harness list-roles [--wave N]
```

| Option | Description |
|--------|-------------|
| `--wave`, `-w` | Filter by wave number |

---

### harness deps

Show dependencies for a role.

```bash
harness deps ROLE_NAME [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--reverse`, `-r` | Show reverse dependencies (roles that depend on this role) |
| `--transitive`, `-t` | Include transitive dependencies |

---

### harness worktrees

List all git worktrees managed by the harness.

```bash
harness worktrees [--json]
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

---

### harness resume

Resume a paused workflow execution.

```bash
harness resume EXECUTION_ID [--breakpoints NODES]
```

| Argument | Description |
|----------|-------------|
| `EXECUTION_ID` | The execution ID from a paused workflow |

| Option | Description |
|--------|-------------|
| `--breakpoints`, `-b` | Comma-separated node names to pause before |

---

### harness graph

Display the box-up-role workflow graph.

```bash
harness graph [--format FORMAT]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--format`, `-f` | `text` | Output format: `text`, `json`, or `mermaid` |

---

### harness check

Run self-checks on the harness database.

```bash
harness check [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--schema/--no-schema` | `--schema` | Validate database schema |
| `--data/--no-data` | `--data` | Validate data integrity |
| `--graph/--no-graph` | `--graph` | Validate dependency graph (cycle detection) |
| `--json` | off | Output as JSON |

---

### harness config

Show the current configuration.

```bash
harness config
```

Displays the resolved configuration including values from `harness.yml`,
environment variables, and defaults.

---

### harness mcp-server

Start the MCP server for Claude Code integration.

```bash
harness mcp-server
```

Usually started automatically by Claude Code via `.claude/settings.json`.
Run manually for debugging.

---

### harness credentials

Manage credential lookups for roles.

```bash
harness credentials
```

---

### harness scan-tokens

Scan for hardcoded tokens and credentials in the codebase.

```bash
harness scan-tokens
```

---

### harness upgrade

Upgrade harness database schema to the latest version.

```bash
harness upgrade
```

---

### Database Subcommands

```bash
harness db stats [--json]         # Database statistics
harness db info                   # File information
harness db backup OUTPUT_PATH     # Create backup
harness db reset [--yes]          # Reset to initial state (DESTRUCTIVE)
harness db clear TABLE [--yes]    # Clear a specific table
harness db vacuum                 # Reclaim disk space
```

### Metrics Subcommands

```bash
harness metrics status [--json]   # Current metric values
harness metrics record NAME VALUE # Record a metric
harness metrics history NAME      # Recent history
harness metrics trend NAME        # Trend analysis
harness metrics list              # List all metrics
harness metrics purge [--days N]  # Purge old records
```

### HOTL Subcommands

```bash
harness hotl start [OPTIONS]      # Start autonomous mode
harness hotl status [--json]      # Current HOTL state
harness hotl stop [--force]       # Stop gracefully or forcefully
harness hotl resume THREAD_ID     # Resume from checkpoint
```

### Install Subcommands

```bash
harness install run [OPTIONS]     # Install into Claude Code
harness install check             # Verify installation
harness install uninstall [--yes] # Remove from Claude Code
harness install upgrade           # Upgrade existing installation
```

---

## Python API

### LangGraphWorkflowRunner

The main workflow execution class.

```python
from harness import LangGraphWorkflowRunner

runner = LangGraphWorkflowRunner(db=state_db, config=harness_config)
result = await runner.execute(
    role_name="common",
    breakpoints={"create_mr"},
    repo_root=Path("/path/to/repo"),
    repo_python="/path/to/.venv/bin/python"
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `db` | `StateDB` | Database instance |
| `config` | `HarnessConfig` | Configuration instance |

**execute() Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `role_name` | `str` | Role to process |
| `breakpoints` | `set[str]` | Nodes to pause before |
| `repo_root` | `Path` | Repository root path |
| `repo_python` | `str` | Python interpreter path |

**Returns:** `dict` with keys `status`, `summary`, `execution_id`, `error`.

---

### BoxUpRoleState

TypedDict defining the workflow state schema.

```python
from harness import BoxUpRoleState
```

**Fields:**

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `role_name` | `str` | last-write | Role being processed |
| `execution_id` | `Optional[int]` | last-write | Database execution ID |
| `worktree_path` | `str` | last-write | Path to git worktree |
| `molecule_passed` | `bool` | last-write | Molecule test result |
| `pytest_passed` | `bool` | last-write | Pytest result |
| `mr_url` | `Optional[str]` | last-write | Merge request URL |
| `issue_url` | `Optional[str]` | last-write | Issue URL |
| `errors` | `list[str]` | `operator.add` | Accumulated errors |
| `state_snapshots` | `list[dict]` | `keep_last_n(10)` | Rolling state history |

---

### StateDB

SQLite state management with graph-queryable patterns.

```python
from harness import StateDB

db = StateDB("harness.db")
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `get_role(name)` | Get a role by name |
| `upsert_role(role)` | Create or update a role |
| `list_roles(wave=None)` | List roles, optionally filtered by wave |
| `get_role_status(name)` | Get full role status view |
| `list_role_statuses()` | Get all role statuses |
| `get_dependencies(role_id)` | Get direct dependencies |
| `get_transitive_deps(role_id)` | Get all transitive dependencies (recursive CTE) |
| `get_reverse_deps(role_id)` | Get roles that depend on this one |
| `upsert_worktree(worktree)` | Create or update a worktree |
| `record_test_run(test_run)` | Record a test execution |
| `get_active_regressions()` | Find active test regressions |
| `create_execution(...)` | Start a new workflow execution |
| `update_execution(...)` | Update execution status |

---

### HarnessStore

LangGraph BaseStore implementation for cross-thread memory.

```python
from harness.dag.store import HarnessStore

store = HarnessStore(db)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `put(namespace, key, value)` | Store a value |
| `get(namespace, key)` | Retrieve a value |
| `delete(namespace, key)` | Delete a value |
| `search(namespace, filter=None)` | Search within a namespace |
| `list_namespaces(prefix=None)` | List available namespaces |

**Namespace Convention:** Use tuples like `("roles", "common")` or
`("waves", "0")`.

---

### CheckpointerWithStateDB

Unified checkpointer that wraps LangGraph's native checkpointer and syncs
with StateDB.

```python
from harness.dag.checkpointer_unified import CheckpointerWithStateDB

async with CheckpointerWithStateDB(db, sqlite_path="harness.db") as checkpointer:
    compiled = graph.compile(checkpointer=checkpointer)
    result = await compiled.ainvoke(initial_state, config)
```

Delegates all checkpoint operations to the underlying checkpointer while
maintaining a copy in `workflow_executions.checkpoint_data`.

---

### ParallelWaveExecutor

Execute roles in wave order with parallel execution within waves.

```python
from harness import ParallelWaveExecutor

executor = ParallelWaveExecutor(db=db, config=config)
results = await executor.execute_all_waves()
```

---

### HarnessConfig

Legacy dataclass-based configuration.

```python
from harness import HarnessConfig

config = HarnessConfig.load()          # Auto-detect config
config = HarnessConfig.load("path")   # Explicit path

errors = config.validate()             # Returns list of error strings
wave, name = config.get_wave_for_role("common")
is_foundation = config.is_foundation_role("common")
config.save("harness.yml")
```

---

## Configuration Models

### HarnessConfigModel

Pydantic-validated root configuration.

```python
from harness.config import HarnessConfigModel

config = HarnessConfigModel.from_yaml("harness.yml")
config = HarnessConfigModel.from_env()  # With ${VAR} interpolation
```

**Fields:** `version`, `project_name`, `gitlab`, `waves`, `checkpointer`,
`hotl`, `db_path`, `log_level`, `breakpoints`

---

### GitLabConfigModel

Pydantic-validated GitLab configuration.

```python
from harness.config import GitLabConfigModel
```

**Fields:** `project_path` (required), `default_branch`, `merge_method`

**Validation:** `project_path` must contain at least one `/`.
`merge_method` must be `merge`, `rebase_merge`, or `ff`.

---

### WaveDefinition

Pydantic-validated wave configuration.

```python
from harness.config import WaveDefinition
```

**Fields:** `wave` (0-9), `roles` (min 1), `parallel` (default true), `name`

---

### CheckpointerConfig

Pydantic-validated checkpointer configuration.

```python
from harness.config import CheckpointerConfig
```

**Fields:** `backend` (sqlite/postgres), `sqlite_path`, `postgres_url`,
`sync_to_statedb`

---

### HOTLConfig

Pydantic-validated HOTL configuration.

```python
from harness.config import HOTLConfig
```

**Fields:** `enabled`, `max_iterations` (>= 1), `notification_interval` (>= 60),
`approval_timeout` (>= 300)

---

## MCP Tools

The MCP server exposes 20+ tools for Claude Code integration. See the
[MCP Tools Reference](api/mcp-tools.md) for the complete tool listing
with parameters and return types.

Key tool categories:

- **Role Management** -- get_role_status, list_role_statuses, sync_roles_from_filesystem
- **Dependencies** -- get_role_dependencies, get_reverse_dependencies
- **Testing** -- run_molecule_tests, run_pytest_tests, record_test_result
- **Git Worktrees** -- create_worktree, list_worktrees, remove_worktree
- **GitLab** -- create_gitlab_issue, create_merge_request
- **Workflows** -- execute_workflow, resume_workflow
- **Metrics** -- record_metric, get_metric_health
- **Database** -- get_database_stats, validate_database
