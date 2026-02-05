# Quick Start

Get up and running with dag-harness in under 5 minutes.

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.11+ | `python3 --version` |
| uv | latest | `uv --version` |
| Git | 2.x+ | `git --version` |
| glab | latest | `glab --version` |

**Optional:**

- GitLab API token (`GITLAB_TOKEN` or `glab auth login`)
- KeePassXC database for credential lookups

## Installation

### From PyPI

```bash
uv pip install dag-harness
```

### From Source

```bash
git clone https://github.com/Jesssullivan/Ansible-DAG-Harness.git
cd dag-harness/harness
uv sync
```

Or install as a CLI tool globally:

```bash
uv tool install ./harness
```

### Verify Installation

```bash
harness --version
# dag-harness 0.3.0
```

## Initialize a Project

Run `harness init` in your Ansible project root to create a `harness.yml` configuration file:

```bash
cd /path/to/your/ansible-project
harness init
```

This creates a `harness.yml` with sensible defaults:

```yaml
db_path: harness.db

gitlab:
  project_path: your-group/your-project
  default_labels:
    - role
    - ansible
    - molecule

worktree:
  base_path: ../worktrees
  branch_prefix: sid/

testing:
  molecule_required: true
  pytest_required: true
  molecule_timeout: 600
  pytest_timeout: 300

waves:
  0:
    name: Foundation
    roles:
      - common
```

Edit `harness.yml` to match your project structure. See the
[Configuration Reference](configuration.md) for all available fields.

## Run Database Migrations

Initialize the SQLite database with the required schema:

```bash
harness bootstrap
```

The bootstrap wizard will:

1. Detect your environment (Python version, git setup)
2. Check for credentials (`GITLAB_TOKEN`, `glab auth`)
3. Validate paths (repo root, database location)
4. Initialize the database with the full schema
5. Install Claude Code integration (MCP server, hooks, skills)
6. Run self-tests to verify everything works

For a non-interactive quick check:

```bash
harness bootstrap --check-only
```

## Sync Roles from Filesystem

Scan your `ansible/roles/` directory and populate the database:

```bash
harness sync
```

This discovers all roles, extracts metadata from `meta/main.yml`, detects
molecule test directories, and assigns wave numbers from your `harness.yml`
configuration.

## Run Your First Workflow

Execute the `box-up-role` workflow for a single role:

```bash
harness box-up-role common
```

This runs the full DAG workflow:

1. **validate_role** -- Verify the role exists and has valid structure
2. **analyze_deps** -- Discover and record role dependencies
3. **check_reverse_deps** -- Warn if downstream roles may be affected
4. **create_worktree** -- Create an isolated git worktree
5. **run_molecule** -- Execute molecule tests
6. **run_pytest** -- Execute pytest tests
7. **create_commit** -- Commit changes in the worktree
8. **push_branch** -- Push to remote
9. **create_issue** -- Create a GitLab issue
10. **create_mr** -- Create a merge request
11. **add_to_merge_train** -- Queue the MR in the merge train
12. **report_summary** -- Output final status

### Dry Run

Preview what the workflow would do without executing:

```bash
harness box-up-role common --dry-run
```

### Breakpoints

Pause the workflow at specific nodes for manual inspection:

```bash
harness box-up-role common --breakpoints run_molecule,create_mr
```

When paused, the workflow prints a resume command:

```
Workflow paused at: create_mr
Resume with: harness resume 42
```

## Verify Status

Check the status of all roles:

```bash
harness status
```

Output shows a table with wave assignment, worktree status, issue state,
MR state, and test results for every role.

Check a specific role:

```bash
harness status common
```

## Next Steps

- [Configuration Reference](configuration.md) -- All harness.yml options
- [Architecture](architecture.md) -- How the system works
- [Claude Code Integration](claude-code-integration.md) -- AI-assisted workflows
- [API Reference](api-reference.md) -- CLI and Python API details
