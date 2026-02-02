# Getting Started

This guide walks you through installing and configuring the DAG Harness.

## Prerequisites

- Python 3.11+
- Git
- [uv](https://github.com/astral-sh/uv) package manager
- GitLab account with API token (for MR creation)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd disposable-dag-langchain
```

### 2. Install Dependencies

```bash
cd harness
uv sync
cd ..
```

### 3. Bootstrap the Harness

The bootstrap command runs an interactive wizard:

```bash
harness bootstrap
```

This will:

1. **Detect environment** - Check Python version, git setup
2. **Check credentials** - Look for GITLAB_TOKEN and other required credentials
3. **Validate paths** - Verify database, worktree, and project paths
4. **Initialize database** - Create SQLite database with schema
5. **Install MCP client integration** - Set up MCP server, hooks, and skills
6. **Run self-tests** - Verify everything is working

### 4. Verify Installation

```bash
harness bootstrap --check-only
```

This shows the current state without making changes.

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GITLAB_TOKEN` | GitLab API token | Yes |
| `HARNESS_DB_PATH` | Database file path | No (default: ./harness/harness.db) |
| `DISCORD_WEBHOOK_URL` | Discord notifications | No |
| `HOTL_EMAIL_TO` | HOTL email notifications | No |
| `KEEPASSXC_DB_PASSWORD` | KeePassXC password | No |

### Setting Credentials

Set the GitLab token before running bootstrap:

```bash
export GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx
```

Or create a `.env` file in the project root:

```
GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx
```

## First Workflow

### 1. Sync Roles

Scan the filesystem for Ansible roles:

```bash
harness sync
```

### 2. List Available Roles

```bash
harness list-roles
harness list-roles --wave 1  # Filter by wave
```

### 3. Check Role Status

```bash
harness status
harness status <role-name>
```

### 4. Execute a Workflow

```bash
harness box-up-role <role-name>
```

Add breakpoints for debugging:

```bash
harness box-up-role <role-name> --breakpoints run_molecule,create_mr
```

## MCP client Integration

After bootstrap, the harness integrates with MCP client:

### MCP Server

The MCP server provides tools for MCP client to interact with the harness:

```bash
# Manual start (usually automatic via settings.json)
harness mcp-server
```

### Slash Commands

Use these slash commands in MCP client:

- `/box-up-role <name>` - Package an Ansible role
- `/hotl start` - Start autonomous operation
- `/observability debug-*` - Debugging commands

### Hooks

Pre/post tool hooks for validation and audit:

- `validate-box-up-env.sh` - Validates GitLab auth before operations
- `notify-box-up-status.sh` - Notifications on failures
- `audit-logger.py` - Records all tool invocations

## Troubleshooting

### Bootstrap Fails

Run with check-only to see detailed status:

```bash
harness bootstrap --check-only
```

### MCP Server Not Responding

Test the MCP server manually:

```bash
uv run --directory ./harness python -m harness.mcp.server
```

### Missing Credentials

Verify credentials are set:

```bash
echo $GITLAB_TOKEN
```

### Database Issues

Reset the database (destructive):

```bash
harness db reset --yes
harness bootstrap
```

## Next Steps

- Read the [Architecture](architecture.md) guide
- Explore [CLI commands](api/cli.md)
- Learn about HOTL (Human Out of The Loop) mode via `harness hotl --help`
