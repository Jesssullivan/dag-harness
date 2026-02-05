# DAG Harness

Self-installing DAG orchestration for Ansible role deployments using LangGraph.

## Features

- **14-Node LangGraph Workflow** - Box-up-role DAG with parallel testing, HITL gates, and merge train integration
- **40+ MCP Tools** - Full Claude Code integration across 8 categories
- **HOTL Mode** - Human Out of The Loop autonomous operation
- **Cost Tracking** - Per-session token usage and cost reporting
- **Self-Installing** - One command bootstrap with MCP client setup

## Installation

### Install via uv (Recommended)

```bash
# From GitHub (latest main)
uv pip install git+https://github.com/Jesssullivan/dag-harness.git#subdirectory=harness

# With extras
uv pip install "dag-harness[dev] @ git+https://github.com/Jesssullivan/dag-harness.git#subdirectory=harness"
```

### Install via pip

```bash
pip install git+https://github.com/Jesssullivan/dag-harness.git#subdirectory=harness
```

### Install from Source

```bash
git clone https://github.com/Jesssullivan/dag-harness.git
cd dag-harness/harness
uv sync  # or: pip install -e .
```

## Quick Start

```bash
# Bootstrap (interactive setup)
harness bootstrap

# Verify installation
harness bootstrap --check-only
```

## CLI Commands

```bash
# Core workflow
harness box-up-role <role>     # Execute workflow for role
harness status [role]          # Show role status
harness list-roles             # List all roles
harness sync                   # Sync from filesystem

# HOTL autonomous mode
harness hotl start             # Start autonomous operation
harness hotl status            # Check status
harness hotl stop              # Stop gracefully

# Resume paused workflows
harness resume <id>            # Resume execution
harness resume <id> --approve  # Approve human gate
harness resume <id> --reject   # Reject with reason

# Database & debugging
harness db stats               # Database statistics
harness check                  # Run self-checks
harness graph                  # Show workflow DAG
harness costs report           # Cost breakdown
```

## Workflow DAG

The `box-up-role` workflow processes Ansible roles through 14 nodes:

```
validate_role → analyze_deps → check_reverse_deps → create_worktree
    ↓
run_molecule ─┬─→ merge_test_results → validate_deploy → create_commit
run_pytest  ──┘
    ↓
push_branch → create_issue → create_mr → human_approval
    ↓
add_to_merge_train → report_summary
```

Key features:
- **Parallel testing** - Molecule and pytest run simultaneously
- **HITL gate** - Human approval before merge train
- **Retry policies** - Exponential backoff for GitLab API calls
- **Checkpointing** - Resume from any node

## MCP Integration

40+ tools across 8 categories for Claude Code:

| Category | Tools | Purpose |
|----------|-------|---------|
| role_management | 7 | List, status, dependencies |
| workflow | 5 | Execution status, HOTL control |
| worktree | 3 | Git worktree management |
| testing | 2 | Test history, regressions |
| agent | 6 | Subagent coordination |
| costs | 3 | Token usage tracking |
| credentials | 1 | Credential discovery |
| search | 2 | Tool search and discovery |

## Directory Structure

```
dag-harness/
├── harness/                    # Python package
│   ├── harness/
│   │   ├── cli.py              # Typer CLI
│   │   ├── dag/                # LangGraph workflow
│   │   ├── db/                 # SQLite state
│   │   ├── mcp/                # MCP server
│   │   ├── hotl/               # Autonomous operation
│   │   ├── costs/              # Cost tracking
│   │   └── bootstrap/          # Self-installation
│   └── tests/                  # 900+ tests
├── docs/                       # Documentation
│   ├── llms.txt                # LLM-friendly reference
│   └── llms.md                 # Detailed LLM docs
└── .gitlab-ci.yml              # CI/CD pipeline
```

## Configuration

Create `harness.yml`:

```yaml
db_path: harness/harness.db
repo_root: /path/to/ansible/roles

gitlab:
  project_path: group/project
  default_assignee: username

worktree:
  base_path: /path/to/worktrees
  branch_prefix: sid/

waves:
  0: { name: Foundation, roles: [common] }
  1: { name: Infrastructure, roles: [...] }
```

## Requirements

- Python 3.11+
- [uv](https://astral.sh/uv/) package manager
- Git with worktree support
- GitLab API access (GITLAB_TOKEN)

## Documentation

- **[Getting Started](docs/getting-started.md)** - Installation and first steps
- **[Architecture](docs/architecture.md)** - System design
- **[CLI Reference](docs/api/cli.md)** - All commands
- **[MCP Tools](docs/api/mcp-tools.md)** - Tool reference
- **[LLM Docs](docs/llms.md)** - For AI assistants

## Author

**Jess Sullivan** — [xoxd.ai](https://xoxd.ai)

Developed by [Tinyland.dev, Inc.](https://tinyland.dev)

## License

MIT — see [LICENSE](LICENSE)
