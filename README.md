# DAG Harness

LangGraph DAG orchestration for Ansible role deployments with GitLab integration.

## Features

- **17-node workflow** with parallel testing, HITL gates, recovery subgraph
- **30 MCP tools** across 9 categories for Claude Code integration
- **HOTL mode** - Human Out of The Loop autonomous operation
- **Checkpointing** - Resume from any node with SQLite persistence

## Installation

```bash
# From source (recommended)
git clone https://github.com/Jesssullivan/dag-harness.git
cd dag-harness/harness
uv sync

# Or via pip
pip install git+https://github.com/Jesssullivan/dag-harness.git#subdirectory=harness
```

## Usage

```bash
harness init                    # Initialize in repo
harness box-up-role <role>      # Execute workflow
harness status                  # Show status
harness hotl start              # Autonomous mode
```

## Workflow

```
validate_role → analyze_deps → check_reverse_deps → create_worktree
                                                          ↓
run_molecule ─┬─→ merge_test_results → validate_deploy → create_commit
run_pytest  ──┘                                              ↓
                push_branch → create_issue → create_mr → human_approval
                                                              ↓
                              add_to_merge_train → report_summary
                                        ↓
                               [recovery subgraph on failure]
```

17 nodes: `validate_role`, `analyze_deps`, `check_reverse_deps`, `create_worktree`, `run_molecule`, `run_pytest`, `merge_test_results`, `validate_deploy`, `create_commit`, `push_branch`, `create_issue`, `create_mr`, `human_approval`, `add_to_merge_train`, `report_summary`, `notify_failure`, `recovery`

## MCP Tools

30 tools defined in `harness/mcp/server.py`:

| Category | Count |
|----------|-------|
| role_management | 7 |
| workflow | 5 |
| agent | 6 |
| worktree | 3 |
| costs | 3 |
| testing | 2 |
| search | 2 |
| credentials | 1 |
| merge_train | 1 |

## Configuration

```yaml
# harness.yml
db_path: harness.db
repo_root: /path/to/ansible/roles

gitlab:
  project_path: group/project
  default_assignee: username

worktree:
  base_path: /path/to/worktrees
  branch_prefix: feature/
```

## Requirements

- Python 3.11+
- Git with worktree support
- GitLab API access (`GITLAB_TOKEN`)

## Tests

```bash
cd harness
uv run pytest                           # 1667 tests
uv run pytest -m "not integration"      # Unit tests only
uv run pytest -m pbt                    # Property-based tests
```

## Documentation

- [Architecture](docs/architecture.md)
- [CLI Reference](docs/api/cli.md)
- [MCP Tools](docs/api/mcp-tools.md)
- [LLM Context](docs/llms.md)

## License

MIT
