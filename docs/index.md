# DAG Harness

A self-installing DAG orchestration system built on LangGraph for automated
Ansible role deployments. Provides MCP server integration for Claude Code,
human-on-the-loop autonomous operation, and wave-based parallel execution.

## Features

- **LangGraph DAG execution** -- Reliable, checkpointed workflow execution
  with conditional routing and retry policies
- **Dual persistence** -- LangGraph checkpointer for graph state, SQLite
  StateDB for application data and observability
- **Cross-thread memory** -- HarnessStore (BaseStore) for sharing state
  between workflow executions
- **Claude Code integration** -- MCP server with 20+ tools, hook-based
  audit logging, and slash command skills
- **Wave-based parallelism** -- Roles execute in dependency-ordered waves
  with parallel execution within each wave
- **Human-on-the-loop (HOTL)** -- Autonomous operation with configurable
  checkpoints, notifications, and approval workflows
- **GitLab integration** -- Idempotent issue/MR creation, merge train
  management, and iteration tracking

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](quickstart.md) | Installation, setup, and first workflow in 5 minutes |
| [Configuration](configuration.md) | Complete harness.yml reference, environment variables, Pydantic models |
| [Architecture](architecture.md) | LangGraph engine, state management, dual persistence, wave execution |
| [Claude Code Integration](claude-code-integration.md) | MCP server setup, hooks, skills, HOTL with Claude Code |
| [API Reference](api-reference.md) | CLI commands, Python API, configuration models |

## Quick Example

```bash
# Install
uv pip install dag-harness

# Initialize in your Ansible project
cd /path/to/ansible-project
harness init
harness bootstrap

# Sync roles from filesystem
harness sync

# Run the box-up-role workflow
harness box-up-role common

# Check status
harness status
```

## Project Links

- [Source Code (GitHub)](https://github.com/Jesssullivan/dag-harness)
- [Source Code (GitLab)](https://gitlab.com/tinyland/projects/dag-harness)
- [Changelog](https://github.com/Jesssullivan/dag-harness/blob/main/CHANGELOG.md)
- [Issue Tracker](https://github.com/Jesssullivan/dag-harness/issues)

## License

MIT License -- see [LICENSE](https://github.com/Jesssullivan/dag-harness/blob/main/LICENSE)
for details.
