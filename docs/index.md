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

## Installation

=== "Quick Install"

    ```bash
    curl -sSL https://raw.githubusercontent.com/Jesssullivan/Ansible-DAG-Harness/main/scripts/bootstrap.sh | bash
    ```

=== "uv"

    ```bash
    # From GitHub (specific version)
    uv tool install git+https://github.com/Jesssullivan/Ansible-DAG-Harness.git@main

    # From GitHub (latest)
    uv tool install git+https://github.com/Jesssullivan/Ansible-DAG-Harness.git
    ```

=== "pip"

    ```bash
    # From GitHub
    pip install git+https://github.com/Jesssullivan/Ansible-DAG-Harness.git@main

    # Direct wheel (fastest)
    pip install https://github.com/Jesssullivan/Ansible-DAG-Harness/releases/download/latest/dag_harness-latest-py3-none-any.whl
    ```

=== "From Source"

    ```bash
    git clone https://github.com/Jesssullivan/Ansible-DAG-Harness.git
    cd dag-harness/harness
    uv sync  # or: pip install -e .
    ```

See [Getting Started](getting-started.md) for detailed installation instructions.

## Quick Start

```bash
# Bootstrap (interactive setup)
harness bootstrap

# Verify installation
harness bootstrap --check-only
```

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

- [Source Code (GitHub)](https://github.com/Jesssullivan/Ansible-DAG-Harness)
- [Source Code (GitLab)](https://gitlab.com/tinyland/projects/dag-harness)
- [Changelog](https://github.com/Jesssullivan/Ansible-DAG-Harness/blob/main/CHANGELOG.md)
- [Issue Tracker](https://github.com/Jesssullivan/Ansible-DAG-Harness/issues)

## License

MIT License -- see [LICENSE](https://github.com/Jesssullivan/Ansible-DAG-Harness/blob/main/LICENSE)
for details.
