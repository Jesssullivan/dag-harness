# Contributing to DAG Harness

Thank you for your interest in contributing!

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://astral.sh/uv/) package manager
- Git

### Clone and Install

```bash
# From GitHub
git clone https://github.com/Jesssullivan/dag-harness.git
cd dag-harness/harness
uv sync --all-extras

# Or from GitLab
# git clone https://gitlab.com/tinyland/projects/dag-harness.git
```

### Verify Setup

```bash
uv run harness --version
uv run pytest --co -q  # List tests without running
```

## Running Tests

### All Tests

```bash
cd harness
uv run pytest
```

### With Coverage

```bash
uv run pytest --cov=harness --cov-report=term-missing
```

### Specific Categories

```bash
# Unit tests only
uv run pytest -m unit

# Property-based tests
uv run pytest -m pbt --hypothesis-show-statistics

# Integration tests
uv run pytest -m integration

# Fast tests (skip slow)
uv run pytest -m "not slow"
```

### Single Test File

```bash
uv run pytest tests/test_state_db.py -v
```

## Code Style

### Linting

```bash
uv run ruff check harness tests
```

### Auto-fix

```bash
uv run ruff check --fix harness tests
```

### Formatting

```bash
uv run ruff format harness tests
```

### Check Only

```bash
uv run ruff format --check harness tests
```

## Using the Justfile

If you have [just](https://github.com/casey/just) installed:

```bash
cd harness
just            # Show available commands
just test       # Run all tests
just lint       # Run linter
just format     # Format code
just check-all  # Full CI check
```

## Project Structure

```
harness/
├── harness/           # Main package
│   ├── cli.py         # Typer CLI entry point
│   ├── config.py      # Configuration loading
│   ├── dag/           # LangGraph workflow
│   │   ├── langgraph_engine.py  # Main DAG
│   │   └── nodes.py   # Node abstractions
│   ├── db/            # Database
│   │   ├── models.py  # Pydantic models
│   │   ├── state.py   # StateDB class
│   │   └── schema.sql # SQLite schema
│   ├── mcp/           # MCP server
│   ├── hotl/          # Autonomous operation
│   ├── bootstrap/     # Self-installation
│   └── costs/         # Cost tracking
├── tests/             # Test suite (900+ tests)
├── pyproject.toml     # Package config
└── justfile           # Task runner
```

## Adding Features

### New CLI Command

1. Add to `harness/cli.py`
2. Follow existing patterns (Typer decorators)
3. Add tests in `tests/test_cli.py`

### New MCP Tool

1. Add to `harness/mcp/server.py`
2. Use appropriate category
3. Add docstring with parameters
4. Add tests

### New DAG Node

1. Add to `harness/dag/langgraph_engine.py`
2. Define state changes
3. Add conditional routing if needed
4. Update workflow graph
5. Add tests

## Docstring Style

We use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is invalid.
    """
```

## Commit Messages

Follow conventional commits:

```
type(scope): description

- feat: New feature
- fix: Bug fix
- docs: Documentation
- test: Tests
- refactor: Code refactoring
- chore: Maintenance
```

Examples:
```
feat(dag): Add parallel test execution
fix(db): Handle null wave values
docs(readme): Update MCP tool count
test(cli): Add bootstrap command tests
```

## Pull Request Process

1. **Fork** the repository
2. **Create branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
3. **Make changes** with tests
4. **Run checks**:
   ```bash
   just check-all  # or manually:
   uv run ruff check harness tests
   uv run ruff format --check harness tests
   uv run pytest
   ```
5. **Commit** with conventional message
6. **Push** and create MR

## CI Pipeline

The GitLab CI runs:

- `lint` - Ruff check and format
- `test` - Full pytest suite
- `test-pbt` - Property-based tests
- `build-docs` - MkDocs build
- `pages` - Deploy to GitLab Pages

All checks must pass before merge.

## Questions?

- Open an issue for bugs or features
- Check existing issues first
- Include reproduction steps for bugs
