# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-03

### Added

- LangGraph workflow engine with 15-node box-up-role graph
- Unified checkpointing (LangGraph + StateDB)
- Cross-thread memory via HarnessStore
- GitLab integration with idempotency, retry, merge trains
- Database migration system (3 migrations)
- Pydantic configuration validation
- HOTL mode with wave-based queue
- Bootstrap `harness init` command
- Claude Code skill integration
- Comprehensive test suite (500+ tests)
- CI/CD release automation (GitHub Actions + GitLab CI)
- Conventional commits tooling (commitizen)

### Changed

- Version bump from 0.2.0 to 0.3.0
- Package data now includes schema.sql and templates

### Fixed

- StateDB connection pattern uses context manager
- LangGraph checkpoint serialization for complex state

## [0.2.0] - 2026-01-30

### Added

- **LangGraph DAG Engine**: Full workflow orchestration with TypedDict state schema, conditional routing, and SQLite checkpointing
- **MCP client Integration**: Self-installing MCP server, hooks (PreToolUse/PostToolUse), and skill definitions
- **StateDB**: SQLite-backed state management with graph-queryable patterns (recursive CTEs for dependency traversal)
- **WorktreeManager**: Git worktree management for parallel role development
- **GitLabClient**: Full GitLab API integration for issues, MRs, iterations, and merge trains
- **Skills**:
  - `/box-up-role`: Package Ansible roles with worktree, tests, and GitLab MR
  - `/hotl`: Human Out of The Loop autonomous operation mode (experimental)
  - `/observability`: Debugging and diagnostic commands
- **Hooks**:
  - `validate-box-up-env.sh`: Validates GitLab auth and KeePass credentials
  - `universal-hook.py`: Central hook dispatcher with rate limiting
  - `notify-box-up-status.sh`: Discord/email notifications on failures
  - `audit-logger.py`: Records all tool invocations
- **CLI Commands**: `harness box-up-role`, `harness status`, `harness sync`, `harness db`, `harness metrics`
- **Test Regression Tracking**: Automatic detection and tracking of test regressions across runs

### Changed

- Refactored DAG nodes to use Python modules directly instead of external shell scripts
- Hook paths now use relative paths (`./`) for MCP client compatibility

### Fixed

- Fixed `${workspaceFolder}` expansion issue in MCP client hooks (MCP client does not expand this variable)
- Fixed MCP server environment variable paths

## [0.1.0] - 2026-01-29

### Added

- Initial project structure
- Basic CLI scaffolding
- SQLite schema for role, dependency, and workflow tracking
