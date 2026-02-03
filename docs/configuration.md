# Configuration Reference

dag-harness is configured through `harness.yml`, environment variables, and
Pydantic-validated models. This document covers every configuration option.

## Configuration Loading Order

Configuration is resolved in the following priority (highest first):

1. **Environment variables** (`HARNESS_*`, `GITLAB_*`)
2. **Config file** (`harness.yml` or path from `HARNESS_CONFIG`)
3. **Defaults** (built into the dataclass/Pydantic models)

The config loader searches for `harness.yml` in these locations:

- `$HARNESS_CONFIG` (if set)
- `./harness.yml`
- `./harness.yaml`
- `.claude/skills/box-up-role/config.yml`
- `.claude/box-up-role/config.yml`
- Parent directories (up to 5 levels)

## harness.yml Format

### Minimal Configuration

```yaml
db_path: harness.db
repo_root: /path/to/ansible-project

gitlab:
  project_path: my-group/my-project

waves:
  0:
    name: Foundation
    roles:
      - common
```

### Full Configuration

```yaml
# Database location (relative to harness.yml or absolute)
db_path: harness.db

# Path to the repository containing Ansible roles
repo_root: /path/to/ansible-project

# GitLab configuration
gitlab:
  project_path: bates-ils/projects/ems/ems-mono
  group_path: bates-ils
  default_assignee: jsullivan2
  default_labels:
    - role
    - ansible
    - molecule
  default_iteration: EMS Upgrade

# Git worktree configuration
worktree:
  base_path: ../worktrees        # Where worktrees are created
  branch_prefix: sid/             # Branch naming prefix

# Testing configuration
testing:
  molecule_required: true         # Require molecule tests to pass
  pytest_required: true           # Require pytest tests to pass
  deploy_target: vmnode876        # Default deployment target
  molecule_timeout: 600           # Timeout in seconds
  pytest_timeout: 300             # Timeout in seconds

# Notification configuration
notifications:
  enabled: false
  discord_webhook_url: null       # Discord webhook for notifications
  email_recipient: null           # Email address for notifications

# Observability configuration
observability:
  langsmith_enabled: false        # Enable LangSmith tracing
  langsmith_project: dag-harness  # LangSmith project name
  anonymize_sensitive: true       # Anonymize sensitive data in traces

# Wave definitions (role execution ordering)
waves:
  0:
    name: Foundation
    roles:
      - common
  1:
    name: Infrastructure Foundation
    roles:
      - windows_prerequisites
      - ems_registry_urls
      - iis-config
  2:
    name: Core Platform
    roles:
      - ems_platform_services
      - ems_web_app
      - database_clone
      - ems_download_artifacts
  3:
    name: Web Applications
    roles:
      - ems_master_calendar
      - ems_master_calendar_api
      - ems_campus_webservice
      - ems_desktop_deploy
  4:
    name: Supporting Services
    roles:
      - grafana_alloy_windows
      - email_infrastructure
      - hrtk_protected_users
      - ems_environment_util
```

## GitLab Configuration

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project_path` | string | **required** | GitLab project path (e.g., `group/project`) |
| `group_path` | string | `""` | GitLab group path for iteration queries |
| `default_assignee` | string | `""` | Default issue/MR assignee username |
| `default_labels` | list[str] | `[]` | Labels applied to new issues and MRs |
| `default_iteration` | string | `""` | Iteration title for new issues |

### Pydantic Model (GitLabConfigModel)

The Pydantic-validated version adds:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_branch` | string | `"main"` | Target branch for MRs |
| `merge_method` | string | `"merge"` | Merge method: `merge`, `rebase_merge`, `ff` |

The `project_path` must contain at least one `/` (namespace/project format).

## Wave Definitions

Waves control execution ordering. Roles in the same wave can run in parallel;
roles in higher waves depend on all lower waves completing first.

### Dictionary Format (harness.yml)

```yaml
waves:
  0:
    name: Foundation
    roles:
      - common
  1:
    name: Infrastructure
    roles:
      - nginx
      - docker
```

### Pydantic Model (WaveDefinition)

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `wave` | int | **required** | 0-9 | Wave number |
| `roles` | list[str] | **required** | min 1 item | Role names in this wave |
| `parallel` | bool | `true` | -- | Whether roles in this wave run in parallel |
| `name` | string | `""` | -- | Human-readable wave name |

**Validation rules:**

- Each role must appear in exactly one wave (no duplicates across waves)
- Wave numbers must be unique
- Wave numbers must be between 0 and 9

## Checkpointer Configuration

Controls how LangGraph workflow state is persisted.

### Pydantic Model (CheckpointerConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `"sqlite"` | Backend type: `sqlite` or `postgres` |
| `sqlite_path` | string | `".harness/checkpoints.db"` | Path for SQLite checkpointer |
| `postgres_url` | string | `null` | PostgreSQL connection string |
| `sync_to_statedb` | bool | `true` | Sync checkpoints to StateDB |

### SQLite (Default)

```yaml
checkpointer:
  backend: sqlite
  sqlite_path: .harness/checkpoints.db
  sync_to_statedb: true
```

SQLite is the default and requires no additional dependencies. Checkpoint data
is stored locally and supports resumable workflows.

### PostgreSQL (Production)

```yaml
checkpointer:
  backend: postgres
  postgres_url: postgresql://user:pass@host:5432/harness
  sync_to_statedb: true
```

PostgreSQL requires the `postgres` optional dependency:

```bash
uv pip install "dag-harness[postgres]"
```

The `postgres_url` must start with `postgresql://` or `postgres://`.

### Dual Persistence

When `sync_to_statedb` is `true` (the default), the `CheckpointerWithStateDB`
wrapper synchronizes checkpoint data between LangGraph's native checkpointer
and the StateDB `workflow_executions.checkpoint_data` column. This provides:

- LangGraph checkpointer as source of truth for graph state
- StateDB copy for observability and external querying
- Graceful degradation if StateDB sync fails

## HOTL Mode Configuration

Human-on-the-loop (HOTL) enables autonomous operation with human oversight.

### Pydantic Model (HOTLConfig)

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `enabled` | bool | `false` | -- | Enable HOTL mode |
| `max_iterations` | int | `50` | >= 1 | Maximum workflow iterations |
| `notification_interval` | int | `300` | >= 60 | Seconds between status notifications |
| `approval_timeout` | int | `3600` | >= 300 | Seconds to wait for human approval |

### Example

```yaml
hotl:
  enabled: true
  max_iterations: 100
  notification_interval: 300
  approval_timeout: 3600
```

## Environment Variables

### Core Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HARNESS_CONFIG` | Path to harness.yml | Auto-detected |
| `HARNESS_DB_PATH` | Database file path | `harness.db` |
| `HARNESS_REPO_ROOT` | Override repo_root from config | From config |
| `HARNESS_BREAKPOINTS` | Enable static breakpoints (`true`/`1`/`yes`) | `false` |

### GitLab Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITLAB_TOKEN` | GitLab personal access token | -- |
| `GITLAB_E2E_TOKEN` | Token for E2E integration tests | -- |
| `GITLAB_PROJECT` | Override gitlab.project_path | From config |
| `GITLAB_GROUP` | Override gitlab.group_path | From config |

### Notification Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DISCORD_WEBHOOK_URL` | Discord webhook URL (enables notifications) | -- |
| `EMAIL_RECIPIENT` | Email address (enables notifications) | -- |

### Observability Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing (`true`) | `false` |
| `LANGCHAIN_PROJECT` | LangSmith project name | `dag-harness` |
| `LANGCHAIN_API_KEY` | LangSmith API key | -- |
| `HARNESS_ANONYMIZE_SENSITIVE` | Anonymize sensitive data (`false` to disable) | `true` |

### KeePass Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KEEPASS_DATABASE_PATH` | Path to KeePassXC database | -- |
| `KEEPASSXC_DB_PASSWORD` | KeePassXC database password | -- |

## Environment Variable Interpolation

The Pydantic config loader supports `${VAR}` interpolation in harness.yml:

```yaml
gitlab:
  project_path: ${GITLAB_PROJECT}

checkpointer:
  postgres_url: ${DATABASE_URL}
```

If a referenced variable is not set, loading raises a `ValueError`.

## HarnessConfigModel (Pydantic Root)

The top-level validated configuration model:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | string | `"1.0"` | Config schema version |
| `project_name` | string | `"dag-harness"` | Project identifier |
| `gitlab` | GitLabConfigModel | **required** | GitLab settings |
| `waves` | list[WaveDefinition] | `[]` | Wave definitions |
| `checkpointer` | CheckpointerConfig | SQLite defaults | Checkpointer settings |
| `hotl` | HOTLConfig | Disabled defaults | HOTL settings |
| `db_path` | string | `".harness/harness.db"` | Database path |
| `log_level` | string | `"INFO"` | Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `breakpoints` | list[str] | `[]` | Static breakpoint node names |

### Loading from YAML

```python
from harness.config import HarnessConfigModel

# Load and validate
config = HarnessConfigModel.from_yaml("harness.yml")

# Load with environment variable interpolation
config = HarnessConfigModel.from_env()
```

### Loading Legacy Config

```python
from harness.config import HarnessConfig

# Auto-detect config file
config = HarnessConfig.load()

# Explicit path
config = HarnessConfig.load("/path/to/harness.yml")

# Validate
errors = config.validate()
if errors:
    print("Config errors:", errors)
```
