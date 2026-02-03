"""
Harness configuration management.

Loads configuration from:
1. Environment variables (HARNESS_* prefix)
2. Config files (harness.yml, .claude/box-up-role/config.yml)
3. Defaults

Includes both legacy dataclass-based config (HarnessConfig) and
Pydantic-based validated config (HarnessConfigModel).
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def find_repo_python(repo_root: Path) -> str:
    """
    Find the Python interpreter for a repository.

    Search order:
    1. repo_root/.venv/bin/python
    2. repo_root/venv/bin/python
    3. UV_PROJECT_ENVIRONMENT if set
    4. sys.executable (fallback)

    Args:
        repo_root: Root directory of the target repository

    Returns:
        Path to Python interpreter
    """
    venv_paths = [
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python3",
        repo_root / "venv" / "bin" / "python",
        repo_root / "venv" / "bin" / "python3",
    ]

    for venv_python in venv_paths:
        if venv_python.exists():
            return str(venv_python)

    # Check UV_PROJECT_ENVIRONMENT
    uv_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_env:
        uv_python = Path(uv_env) / "bin" / "python"
        if uv_python.exists():
            return str(uv_python)

    # Fallback to current interpreter
    return sys.executable


@dataclass
class GitLabConfig:
    """GitLab configuration."""

    base_url: str = "https://gitlab.com/api/v4"
    project_path: str = "bates-ils/projects/ems/ems-mono"
    group_path: str = "bates-ils"
    default_assignee: str = "jsullivan2"
    default_labels: list[str] = field(default_factory=lambda: ["role", "ansible", "molecule"])
    default_iteration: str = "EMS Upgrade"
    token_env_var: str = "GITLAB_TOKEN"  # Environment variable to read token from


@dataclass
class WorktreeConfig:
    """Git worktree configuration."""

    base_path: str = ".."
    branch_prefix: str = "sid/"


@dataclass
class TestingConfig:
    """Testing configuration."""

    molecule_required: bool = True
    pytest_required: bool = True
    deploy_target: str = "vmnode852"
    molecule_timeout: int = 600
    pytest_timeout: int = 300


@dataclass
class NotificationConfig:
    """Notification configuration."""

    discord_webhook_url: str | None = None
    email_recipient: str | None = None
    email_from: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_username: str | None = None
    smtp_password: str | None = None
    enabled: bool = False


@dataclass
class ObservabilityConfig:
    """
    Observability configuration for LangSmith tracing.

    Environment variables:
    - LANGCHAIN_TRACING_V2: Set to "true" to enable LangSmith tracing
    - LANGCHAIN_PROJECT: Project name in LangSmith (default: "dag-harness")
    - LANGCHAIN_API_KEY: LangSmith API key (required if tracing enabled)
    - HARNESS_ANONYMIZE_SENSITIVE: Set to "false" to disable sensitive data anonymization
    """

    langsmith_enabled: bool = False  # Set from LANGCHAIN_TRACING_V2
    langsmith_project: str = "dag-harness"  # Set from LANGCHAIN_PROJECT
    anonymize_sensitive: bool = True  # Anonymize sensitive data before sending

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Load observability config from environment variables."""
        return cls(
            langsmith_enabled=os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true",
            langsmith_project=os.environ.get("LANGCHAIN_PROJECT", "dag-harness"),
            anonymize_sensitive=os.environ.get("HARNESS_ANONYMIZE_SENSITIVE", "true").lower()
            != "false",
        )


@dataclass
class HarnessConfig:
    """Main harness configuration."""

    db_path: str = "harness.db"
    repo_root: str = "."
    gitlab: GitLabConfig = field(default_factory=GitLabConfig)
    worktree: WorktreeConfig = field(default_factory=WorktreeConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # Wave definitions
    waves: dict[int, dict] = field(
        default_factory=lambda: {
            0: {"name": "Foundation", "roles": ["common"]},
            1: {
                "name": "Infrastructure Foundation",
                "roles": ["windows_prerequisites", "ems_registry_urls", "iis-config"],
            },
            2: {
                "name": "Core Platform",
                "roles": [
                    "ems_platform_services",
                    "ems_web_app",
                    "database_clone",
                    "ems_download_artifacts",
                ],
            },
            3: {
                "name": "Web Applications",
                "roles": [
                    "ems_master_calendar",
                    "ems_master_calendar_api",
                    "ems_campus_webservice",
                    "ems_desktop_deploy",
                ],
            },
            4: {
                "name": "Supporting Services",
                "roles": [
                    "grafana_alloy_windows",
                    "email_infrastructure",
                    "hrtk_protected_users",
                    "ems_environment_util",
                ],
            },
        }
    )

    # Cached repo_python path (set after load)
    _repo_python: str | None = field(default=None, repr=False)

    def __post_init__(self):
        """Resolve repo_root to absolute path after initialization."""
        self.repo_root = str(Path(self.repo_root).resolve())
        self._repo_python = None  # Will be computed lazily

    @property
    def repo_python(self) -> str:
        """Get the Python interpreter for the target repository."""
        if self._repo_python is None:
            self._repo_python = find_repo_python(Path(self.repo_root))
        return self._repo_python

    @classmethod
    def load(cls, config_path: str | None = None) -> "HarnessConfig":
        """Load configuration from file and environment."""
        config = cls()

        # Check HARNESS_CONFIG environment variable first
        if not config_path:
            config_path = os.environ.get("HARNESS_CONFIG")

        # Try to find config file
        paths_to_try = [
            config_path,
            "harness.yml",
            "harness.yaml",
            ".claude/skills/box-up-role/config.yml",
            ".claude/box-up-role/config.yml",
        ]

        # Also search up from CWD to find repo root
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents)[:5]:
            paths_to_try.append(str(parent / "harness.yml"))
            paths_to_try.append(str(parent / ".harness" / "config.yml"))

        for path in paths_to_try:
            if path and Path(path).exists():
                config = cls._load_from_file(path)
                break

        # Override with environment variables
        config._load_from_env()

        # Ensure repo_root is absolute
        config.repo_root = str(Path(config.repo_root).resolve())

        return config

    @classmethod
    def _load_from_file(cls, path: str) -> "HarnessConfig":
        """Load configuration from YAML file.
        
        Relative paths in the config (repo_root, db_path) are resolved
        relative to the config file's directory, not the CWD.
        """
        config_path = Path(path).resolve()
        config_dir = config_path.parent

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # Resolve paths relative to config file directory
        if "db_path" in data:
            db_path = Path(data["db_path"])
            if not db_path.is_absolute():
                db_path = config_dir / db_path
            config.db_path = str(db_path)
        if "repo_root" in data:
            repo_root = Path(data["repo_root"])
            if not repo_root.is_absolute():
                repo_root = config_dir / repo_root
            config.repo_root = str(repo_root)

        if "gitlab" in data:
            gl = data["gitlab"]
            config.gitlab = GitLabConfig(
                base_url=gl.get("base_url", config.gitlab.base_url),
                project_path=gl.get("project_path", config.gitlab.project_path),
                group_path=gl.get("group_path", config.gitlab.group_path),
                default_assignee=gl.get("default_assignee", config.gitlab.default_assignee),
                default_labels=gl.get("default_labels", config.gitlab.default_labels),
                default_iteration=gl.get("default_iteration", config.gitlab.default_iteration),
                token_env_var=gl.get("token_env_var", config.gitlab.token_env_var),
            )

        if "worktree" in data:
            wt = data["worktree"]
            config.worktree = WorktreeConfig(
                base_path=wt.get("base_path", config.worktree.base_path),
                branch_prefix=wt.get("branch_prefix", config.worktree.branch_prefix),
            )

        if "testing" in data:
            t = data["testing"]
            config.testing = TestingConfig(
                molecule_required=t.get("molecule_required", config.testing.molecule_required),
                pytest_required=t.get("pytest_required", config.testing.pytest_required),
                deploy_target=t.get("deploy_target", config.testing.deploy_target),
                molecule_timeout=t.get("molecule_timeout", config.testing.molecule_timeout),
                pytest_timeout=t.get("pytest_timeout", config.testing.pytest_timeout),
            )

        if "notifications" in data:
            n = data["notifications"]
            config.notifications = NotificationConfig(
                discord_webhook_url=n.get("discord_webhook_url"),
                email_recipient=n.get("email_recipient"),
                enabled=n.get("enabled", False),
            )

        if "waves" in data:
            config.waves = data["waves"]

        return config

    def _load_from_env(self) -> None:
        """Override configuration from environment variables."""
        if os.environ.get("HARNESS_DB_PATH"):
            self.db_path = os.environ["HARNESS_DB_PATH"]

        # NEW: Support HARNESS_REPO_ROOT environment variable
        if os.environ.get("HARNESS_REPO_ROOT"):
            self.repo_root = os.environ["HARNESS_REPO_ROOT"]

        if os.environ.get("GITLAB_PROJECT"):
            self.gitlab.project_path = os.environ["GITLAB_PROJECT"]

        if os.environ.get("GITLAB_GROUP"):
            self.gitlab.group_path = os.environ["GITLAB_GROUP"]

        if os.environ.get("DISCORD_WEBHOOK_URL"):
            self.notifications.discord_webhook_url = os.environ["DISCORD_WEBHOOK_URL"]
            self.notifications.enabled = True

        if os.environ.get("EMAIL_RECIPIENT"):
            self.notifications.email_recipient = os.environ["EMAIL_RECIPIENT"]
            self.notifications.enabled = True

        # Load observability config from environment
        self.observability = ObservabilityConfig.from_env()

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        repo_root = Path(self.repo_root)
        if not repo_root.exists():
            errors.append(f"repo_root does not exist: {self.repo_root}")
        elif not (repo_root / "ansible" / "roles").exists():
            errors.append(f"No ansible/roles directory in repo_root: {self.repo_root}")

        # Validate db_path is writable
        db_path = Path(self.db_path)
        if not db_path.is_absolute():
            db_path = repo_root / db_path
        db_parent = db_path.parent
        if not db_parent.exists():
            try:
                db_parent.mkdir(parents=True)
            except OSError as e:
                errors.append(f"Cannot create database directory {db_parent}: {e}")

        return errors

    def get_wave_for_role(self, role_name: str) -> tuple[int, str]:
        """Get wave number and name for a role."""
        for wave_num, wave_info in self.waves.items():
            if role_name in wave_info.get("roles", []):
                return wave_num, wave_info.get("name", f"Wave {wave_num}")
        return 0, "Unassigned"

    def is_foundation_role(self, role_name: str) -> bool:
        """Check if a role is a foundation role (Wave 0 with no dependencies)."""
        wave_num, _ = self.get_wave_for_role(role_name)
        return wave_num == 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "db_path": self.db_path,
            "repo_root": self.repo_root,
            "gitlab": {
                "base_url": self.gitlab.base_url,
                "project_path": self.gitlab.project_path,
                "group_path": self.gitlab.group_path,
                "default_assignee": self.gitlab.default_assignee,
                "default_labels": self.gitlab.default_labels,
                "default_iteration": self.gitlab.default_iteration,
                "token_env_var": self.gitlab.token_env_var,
            },
            "worktree": {
                "base_path": self.worktree.base_path,
                "branch_prefix": self.worktree.branch_prefix,
            },
            "testing": {
                "molecule_required": self.testing.molecule_required,
                "pytest_required": self.testing.pytest_required,
                "deploy_target": self.testing.deploy_target,
                "molecule_timeout": self.testing.molecule_timeout,
                "pytest_timeout": self.testing.pytest_timeout,
            },
            "notifications": {
                "discord_webhook_url": self.notifications.discord_webhook_url,
                "email_recipient": self.notifications.email_recipient,
                "enabled": self.notifications.enabled,
            },
            "observability": {
                "langsmith_enabled": self.observability.langsmith_enabled,
                "langsmith_project": self.observability.langsmith_project,
                "anonymize_sensitive": self.observability.anonymize_sensitive,
            },
            "waves": self.waves,
        }

    def save(self, path: str = "harness.yml") -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# =============================================================================
# PYDANTIC CONFIGURATION MODELS (Validated)
# =============================================================================

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values.

    Args:
        value: String potentially containing ${VAR} patterns.

    Returns:
        String with environment variables substituted.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")
        return env_value

    return _ENV_VAR_PATTERN.sub(replacer, value)


def _interpolate_dict(data: dict) -> dict:
    """Recursively interpolate environment variables in a dictionary."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = _interpolate_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _interpolate_dict(value)
        elif isinstance(value, list):
            result[key] = _interpolate_list(value)
        else:
            result[key] = value
    return result


def _interpolate_list(data: list) -> list:
    """Recursively interpolate environment variables in a list."""
    result = []
    for item in data:
        if isinstance(item, str):
            result.append(_interpolate_env_vars(item))
        elif isinstance(item, dict):
            result.append(_interpolate_dict(item))
        elif isinstance(item, list):
            result.append(_interpolate_list(item))
        else:
            result.append(item)
    return result


class GitLabConfigModel(BaseModel):
    """GitLab project configuration."""

    project_path: str  # e.g., "tinyland/projects/dag-harness"
    default_branch: str = "main"
    merge_method: str = "merge"  # merge, rebase_merge, ff

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, v: str) -> str:
        """Must be namespace/project format."""
        if "/" not in v or v.count("/") < 1:
            raise ValueError("project_path must be in 'namespace/project' format")
        return v

    @field_validator("merge_method")
    @classmethod
    def validate_merge_method(cls, v: str) -> str:
        """Must be a valid merge method."""
        valid_methods = {"merge", "rebase_merge", "ff"}
        if v not in valid_methods:
            raise ValueError(f"merge_method must be one of {valid_methods}")
        return v


class WaveDefinition(BaseModel):
    """Wave configuration for role ordering."""

    wave: int = Field(ge=0, le=9)
    roles: list[str] = Field(min_length=1)
    parallel: bool = True
    name: str = ""


class CheckpointerConfig(BaseModel):
    """Checkpointer configuration."""

    backend: str = "sqlite"  # sqlite, postgres
    sqlite_path: str = ".harness/checkpoints.db"
    postgres_url: Optional[str] = None
    sync_to_statedb: bool = True

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Must be a valid backend."""
        valid_backends = {"sqlite", "postgres"}
        if v not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        return v

    @field_validator("postgres_url")
    @classmethod
    def validate_postgres_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate postgres URL format if provided."""
        if v is not None:
            if not v.startswith(("postgresql://", "postgres://")):
                raise ValueError(
                    "postgres_url must start with 'postgresql://' or 'postgres://'"
                )
        return v


class HOTLConfig(BaseModel):
    """Human-on-the-loop configuration."""

    enabled: bool = False
    max_iterations: int = Field(default=50, ge=1)
    notification_interval: int = Field(default=300, ge=60)
    approval_timeout: int = Field(default=3600, ge=300)


class HarnessConfigModel(BaseModel):
    """Root configuration model for harness.yml.

    Provides Pydantic-based validation for harness configuration,
    complementing the legacy dataclass-based HarnessConfig.
    """

    version: str = "1.0"
    project_name: str = "dag-harness"
    gitlab: GitLabConfigModel
    waves: list[WaveDefinition] = []
    checkpointer: CheckpointerConfig = CheckpointerConfig()
    hotl: HOTLConfig = HOTLConfig()
    db_path: str = ".harness/harness.db"
    log_level: str = "INFO"
    breakpoints: list[str] = []

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Must be a valid log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @model_validator(mode="after")
    def validate_wave_roles_unique(self) -> "HarnessConfigModel":
        """All roles across waves must be unique."""
        all_roles: list[str] = []
        for wave in self.waves:
            for role in wave.roles:
                if role in all_roles:
                    raise ValueError(f"Role '{role}' appears in multiple waves")
                all_roles.append(role)
        return self

    @model_validator(mode="after")
    def validate_wave_numbers_unique(self) -> "HarnessConfigModel":
        """All wave numbers must be unique."""
        wave_numbers: list[int] = []
        for wave in self.waves:
            if wave.wave in wave_numbers:
                raise ValueError(f"Wave number {wave.wave} appears multiple times")
            wave_numbers.append(wave.wave)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HarnessConfigModel":
        """Load and validate config from YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated HarnessConfigModel instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            pydantic.ValidationError: If the config is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Convert wave format from dict-of-dicts to list-of-WaveDefinition
        if "waves" in data and isinstance(data["waves"], dict):
            wave_list = []
            for wave_num, wave_info in data["waves"].items():
                wave_def = {
                    "wave": int(wave_num),
                    "roles": wave_info.get("roles", []),
                    "parallel": wave_info.get("parallel", True),
                    "name": wave_info.get("name", ""),
                }
                wave_list.append(wave_def)
            data["waves"] = wave_list

        # Ensure gitlab section exists with project_path
        if "gitlab" not in data:
            data["gitlab"] = {"project_path": "default/project"}
        elif "project_path" not in data["gitlab"]:
            data["gitlab"]["project_path"] = "default/project"

        return cls(**data)

    @classmethod
    def from_env(cls) -> "HarnessConfigModel":
        """Load config with environment variable interpolation.

        Loads harness.yml (searching standard paths) and replaces
        ${VAR} patterns with environment variable values.

        Returns:
            Validated HarnessConfigModel with env vars interpolated.
        """
        # Find config file
        paths_to_try = [
            "harness.yml",
            "harness.yaml",
            ".harness/config.yml",
        ]

        config_path_env = os.environ.get("HARNESS_CONFIG")
        if config_path_env:
            paths_to_try.insert(0, config_path_env)

        # Search up from CWD
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents)[:5]:
            paths_to_try.append(str(parent / "harness.yml"))

        found_path = None
        for p in paths_to_try:
            if p and Path(p).exists():
                found_path = Path(p)
                break

        if found_path is None:
            raise FileNotFoundError(
                "No harness.yml found. Searched: " + ", ".join(filter(None, paths_to_try))
            )

        with open(found_path) as f:
            data = yaml.safe_load(f) or {}

        # Interpolate environment variables
        data = _interpolate_dict(data)

        # Convert wave format from dict-of-dicts to list-of-WaveDefinition
        if "waves" in data and isinstance(data["waves"], dict):
            wave_list = []
            for wave_num, wave_info in data["waves"].items():
                wave_def = {
                    "wave": int(wave_num),
                    "roles": wave_info.get("roles", []),
                    "parallel": wave_info.get("parallel", True),
                    "name": wave_info.get("name", ""),
                }
                wave_list.append(wave_def)
            data["waves"] = wave_list

        # Ensure gitlab section
        if "gitlab" not in data:
            data["gitlab"] = {"project_path": "default/project"}
        elif "project_path" not in data["gitlab"]:
            data["gitlab"]["project_path"] = "default/project"

        return cls(**data)
