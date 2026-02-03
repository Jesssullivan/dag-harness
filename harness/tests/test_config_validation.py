"""Tests for Pydantic configuration validation models."""

import json
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from harness.config import (
    CheckpointerConfig,
    GitLabConfigModel,
    HOTLConfig,
    HarnessConfigModel,
    WaveDefinition,
    _interpolate_env_vars,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_config_data() -> dict:
    """Minimal valid configuration data."""
    return {
        "version": "1.0",
        "project_name": "test-project",
        "gitlab": {
            "project_path": "tinyland/projects/dag-harness",
            "default_branch": "main",
            "merge_method": "merge",
        },
        "waves": [
            {"wave": 0, "roles": ["common"], "parallel": False, "name": "Foundation"},
            {
                "wave": 1,
                "roles": ["iis-config", "windows_prerequisites"],
                "parallel": True,
                "name": "Infrastructure",
            },
        ],
        "db_path": ".harness/harness.db",
        "log_level": "INFO",
    }


@pytest.fixture
def valid_yaml_file(tmp_path: Path, valid_config_data: dict) -> Path:
    """Write valid config to a temporary YAML file."""
    config_path = tmp_path / "harness.yml"
    with open(config_path, "w") as f:
        yaml.dump(valid_config_data, f)
    return config_path


@pytest.fixture
def valid_yaml_file_dict_waves(tmp_path: Path) -> Path:
    """Write config with dict-style waves (as used in existing harness.yml)."""
    data = {
        "project_name": "test-project",
        "gitlab": {"project_path": "bates-ils/projects/ems/ems-mono"},
        "waves": {
            0: {"name": "Foundation", "roles": ["common"]},
            1: {"name": "Infrastructure", "roles": ["iis-config", "windows_prerequisites"]},
        },
        "db_path": "harness.db",
    }
    config_path = tmp_path / "harness.yml"
    with open(config_path, "w") as f:
        yaml.dump(data, f)
    return config_path


# ============================================================================
# TESTS: Valid configuration
# ============================================================================


class TestValidConfig:
    """Tests for valid configuration loading."""

    def test_valid_config_loads(self, valid_config_data: dict):
        """Valid YAML config data loads without errors."""
        config = HarnessConfigModel(**valid_config_data)
        assert config.project_name == "test-project"
        assert config.gitlab.project_path == "tinyland/projects/dag-harness"
        assert len(config.waves) == 2
        assert config.waves[0].roles == ["common"]

    def test_from_yaml_file(self, valid_yaml_file: Path):
        """Load from actual YAML file."""
        config = HarnessConfigModel.from_yaml(valid_yaml_file)
        assert config.project_name == "test-project"
        assert config.gitlab.project_path == "tinyland/projects/dag-harness"
        assert len(config.waves) == 2

    def test_from_yaml_dict_waves(self, valid_yaml_file_dict_waves: Path):
        """Load from YAML file with dict-style waves (backward compat)."""
        config = HarnessConfigModel.from_yaml(valid_yaml_file_dict_waves)
        assert len(config.waves) == 2
        wave_numbers = {w.wave for w in config.waves}
        assert wave_numbers == {0, 1}

    def test_default_values(self):
        """Unspecified fields get defaults."""
        config = HarnessConfigModel(
            gitlab=GitLabConfigModel(project_path="ns/project"),
        )
        assert config.version == "1.0"
        assert config.project_name == "dag-harness"
        assert config.db_path == ".harness/harness.db"
        assert config.log_level == "INFO"
        assert config.waves == []
        assert config.breakpoints == []
        assert config.checkpointer.backend == "sqlite"
        assert config.checkpointer.sync_to_statedb is True
        assert config.hotl.enabled is False
        assert config.hotl.max_iterations == 50
        assert config.hotl.notification_interval == 300
        assert config.hotl.approval_timeout == 3600

    def test_gitlab_config_valid(self):
        """Valid GitLab config with various path formats."""
        cfg = GitLabConfigModel(project_path="namespace/project")
        assert cfg.project_path == "namespace/project"

        # Multi-level path
        cfg2 = GitLabConfigModel(project_path="bates-ils/projects/ems/ems-mono")
        assert cfg2.project_path == "bates-ils/projects/ems/ems-mono"

    def test_wave_definition_valid(self):
        """Valid wave definitions."""
        wave = WaveDefinition(wave=0, roles=["common"])
        assert wave.wave == 0
        assert wave.roles == ["common"]
        assert wave.parallel is True  # default

    def test_checkpointer_defaults(self):
        """Checkpointer config defaults."""
        cp = CheckpointerConfig()
        assert cp.backend == "sqlite"
        assert cp.sqlite_path == ".harness/checkpoints.db"
        assert cp.postgres_url is None
        assert cp.sync_to_statedb is True

    def test_hotl_defaults(self):
        """HOTL config defaults."""
        hotl = HOTLConfig()
        assert hotl.enabled is False
        assert hotl.max_iterations == 50
        assert hotl.notification_interval == 300
        assert hotl.approval_timeout == 3600


# ============================================================================
# TESTS: Validation errors
# ============================================================================


class TestValidationErrors:
    """Tests for configuration validation errors."""

    def test_invalid_project_path(self):
        """Bad project path raises ValidationError."""
        with pytest.raises(ValidationError, match="project_path must be in"):
            GitLabConfigModel(project_path="no-slash-here")

    def test_invalid_project_path_empty(self):
        """Empty project path raises ValidationError."""
        with pytest.raises(ValidationError):
            GitLabConfigModel(project_path="")

    def test_empty_wave_roles(self):
        """Empty roles list raises error."""
        with pytest.raises(ValidationError, match="too_short"):
            WaveDefinition(wave=0, roles=[])

    def test_duplicate_wave_roles(self):
        """Same role in two waves raises error."""
        with pytest.raises(ValidationError, match="appears in multiple waves"):
            HarnessConfigModel(
                gitlab=GitLabConfigModel(project_path="ns/project"),
                waves=[
                    WaveDefinition(wave=0, roles=["common"]),
                    WaveDefinition(wave=1, roles=["common", "other"]),
                ],
            )

    def test_duplicate_wave_numbers(self):
        """Duplicate wave numbers raise error."""
        with pytest.raises(ValidationError, match="appears multiple times"):
            HarnessConfigModel(
                gitlab=GitLabConfigModel(project_path="ns/project"),
                waves=[
                    WaveDefinition(wave=0, roles=["common"]),
                    WaveDefinition(wave=0, roles=["other"]),
                ],
            )

    def test_wave_number_out_of_range(self):
        """Wave number outside 0-9 raises error."""
        with pytest.raises(ValidationError):
            WaveDefinition(wave=10, roles=["common"])

        with pytest.raises(ValidationError):
            WaveDefinition(wave=-1, roles=["common"])

    def test_invalid_merge_method(self):
        """Invalid merge method raises error."""
        with pytest.raises(ValidationError, match="merge_method must be one of"):
            GitLabConfigModel(project_path="ns/project", merge_method="squash")

    def test_invalid_log_level(self):
        """Invalid log level raises error."""
        with pytest.raises(ValidationError, match="log_level must be one of"):
            HarnessConfigModel(
                gitlab=GitLabConfigModel(project_path="ns/project"),
                log_level="VERBOSE",
            )

    def test_invalid_checkpointer_backend(self):
        """Invalid checkpointer backend raises error."""
        with pytest.raises(ValidationError, match="backend must be one of"):
            CheckpointerConfig(backend="redis")

    def test_postgres_url_validation(self):
        """Invalid postgres URL raises error."""
        with pytest.raises(ValidationError, match="postgres_url must start with"):
            CheckpointerConfig(postgres_url="mysql://localhost/db")

    def test_postgres_url_valid(self):
        """Valid postgres URLs are accepted."""
        cp = CheckpointerConfig(postgres_url="postgresql://user:pass@localhost:5432/db")
        assert cp.postgres_url == "postgresql://user:pass@localhost:5432/db"

        cp2 = CheckpointerConfig(postgres_url="postgres://localhost/db")
        assert cp2.postgres_url == "postgres://localhost/db"

    def test_hotl_max_iterations_min(self):
        """HOTL max_iterations must be >= 1."""
        with pytest.raises(ValidationError):
            HOTLConfig(max_iterations=0)

    def test_hotl_notification_interval_min(self):
        """HOTL notification_interval must be >= 60."""
        with pytest.raises(ValidationError):
            HOTLConfig(notification_interval=30)

    def test_hotl_approval_timeout_min(self):
        """HOTL approval_timeout must be >= 300."""
        with pytest.raises(ValidationError):
            HOTLConfig(approval_timeout=100)

    def test_from_yaml_missing_file(self):
        """Loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            HarnessConfigModel.from_yaml("/nonexistent/harness.yml")


# ============================================================================
# TESTS: Environment variable interpolation
# ============================================================================


class TestEnvInterpolation:
    """Tests for environment variable interpolation."""

    def test_env_interpolation_home(self, monkeypatch):
        """${HOME} replaced with env value."""
        monkeypatch.setenv("HOME", "/Users/testuser")
        result = _interpolate_env_vars("${HOME}/projects")
        assert result == "/Users/testuser/projects"

    def test_env_interpolation_multiple(self, monkeypatch):
        """Multiple ${VAR} patterns replaced."""
        monkeypatch.setenv("USER", "alice")
        monkeypatch.setenv("HOST", "example.com")
        result = _interpolate_env_vars("${USER}@${HOST}")
        assert result == "alice@example.com"

    def test_env_interpolation_missing_var(self, monkeypatch):
        """Missing env var raises ValueError."""
        monkeypatch.delenv("SURELY_MISSING_VAR_XYZ", raising=False)
        with pytest.raises(ValueError, match="Environment variable.*is not set"):
            _interpolate_env_vars("${SURELY_MISSING_VAR_XYZ}")

    def test_env_interpolation_no_vars(self):
        """Strings without ${} are returned unchanged."""
        result = _interpolate_env_vars("plain string")
        assert result == "plain string"

    def test_env_interpolation_in_yaml(self, tmp_path, monkeypatch):
        """Full YAML loading with env var interpolation."""
        monkeypatch.setenv("TEST_DB_PATH", "/tmp/test.db")
        monkeypatch.setenv("TEST_PROJECT", "myns/myproject")

        data = {
            "project_name": "test",
            "gitlab": {"project_path": "${TEST_PROJECT}"},
            "db_path": "${TEST_DB_PATH}",
        }
        config_path = tmp_path / "harness.yml"
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        # Use from_env approach: set HARNESS_CONFIG and chdir
        monkeypatch.setenv("HARNESS_CONFIG", str(config_path))
        config = HarnessConfigModel.from_env()
        assert config.db_path == "/tmp/test.db"
        assert config.gitlab.project_path == "myns/myproject"


# ============================================================================
# TESTS: JSON Schema generation
# ============================================================================


class TestJsonSchema:
    """Tests for JSON Schema generation."""

    def test_json_schema_generation(self):
        """model_json_schema() produces valid schema."""
        schema = HarnessConfigModel.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "title" in schema
        assert schema["title"] == "HarnessConfigModel"

        # Check key properties are present
        props = schema["properties"]
        assert "version" in props
        assert "project_name" in props
        assert "gitlab" in props
        assert "waves" in props
        assert "db_path" in props
        assert "log_level" in props
        assert "checkpointer" in props
        assert "hotl" in props

    def test_json_schema_is_valid_json(self):
        """Schema can be serialized to and from JSON."""
        schema = HarnessConfigModel.model_json_schema()
        json_str = json.dumps(schema, indent=2)
        parsed = json.loads(json_str)
        assert parsed == schema

    def test_json_schema_has_definitions(self):
        """Schema includes definitions for nested models."""
        schema = HarnessConfigModel.model_json_schema()
        # Pydantic v2 uses $defs
        defs_key = "$defs" if "$defs" in schema else "definitions"
        assert defs_key in schema
        defs = schema[defs_key]
        assert "GitLabConfigModel" in defs
        assert "WaveDefinition" in defs
        assert "CheckpointerConfig" in defs
        assert "HOTLConfig" in defs


# ============================================================================
# TESTS: Wave ordering
# ============================================================================


class TestWaveOrdering:
    """Tests for wave ordering validation."""

    def test_wave_ordering_valid(self):
        """Waves with unique numbers are valid."""
        config = HarnessConfigModel(
            gitlab=GitLabConfigModel(project_path="ns/project"),
            waves=[
                WaveDefinition(wave=0, roles=["common"]),
                WaveDefinition(wave=2, roles=["web_app"]),
                WaveDefinition(wave=5, roles=["monitoring"]),
            ],
        )
        assert len(config.waves) == 3

    def test_wave_ordering_duplicate_numbers(self):
        """Duplicate wave numbers raise error."""
        with pytest.raises(ValidationError, match="appears multiple times"):
            HarnessConfigModel(
                gitlab=GitLabConfigModel(project_path="ns/project"),
                waves=[
                    WaveDefinition(wave=1, roles=["role_a"]),
                    WaveDefinition(wave=1, roles=["role_b"]),
                ],
            )

    def test_wave_max_boundary(self):
        """Wave number at max boundary (9) is valid."""
        wave = WaveDefinition(wave=9, roles=["final_role"])
        assert wave.wave == 9

    def test_wave_min_boundary(self):
        """Wave number at min boundary (0) is valid."""
        wave = WaveDefinition(wave=0, roles=["foundation"])
        assert wave.wave == 0


# ============================================================================
# TESTS: Log level normalization
# ============================================================================


class TestLogLevel:
    """Tests for log level validation and normalization."""

    def test_log_level_case_insensitive(self):
        """Log level is normalized to uppercase."""
        config = HarnessConfigModel(
            gitlab=GitLabConfigModel(project_path="ns/project"),
            log_level="debug",
        )
        assert config.log_level == "DEBUG"

    def test_log_level_all_valid(self):
        """All standard log levels are accepted."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = HarnessConfigModel(
                gitlab=GitLabConfigModel(project_path="ns/project"),
                log_level=level,
            )
            assert config.log_level == level
