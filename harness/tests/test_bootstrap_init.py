"""Tests for harness init bootstrap command."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from harness.bootstrap.init import (
    GITIGNORE_ENTRIES,
    _detect_git_root,
    _scan_roles,
    _update_gitignore,
    init_harness,
)
from harness.cli import app
from harness.db.state import StateDB

runner = CliRunner()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository."""
    subprocess.run(
        ["git", "init", str(tmp_path)],
        capture_output=True,
        check=True,
    )
    return tmp_path


@pytest.fixture
def git_repo_with_roles(git_repo: Path) -> Path:
    """Create a git repo with ansible/roles/ populated."""
    roles_dir = git_repo / "ansible" / "roles"

    # Create several role directories with typical structure
    for role_name in ["common", "iis-config", "ems_web_app"]:
        role_path = roles_dir / role_name
        (role_path / "tasks").mkdir(parents=True)
        (role_path / "tasks" / "main.yml").write_text("---\n- name: test\n  debug:\n    msg: hi\n")

    # Give common a meta/main.yml with galaxy_info
    meta_dir = roles_dir / "common" / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "main.yml").write_text(
        "---\ngalaxy_info:\n  description: Common configuration\ndependencies: []\n"
    )

    # Give ems_web_app a molecule directory
    (roles_dir / "ems_web_app" / "molecule").mkdir(parents=True)

    return git_repo


class TestInitCreatesHarnessDir:
    """Test that init creates the .harness/ directory."""

    @pytest.mark.unit
    def test_init_creates_harness_dir(self, git_repo: Path):
        harness_dir = git_repo / ".harness"
        assert not harness_dir.exists()

        result = init_harness(repo_root=git_repo)

        assert harness_dir.exists()
        assert harness_dir.is_dir()
        assert result["harness_dir"] == str(harness_dir)


class TestInitCreatesDatabase:
    """Test that init creates and initializes the StateDB."""

    @pytest.mark.unit
    def test_init_creates_database(self, git_repo: Path):
        result = init_harness(repo_root=git_repo)

        db_path = Path(result["db_path"])
        assert db_path.exists()
        assert db_path.name == "harness.db"

        # Verify schema was applied by opening the database
        db = StateDB(db_path)
        schema_info = db.validate_schema()
        assert schema_info["valid"]


class TestInitGeneratesConfig:
    """Test that init generates a harness.yml configuration file."""

    @pytest.mark.unit
    def test_init_generates_config(self, git_repo: Path):
        result = init_harness(repo_root=git_repo)

        config_path = Path(result["config_path"])
        assert config_path.exists()
        assert result["config_created"] is True

        # Verify it is valid YAML
        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "db_path" in data
        assert "repo_root" in data

    @pytest.mark.unit
    def test_init_custom_config_path(self, git_repo: Path):
        custom = str(git_repo / "custom-config.yml")
        result = init_harness(repo_root=git_repo, config_path=custom)

        assert Path(custom).exists()
        assert result["config_path"] == custom


class TestInitDetectsRoles:
    """Test that init detects roles in ansible/roles/."""

    @pytest.mark.unit
    def test_init_detects_roles(self, git_repo_with_roles: Path):
        result = init_harness(repo_root=git_repo_with_roles)

        detected = result["roles_detected"]
        assert len(detected) == 3
        assert "common" in detected
        assert "iis-config" in detected
        assert "ems_web_app" in detected

    @pytest.mark.unit
    def test_init_roles_populated_in_db(self, git_repo_with_roles: Path):
        result = init_harness(repo_root=git_repo_with_roles)

        db = StateDB(result["db_path"])
        roles = db.list_roles()
        role_names = [r.name for r in roles]

        assert "common" in role_names
        assert "ems_web_app" in role_names

    @pytest.mark.unit
    def test_init_detects_molecule(self, git_repo_with_roles: Path):
        result = init_harness(repo_root=git_repo_with_roles)

        db = StateDB(result["db_path"])
        ems_web = db.get_role("ems_web_app")
        common = db.get_role("common")

        assert ems_web.has_molecule_tests is True
        assert common.has_molecule_tests is False

    @pytest.mark.unit
    def test_init_reads_role_description(self, git_repo_with_roles: Path):
        result = init_harness(repo_root=git_repo_with_roles)

        db = StateDB(result["db_path"])
        common = db.get_role("common")
        assert common.description == "Common configuration"

    @pytest.mark.unit
    def test_init_no_detect_roles(self, git_repo_with_roles: Path):
        result = init_harness(repo_root=git_repo_with_roles, no_detect_roles=True)
        assert result["roles_detected"] == []

    @pytest.mark.unit
    def test_init_no_roles_dir(self, git_repo: Path):
        """No ansible/roles/ directory should not cause an error."""
        result = init_harness(repo_root=git_repo)
        assert result["roles_detected"] == []


class TestInitUpdatesGitignore:
    """Test that init updates .gitignore with .harness/ entries."""

    @pytest.mark.unit
    def test_init_updates_gitignore(self, git_repo: Path):
        result = init_harness(repo_root=git_repo)
        assert result["gitignore_updated"] is True

        gitignore = (git_repo / ".gitignore").read_text()
        assert ".harness/*.db" in gitignore
        assert ".harness/*.log" in gitignore

    @pytest.mark.unit
    def test_init_appends_to_existing_gitignore(self, git_repo: Path):
        existing = "*.pyc\n__pycache__/\n"
        (git_repo / ".gitignore").write_text(existing)

        init_harness(repo_root=git_repo)

        gitignore = (git_repo / ".gitignore").read_text()
        assert "*.pyc" in gitignore
        assert ".harness/*.db" in gitignore

    @pytest.mark.unit
    def test_init_does_not_duplicate_gitignore(self, git_repo: Path):
        """If .harness entries already exist, don't add them again."""
        (git_repo / ".gitignore").write_text(".harness/*.db\n")

        result = init_harness(repo_root=git_repo)
        assert result["gitignore_updated"] is False

        content = (git_repo / ".gitignore").read_text()
        assert content.count(".harness/*.db") == 1


class TestInitForceOverwrites:
    """Test that --force overwrites existing configuration."""

    @pytest.mark.unit
    def test_init_force_overwrites(self, git_repo: Path):
        # First init
        init_harness(repo_root=git_repo)
        config_path = git_repo / "harness.yml"
        original_content = config_path.read_text()

        # Modify the config
        config_path.write_text("db_path: custom.db\n")
        assert config_path.read_text() != original_content

        # Re-init with force
        result = init_harness(repo_root=git_repo, force=True)
        assert result["config_created"] is True

        # Config should be regenerated
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "repo_root" in data

    @pytest.mark.unit
    def test_init_without_force_preserves_config(self, git_repo: Path):
        init_harness(repo_root=git_repo)

        config_path = git_repo / "harness.yml"
        config_path.write_text("db_path: custom.db\n")

        result = init_harness(repo_root=git_repo, force=False)
        assert result["config_created"] is False

        # Original custom content should be preserved
        content = config_path.read_text()
        assert "custom.db" in content


class TestInitIdempotent:
    """Test that running init twice doesn't corrupt state."""

    @pytest.mark.unit
    def test_init_idempotent(self, git_repo_with_roles: Path):
        result1 = init_harness(repo_root=git_repo_with_roles)
        result2 = init_harness(repo_root=git_repo_with_roles)

        # Both should succeed
        assert result1["repo_root"] == result2["repo_root"]
        assert result1["db_path"] == result2["db_path"]

        # Database should still be valid
        db = StateDB(result2["db_path"])
        schema = db.validate_schema()
        assert schema["valid"]

        # Roles should not be duplicated (upsert semantics)
        roles = db.list_roles()
        role_names = [r.name for r in roles]
        assert len(role_names) == len(set(role_names))

    @pytest.mark.unit
    def test_init_idempotent_gitignore(self, git_repo: Path):
        init_harness(repo_root=git_repo)
        init_harness(repo_root=git_repo)

        content = (git_repo / ".gitignore").read_text()
        # Count the sentinel line exactly (not substrings like *.db-journal)
        assert content.count(".harness/*.db\n") == 1


class TestInitNoGitRepo:
    """Test graceful error when not inside a git repo."""

    @pytest.mark.unit
    def test_init_no_git_repo(self, tmp_path: Path):
        """Should raise RuntimeError outside a git repository."""
        non_git = tmp_path / "not-a-repo"
        non_git.mkdir()

        with pytest.raises(RuntimeError, match="Not a git repository"):
            init_harness(repo_root=non_git)

    @pytest.mark.unit
    def test_init_no_git_repo_autodetect(self, tmp_path: Path, monkeypatch):
        """Should raise RuntimeError when git rev-parse fails."""
        monkeypatch.chdir(tmp_path)

        # Mock git rev-parse to fail
        with patch("harness.bootstrap.init.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git")
            with pytest.raises(RuntimeError, match="Not inside a git repository"):
                init_harness()


class TestInitCLI:
    """Test the CLI integration of the init command."""

    @pytest.mark.unit
    def test_cli_init_basic(self, git_repo_with_roles: Path, monkeypatch):
        monkeypatch.chdir(git_repo_with_roles)
        result = runner.invoke(app, ["init", "--repo-root", str(git_repo_with_roles)])
        assert result.exit_code == 0
        assert "Repository root" in result.stdout

    @pytest.mark.unit
    def test_cli_init_force(self, git_repo: Path, monkeypatch):
        monkeypatch.chdir(git_repo)
        # First init
        runner.invoke(app, ["init", "--repo-root", str(git_repo)])
        # Second with --force
        result = runner.invoke(app, ["init", "--repo-root", str(git_repo), "--force"])
        assert result.exit_code == 0

    @pytest.mark.unit
    def test_cli_init_not_git_repo(self, tmp_path: Path, monkeypatch):
        non_git = tmp_path / "nope"
        non_git.mkdir()
        monkeypatch.chdir(non_git)
        result = runner.invoke(app, ["init", "--repo-root", str(non_git)])
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestHelperFunctions:
    """Test internal helper functions."""

    @pytest.mark.unit
    def test_scan_roles_empty_dir(self, tmp_path: Path):
        roles_dir = tmp_path / "roles"
        roles_dir.mkdir()
        assert _scan_roles(roles_dir) == []

    @pytest.mark.unit
    def test_scan_roles_skips_hidden(self, tmp_path: Path):
        roles_dir = tmp_path / "roles"
        roles_dir.mkdir()
        (roles_dir / ".hidden").mkdir()
        (roles_dir / "_private").mkdir()
        (roles_dir / "valid_role").mkdir()

        result = _scan_roles(roles_dir)
        assert result == ["valid_role"]

    @pytest.mark.unit
    def test_scan_roles_nonexistent(self, tmp_path: Path):
        assert _scan_roles(tmp_path / "nope") == []

    @pytest.mark.unit
    def test_update_gitignore_creates_file(self, tmp_path: Path):
        assert not (tmp_path / ".gitignore").exists()
        updated = _update_gitignore(tmp_path)
        assert updated is True
        assert (tmp_path / ".gitignore").exists()
        assert ".harness/*.db" in (tmp_path / ".gitignore").read_text()

    @pytest.mark.unit
    def test_detect_git_root_explicit(self, git_repo: Path):
        root = _detect_git_root(git_repo)
        assert root == git_repo.resolve()

    @pytest.mark.unit
    def test_detect_git_root_invalid(self, tmp_path: Path):
        bad_path = tmp_path / "empty"
        bad_path.mkdir()
        with pytest.raises(RuntimeError):
            _detect_git_root(bad_path)
