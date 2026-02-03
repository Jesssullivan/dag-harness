"""Structural tests for release automation and CI/CD configuration.

These tests verify that version numbers, changelog entries, and CI config
files are consistent and present, preventing broken releases.
"""

import re
from pathlib import Path

import pytest

# Root of the harness package (harness/)
HARNESS_DIR = Path(__file__).parent.parent
# Root of the repo (dag-harness/)
REPO_ROOT = HARNESS_DIR.parent


def _get_pyproject_version() -> str:
    """Extract version from pyproject.toml."""
    pyproject = HARNESS_DIR / "pyproject.toml"
    content = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match, "Could not find version in pyproject.toml"
    return match.group(1)


def _get_init_version() -> str:
    """Extract __version__ from harness/__init__.py."""
    init_file = HARNESS_DIR / "harness" / "__init__.py"
    content = init_file.read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match, "Could not find __version__ in harness/__init__.py"
    return match.group(1)


@pytest.mark.unit
class TestVersionConsistency:
    """Verify version numbers are consistent across the project."""

    def test_version_in_pyproject_matches_init(self):
        """Version in pyproject.toml must match __version__ in __init__.py."""
        pyproject_version = _get_pyproject_version()
        init_version = _get_init_version()
        assert pyproject_version == init_version, (
            f"Version mismatch: pyproject.toml has '{pyproject_version}' "
            f"but harness/__init__.py has '{init_version}'"
        )

    def test_version_is_valid_semver(self):
        """Version must be valid semantic versioning."""
        version = _get_pyproject_version()
        # Match major.minor.patch with optional pre-release
        assert re.match(r"^\d+\.\d+\.\d+(-[\w.]+)?$", version), (
            f"Version '{version}' is not valid semver"
        )


@pytest.mark.unit
class TestChangelog:
    """Verify CHANGELOG.md is present and up to date."""

    def test_changelog_exists(self):
        """CHANGELOG.md must exist at the repo root."""
        changelog = REPO_ROOT / "CHANGELOG.md"
        assert changelog.exists(), (
            f"CHANGELOG.md not found at {changelog}. "
            "Create one following Keep a Changelog format."
        )

    def test_changelog_has_current_version(self):
        """CHANGELOG.md must contain an entry for the current version."""
        changelog = REPO_ROOT / "CHANGELOG.md"
        if not changelog.exists():
            pytest.skip("CHANGELOG.md does not exist")

        version = _get_pyproject_version()
        content = changelog.read_text()
        assert f"[{version}]" in content, (
            f"CHANGELOG.md does not contain an entry for version {version}. "
            f"Add a '## [{version}]' section before releasing."
        )

    def test_changelog_format(self):
        """CHANGELOG.md should follow Keep a Changelog format."""
        changelog = REPO_ROOT / "CHANGELOG.md"
        if not changelog.exists():
            pytest.skip("CHANGELOG.md does not exist")

        content = changelog.read_text()
        # Should have at least one version header
        assert re.search(r"^## \[\d+\.\d+\.\d+\]", content, re.MULTILINE), (
            "CHANGELOG.md should have version headers in format '## [x.y.z]'"
        )


@pytest.mark.unit
class TestCIConfig:
    """Verify CI/CD configuration files are present and valid."""

    def test_ci_config_exists(self):
        """At least one CI config file must exist."""
        github_ci = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        gitlab_ci = REPO_ROOT / ".gitlab-ci.yml"
        assert github_ci.exists() or gitlab_ci.exists(), (
            "No CI configuration found. Expected .github/workflows/ci.yml "
            "or .gitlab-ci.yml"
        )

    def test_github_ci_exists(self):
        """GitHub Actions CI workflow should exist."""
        ci_yml = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_yml.exists(), (
            f"GitHub Actions CI workflow not found at {ci_yml}"
        )

    def test_github_release_exists(self):
        """GitHub Actions release workflow should exist."""
        release_yml = REPO_ROOT / ".github" / "workflows" / "release.yml"
        assert release_yml.exists(), (
            f"GitHub Actions release workflow not found at {release_yml}"
        )

    def test_gitlab_ci_exists(self):
        """GitLab CI config should exist."""
        gitlab_ci = REPO_ROOT / ".gitlab-ci.yml"
        assert gitlab_ci.exists(), (
            f"GitLab CI config not found at {gitlab_ci}"
        )

    def test_ci_runs_tests(self):
        """CI config must reference pytest for running tests."""
        github_ci = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        gitlab_ci = REPO_ROOT / ".gitlab-ci.yml"

        found_pytest = False
        for ci_file in [github_ci, gitlab_ci]:
            if ci_file.exists():
                content = ci_file.read_text()
                if "pytest" in content:
                    found_pytest = True
                    break

        assert found_pytest, (
            "No CI config file references pytest. "
            "CI must run tests as part of the pipeline."
        )

    def test_ci_runs_linter(self):
        """CI config should reference ruff for linting."""
        github_ci = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        gitlab_ci = REPO_ROOT / ".gitlab-ci.yml"

        found_ruff = False
        for ci_file in [github_ci, gitlab_ci]:
            if ci_file.exists():
                content = ci_file.read_text()
                if "ruff" in content:
                    found_ruff = True
                    break

        assert found_ruff, (
            "No CI config file references ruff linter. "
            "CI should run linting as part of the pipeline."
        )


@pytest.mark.unit
class TestReleaseTooling:
    """Verify release tooling configuration."""

    def test_pyproject_has_commitizen(self):
        """pyproject.toml should have commitizen configuration."""
        pyproject = HARNESS_DIR / "pyproject.toml"
        content = pyproject.read_text()
        assert "[tool.commitizen]" in content, (
            "pyproject.toml missing [tool.commitizen] section. "
            "Add commitizen config for conventional commits."
        )

    def test_pyproject_has_build_system(self):
        """pyproject.toml must have a build-system section."""
        pyproject = HARNESS_DIR / "pyproject.toml"
        content = pyproject.read_text()
        assert "[build-system]" in content, (
            "pyproject.toml missing [build-system] section"
        )

    def test_pyproject_has_version(self):
        """pyproject.toml must have a version field."""
        version = _get_pyproject_version()
        assert version, "pyproject.toml must have a version"

    def test_publish_workflow_uses_trusted_publishing(self):
        """Release workflow should use PyPI trusted publishing."""
        publish_yml = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
        release_yml = REPO_ROOT / ".github" / "workflows" / "release.yml"

        found_trusted = False
        for wf in [publish_yml, release_yml]:
            if wf.exists():
                content = wf.read_text()
                if "id-token: write" in content:
                    found_trusted = True
                    break

        assert found_trusted, (
            "No release workflow uses PyPI trusted publishing (id-token: write). "
            "Trusted publishing is the recommended approach for PyPI."
        )
