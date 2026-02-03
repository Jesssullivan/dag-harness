"""Tests for npm integration script generation."""

import json
import os
import stat
from pathlib import Path

import pytest


@pytest.mark.unit
class TestGetNpmScripts:
    """Tests for get_npm_scripts()."""

    def test_get_npm_scripts_returns_dict(self) -> None:
        """Returns a non-empty dict of script entries."""
        from scripts.npm_integration import get_npm_scripts

        scripts = get_npm_scripts()
        assert isinstance(scripts, dict)
        assert len(scripts) > 0

    def test_all_scripts_start_with_harness(self) -> None:
        """All keys start with 'harness:' prefix."""
        from scripts.npm_integration import get_npm_scripts

        scripts = get_npm_scripts()
        for key in scripts:
            assert key.startswith("harness:"), f"Script key '{key}' does not start with 'harness:'"

    def test_scripts_use_uv_run(self) -> None:
        """All commands use 'uv run' for virtual environment management."""
        from scripts.npm_integration import get_npm_scripts

        scripts = get_npm_scripts()
        for key, cmd in scripts.items():
            assert "uv run" in cmd, f"Script '{key}' does not use 'uv run': {cmd}"

    def test_scripts_change_to_harness_dir(self) -> None:
        """All commands cd into the harness directory first."""
        from scripts.npm_integration import get_npm_scripts

        scripts = get_npm_scripts()
        for key, cmd in scripts.items():
            assert cmd.startswith("cd harness && "), (
                f"Script '{key}' does not cd to harness dir: {cmd}"
            )

    def test_scripts_contain_expected_entries(self) -> None:
        """Essential harness commands are present."""
        from scripts.npm_integration import get_npm_scripts

        scripts = get_npm_scripts()
        expected_keys = [
            "harness:init",
            "harness:status",
            "harness:box-up",
            "harness:list-roles",
            "harness:graph",
            "harness:costs",
            "harness:hotl:start",
            "harness:hotl:stop",
            "harness:test",
        ]
        for key in expected_keys:
            assert key in scripts, f"Expected script '{key}' not found"


@pytest.mark.unit
class TestGenerateSnippet:
    """Tests for generate_package_json_snippet()."""

    def test_generate_snippet_valid_json(self) -> None:
        """Output is valid JSON that can be parsed."""
        from scripts.npm_integration import generate_package_json_snippet

        snippet = generate_package_json_snippet()
        parsed = json.loads(snippet)
        assert isinstance(parsed, dict)

    def test_generate_snippet_contains_scripts(self) -> None:
        """Generated snippet contains the harness script entries."""
        from scripts.npm_integration import generate_package_json_snippet

        snippet = generate_package_json_snippet()
        parsed = json.loads(snippet)
        # Filter out comment keys (start with //)
        script_keys = [k for k in parsed if not k.startswith("//")]
        assert len(script_keys) > 0
        for key in script_keys:
            assert key.startswith("harness:")

    def test_generate_snippet_contains_section_header(self) -> None:
        """Generated snippet includes the section header comments."""
        from scripts.npm_integration import generate_package_json_snippet

        snippet = generate_package_json_snippet()
        parsed = json.loads(snippet)
        comment_keys = [k for k in parsed if k.startswith("//")]
        assert len(comment_keys) > 0


@pytest.mark.unit
class TestSetupScript:
    """Tests for setup.sh existence and permissions."""

    def test_setup_script_exists(self) -> None:
        """setup.sh exists in the scripts directory."""
        setup_path = Path(__file__).parent.parent / "scripts" / "setup.sh"
        assert setup_path.exists(), f"setup.sh not found at {setup_path}"

    def test_setup_script_is_executable(self) -> None:
        """setup.sh has executable permissions."""
        setup_path = Path(__file__).parent.parent / "scripts" / "setup.sh"
        file_stat = os.stat(setup_path)
        assert file_stat.st_mode & stat.S_IXUSR, "setup.sh is not executable by owner"

    def test_setup_script_has_bash_shebang(self) -> None:
        """setup.sh starts with a bash shebang line."""
        setup_path = Path(__file__).parent.parent / "scripts" / "setup.sh"
        content = setup_path.read_text()
        assert content.startswith("#!/usr/bin/env bash"), (
            "setup.sh missing bash shebang"
        )

    def test_setup_script_uses_strict_mode(self) -> None:
        """setup.sh uses set -euo pipefail for safety."""
        setup_path = Path(__file__).parent.parent / "scripts" / "setup.sh"
        content = setup_path.read_text()
        assert "set -euo pipefail" in content, (
            "setup.sh missing strict mode (set -euo pipefail)"
        )
