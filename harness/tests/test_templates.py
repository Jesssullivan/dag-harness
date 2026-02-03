"""Tests for GitLab MR template rendering."""

import pytest


class TestRenderMrDescription:
    """Tests for render_mr_description function."""

    def test_new_role_template(self):
        """New role shows NEW ROLE header and full implementation details."""
        from harness.gitlab.templates import render_mr_description

        ctx = {
            "role_name": "test_role",
            "issue_iid": "123",
            "wave": 1,
            "wave_name": "Infrastructure Foundation",
            "is_new_role": True,
            "role_diff_stat": None,
            "molecule_passed": True,
            "molecule_duration": 60,
            "deploy_target": "vmnode852",
            "tags": ["test"],
            "credentials": [],
            "explicit_deps": [],
            "reverse_deps": [],
            "assignee": "testuser",
        }

        result = render_mr_description(ctx)

        assert "**NEW ROLE**" in result
        assert "Closes #123" in result
        assert "Wave**: 1 (Infrastructure Foundation)" in result
        assert "✅ PASSED" in result
        assert "60s" in result
        assert "ansible/roles/test_role/" in result

    def test_existing_role_template_with_changes(self):
        """Existing role with changes shows VALIDATION header and diff stat."""
        from harness.gitlab.templates import render_mr_description

        ctx = {
            "role_name": "common",
            "issue_iid": "456",
            "wave": 0,
            "wave_name": "Foundation",
            "is_new_role": False,
            "role_diff_stat": " tasks/main.yml | 5 +++++\n 1 file changed",
            "molecule_passed": True,
            "molecule_duration": 120,
            "deploy_target": "vmnode852",
            "tags": ["common"],
            "credentials": [],
            "explicit_deps": [],
            "reverse_deps": ["windows_prerequisites"],
            "assignee": "testuser",
        }

        result = render_mr_description(ctx)

        assert "**VALIDATION**" in result
        assert "**NEW ROLE**" not in result
        assert "Changes from main" in result
        assert "tasks/main.yml | 5" in result
        assert "windows_prerequisites" in result

    def test_existing_role_no_changes(self):
        """Existing role without changes shows validation-only message."""
        from harness.gitlab.templates import render_mr_description

        ctx = {
            "role_name": "common",
            "issue_iid": "789",
            "wave": 0,
            "wave_name": "Foundation",
            "is_new_role": False,
            "role_diff_stat": None,
            "molecule_passed": True,
            "molecule_duration": 90,
            "deploy_target": "vmnode852",
            "tags": [],
            "credentials": [],
            "explicit_deps": [],
            "reverse_deps": [],
            "assignee": "testuser",
        }

        result = render_mr_description(ctx)

        assert "**VALIDATION**" in result
        assert "No file changes detected" in result

    def test_failed_molecule_tests(self):
        """Failed molecule tests show error details."""
        from harness.gitlab.templates import render_mr_description

        ctx = {
            "role_name": "broken_role",
            "issue_iid": "999",
            "wave": 2,
            "wave_name": "Core Platform",
            "is_new_role": True,
            "role_diff_stat": None,
            "molecule_passed": False,
            "molecule_duration": 45,
            "molecule_stderr": "TASK [verify] failed: expected service running",
            "deploy_target": "vmnode852",
            "tags": [],
            "credentials": [],
            "explicit_deps": [],
            "reverse_deps": [],
            "assignee": "testuser",
        }

        result = render_mr_description(ctx)

        assert "❌ FAILED" in result
        assert "Error Output" in result
        assert "expected service running" in result

    def test_skipped_molecule_tests(self):
        """Skipped molecule tests show warning message."""
        from harness.gitlab.templates import render_mr_description

        ctx = {
            "role_name": "no_tests_role",
            "issue_iid": "111",
            "wave": 4,
            "wave_name": "Supporting Services",
            "is_new_role": True,
            "role_diff_stat": None,
            "molecule_skipped": True,
            "deploy_target": "vmnode852",
            "tags": [],
            "credentials": [],
            "explicit_deps": [],
            "reverse_deps": [],
            "assignee": "testuser",
        }

        result = render_mr_description(ctx)

        assert "No molecule tests configured" in result

    def test_credentials_rendered(self):
        """Credentials are rendered in the MR description."""
        from harness.gitlab.templates import render_mr_description

        ctx = {
            "role_name": "cred_role",
            "issue_iid": "222",
            "wave": 2,
            "wave_name": "Core Platform",
            "is_new_role": True,
            "role_diff_stat": None,
            "molecule_passed": True,
            "molecule_duration": 30,
            "deploy_target": "vmnode852",
            "tags": [],
            "credentials": [
                {"entry": "EMS/Admin", "purpose": "Admin access"},
                {"entry": "SQL/ReadOnly", "purpose": "Database queries"},
            ],
            "explicit_deps": [],
            "reverse_deps": [],
            "assignee": "testuser",
        }

        result = render_mr_description(ctx)

        assert "EMS/Admin" in result
        assert "Admin access" in result
        assert "SQL/ReadOnly" in result

    def test_dependencies_rendered(self):
        """Dependencies are rendered in the MR description."""
        from harness.gitlab.templates import render_mr_description

        ctx = {
            "role_name": "dep_role",
            "issue_iid": "333",
            "wave": 3,
            "wave_name": "Web Applications",
            "is_new_role": True,
            "role_diff_stat": None,
            "molecule_passed": True,
            "molecule_duration": 60,
            "deploy_target": "vmnode852",
            "tags": [],
            "credentials": [],
            "explicit_deps": ["common", "iis-config"],
            "reverse_deps": [],
            "assignee": "testuser",
        }

        result = render_mr_description(ctx)

        assert "Depends on: `common`" in result
        assert "Depends on: `iis-config`" in result


class TestFormatTestEvidence:
    """Tests for _format_test_evidence helper."""

    def test_passed_test_evidence(self):
        """Passed tests show success status."""
        from harness.gitlab.templates import _format_test_evidence

        ctx = {
            "role_name": "test_role",
            "molecule_passed": True,
            "molecule_duration": 120,
            "deploy_target": "vmnode852",
        }

        result = _format_test_evidence(ctx)

        assert "✅ PASSED" in result
        assert "120s" in result
        assert "vmnode852" in result

    def test_failed_test_evidence_truncation(self):
        """Long error output is truncated."""
        from harness.gitlab.templates import _format_test_evidence

        long_error = "A" * 3000  # Longer than 2000 char limit

        ctx = {
            "role_name": "test_role",
            "molecule_passed": False,
            "molecule_duration": 30,
            "molecule_stderr": long_error,
            "deploy_target": "vmnode852",
        }

        result = _format_test_evidence(ctx)

        assert "❌ FAILED" in result
        assert "... (truncated)" in result
        assert len(result) < len(long_error) + 500  # Reasonable size

    def test_skipped_test_evidence(self):
        """Skipped tests show warning."""
        from harness.gitlab.templates import _format_test_evidence

        ctx = {"molecule_skipped": True}

        result = _format_test_evidence(ctx)

        assert "No molecule tests configured" in result
