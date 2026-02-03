"""Template rendering for GitLab issues and MRs.

Uses Jinja2 to render MR descriptions from bundled templates with
test evidence, role metadata, and conditional NEW/EXISTING sections.
"""

from __future__ import annotations

from typing import Any

import jinja2

from harness.assets import loader


def render_mr_description(ctx: dict[str, Any]) -> str:
    """Render MR description from bundled template with test evidence.

    Args:
        ctx: Template context containing:
            - role_name: Name of the Ansible role
            - issue_iid: GitLab issue IID to close
            - wave: Wave number (0-8)
            - wave_name: Wave name (e.g., "Foundation")
            - is_new_role: Whether this is a new role (not on origin/main)
            - role_diff_stat: Git diff stat output for existing roles
            - molecule_passed: Whether molecule tests passed
            - molecule_duration: Test duration in seconds
            - molecule_stderr: Test error output (if failed)
            - molecule_skipped: Whether tests were skipped
            - deploy_target: Target VM for deployment (e.g., "vmnode852")
            - tags: Role tags for deployment
            - credentials: List of credential info dicts
            - explicit_deps: List of upstream dependency role names
            - reverse_deps: List of downstream dependency role names
            - assignee: GitLab username for assignment

    Returns:
        Rendered MR description as markdown string.
    """
    template_text = loader.read_text("skills/box-up-role/templates/mr.md")
    template = jinja2.Template(template_text)

    # Format test evidence section
    test_evidence = _format_test_evidence(ctx)

    return template.render(
        role_name=ctx.get("role_name"),
        issue_iid=ctx.get("issue_iid"),
        wave_number=ctx.get("wave", 0),
        wave_name=ctx.get("wave_name", "Unassigned"),
        role_tags=",".join(ctx.get("tags", [])) if ctx.get("tags") else ctx.get("role_name", ""),
        deploy_target=ctx.get("deploy_target", "vmnode852"),
        credentials=ctx.get("credentials", []),
        explicit_deps=ctx.get("explicit_deps", []),
        reverse_deps=ctx.get("reverse_deps", []),
        assignee=ctx.get("assignee", ""),
        is_new_role=ctx.get("is_new_role", True),
        role_diff_stat=ctx.get("role_diff_stat"),
        test_evidence=test_evidence,
    )


def _format_test_evidence(ctx: dict[str, Any]) -> str:
    """Format molecule test results for MR description.

    Args:
        ctx: Template context with molecule test results.

    Returns:
        Formatted test evidence section as markdown.
    """
    if ctx.get("molecule_skipped"):
        return "> ⚠️ No molecule tests configured for this role."

    passed = ctx.get("molecule_passed", False)
    duration = ctx.get("molecule_duration", 0)
    role_name = ctx.get("role_name", "unknown")
    deploy_target = ctx.get("deploy_target", "vmnode852")

    status_icon = "✅" if passed else "❌"
    status_text = "PASSED" if passed else "FAILED"

    evidence = f"""### Molecule Test Results

| Field | Value |
|-------|-------|
| **Status** | {status_icon} {status_text} |
| **Duration** | {duration}s |
| **Target** | {deploy_target} |
| **Command** | `npm run molecule:role --role={role_name}` |
"""

    if not passed:
        stderr = ctx.get("molecule_stderr", "")
        if stderr:
            # Truncate to avoid massive MR descriptions
            truncated = stderr[:2000]
            if len(stderr) > 2000:
                truncated += "\n... (truncated)"
            evidence += f"""
<details>
<summary>Error Output (click to expand)</summary>

```
{truncated}
```
</details>
"""

    return evidence


def render_issue_description(ctx: dict[str, Any]) -> str:
    """Render issue description from bundled template.

    Args:
        ctx: Template context for issue.

    Returns:
        Rendered issue description as markdown string.
    """
    template_text = loader.read_text("skills/box-up-role/templates/issue.md")
    template = jinja2.Template(template_text)

    return template.render(
        role_name=ctx.get("role_name"),
        wave_number=ctx.get("wave", 0),
        wave_name=ctx.get("wave_name", "Unassigned"),
        deploy_target=ctx.get("deploy_target", "vmnode852"),
        explicit_deps=ctx.get("explicit_deps", []),
        reverse_deps=ctx.get("reverse_deps", []),
        credentials=ctx.get("credentials", []),
    )
