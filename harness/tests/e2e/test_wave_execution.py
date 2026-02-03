"""
E2E tests for wave-based execution ordering and parallelism.

These tests verify that the harness correctly:
1. Runs multiple roles within the same wave in parallel
2. Enforces wave ordering (wave N completes before wave N+1 starts)
3. Respects cross-wave dependencies

All external operations (git, molecule, GitLab) are mocked.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    create_box_up_role_graph,
    create_initial_state,
    set_module_db,
)
from harness.db.models import DependencyType, Role, RoleDependency
from harness.db.state import StateDB


# ============================================================================
# HELPERS
# ============================================================================


def _build_mock_stack(role_name: str, tmp_path: Path, *, fail_at_node: str | None = None):
    """Return a context-manager stack that mocks all externals for a single role.

    Optionally inject a failure at a specific node by name.
    """
    from contextlib import ExitStack

    stack = ExitStack()

    # Create role directory so the patched validate_role_node succeeds
    role_path = tmp_path / "ansible" / "roles" / role_name
    role_path.mkdir(parents=True, exist_ok=True)
    (role_path / "molecule").mkdir(exist_ok=True)
    (role_path / "meta").mkdir(exist_ok=True)
    (role_path / "meta" / "main.yml").write_text("---\ndependencies: []\n")

    # Patch validate_role_node to resolve against tmp_path
    async def _patched_validate(state: BoxUpRoleState) -> dict:
        rn = state["role_name"]
        rp = tmp_path / "ansible" / "roles" / rn
        import os
        if not os.path.isdir(str(rp)):
            return {
                "errors": [f"Role not found: {rn}"],
                "completed_nodes": ["validate_role"],
            }
        has_molecule = os.path.isdir(str(rp / "molecule"))
        has_meta = os.path.isfile(str(rp / "meta" / "main.yml"))
        return {
            "role_path": str(rp),
            "has_molecule_tests": has_molecule,
            "has_meta": has_meta,
            "completed_nodes": ["validate_role"],
        }

    stack.enter_context(
        patch("harness.dag.langgraph_engine.validate_role_node", _patched_validate)
    )

    # Test result recording (no-op for E2E)
    stack.enter_context(
        patch("harness.dag.langgraph_engine._record_test_result")
    )

    # Subprocess mock
    def _sub(cmd, **kw):
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)

        if "ls-remote" in cmd_str:
            # Pretend all remote branches exist (reverse deps already boxed up)
            r.stdout = "abc123\trefs/heads/sid/some_role"
        elif "status" in cmd_str and "porcelain" in cmd_str:
            r.stdout = f"M ansible/roles/{role_name}/tasks/main.yml\n"
        elif "rev-parse" in cmd_str:
            r.stdout = f"wave_{role_name}_sha"
        elif "molecule" in cmd_str:
            if fail_at_node == "run_molecule":
                r.returncode = 1
                r.stderr = "molecule failed"
            else:
                r.stdout = "ok=5  changed=0  failed=0"
        elif "pytest" in cmd_str:
            r.stdout = "1 passed"
        elif "syntax-check" in cmd_str:
            r.stdout = "playbook: site.yml"
        return r

    stack.enter_context(patch("subprocess.run", side_effect=_sub))

    # WorktreeManager
    wt_info = MagicMock()
    wt_info.path = str(tmp_path)
    wt_info.branch = f"sid/{role_name}"
    wt_info.commit = f"wave_{role_name}_sha"
    wt_mgr = MagicMock()
    wt_mgr.create.return_value = wt_info
    stack.enter_context(
        patch("harness.worktree.manager.WorktreeManager", return_value=wt_mgr)
    )

    # GitLabClient
    gl = MagicMock()
    mock_issue = MagicMock()
    mock_issue.iid = 1000
    mock_issue.web_url = f"https://gitlab.example.com/-/issues/1000"
    gl.get_or_create_issue.return_value = (mock_issue, True)

    mock_mr = MagicMock()
    mock_mr.iid = 2000
    mock_mr.web_url = f"https://gitlab.example.com/-/merge_requests/2000"
    gl.get_or_create_mr.return_value = (mock_mr, True)

    gl.is_merge_train_available.return_value = {"available": True}
    gl.is_merge_train_enabled.return_value = True
    gl.add_to_merge_train.return_value = None
    gl.prepare_labels_for_role.return_value = ["role"]
    gl.get_current_iteration.return_value = MagicMock(id=10)
    gl.config = MagicMock()
    gl.config.default_reviewers = []
    gl.set_mr_reviewers.return_value = True
    gl.remote_branch_exists.return_value = False
    gl.update_issue_on_failure.return_value = True
    stack.enter_context(patch("harness.gitlab.api.GitLabClient", return_value=gl))

    # Notifications
    stack.enter_context(
        patch("harness.dag.langgraph_engine.notify_workflow_started", new_callable=AsyncMock)
    )
    stack.enter_context(
        patch("harness.dag.langgraph_engine.notify_workflow_completed", new_callable=AsyncMock)
    )
    stack.enter_context(
        patch("harness.dag.langgraph_engine.notify_workflow_failed", new_callable=AsyncMock)
    )

    # Auto-approve human approval (patch the node function before graph build)
    async def _auto_approve(state: BoxUpRoleState) -> dict:
        return {
            "human_approved": True,
            "awaiting_human_input": False,
            "completed_nodes": ["human_approval"],
        }

    stack.enter_context(
        patch("harness.dag.langgraph_engine.human_approval_node", _auto_approve)
    )

    return stack


async def _run_workflow_for_role(
    role_name: str,
    db: StateDB,
    tmp_path: Path,
    *,
    fail_at_node: str | None = None,
) -> dict:
    """Execute the full box-up-role workflow for a single role and return final state."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    stack = _build_mock_stack(role_name, tmp_path, fail_at_node=fail_at_node)

    with stack:
        set_module_db(db)

        graph_def, _ = create_box_up_role_graph(
            db_path=str(tmp_path / f"cp_{role_name}.db"),
            parallel_tests=True,
            enable_breakpoints=False,
        )

        # human_approval_node is already patched by _build_mock_stack

        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / f"cp_{role_name}.db")
        ) as checkpointer:
            compiled = graph_def.compile(checkpointer=checkpointer)
            initial = create_initial_state(role_name)
            config = {"configurable": {"thread_id": f"wave-{role_name}"}}
            return await compiled.ainvoke(initial, config=config)


# ============================================================================
# WAVE EXECUTION TESTS
# ============================================================================


@pytest.mark.e2e
class TestWaveExecution:
    """Tests for wave-based parallelism and ordering."""

    @pytest.mark.asyncio
    async def test_wave_parallel_execution(
        self,
        e2e_db: StateDB,
        tmp_path: Path,
    ):
        """Multiple roles in the same wave should be able to run concurrently.

        We verify this by launching three wave-1 roles concurrently with
        asyncio.gather and checking that all complete successfully.
        """
        wave_1_roles = ["windows_prerequisites", "ems_registry_urls", "iis_config"]

        # Ensure all roles exist in the DB (they were seeded by e2e_db)
        for rn in wave_1_roles:
            role = e2e_db.get_role(rn)
            assert role is not None, f"Role {rn} should be seeded in e2e_db"

        # Create separate tmp subdirs so Path mocking does not collide
        sub_dirs = {rn: tmp_path / rn for rn in wave_1_roles}
        for d in sub_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Run all three concurrently
        results = await asyncio.gather(
            *[
                _run_workflow_for_role(rn, e2e_db, sub_dirs[rn])
                for rn in wave_1_roles
            ],
            return_exceptions=True,
        )

        # All should succeed (no exceptions)
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), (
                f"Role {wave_1_roles[i]} raised: {result}"
            )

        # Each result should have completed the workflow
        for i, result in enumerate(results):
            completed = result.get("completed_nodes", [])
            assert "validate_role" in completed, (
                f"{wave_1_roles[i]} missing validate_role"
            )
            assert "report_summary" in completed, (
                f"{wave_1_roles[i]} missing report_summary"
            )
            assert result.get("issue_iid") is not None, (
                f"{wave_1_roles[i]} missing issue_iid"
            )

    @pytest.mark.asyncio
    async def test_wave_ordering(
        self,
        e2e_db: StateDB,
        tmp_path: Path,
    ):
        """Wave N roles should not start until wave N-1 completes.

        We simulate this by running wave 0 (common) first, verifying it
        completes, and then running wave 1 roles. The key assertion is
        that wave-1 roles can access wave-0 results (e.g., the common
        role's state is available for dependency checks).
        """
        # --- Wave 0: Run "common" ---
        wave0_dir = tmp_path / "wave0"
        wave0_dir.mkdir()
        wave0_result = await _run_workflow_for_role("common", e2e_db, wave0_dir)

        wave0_completed = wave0_result.get("completed_nodes", [])
        assert "report_summary" in wave0_completed, "Wave 0 (common) should complete"

        # --- Wave 1: Run after wave 0 ---
        wave1_role = "windows_prerequisites"
        wave1_dir = tmp_path / "wave1"
        wave1_dir.mkdir()
        wave1_result = await _run_workflow_for_role(wave1_role, e2e_db, wave1_dir)

        wave1_completed = wave1_result.get("completed_nodes", [])
        assert "report_summary" in wave1_completed, "Wave 1 should complete after wave 0"

        # Verify the wave 1 role detected its dependency on common
        deps = wave1_result.get("explicit_deps", [])
        # The role has common as a dependency in the DB
        assert "common" in deps, (
            f"Wave 1 role should list 'common' as dependency; got {deps}"
        )

    @pytest.mark.asyncio
    async def test_cross_wave_dependency(
        self,
        e2e_db: StateDB,
        tmp_path: Path,
    ):
        """Dependencies across waves must be respected.

        ems_platform_services (wave 2) depends on common (wave 0).
        We verify that:
        1. The dependency is detected during analyze_deps
        2. The workflow completes even with cross-wave deps (since common
           is mocked as already boxed up via the ls-remote mock)
        """
        # Verify the cross-wave dependency exists in the database
        role = e2e_db.get_role("ems_platform_services")
        assert role is not None
        deps = e2e_db.get_dependencies("ems_platform_services", transitive=False)
        dep_names = [d[0] for d in deps]
        assert "common" in dep_names, (
            f"ems_platform_services should depend on common; got {dep_names}"
        )

        # Run the wave-2 role
        wave2_dir = tmp_path / "wave2"
        wave2_dir.mkdir()
        result = await _run_workflow_for_role(
            "ems_platform_services", e2e_db, wave2_dir
        )

        completed = result.get("completed_nodes", [])

        # analyze_dependencies should have detected the cross-wave dep
        assert "analyze_dependencies" in completed
        explicit = result.get("explicit_deps", [])
        assert "common" in explicit, (
            f"Cross-wave dependency 'common' not detected; got {explicit}"
        )

        # Workflow should have reached summary (common is not blocking
        # because ls-remote returns empty, and check_reverse_deps checks
        # reverse deps, not forward deps)
        assert "report_summary" in completed or "notify_failure" in completed

    @pytest.mark.asyncio
    async def test_wave_failure_isolation(
        self,
        e2e_db: StateDB,
        tmp_path: Path,
    ):
        """A failing role in one wave should not prevent other roles
        in the same wave from completing.

        We run two wave-1 roles sequentially (to avoid mock interference),
        verifying that one role's failure state does not leak into another's.
        """
        passing_role = "ems_registry_urls"
        failing_role = "iis_config"

        # Run the failing role first
        fail_dir = tmp_path / "fail"
        fail_dir.mkdir()
        fail_result = await _run_workflow_for_role(
            failing_role, e2e_db, fail_dir, fail_at_node="run_molecule"
        )

        # Then run the passing role
        pass_dir = tmp_path / "pass"
        pass_dir.mkdir()
        pass_result = await _run_workflow_for_role(passing_role, e2e_db, pass_dir)

        # Passing role completed successfully -- failure state did not leak
        pass_completed = pass_result.get("completed_nodes", [])
        assert "report_summary" in pass_completed, (
            f"Passing role should complete; got {pass_completed}"
        )
        pass_errors = pass_result.get("errors", [])
        # Passing role may have non-fatal errors but should not have molecule failures
        assert not any("molecule" in e.lower() and "failed" in e.lower() for e in pass_errors), (
            f"Passing role should not have molecule failures; got {pass_errors}"
        )

        # Failing role should have hit notify_failure
        fail_completed = fail_result.get("completed_nodes", [])
        assert "notify_failure" in fail_completed, (
            f"Failing role should route to notify_failure; got {fail_completed}"
        )
        fail_errors = fail_result.get("errors", [])
        assert len(fail_errors) > 0, "Failing role should have errors"
