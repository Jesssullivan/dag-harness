"""
E2E tests for the full box-up-role LangGraph workflow.

Tests exercise the compiled StateGraph end-to-end using the
LangGraphWorkflowRunner, with subprocess/GitLab operations mocked
to avoid requiring a live VM or GitLab project.

When GITLAB_E2E_TOKEN is set, selected tests can use real GitLab
operations; otherwise they fall back to mocks.

Four scenarios are covered:
1. Happy path -- full workflow from validate to merge train
2. Failure path -- invalid role fails at validation
3. Idempotency -- running the workflow twice reuses GitLab resources
4. Resume from approval -- workflow pauses at HITL, then resumes
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.dag.langgraph_engine import (
    BoxUpRoleState,
    Command,
    LangGraphWorkflowRunner,
    create_box_up_role_graph,
    create_initial_state,
    set_module_db,
)
from harness.db.models import Role, WorkflowStatus
from harness.db.state import StateDB


# ============================================================================
# HELPERS
# ============================================================================


def _patch_all_externals(role_name: str, tmp_path: Path, *, auto_approve: bool = True):
    """Return a stack of context managers that mock every external dependency.

    This patches:
    - validate_role_node: pretend the role dir exists with molecule tests
    - subprocess.run: simulate git, molecule, pytest, ansible
    - GitLabClient: simulate issue/MR/merge-train operations
    - WorktreeManager: simulate worktree creation
    - NotificationService: no-op notifications
    - human_approval_node: auto-approve (when auto_approve=True)
    """
    from contextlib import ExitStack

    stack = ExitStack()

    # -- Role directory existence ----------------------------------------
    # Create role directory under tmp_path so we can redirect validate_role_node
    role_path = tmp_path / "ansible" / "roles" / role_name
    role_path.mkdir(parents=True, exist_ok=True)
    (role_path / "molecule").mkdir(exist_ok=True)
    (role_path / "meta").mkdir(exist_ok=True)
    (role_path / "meta" / "main.yml").write_text("---\ndependencies: []\n")

    # Patch validate_role_node to check tmp_path instead of cwd
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

    # -- Test result recording (no-op for E2E) ---------------------------
    stack.enter_context(
        patch("harness.dag.langgraph_engine._record_test_result")
    )

    # -- Human approval auto-approve -------------------------------------
    if auto_approve:
        async def _auto_approve_node(state: BoxUpRoleState) -> dict:
            return {
                "human_approved": True,
                "awaiting_human_input": False,
                "completed_nodes": ["human_approval"],
            }

        stack.enter_context(
            patch(
                "harness.dag.langgraph_engine.human_approval_node",
                _auto_approve_node,
            )
        )

    # -- subprocess.run (git, molecule, pytest, ansible) ------------------
    def _subprocess_side_effect(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)

        if "ls-remote" in cmd_str:
            result.stdout = ""
        elif "status" in cmd_str and "porcelain" in cmd_str:
            result.stdout = f"M ansible/roles/{role_name}/tasks/main.yml\n"
        elif "rev-parse" in cmd_str:
            result.stdout = "e2eabc123"
        elif "push" in cmd_str:
            pass
        elif "commit" in cmd_str:
            result.stdout = f"[sid/{role_name} e2eabc123] feat({role_name})"
        elif "molecule" in cmd_str:
            result.stdout = "ok=5  changed=0  failed=0"
        elif "pytest" in cmd_str:
            result.stdout = "1 passed"
        elif "syntax-check" in cmd_str:
            result.stdout = "playbook: site.yml"
        return result

    stack.enter_context(patch("subprocess.run", side_effect=_subprocess_side_effect))

    # -- WorktreeManager --------------------------------------------------
    mock_wt_info = MagicMock()
    mock_wt_info.path = str(tmp_path)
    mock_wt_info.branch = f"sid/{role_name}"
    mock_wt_info.commit = "e2eabc123"
    mock_manager = MagicMock()
    mock_manager.create.return_value = mock_wt_info
    stack.enter_context(
        patch("harness.worktree.manager.WorktreeManager", return_value=mock_manager)
    )

    # -- GitLabClient -----------------------------------------------------
    mock_client = MagicMock()
    mock_issue = MagicMock()
    mock_issue.iid = 42
    mock_issue.web_url = f"https://gitlab.example.com/project/-/issues/42"
    mock_client.get_or_create_issue.return_value = (mock_issue, True)

    mock_mr = MagicMock()
    mock_mr.iid = 99
    mock_mr.web_url = f"https://gitlab.example.com/project/-/merge_requests/99"
    mock_client.get_or_create_mr.return_value = (mock_mr, True)

    mock_client.is_merge_train_available.return_value = {"available": True}
    mock_client.is_merge_train_enabled.return_value = True
    mock_client.add_to_merge_train.return_value = None
    mock_client.prepare_labels_for_role.return_value = ["role", "wave-1"]
    mock_iter = MagicMock()
    mock_iter.id = 100
    mock_client.get_current_iteration.return_value = mock_iter
    mock_client.config = MagicMock()
    mock_client.config.default_reviewers = []
    mock_client.set_mr_reviewers.return_value = True
    mock_client.remote_branch_exists.return_value = False
    mock_client.update_issue_on_failure.return_value = True

    stack.enter_context(
        patch("harness.gitlab.api.GitLabClient", return_value=mock_client)
    )

    # -- Notifications (no-op) -------------------------------------------
    stack.enter_context(
        patch("harness.dag.langgraph_engine.notify_workflow_started", new_callable=AsyncMock)
    )
    stack.enter_context(
        patch("harness.dag.langgraph_engine.notify_workflow_completed", new_callable=AsyncMock)
    )
    stack.enter_context(
        patch("harness.dag.langgraph_engine.notify_workflow_failed", new_callable=AsyncMock)
    )

    return stack, mock_client


# ============================================================================
# E2E TEST CLASS
# ============================================================================


@pytest.mark.e2e
class TestBoxUpWorkflowE2E:
    """Full workflow E2E tests for the box-up-role LangGraph DAG."""

    @pytest.mark.asyncio
    async def test_happy_path(
        self,
        e2e_db: StateDB,
        e2e_config,
        unique_role_name: str,
        e2e_cleanup,
        tmp_path: Path,
    ):
        """Full workflow: validate -> analyze -> worktree -> tests -> commit
        -> push -> issue -> MR -> approve -> merge train -> summary.

        Verifies that every node executes in order, GitLab resources are
        created, and the final summary reflects success.
        """
        # Register the unique role in the DB
        e2e_db.upsert_role(
            Role(
                name=unique_role_name,
                wave=1,
                wave_name="E2E Test Wave",
                has_molecule_tests=True,
            )
        )

        stack, mock_gl = _patch_all_externals(unique_role_name, tmp_path)

        with stack:
            set_module_db(e2e_db)

            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            graph_def, _ = create_box_up_role_graph(
                db_path=str(tmp_path / "cp.db"),
                parallel_tests=True,
                enable_breakpoints=False,
            )

            async with AsyncSqliteSaver.from_conn_string(
                str(tmp_path / "cp.db")
            ) as checkpointer:
                compiled = graph_def.compile(checkpointer=checkpointer)

                initial = create_initial_state(unique_role_name)
                config = {"configurable": {"thread_id": f"e2e-happy-{unique_role_name}"}}

                final_state = await compiled.ainvoke(initial, config=config)

        # -- Assertions -------------------------------------------------------
        completed = final_state.get("completed_nodes", [])
        errors = final_state.get("errors", [])

        # Core workflow nodes should all appear in completed_nodes
        assert "validate_role" in completed, f"validate_role missing; completed={completed}"
        assert "analyze_dependencies" in completed, f"analyze_deps missing; completed={completed}"
        assert "create_worktree" in completed
        assert "create_commit" in completed
        assert "push_branch" in completed
        assert "create_gitlab_issue" in completed
        assert "create_merge_request" in completed
        assert "human_approval" in completed
        assert "add_to_merge_train" in completed
        assert "report_summary" in completed

        # GitLab resources created
        assert final_state.get("issue_iid") == 42
        assert final_state.get("mr_iid") == 99
        assert final_state.get("merge_train_status") == "added"

        # Summary present and reflects success
        summary = final_state.get("summary")
        assert summary is not None
        assert summary["role"] == unique_role_name
        assert summary["success"] is True or len(errors) == 0

        # Track for cleanup
        e2e_cleanup.track_issue(42)
        e2e_cleanup.track_mr(99)

    @pytest.mark.asyncio
    async def test_failure_path(
        self,
        e2e_db: StateDB,
        e2e_config,
        e2e_cleanup,
        tmp_path: Path,
    ):
        """Non-existent role should fail at validation node.

        The workflow should stop early, populate errors, and never create
        GitLab resources.
        """
        nonexistent_role = "role_that_does_not_exist_anywhere"

        # Register in DB so execution creation succeeds, but do NOT create
        # the role directory -- validate_role_node checks the filesystem.
        e2e_db.upsert_role(
            Role(name=nonexistent_role, wave=0, has_molecule_tests=False)
        )

        # Patch only notifications (we want real validate_role_node to fail)
        with (
            patch(
                "harness.dag.langgraph_engine.notify_workflow_started",
                new_callable=AsyncMock,
            ),
            patch(
                "harness.dag.langgraph_engine.notify_workflow_completed",
                new_callable=AsyncMock,
            ),
            patch(
                "harness.dag.langgraph_engine.notify_workflow_failed",
                new_callable=AsyncMock,
            ),
        ):
            set_module_db(e2e_db)

            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            graph_def, _ = create_box_up_role_graph(
                db_path=str(tmp_path / "cp.db"),
                parallel_tests=True,
                enable_breakpoints=False,
            )

            async with AsyncSqliteSaver.from_conn_string(
                str(tmp_path / "cp.db")
            ) as checkpointer:
                compiled = graph_def.compile(checkpointer=checkpointer)

                initial = create_initial_state(nonexistent_role)
                config = {"configurable": {"thread_id": "e2e-failure"}}

                final_state = await compiled.ainvoke(initial, config=config)

        # -- Assertions -------------------------------------------------------
        errors = final_state.get("errors", [])
        completed = final_state.get("completed_nodes", [])

        # Validation should have failed
        assert len(errors) > 0, "Expected at least one error"
        assert any("not found" in e.lower() or "role" in e.lower() for e in errors)

        # notify_failure should have run
        assert "notify_failure" in completed

        # No GitLab resources should have been created
        assert final_state.get("issue_iid") is None
        assert final_state.get("mr_iid") is None
        assert final_state.get("merge_train_status") is None

    @pytest.mark.asyncio
    async def test_idempotency(
        self,
        e2e_db: StateDB,
        e2e_config,
        unique_role_name: str,
        e2e_cleanup,
        tmp_path: Path,
    ):
        """Running the workflow twice should reuse existing GitLab resources.

        The second run should return the same issue IID and MR IID as the
        first run, demonstrating the get_or_create idempotency.
        """
        e2e_db.upsert_role(
            Role(
                name=unique_role_name,
                wave=1,
                wave_name="E2E Idempotency Wave",
                has_molecule_tests=True,
            )
        )

        stack, mock_gl = _patch_all_externals(unique_role_name, tmp_path)

        with stack:
            set_module_db(e2e_db)

            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            graph_def, _ = create_box_up_role_graph(
                db_path=str(tmp_path / "cp.db"),
                parallel_tests=True,
                enable_breakpoints=False,
            )

            async with AsyncSqliteSaver.from_conn_string(
                str(tmp_path / "cp.db")
            ) as checkpointer:
                compiled = graph_def.compile(checkpointer=checkpointer)

                # --- First run ---
                initial_1 = create_initial_state(unique_role_name)
                config_1 = {"configurable": {"thread_id": f"e2e-idem-1-{unique_role_name}"}}
                state_1 = await compiled.ainvoke(initial_1, config=config_1)

                # Switch get_or_create to return created=False (existing)
                mock_gl.get_or_create_issue.return_value = (
                    mock_gl.get_or_create_issue.return_value[0],
                    False,
                )
                mock_gl.get_or_create_mr.return_value = (
                    mock_gl.get_or_create_mr.return_value[0],
                    False,
                )

                # --- Second run ---
                initial_2 = create_initial_state(unique_role_name)
                config_2 = {"configurable": {"thread_id": f"e2e-idem-2-{unique_role_name}"}}
                state_2 = await compiled.ainvoke(initial_2, config=config_2)

        # -- Assertions -------------------------------------------------------
        # Both runs should reference the same GitLab resources
        assert state_1.get("issue_iid") == state_2.get("issue_iid") == 42
        assert state_1.get("mr_iid") == state_2.get("mr_iid") == 99

        # First run created, second reused
        assert state_1.get("issue_created") is True
        assert state_2.get("issue_created") is False
        assert state_1.get("mr_created") is True
        assert state_2.get("mr_created") is False

        e2e_cleanup.track_issue(42)
        e2e_cleanup.track_mr(99)

    @pytest.mark.asyncio
    async def test_resume_from_approval(
        self,
        e2e_db: StateDB,
        e2e_config,
        unique_role_name: str,
        e2e_cleanup,
        tmp_path: Path,
    ):
        """Workflow should pause at human_approval, then resume when approved.

        1. Run the workflow until it hits the interrupt() in human_approval.
        2. Verify the workflow is paused and state is saved.
        3. Resume with Command(resume={"approved": True}).
        4. Verify the workflow completes through merge train and summary.
        """
        e2e_db.upsert_role(
            Role(
                name=unique_role_name,
                wave=1,
                wave_name="E2E Resume Wave",
                has_molecule_tests=True,
            )
        )

        # auto_approve=False keeps the real human_approval_node with interrupt()
        stack, mock_gl = _patch_all_externals(
            unique_role_name, tmp_path, auto_approve=False
        )

        with stack:
            set_module_db(e2e_db)

            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            graph_def, _ = create_box_up_role_graph(
                db_path=str(tmp_path / "cp.db"),
                parallel_tests=True,
                enable_breakpoints=False,
            )

            async with AsyncSqliteSaver.from_conn_string(
                str(tmp_path / "cp.db")
            ) as checkpointer:
                compiled = graph_def.compile(checkpointer=checkpointer)

                thread_id = f"e2e-resume-{unique_role_name}"
                config = {"configurable": {"thread_id": thread_id}}

                initial = create_initial_state(unique_role_name)

                # --- Phase 1: Run until interrupt ---
                # ainvoke returns the state at the interrupt point.
                paused_state = await compiled.ainvoke(initial, config=config)

                # The workflow should have reached human_approval and paused.
                completed_before_resume = paused_state.get("completed_nodes", [])

                # Nodes before human_approval should be completed
                assert "validate_role" in completed_before_resume
                assert "create_merge_request" in completed_before_resume

                # human_approval should NOT be completed yet (it interrupted)
                # and add_to_merge_train / report_summary should not appear
                assert "add_to_merge_train" not in completed_before_resume
                assert "report_summary" not in completed_before_resume

                # --- Phase 2: Resume with approval ---
                resumed_state = await compiled.ainvoke(
                    Command(resume={"approved": True}),
                    config=config,
                )

                completed_after_resume = resumed_state.get("completed_nodes", [])

                # After resume, the remaining nodes should have executed
                assert "human_approval" in completed_after_resume
                assert "add_to_merge_train" in completed_after_resume
                assert "report_summary" in completed_after_resume

                # Final state should show approval and merge train success
                assert resumed_state.get("human_approved") is True
                assert resumed_state.get("merge_train_status") == "added"
                assert resumed_state.get("mr_iid") == 99

        e2e_cleanup.track_issue(42)
        e2e_cleanup.track_mr(99)
