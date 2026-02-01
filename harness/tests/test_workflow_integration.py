"""Integration tests for workflow execution and checkpointing."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from harness.db.state import StateDB
from harness.db.models import (
    Role, WorkflowStatus, NodeStatus, TestType, TestStatus, TestRun
)


class TestWorkflowExecution:
    """Tests for workflow execution lifecycle."""

    @pytest.mark.unit
    def test_create_workflow_definition(self, in_memory_db):
        """Test creating a workflow definition."""
        nodes = [
            {"id": "start", "type": "start"},
            {"id": "analyze", "type": "task"},
            {"id": "end", "type": "end"}
        ]
        edges = [
            {"from": "start", "to": "analyze"},
            {"from": "analyze", "to": "end"}
        ]

        workflow_id = in_memory_db.create_workflow_definition(
            name="test-workflow",
            description="Test workflow",
            nodes=nodes,
            edges=edges
        )

        assert workflow_id > 0

    @pytest.mark.unit
    def test_create_execution(self, workflow_db):
        """Test creating a workflow execution."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        assert execution_id > 0

    @pytest.mark.unit
    def test_create_execution_unknown_workflow(self, db_with_roles):
        """Test creating execution for unknown workflow fails."""
        with pytest.raises(ValueError, match="not found"):
            db_with_roles.create_execution(
                workflow_name="nonexistent",
                role_name="common"
            )

    @pytest.mark.unit
    def test_create_execution_unknown_role(self, workflow_db):
        """Test creating execution for unknown role fails."""
        with pytest.raises(ValueError, match="not found"):
            workflow_db.create_execution(
                workflow_name="box-up-role",
                role_name="nonexistent"
            )

    @pytest.mark.unit
    def test_update_execution_status(self, workflow_db):
        """Test updating execution status."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.RUNNING,
            current_node="analyze"
        )

        # Verify status updated (would need a get_execution method)
        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status, current_node FROM workflow_executions WHERE id = ?",
                (execution_id,)
            ).fetchone()
            assert row["status"] == "running"
            assert row["current_node"] == "analyze"

    @pytest.mark.unit
    def test_update_execution_completed(self, workflow_db):
        """Test marking execution as completed."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.COMPLETED
        )

        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status, completed_at FROM workflow_executions WHERE id = ?",
                (execution_id,)
            ).fetchone()
            assert row["status"] == "completed"
            assert row["completed_at"] is not None

    @pytest.mark.unit
    def test_update_execution_failed(self, workflow_db):
        """Test marking execution as failed with error."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.FAILED,
            error_message="Test error"
        )

        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status, error_message FROM workflow_executions WHERE id = ?",
                (execution_id,)
            ).fetchone()
            assert row["status"] == "failed"
            assert row["error_message"] == "Test error"


class TestWorkflowCheckpoint:
    """Tests for workflow checkpoint and resume."""

    @pytest.mark.unit
    def test_checkpoint_execution(self, workflow_db):
        """Test saving checkpoint data."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        checkpoint_data = {
            "current_node": "analyze",
            "completed_nodes": ["start"],
            "node_outputs": {"start": {"status": "ok"}}
        }

        workflow_db.checkpoint_execution(execution_id, checkpoint_data)

        # Verify checkpoint saved
        checkpoint = workflow_db.get_checkpoint(execution_id)
        assert checkpoint is not None
        assert checkpoint["current_node"] == "analyze"
        assert "start" in checkpoint["completed_nodes"]

    @pytest.mark.unit
    def test_get_checkpoint_none(self, workflow_db):
        """Test getting checkpoint when none exists."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        checkpoint = workflow_db.get_checkpoint(execution_id)
        assert checkpoint is None

    @pytest.mark.unit
    def test_checkpoint_overwrite(self, workflow_db):
        """Test overwriting checkpoint data."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        # First checkpoint
        workflow_db.checkpoint_execution(execution_id, {"step": 1})

        # Update checkpoint
        workflow_db.checkpoint_execution(execution_id, {"step": 2, "data": "new"})

        checkpoint = workflow_db.get_checkpoint(execution_id)
        assert checkpoint["step"] == 2
        assert checkpoint["data"] == "new"


class TestNodeExecution:
    """Tests for node execution tracking."""

    @pytest.mark.unit
    def test_update_node_execution_start(self, workflow_db):
        """Test recording node execution start."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        node_id = workflow_db.update_node_execution(
            execution_id,
            node_name="analyze",
            status=NodeStatus.RUNNING,
            input_data={"role": "common"}
        )

        assert node_id > 0

    @pytest.mark.unit
    def test_update_node_execution_complete(self, workflow_db):
        """Test recording node execution completion."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        # Start node
        workflow_db.update_node_execution(
            execution_id,
            node_name="analyze",
            status=NodeStatus.RUNNING
        )

        # Complete node
        workflow_db.update_node_execution(
            execution_id,
            node_name="analyze",
            status=NodeStatus.COMPLETED,
            output_data={"result": "success"}
        )

        # Verify
        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status, output_data FROM node_executions WHERE node_name = ?",
                ("analyze",)
            ).fetchone()
            assert row["status"] == "completed"
            assert "success" in row["output_data"]

    @pytest.mark.unit
    def test_update_node_execution_failed(self, workflow_db):
        """Test recording node execution failure."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        workflow_db.update_node_execution(
            execution_id,
            node_name="test",
            status=NodeStatus.FAILED,
            error_message="Test failed"
        )

        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status, error_message FROM node_executions WHERE node_name = ?",
                ("test",)
            ).fetchone()
            assert row["status"] == "failed"
            assert row["error_message"] == "Test failed"

    @pytest.mark.unit
    def test_node_execution_retry_count(self, workflow_db):
        """Test retry count increments on failure and retry."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        # First attempt fails
        workflow_db.update_node_execution(
            execution_id,
            node_name="flaky_test",
            status=NodeStatus.FAILED
        )

        # Retry
        workflow_db.update_node_execution(
            execution_id,
            node_name="flaky_test",
            status=NodeStatus.RUNNING
        )

        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT retry_count FROM node_executions WHERE node_name = ?",
                ("flaky_test",)
            ).fetchone()
            assert row["retry_count"] >= 1


class TestTestRuns:
    """Tests for test run tracking."""

    @pytest.mark.unit
    def test_create_test_run(self, db_with_roles):
        """Test creating a test run."""
        role = db_with_roles.get_role("common")
        test_run = TestRun(
            role_id=role.id,
            test_type=TestType.MOLECULE,
            status=TestStatus.RUNNING
        )

        run_id = db_with_roles.create_test_run(test_run)
        assert run_id > 0

    @pytest.mark.unit
    def test_update_test_run(self, db_with_roles):
        """Test updating test run status."""
        role = db_with_roles.get_role("common")
        test_run = TestRun(
            role_id=role.id,
            test_type=TestType.MOLECULE,
            status=TestStatus.RUNNING
        )

        run_id = db_with_roles.create_test_run(test_run)

        db_with_roles.update_test_run(
            run_id,
            status=TestStatus.PASSED,
            duration_seconds=120
        )

        # Verify
        with db_with_roles.connection() as conn:
            row = conn.execute(
                "SELECT status, duration_seconds FROM test_runs WHERE id = ?",
                (run_id,)
            ).fetchone()
            assert row["status"] == "passed"
            assert row["duration_seconds"] == 120

    @pytest.mark.unit
    def test_get_recent_test_runs(self, db_with_roles):
        """Test getting recent test runs for a role."""
        role = db_with_roles.get_role("common")

        # Create multiple test runs
        for status in [TestStatus.PASSED, TestStatus.FAILED, TestStatus.PASSED]:
            test_run = TestRun(
                role_id=role.id,
                test_type=TestType.MOLECULE,
                status=status
            )
            db_with_roles.create_test_run(test_run)

        runs = db_with_roles.get_recent_test_runs("common", limit=10)
        assert len(runs) == 3


class TestNotificationDelivery:
    """Tests for notification delivery (mocked)."""

    @pytest.mark.unit
    def test_discord_notification_sync(self, notification_config, mock_httpx_client):
        """Test sync Discord notification."""
        from harness.hotl.notifications import NotificationService

        service = NotificationService(notification_config)

        with patch.object(service, '_get_sync_client', return_value=mock_httpx_client):
            result = service.send_discord_sync(
                title="Test Notification",
                description="This is a test",
                color=0x00ff00
            )

            assert result is True
            mock_httpx_client.post.assert_called_once()

    @pytest.mark.unit
    def test_discord_notification_no_webhook(self):
        """Test Discord notification without webhook configured."""
        from harness.hotl.notifications import NotificationService, NotificationConfig

        config = NotificationConfig()  # No webhook URL
        service = NotificationService(config)

        result = service.send_discord_sync(
            title="Test",
            description="Test"
        )

        assert result is False

    @pytest.mark.unit
    def test_status_update_sync(self, notification_config, mock_httpx_client):
        """Test sync status update notification."""
        from harness.hotl.notifications import NotificationService

        service = NotificationService(notification_config)

        state = {
            "phase": "testing",
            "iteration_count": 5,
            "max_iterations": 10,
            "completed_tasks": [1, 2, 3],
            "failed_tasks": [],
            "errors": [],
            "warnings": ["Minor issue"]
        }

        with patch.object(service, '_get_sync_client', return_value=mock_httpx_client):
            results = service.send_status_update_sync(state, "Test summary")

            assert "discord" in results
            assert results["discord"] is True

    @pytest.mark.unit
    def test_status_update_with_errors(self, notification_config, mock_httpx_client):
        """Test status update with errors uses red color."""
        from harness.hotl.notifications import NotificationService

        service = NotificationService(notification_config)

        state = {
            "phase": "failed",
            "iteration_count": 3,
            "max_iterations": 10,
            "completed_tasks": [1],
            "failed_tasks": [2],
            "errors": ["Critical error occurred"],
            "warnings": []
        }

        with patch.object(service, '_get_sync_client', return_value=mock_httpx_client):
            results = service.send_status_update_sync(state, "Error summary")

            # Should still send
            assert "discord" in results

    @pytest.mark.asyncio
    async def test_discord_notification_async(self, notification_config, mock_async_httpx_client):
        """Test async Discord notification."""
        from harness.hotl.notifications import NotificationService

        service = NotificationService(notification_config)
        service._async_client = mock_async_httpx_client

        result = await service.send_discord(
            title="Async Test",
            description="Async notification test"
        )

        assert result is True


class TestErrorHandlingPaths:
    """Tests for error handling in workflows."""

    @pytest.mark.unit
    def test_workflow_cancellation(self, workflow_db):
        """Test cancelling a running workflow."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.RUNNING
        )

        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.CANCELLED
        )

        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status FROM workflow_executions WHERE id = ?",
                (execution_id,)
            ).fetchone()
            assert row["status"] == "cancelled"

    @pytest.mark.unit
    def test_node_skipped(self, workflow_db):
        """Test marking a node as skipped."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        workflow_db.update_node_execution(
            execution_id,
            node_name="optional_step",
            status=NodeStatus.SKIPPED
        )

        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status FROM node_executions WHERE node_name = ?",
                ("optional_step",)
            ).fetchone()
            assert row["status"] == "skipped"

    @pytest.mark.unit
    def test_test_regression_tracking(self, db_with_roles):
        """Test regression is tracked on test failure."""
        role = db_with_roles.get_role("common")
        test_run = TestRun(
            role_id=role.id,
            test_type=TestType.MOLECULE,
            status=TestStatus.FAILED
        )
        run_id = db_with_roles.create_test_run(test_run)

        regression_id = db_with_roles.record_test_failure(
            role_name="common",
            test_name="test_smoke",
            test_type=TestType.MOLECULE,
            test_run_id=run_id,
            error_message="Assertion failed"
        )

        assert regression_id > 0

        regressions = db_with_roles.get_active_regressions()
        assert len(regressions) >= 1

    @pytest.mark.unit
    def test_test_success_resolves_regression(self, db_with_roles):
        """Test successful test resolves regression."""
        role = db_with_roles.get_role("common")

        # Create failed run
        failed_run = TestRun(
            role_id=role.id,
            test_type=TestType.MOLECULE,
            status=TestStatus.FAILED
        )
        failed_run_id = db_with_roles.create_test_run(failed_run)

        # Record multiple failures
        for _ in range(3):
            db_with_roles.record_test_failure(
                role_name="common",
                test_name="test_regression",
                test_type=TestType.MOLECULE,
                test_run_id=failed_run_id
            )

        # Create passing run
        passed_run = TestRun(
            role_id=role.id,
            test_type=TestType.MOLECULE,
            status=TestStatus.PASSED
        )
        passed_run_id = db_with_roles.create_test_run(passed_run)

        # Record success
        db_with_roles.record_test_success(
            role_name="common",
            test_name="test_regression",
            test_type=TestType.MOLECULE,
            test_run_id=passed_run_id
        )

        # Check regression status
        regression = db_with_roles.get_regression(
            "common",
            "test_regression",
            TestType.MOLECULE
        )
        assert regression.status.value == "resolved"


class TestEndToEndBoxUpRole:
    """End-to-end tests for box-up-role workflow with mocks."""

    @pytest.mark.integration
    def test_box_up_role_success_flow(self, workflow_db):
        """Test successful box-up-role workflow execution."""
        # Create execution
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        # Simulate workflow progression
        nodes = ["start", "analyze", "test", "end"]

        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.RUNNING,
            current_node=nodes[0]
        )

        for i, node in enumerate(nodes):
            # Start node
            workflow_db.update_node_execution(
                execution_id,
                node_name=node,
                status=NodeStatus.RUNNING
            )

            # Complete node
            workflow_db.update_node_execution(
                execution_id,
                node_name=node,
                status=NodeStatus.COMPLETED,
                output_data={"step": i + 1}
            )

            # Checkpoint
            workflow_db.checkpoint_execution(execution_id, {
                "completed_nodes": nodes[:i + 1],
                "current_index": i + 1
            })

            if i < len(nodes) - 1:
                workflow_db.update_execution_status(
                    execution_id,
                    status=WorkflowStatus.RUNNING,
                    current_node=nodes[i + 1]
                )

        # Complete workflow
        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.COMPLETED
        )

        # Verify final state
        checkpoint = workflow_db.get_checkpoint(execution_id)
        assert len(checkpoint["completed_nodes"]) == 4

    @pytest.mark.integration
    def test_box_up_role_resume_from_checkpoint(self, workflow_db):
        """Test resuming workflow from checkpoint."""
        # Create execution
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        # Simulate partial progress
        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.RUNNING,
            current_node="analyze"
        )

        workflow_db.update_node_execution(
            execution_id,
            node_name="start",
            status=NodeStatus.COMPLETED
        )

        workflow_db.checkpoint_execution(execution_id, {
            "completed_nodes": ["start"],
            "current_node": "analyze",
            "resume_data": {"partial": True}
        })

        # Simulate interruption - mark as paused
        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.PAUSED
        )

        # Resume from checkpoint
        checkpoint = workflow_db.get_checkpoint(execution_id)
        assert checkpoint is not None
        assert "start" in checkpoint["completed_nodes"]
        assert checkpoint["current_node"] == "analyze"

        # Continue execution
        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.RUNNING,
            current_node="analyze"
        )

        workflow_db.update_node_execution(
            execution_id,
            node_name="analyze",
            status=NodeStatus.COMPLETED
        )

    @pytest.mark.integration
    def test_box_up_role_failure_and_recovery(self, workflow_db):
        """Test workflow failure and partial recovery."""
        execution_id = workflow_db.create_execution(
            workflow_name="box-up-role",
            role_name="common"
        )

        # Progress to test node
        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.RUNNING,
            current_node="test"
        )

        workflow_db.update_node_execution(
            execution_id,
            node_name="start",
            status=NodeStatus.COMPLETED
        )

        workflow_db.update_node_execution(
            execution_id,
            node_name="analyze",
            status=NodeStatus.COMPLETED
        )

        # Test node fails
        workflow_db.update_node_execution(
            execution_id,
            node_name="test",
            status=NodeStatus.FAILED,
            error_message="Molecule test failed"
        )

        workflow_db.update_execution_status(
            execution_id,
            status=WorkflowStatus.FAILED,
            error_message="Molecule test failed"
        )

        # Verify failure state
        with workflow_db.connection() as conn:
            row = conn.execute(
                "SELECT status, error_message FROM workflow_executions WHERE id = ?",
                (execution_id,)
            ).fetchone()
            assert row["status"] == "failed"
            assert "Molecule" in row["error_message"]
