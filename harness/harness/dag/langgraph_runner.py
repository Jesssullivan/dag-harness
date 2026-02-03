"""
LangGraph workflow runner with database integration.

Provides:
- Easy execution interface
- State persistence to StateDB
- Resume from checkpoint support
- Event emission for observability
- Regression tracking via module-level db
- Configurable checkpointer (SQLite or PostgreSQL)
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig

from harness.config import NotificationConfig
from harness.dag.langgraph_builder import create_box_up_role_graph
from harness.dag.langgraph_state import (
    BoxUpRoleState,
    create_initial_state,
    set_module_config,
    set_module_db,
)
from harness.dag.store import HarnessStore
from harness.db.models import WorkflowStatus
from harness.db.state import StateDB
from harness.notifications import (
    NotificationService,
    notify_workflow_completed,
    notify_workflow_failed,
    notify_workflow_started,
)


class LangGraphWorkflowRunner:
    """
    Wrapper for LangGraph workflow execution with database integration.

    Provides:
    - Easy execution interface
    - State persistence to StateDB
    - Resume from checkpoint support
    - Event emission for observability
    - Regression tracking via module-level db
    - Configurable checkpointer (SQLite or PostgreSQL)

    Checkpointer Configuration (Task #23):
        The runner supports multiple checkpointer backends:
        - SQLite (default): Uses db_path for local development
        - PostgreSQL: Uses checkpointer_factory for production

        To use PostgreSQL checkpointing:
            from harness.dag.checkpointer import CheckpointerFactory

            runner = LangGraphWorkflowRunner(
                db=db,
                checkpointer_factory=CheckpointerFactory,
                postgres_url="postgresql://user:pass@host/db"
            )

        Or configure via environment:
            export POSTGRES_URL=postgresql://user:pass@host/db
            runner = LangGraphWorkflowRunner(db=db, use_postgres=True)
    """

    def __init__(
        self,
        db: StateDB,
        db_path: str = "harness.db",
        notification_config: NotificationConfig | None = None,
        checkpointer_factory: Any | None = None,
        postgres_url: str | None = None,
        use_postgres: bool = False,
        store: HarnessStore | None = None,
    ):
        """
        Initialize the workflow runner.

        Args:
            db: StateDB instance for workflow tracking
            db_path: Path to SQLite database (default checkpointer)
            notification_config: Optional notification configuration
            checkpointer_factory: Optional CheckpointerFactory class for custom checkpointer
            postgres_url: Optional PostgreSQL URL (overrides environment)
            use_postgres: If True, prefer PostgreSQL from environment variable
            store: Optional HarnessStore for cross-thread memory persistence
        """
        self.db = db
        self.db_path = db_path
        self._graph = None
        self._checkpointer = None
        self._checkpointer_factory = checkpointer_factory
        self._postgres_url = postgres_url
        self._use_postgres = use_postgres or postgres_url is not None
        self._store = store
        # Set module-level db and config for node access
        set_module_db(db)
        # Load harness config from harness.yml for GitLab settings
        try:
            from harness.config import HarnessConfig

            set_module_config(HarnessConfig.load())
        except Exception:
            pass  # Config not required, nodes fall back to defaults
        # Initialize notification service
        self._notification_service = NotificationService(
            notification_config or NotificationConfig(enabled=False)
        )

    async def _get_graph(self):
        """Lazily create and cache the compiled graph with appropriate checkpointer."""
        if self._graph is None:
            graph, breakpoints = create_box_up_role_graph(self.db_path)

            # Determine which checkpointer to use
            checkpointer = None

            if self._checkpointer_factory is not None:
                # Use provided factory
                try:
                    checkpointer = await self._checkpointer_factory.create_async(
                        postgres_url=self._postgres_url,
                        sqlite_path=self.db_path,
                        fallback_to_sqlite=True,
                    )
                    self._checkpointer = checkpointer
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Failed to create checkpointer from factory: {e}. "
                        "Falling back to in-memory."
                    )
            elif self._use_postgres:
                # Try to use PostgreSQL from environment
                try:
                    from harness.dag.checkpointer import CheckpointerFactory

                    checkpointer = await CheckpointerFactory.create_async(
                        postgres_url=self._postgres_url,
                        sqlite_path=self.db_path,
                        fallback_to_sqlite=True,
                    )
                    self._checkpointer = checkpointer
                except ImportError:
                    pass  # checkpointer module not available
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Failed to create PostgreSQL checkpointer: {e}. "
                        "Compiling without checkpointer."
                    )

            # Build compile kwargs
            compile_kwargs = {}
            if checkpointer is not None:
                compile_kwargs["checkpointer"] = checkpointer
            if breakpoints:
                compile_kwargs["interrupt_before"] = breakpoints
            if self._store is not None:
                compile_kwargs["store"] = self._store

            # Compile with or without checkpointer/breakpoints/store
            self._graph = graph.compile(**compile_kwargs)

        return self._graph

    async def execute(
        self, role_name: str, resume_from: int | None = None, config: RunnableConfig | None = None
    ) -> dict:
        """
        Execute the box-up-role workflow.

        Args:
            role_name: Name of the role to process
            resume_from: Optional execution ID to resume from
            config: Optional LangGraph configuration

        Returns:
            Final state after execution
        """
        graph = await self._get_graph()

        # Create or restore initial state
        if resume_from:
            checkpoint = self.db.get_checkpoint(resume_from)
            if checkpoint:
                initial_state = checkpoint.get("state", {})
                initial_state["role_name"] = role_name
                execution_id = resume_from
            else:
                raise ValueError(f"No checkpoint found for execution {resume_from}")
        else:
            # Create new execution record
            self.db.create_workflow_definition(
                "box_up_role_langgraph",
                "LangGraph-based box-up-role workflow",
                nodes=[
                    {"name": n}
                    for n in [
                        "validate_role",
                        "analyze_deps",
                        "check_reverse_deps",
                        "create_worktree",
                        "run_molecule",
                        "run_pytest",
                        "validate_deploy",
                        "create_commit",
                        "push_branch",
                        "create_issue",
                        "create_mr",
                        "add_to_merge_train",
                        "report_summary",
                        "notify_failure",
                    ]
                ],
                edges=[],
            )
            execution_id = self.db.create_execution("box_up_role_langgraph", role_name)
            initial_state = create_initial_state(role_name, execution_id)

        # Update execution status
        self.db.update_execution_status(
            execution_id, WorkflowStatus.RUNNING, current_node="validate_role"
        )

        # Send workflow started notification
        await notify_workflow_started(self._notification_service, role_name, execution_id)

        try:
            # Execute the graph
            final_state = await graph.ainvoke(initial_state, config=config or {})

            # Save checkpoint
            self.db.checkpoint_execution(
                execution_id,
                {
                    "state": dict(final_state),
                    "completed_nodes": final_state.get("completed_nodes", []),
                },
            )

            # Update final status and send notifications
            if final_state.get("errors"):
                error_msg = "; ".join(final_state.get("errors", []))
                self.db.update_execution_status(
                    execution_id, WorkflowStatus.FAILED, error_message=error_msg
                )
                await notify_workflow_failed(
                    self._notification_service,
                    role_name,
                    execution_id,
                    error=error_msg,
                    failed_node=final_state.get("current_node"),
                )
            else:
                self.db.update_execution_status(execution_id, WorkflowStatus.COMPLETED)
                await notify_workflow_completed(
                    self._notification_service,
                    role_name,
                    execution_id,
                    summary=final_state.get("summary", {}),
                )

            return {
                "status": "completed" if not final_state.get("errors") else "failed",
                "execution_id": execution_id,
                "state": dict(final_state),
                "summary": final_state.get("summary"),
            }

        except Exception as e:
            self.db.update_execution_status(
                execution_id, WorkflowStatus.FAILED, error_message=str(e)
            )
            await notify_workflow_failed(
                self._notification_service,
                role_name,
                execution_id,
                error=str(e),
                failed_node="unknown",
            )

            # Try to update issue with failure info if an issue exists
            # This handles unexpected errors that bypass the normal failure path
            issue_iid = initial_state.get("issue_iid")
            if issue_iid:
                try:
                    from harness.gitlab.api import GitLabClient

                    client = GitLabClient(self.db)
                    client.update_issue_on_failure(issue_iid, str(e))
                except Exception:
                    pass  # Best effort - don't let this cause additional failures

            return {"status": "error", "execution_id": execution_id, "error": str(e)}

    # =========================================================================
    # DAG Modification Entrypoints
    # =========================================================================

    def modify_edge(self, from_node: str, condition: str, new_target: str) -> bool:
        """
        Store an edge modification to be applied on next graph build.

        Args:
            from_node: Source node name
            condition: Edge condition (e.g., "success", "failure", "default")
            new_target: New target node name

        Returns:
            True if modification was stored successfully
        """
        if not hasattr(self, "_edge_modifications"):
            self._edge_modifications: list[dict] = []

        self._edge_modifications.append(
            {
                "type": "modify",
                "from": from_node,
                "condition": condition,
                "target": new_target,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return True

    def insert_node_after(
        self,
        existing_node: str,
        new_node: str,
        node_func: Callable[[BoxUpRoleState], dict],
        description: str,
    ) -> bool:
        """
        Store a node insertion to be applied on next graph build.

        Args:
            existing_node: Node to insert after
            new_node: Name for the new node
            node_func: Function to execute for the new node
            description: Node description for documentation

        Returns:
            True if insertion was stored successfully
        """
        if not hasattr(self, "_node_insertions"):
            self._node_insertions: list[dict] = []

        self._node_insertions.append(
            {
                "after": existing_node,
                "name": new_node,
                "func": node_func,
                "description": description,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return True

    def remove_node(self, node_name: str) -> bool:
        """
        Store a node removal to be applied on next graph build.

        Args:
            node_name: Node to remove

        Returns:
            True if removal was stored successfully
        """
        if not hasattr(self, "_node_removals"):
            self._node_removals: list[str] = []

        if node_name not in self._node_removals:
            self._node_removals.append(node_name)
        return True

    def export_graph(self) -> dict:
        """
        Export the graph definition as a serializable dict.

        Returns:
            Dict containing graph structure and pending modifications
        """
        # Get node list from the default node order
        default_nodes = [
            "validate_role",
            "analyze_deps",
            "check_reverse_deps",
            "create_worktree",
            "run_molecule",
            "run_pytest",
            "validate_deploy",
            "create_commit",
            "push_branch",
            "create_issue",
            "create_mr",
            "add_to_merge_train",
            "report_summary",
            "notify_failure",
        ]

        return {
            "workflow_type": "box_up_role",
            "entry_point": "validate_role",
            "nodes": default_nodes,
            "pending_modifications": {
                "edge_modifications": getattr(self, "_edge_modifications", []),
                "node_insertions": [
                    {
                        "after": ins["after"],
                        "name": ins["name"],
                        "description": ins["description"],
                        "timestamp": ins["timestamp"],
                    }
                    for ins in getattr(self, "_node_insertions", [])
                ],
                "node_removals": getattr(self, "_node_removals", []),
            },
            "exported_at": datetime.utcnow().isoformat(),
        }

    def import_graph(self, definition: dict) -> None:
        """
        Import a graph definition from a serialized dict.

        Note: Node functions cannot be serialized, so insertions
        from imported definitions will need their functions re-attached.

        Args:
            definition: Previously exported graph definition
        """
        mods = definition.get("pending_modifications", {})

        self._edge_modifications = mods.get("edge_modifications", [])
        self._node_removals = mods.get("node_removals", [])

        # For insertions, we store the metadata but func must be re-attached
        self._node_insertions = [
            {
                "after": ins["after"],
                "name": ins["name"],
                "func": None,  # Must be re-attached via register_node_func
                "description": ins["description"],
                "timestamp": ins.get("timestamp", datetime.utcnow().isoformat()),
            }
            for ins in mods.get("node_insertions", [])
        ]

    def register_node_func(self, node_name: str, func: Callable[[BoxUpRoleState], dict]) -> bool:
        """
        Register a function for a node (used after import_graph).

        Args:
            node_name: Name of the node
            func: Function to execute for this node

        Returns:
            True if registration was successful
        """
        if not hasattr(self, "_node_insertions"):
            return False

        for insertion in self._node_insertions:
            if insertion["name"] == node_name:
                insertion["func"] = func
                return True
        return False

    def clear_modifications(self) -> None:
        """Clear all pending modifications."""
        self._edge_modifications = []
        self._node_insertions = []
        self._node_removals = []

    def get_pending_modifications_count(self) -> dict[str, int]:
        """Get count of pending modifications by type."""
        return {
            "edge_modifications": len(getattr(self, "_edge_modifications", [])),
            "node_insertions": len(getattr(self, "_node_insertions", [])),
            "node_removals": len(getattr(self, "_node_removals", [])),
        }


__all__ = [
    "LangGraphWorkflowRunner",
]
