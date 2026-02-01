"""Pydantic models for database entities."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# Custom exceptions
class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected in the role graph."""

    def __init__(self, message: str, cycle_path: Optional[list[str]] = None):
        super().__init__(message)
        self.cycle_path = cycle_path or []

    def __str__(self) -> str:
        if self.cycle_path:
            return f"{self.args[0]}"
        return super().__str__()


class DependencyType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CREDENTIAL = "credential"


class WorktreeStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    DIRTY = "dirty"
    MERGED = "merged"
    PRUNED = "pruned"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TestType(str, Enum):
    MOLECULE = "molecule"
    PYTEST = "pytest"
    DEPLOY = "deploy"


class TestStatus(str, Enum):
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Role(BaseModel):
    """Ansible role with wave placement and metadata."""
    id: Optional[int] = None
    name: str
    wave: int = Field(ge=0, le=4)
    wave_name: Optional[str] = None
    description: Optional[str] = None
    molecule_path: Optional[str] = None
    has_molecule_tests: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RoleDependency(BaseModel):
    """Dependency relationship between roles."""
    id: Optional[int] = None
    role_id: int
    depends_on_id: int
    dependency_type: DependencyType
    source_file: Optional[str] = None
    created_at: Optional[datetime] = None


class Credential(BaseModel):
    """Credential requirement for a role."""
    id: Optional[int] = None
    role_id: int
    entry_name: str
    purpose: Optional[str] = None
    is_base58: bool = False
    attribute: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    created_at: Optional[datetime] = None


class Worktree(BaseModel):
    """Git worktree for parallel development."""
    id: Optional[int] = None
    role_id: int
    path: str
    branch: str
    base_commit: Optional[str] = None
    current_commit: Optional[str] = None
    commits_ahead: int = 0
    commits_behind: int = 0
    uncommitted_changes: int = 0
    status: WorktreeStatus = WorktreeStatus.ACTIVE
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Iteration(BaseModel):
    """GitLab iteration."""
    id: int
    title: Optional[str] = None
    state: str = "opened"
    start_date: Optional[str] = None
    due_date: Optional[str] = None
    group_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Issue(BaseModel):
    """GitLab issue."""
    id: int
    iid: int
    role_id: Optional[int] = None
    iteration_id: Optional[int] = None
    title: str
    state: str = "opened"
    web_url: Optional[str] = None
    labels: Optional[str] = None
    assignee: Optional[str] = None
    weight: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MergeRequest(BaseModel):
    """GitLab merge request."""
    id: int
    iid: int
    role_id: Optional[int] = None
    issue_id: Optional[int] = None
    source_branch: str
    target_branch: str = "main"
    title: str
    state: str = "opened"
    web_url: Optional[str] = None
    merge_status: Optional[str] = None
    squash_on_merge: bool = True
    remove_source_branch: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WorkflowDefinition(BaseModel):
    """DAG workflow definition."""
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    nodes_json: str
    edges_json: str
    created_at: Optional[datetime] = None


class WorkflowExecution(BaseModel):
    """Workflow execution instance."""
    id: Optional[int] = None
    workflow_id: int
    role_id: int
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_node: Optional[str] = None
    checkpoint_data: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class NodeExecution(BaseModel):
    """Individual node execution within a workflow."""
    id: Optional[int] = None
    execution_id: int
    node_name: str
    status: NodeStatus = NodeStatus.PENDING
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    created_at: Optional[datetime] = None


class TestRun(BaseModel):
    """Test execution record."""
    id: Optional[int] = None
    role_id: int
    worktree_id: Optional[int] = None
    execution_id: Optional[int] = None
    test_type: TestType
    status: TestStatus
    duration_seconds: Optional[int] = None
    log_path: Optional[str] = None
    output_json: Optional[str] = None
    commit_sha: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class TestCase(BaseModel):
    """Individual test case result."""
    id: Optional[int] = None
    test_run_id: int
    name: str
    status: str
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    failure_output: Optional[str] = None
    created_at: Optional[datetime] = None


class RoleStatusView(BaseModel):
    """Aggregated role status from v_role_status view."""
    id: int
    name: str
    wave: int
    wave_name: Optional[str] = None
    worktree_status: Optional[str] = None
    commits_ahead: Optional[int] = None
    commits_behind: Optional[int] = None
    issue_state: Optional[str] = None
    issue_url: Optional[str] = None
    mr_state: Optional[str] = None
    mr_url: Optional[str] = None
    passed_tests: int = 0
    failed_tests: int = 0


# ============================================================================
# SEE/ACP CONTEXT CONTROL MODELS
# ============================================================================

class CapabilityStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


class ExecutionContext(BaseModel):
    """MCP client session execution context."""
    id: Optional[int] = None
    session_id: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    capabilities: Optional[str] = None  # JSON array
    metadata: Optional[str] = None  # JSON object
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class ContextCapability(BaseModel):
    """Fine-grained capability grant for a context."""
    id: Optional[int] = None
    context_id: int
    capability: str  # e.g., 'write:roles', 'execute:molecule'
    scope: Optional[str] = None  # Optional scope restriction
    granted_by: str = "system"
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


class ToolInvocation(BaseModel):
    """Tool invocation tracking for telemetry."""
    id: Optional[int] = None
    context_id: Optional[int] = None
    tool_name: str
    arguments: Optional[str] = None  # JSON object
    result: Optional[str] = None  # JSON object or error
    status: str = "pending"
    duration_ms: Optional[int] = None
    blocked_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ============================================================================
# TEST REGRESSION MODELS
# ============================================================================

class RegressionStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    FLAKY = "flaky"
    KNOWN_ISSUE = "known_issue"


class TestRegression(BaseModel):
    """Test regression tracking."""
    id: Optional[int] = None
    role_id: int
    test_name: str
    test_type: Optional[TestType] = None
    first_failure_run_id: Optional[int] = None
    resolved_run_id: Optional[int] = None
    failure_count: int = 1
    consecutive_failures: int = 1
    last_failure_at: Optional[datetime] = None
    last_error_message: Optional[str] = None
    status: RegressionStatus = RegressionStatus.ACTIVE
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MergeTrainStatus(str, Enum):
    QUEUED = "queued"
    MERGING = "merging"
    MERGED = "merged"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MergeTrainEntry(BaseModel):
    """Merge train queue entry."""
    id: Optional[int] = None
    mr_id: int
    position: Optional[int] = None
    target_branch: str = "main"
    status: MergeTrainStatus = MergeTrainStatus.QUEUED
    pipeline_id: Optional[int] = None
    pipeline_status: Optional[str] = None
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ActiveRegressionView(BaseModel):
    """Active regression from v_active_regressions view."""
    id: int
    role_name: str
    wave: int
    test_name: str
    test_type: Optional[str] = None
    failure_count: int
    consecutive_failures: int
    last_failure_at: Optional[datetime] = None
    last_error_message: Optional[str] = None
    status: str
    notes: Optional[str] = None


# ============================================================================
# AGENT SESSION MODELS (HOTL Claude Code Integration)
# ============================================================================

class AgentSessionStatus(str, Enum):
    """Status of a Claude Code agent session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"
    CANCELLED = "cancelled"


class AgentFileChangeType(str, Enum):
    """Type of file change made by an agent."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class AgentSessionModel(BaseModel):
    """Database model for agent sessions."""
    id: str  # UUID string
    execution_id: Optional[int] = None
    task: str
    status: AgentSessionStatus = AgentSessionStatus.PENDING
    output: Optional[str] = None
    error_message: Optional[str] = None
    intervention_reason: Optional[str] = None
    context_json: Optional[str] = None
    progress_json: Optional[str] = None
    working_dir: str
    pid: Optional[int] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentFileChange(BaseModel):
    """Database model for agent file changes."""
    id: Optional[int] = None
    session_id: str
    file_path: str
    change_type: AgentFileChangeType
    diff: Optional[str] = None
    old_path: Optional[str] = None
    created_at: Optional[datetime] = None


class AgentSessionView(BaseModel):
    """View model for agent sessions with computed fields."""
    id: str
    execution_id: Optional[int] = None
    task: str
    status: str
    working_dir: str
    intervention_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    pid: Optional[int] = None
    file_change_count: int = 0
    duration_seconds: Optional[float] = None
