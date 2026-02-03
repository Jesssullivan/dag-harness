# Architecture

This document describes the internal architecture of dag-harness, a LangGraph-based
DAG orchestration system for Ansible role deployments.

## System Overview

```
+-------------------------------------------------------------+
|                      Claude Code                            |
|  +-----------+  +-----------+  +--------------------------+ |
|  |  Skills   |  |   Hooks   |  |      MCP Server          | |
|  |/box-up-   |  |PreToolUse |  |    (dag-harness)         | |
|  |  role     |  |PostToolUse|  |                          | |
|  +-----------+  +-----------+  +------------+-------------+ |
+----------------------------------------------|--------------+
                                               |
+----------------------------------------------|--------------+
|                  Harness Core                |              |
|  +-------------------------------------------v-----------+  |
|  |                   CLI (Typer/Rich)                     |  |
|  |  box-up-role | status | sync | hotl | bootstrap       |  |
|  +----------------------------------------------------+  |  |
|                           |                                 |
|  +------------------------v-----------------------------+   |
|  |              DAG Engine (LangGraph)                  |   |
|  |  +----------+  +----------+  +----------+           |   |
|  |  | validate |->| analyze  |->| worktree |-> ...     |   |
|  |  +----------+  +----------+  +----------+           |   |
|  |              SqliteSaver (checkpointing)             |   |
|  +------------------------------------------------------+   |
|                           |                                 |
|  +------------------------v-----------------------------+   |
|  |                  StateDB (SQLite)                    |   |
|  |  roles | worktrees | test_runs | executions | store  |   |
|  +------------------------------------------------------+   |
+-------------------------------------------------------------+
```

## LangGraph Workflow Engine

The core execution engine is built on [LangGraph](https://langchain-ai.github.io/langgraph/),
providing reliable state machine execution with built-in checkpointing.

### Key Design Decisions

- **LangGraph 1.0.x** -- Uses stable release with RetryPolicy support
- **Async execution** -- All workflow nodes are async functions
- **TypedDict state** -- Schema-driven state with Annotated reducers
- **SqliteSaver** -- Persistent checkpoints for resume capability
- **Conditional routing** -- Smart branching based on test results and failures

### Graph Compilation

The workflow graph is defined using `StateGraph` and compiled with a checkpointer:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

graph = StateGraph(BoxUpRoleState)
graph.add_node("validate_role", validate_role_node)
graph.add_node("analyze_deps", analyze_deps_node)
# ... more nodes

graph.add_edge("validate_role", "analyze_deps")
graph.add_conditional_edges("run_molecule", route_after_tests)

compiled = graph.compile(checkpointer=AsyncSqliteSaver.from_conn_string("harness.db"))
```

### RetryPolicy

External API and subprocess nodes use LangGraph's `RetryPolicy` for resilience:

```python
from langgraph.types import RetryPolicy

graph.add_node(
    "create_mr",
    create_mr_node,
    retry_policy=RetryPolicy(max_attempts=3)
)
```

Default behavior retries on 5xx HTTP errors and skips 4xx client errors.

## State Management

### BoxUpRoleState

The primary state schema for the `box-up-role` workflow:

```python
class BoxUpRoleState(TypedDict, total=False):
    role_name: str
    execution_id: Optional[int]
    worktree_path: str
    molecule_passed: bool
    pytest_passed: bool
    mr_url: Optional[str]
    issue_url: Optional[str]
    errors: Annotated[list[str], operator.add]
    state_snapshots: Annotated[list[dict], keep_last_n(10)]
```

### Annotated Reducers

LangGraph uses `Annotated` types to define how state fields are merged across
node executions:

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `operator.add` | Append new items to list | `errors` -- accumulate all errors |
| `keep_last_n(n)` | Keep only last N items | `state_snapshots` -- rolling history |
| Default | Last write wins | Scalar fields like `role_name` |

The `keep_last_n` reducer prevents unbounded state growth for list fields that
record history.

### Node Functions

Each node is an async function that receives state and returns a partial update:

```python
async def validate_role_node(state: BoxUpRoleState) -> BoxUpRoleState:
    role_path = Path(state["repo_root"]) / "ansible" / "roles" / state["role_name"]
    if not role_path.exists():
        return {"errors": [f"Role not found: {state['role_name']}"]}
    return {"role_path": str(role_path), "has_molecule_tests": True}
```

Nodes return only the fields they modify. The LangGraph engine merges updates
using the configured reducers.

## Dual Persistence

dag-harness uses two complementary persistence layers:

### LangGraph Checkpointer

- **Purpose**: Workflow state for pause/resume
- **Backend**: SQLite (default) or PostgreSQL (production)
- **Data**: Full graph state at each checkpoint
- **Access**: LangGraph internal APIs

### StateDB (Application Database)

- **Purpose**: Application state, metadata, and observability
- **Backend**: SQLite with WAL mode
- **Data**: Roles, worktrees, test runs, issues, MRs, metrics
- **Access**: Python API via `StateDB` class

### CheckpointerWithStateDB

The `CheckpointerWithStateDB` class bridges both systems:

```python
async with CheckpointerWithStateDB(db, sqlite_path="harness.db") as checkpointer:
    compiled = graph.compile(checkpointer=checkpointer)
    await compiled.ainvoke(initial_state, config)
```

On every `put()` call, it:

1. Delegates to the underlying LangGraph checkpointer (source of truth)
2. Serializes checkpoint data to `workflow_executions.checkpoint_data` in StateDB
3. Gracefully degrades if StateDB sync fails

### CheckpointerFactory

For production deployments, the `CheckpointerFactory` manages backend selection:

```python
from harness.dag.checkpointer import CheckpointerFactory

# Async factory (preferred) -- auto-detects PostgreSQL or falls back to SQLite
checkpointer = await CheckpointerFactory.create_async()

# Includes connection pooling, health checks, and migration utilities
```

## Cross-Thread Memory (HarnessStore)

The `HarnessStore` implements LangGraph's `BaseStore` interface, enabling
cross-thread memory persistence backed by StateDB.

### Namespace Structure

| Namespace | Example | Purpose |
|-----------|---------|---------|
| `("roles", "<name>")` | `("roles", "common")` | Role-specific memory |
| `("waves", "<n>")` | `("waves", "0")` | Wave-level coordination |
| `("users", "<id>")` | `("users", "jsullivan2")` | User preferences |
| `("workflows", "<type>")` | `("workflows", "box-up-role")` | Workflow execution memory |

### Usage

```python
from harness.dag.store import HarnessStore
from harness.db.state import StateDB

db = StateDB("harness.db")
store = HarnessStore(db)

# Store data
store.put(("roles", "common"), "status", {"last_run": "2026-02-03", "passed": True})

# Retrieve data
item = store.get(("roles", "common"), "status")
print(item.value)  # {"last_run": "2026-02-03", "passed": True}

# Search within namespace
results = store.search(("roles",), filter={"passed": True})
```

The store is backed by the `store_items` table in SQLite with JSON-encoded
namespace tuples and values.

## GitLab Integration

### Idempotent Operations

All GitLab operations (issue creation, MR creation, merge train) are idempotent.
The system checks for existing resources before creating new ones, using the
role name and branch as deduplication keys.

### Retry Strategy

GitLab API calls use exponential backoff with jitter:

- HTTP 429 (rate limited): Wait and retry
- HTTP 5xx (server error): Retry up to 3 times
- HTTP 4xx (client error): Fail immediately

### Merge Trains

Merge requests are queued into GitLab merge trains for ordered integration.
The `merge_train_entries` table tracks queue position, pipeline status, and
completion state.

## Wave-Based Parallel Execution

The `ParallelWaveExecutor` processes roles in wave order:

```
Wave 0: [common]                    -- Sequential (foundation)
Wave 1: [windows_prerequisites, ems_registry_urls, iis-config]  -- Parallel
Wave 2: [ems_platform_services, ems_web_app, database_clone]    -- Parallel
Wave 3: [ems_master_calendar, ems_campus_webservice, ...]       -- Parallel
Wave 4: [grafana_alloy_windows, email_infrastructure, ...]      -- Parallel
```

Within each wave, roles execute in parallel (configurable via `parallel: false`
in wave definitions). The executor waits for all roles in a wave to complete
before advancing to the next wave.

### Parallel Test Execution

Within a single role workflow, molecule and pytest tests run in parallel using
LangGraph's `Send` API:

```python
from langgraph.types import Send

def route_to_parallel_tests(state):
    sends = []
    if state.get("has_molecule_tests"):
        sends.append(Send("run_molecule", state))
    if state.get("has_pytest_tests"):
        sends.append(Send("run_pytest", state))
    return sends
```

The `merge_test_results_node` aggregates results from parallel test branches.
Target improvement: 30%+ time reduction for the test phase when both test
types are available.

## Interrupt/Resume Workflow

### Human-in-the-Loop (HITL)

The workflow supports pausing at any node for human review:

```python
from langgraph.types import interrupt

async def human_approval_node(state: BoxUpRoleState) -> BoxUpRoleState:
    response = interrupt({"question": "Approve merge request creation?"})
    return {"human_approved": response.get("approved", False)}
```

### Static Breakpoints

Default breakpoints pause before critical operations:

```python
DEFAULT_BREAKPOINTS = [
    "human_approval",      # Always pause
    "create_mr",           # Before MR creation
    "add_to_merge_train",  # Before merge train
]
```

Enable via environment variable:

```bash
export HARNESS_BREAKPOINTS=true
```

Or per-execution:

```bash
harness box-up-role common --breakpoints run_molecule,create_mr
```

### Resume

Paused workflows store their full state in the checkpointer. Resume with:

```bash
harness resume <execution-id>
```

The `Command(resume=...)` pattern provides human input back to the paused node.

## Database Schema

The SQLite database uses an adjacency list pattern with recursive CTEs for
dependency traversal. Key tables:

| Table | Purpose |
|-------|---------|
| `roles` | Ansible role metadata and wave placement |
| `role_dependencies` | Dependency graph (adjacency list) |
| `worktrees` | Git worktree tracking |
| `workflow_executions` | Workflow execution history and checkpoints |
| `node_executions` | Per-node execution records |
| `test_runs` | Test execution history |
| `test_regressions` | Regression detection and tracking |
| `issues` | GitLab issue metadata |
| `merge_requests` | GitLab MR metadata |
| `merge_train_entries` | Merge train queue state |
| `store_items` | Cross-thread memory (HarnessStore) |
| `token_usage` | Cost tracking per session |
| `audit_log` | Mutation audit trail |
| `agent_sessions` | HOTL Claude Code session tracking |
| `execution_contexts` | Context control for session management |

### Graph Traversal

Dependencies are queried using recursive CTEs:

```sql
-- Get all transitive dependencies of a role
WITH RECURSIVE deps(role_id, depth) AS (
    SELECT depends_on_id, 1 FROM role_dependencies WHERE role_id = ?
    UNION ALL
    SELECT rd.depends_on_id, d.depth + 1
    FROM role_dependencies rd
    JOIN deps d ON rd.role_id = d.role_id
    WHERE d.depth < 10
)
SELECT DISTINCT r.name, d.depth
FROM deps d JOIN roles r ON d.role_id = r.id
ORDER BY d.depth;
```

### Views

Pre-built views for common queries:

| View | Purpose |
|------|---------|
| `v_role_status` | Role status with worktree, issue, MR, and test counts |
| `v_dependency_graph` | Dependency edges for visualization |
| `v_active_regressions` | Active and flaky test regressions |
| `v_context_capabilities` | Session capabilities with expiration status |
| `v_agent_sessions` | HOTL sessions with file change counts |
| `v_daily_costs` | Token usage cost by day and model |
| `v_session_costs` | Token usage cost by session |
