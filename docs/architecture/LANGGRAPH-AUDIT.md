# LangGraph Implementation Audit

**Date**: 2026-02-03 | **Sprint**: Week 1, Day 1 | **Status**: Complete

## Executive Summary

The dag-harness project has a **mature LangGraph implementation** with most SOTA patterns already in place. Key findings:

| Feature | Status | Notes |
|---------|--------|-------|
| Native Checkpointer | ✅ Implemented | Uses AsyncSqliteSaver/PostgresSaver |
| Reducer-Driven State | ✅ Implemented | 9 fields use `operator.add` |
| HOTL with interrupt() | ✅ Implemented | human_approval_node uses interrupt() |
| Command(resume=...) | ✅ Implemented | CLI resume uses typed payloads |
| Send API Parallelism | ✅ Implemented | Test fan-out uses Send API |
| Custom StateDB | ⚠️ Dual Persistence | Separate from LangGraph checkpoints |
| Store Interface | ❌ Not Implemented | No cross-thread memory |

**Primary Gap**: Dual persistence (LangGraph checkpointer + custom StateDB) creates complexity. Consider unifying via custom checkpointer wrapper.

---

## Checkpoint Storage

### 1. Native LangGraph Checkpointer Usage

**Yes, native checkpointers are used.** From `checkpointer.py`:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
```

The `CheckpointerFactory` creates either `AsyncSqliteSaver` or `SqliteSaver` as fallback when PostgreSQL is unavailable. It also supports `AsyncPostgresSaver`/`PostgresSaver` when configured.

### 2. Dual Persistence Architecture

**Yes, there is dual persistence causing complexity:**

| System | File | Database | Purpose |
|--------|------|----------|---------|
| LangGraph Checkpointer | `checkpointer.py` | `harness.db` | Graph state checkpoints |
| Custom StateDB | `state.py` | `harness.db` (shared) | Application domain state |

Both use `harness.db` but maintain separate schemas:
- LangGraph creates: `checkpoints`, `checkpoint_writes`
- StateDB creates: 21 application-specific tables

The `workflow_executions.checkpoint_data` column in StateDB duplicates some LangGraph checkpoint functionality.

### 3. Checkpointer Factory Structure

| Method | Returns | Behavior |
|--------|---------|----------|
| `create_async()` | `BaseCheckpointSaver` | Prefers PostgreSQL, falls back to SQLite |
| `create_sync()` | `BaseCheckpointSaver` | Same preference chain, sync version |

Additional utilities:
- `CheckpointerContext` - Async context manager for lifecycle
- `check_postgres_health()` - Health check with latency metrics
- `cleanup_old_checkpoints()` - Removes checkpoints older than N days
- `migrate_sqlite_to_postgres()` - One-way migration utility

### 4. Tables Storing Workflow State

**LangGraph Tables:**
| Table | Purpose |
|-------|---------|
| `checkpoints` | Graph state snapshots (thread_id, checkpoint_id, blob) |
| `checkpoint_writes` | Write operations for incremental updates |

**StateDB Tables (21 total):**
| Table | Purpose |
|-------|---------|
| `workflow_definitions` | DAG structure (nodes_json, edges_json) |
| `workflow_executions` | Execution instances with `checkpoint_data` |
| `node_executions` | Individual node execution records |
| `roles` | Ansible roles being processed |
| `worktrees` | Git worktree state per role |
| `test_runs` / `test_cases` | Test execution results |
| `agent_sessions` | Claude Code subagent sessions for HOTL |
| `token_usage` | Cost tracking per session |
| `audit_log` | All state mutations |

---

## State Schema and Reducers

### BoxUpRoleState Definition

The `BoxUpRoleState` is a `TypedDict` with `total=False` (all fields optional) containing 45 fields organized into categories:

### Fields Using `Annotated[list, operator.add]` Reducers (9 total)

| Field | Type | Purpose |
|-------|------|---------|
| `explicit_deps` | `Annotated[list[str], operator.add]` | Direct dependencies |
| `implicit_deps` | `Annotated[list[str], operator.add]` | Transitive dependencies |
| `reverse_deps` | `Annotated[list[str], operator.add]` | Roles that depend on this |
| `credentials` | `Annotated[list[dict], operator.add]` | Required credentials |
| `tags` | `Annotated[list[str], operator.add]` | Ansible tags |
| `blocking_deps` | `Annotated[list[str], operator.add]` | Deps blocking execution |
| `parallel_tests_completed` | `Annotated[list[str], operator.add]` | Completed test names |
| `completed_nodes` | `Annotated[list[str], operator.add]` | Workflow progress |
| `errors` | `Annotated[list[str], operator.add]` | Accumulated errors |

### Field Categories

**Role Identification**: `role_name`, `execution_id`

**Role Metadata**: `role_path`, `has_molecule_tests`, `has_meta`, `wave`, `wave_name`

**Worktree/Git State**: `worktree_path`, `branch`, `commit_sha`, `commit_message`, `pushed`

**Test Results**: `molecule_passed`, `molecule_skipped`, `molecule_duration`, `molecule_output`, `pytest_passed`, `pytest_skipped`, `pytest_duration`, `deploy_passed`, `deploy_skipped`

**Parallel Execution**: `all_tests_passed`, `test_phase_start_time`, `test_phase_duration`, `parallel_execution_enabled`

**GitLab Integration**: `issue_url`, `issue_iid`, `issue_created`, `mr_url`, `mr_iid`, `mr_created`, `reviewers_set`, `iteration_assigned`, `merge_train_status`, `branch_existed`

**Workflow Control**: `current_node`, `summary`

**Human-in-the-Loop**: `human_approved`, `human_rejection_reason`, `awaiting_human_input`

### Missing Reducers Assessment

**All list fields that need accumulation already have `operator.add` reducers.** The schema is well-designed.

### Custom Reducers

**None implemented.** The codebase exclusively uses the built-in `operator.add` reducer.

---

## HOTL Implementation

### 1. interrupt() Usage

The `interrupt()` function is used in `human_approval_node` (lines 1131-1206):

```python
from langgraph.types import interrupt

async def human_approval_node(state: BoxUpRoleState) -> dict:
    approval = interrupt({
        "question": f"Approve merge train for role '{role_name}'?",
        "context": approval_context,
        "instructions": (
            f"Review the MR at {mr_url} and approve or reject.\n"
            f"Resume with: harness resume <execution_id> --approve\n"
            f"Or reject:   harness resume <execution_id> --reject --reason '...'"
        ),
    })

    if isinstance(approval, dict):
        approved = approval.get("approved", False)
        reason = approval.get("reason", "")
    # ...
```

### 2. Approval Gates Location

The approval gate is at the **MR creation to merge train transition**:

```
create_mr → human_approval → add_to_merge_train
                 ↓ (rejected)
            notify_failure
```

Routing logic:
- `should_continue_after_mr()` → routes to `human_approval` if MR exists
- `should_continue_after_human_approval()` → routes to `add_to_merge_train` if approved

### 3. Command(resume=...) Pattern

**Yes, fully implemented** in CLI's resume handler:

```python
async def _resume_with_command(db, execution_id: int, role_name: str, resume_value: dict):
    from langgraph.types import Command

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        thread_config = {"configurable": {"thread_id": str(execution_id)}}
        result = await compiled.ainvoke(Command(resume=resume_value), config=thread_config)
```

Resume value structure: `{"approved": bool, "reason": str}`

### 4. CLI Resumption Commands

```bash
harness resume 123 --approve          # Approve and continue
harness resume 123 --reject           # Reject without reason
harness resume 123 --reject --reason "Tests need more coverage"
```

### 5. HOTL Supervisor (Separate System)

The HOTL Supervisor in `/harness/hotl/supervisor.py` implements autonomous operation using a **separate LangGraph workflow** with:
- State flags (`stop_requested`, `pause_requested`) instead of interrupt()
- Phase tracking: IDLE → RESEARCHING → PLANNING → EXECUTING → TESTING → etc.
- External control via `request_stop()`, `request_pause()` methods

---

## Parallel Execution

### 1. Wave-Level Execution

**Waves execute sequentially; roles within each wave execute in parallel:**

```python
# From parallel.py
semaphore = asyncio.Semaphore(self.max_concurrent)  # Default: 3
tasks = [self._execute_role(role, wave, semaphore, progress) for role in roles]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 2. LangGraph Send API Usage

**Yes, Send API is used for test fan-out:**

```python
def route_to_parallel_tests(state: BoxUpRoleState) -> list[Send]:
    sends = []
    if has_molecule:
        sends.append(Send("run_molecule", {**state, "test_phase_start_time": test_phase_start}))
    sends.append(Send("run_pytest", {**state, "test_phase_start_time": test_phase_start}))
    return sends
```

Graph wiring:
```python
graph.add_conditional_edges("create_worktree", route_to_parallel_tests,
                            ["run_molecule", "run_pytest", "merge_test_results"])
graph.add_edge("run_molecule", "merge_test_results")
graph.add_edge("run_pytest", "merge_test_results")
```

### 3. Test Result Merging

The `merge_test_results_node()` function:
- Reads individual test results from state
- Calculates combined pass/fail: `all_passed = molecule_ok and pytest_ok`
- Computes performance metrics (time saved vs sequential)
- Target: 30%+ time reduction when both test types exist

### 4. Summary

| Level | Mechanism | Concurrency Control |
|-------|-----------|---------------------|
| Waves | Sequential loop | None - one at a time |
| Roles within wave | `asyncio.gather()` | `asyncio.Semaphore(3)` |
| Tests within role | LangGraph `Send` API | LangGraph handles internally |

---

## Graph Structure

### Node Functions (16 total)

| Node | Purpose | Retry Policy |
|------|---------|--------------|
| `validate_role_node` | Validate role existence, extract metadata | None |
| `analyze_deps_node` | Analyze dependencies via StateDB | None |
| `check_reverse_deps_node` | Check reverse deps are boxed | None |
| `create_worktree_node` | Create git worktree | None |
| `run_molecule_node` | Run molecule tests | SUBPROCESS_RETRY |
| `run_pytest_node` | Run pytest tests | SUBPROCESS_RETRY |
| `merge_test_results_node` | Merge parallel test results | None |
| `validate_deploy_node` | Validate deployment config | None |
| `create_commit_node` | Create git commit | None |
| `push_branch_node` | Push to origin | GIT_RETRY |
| `create_issue_node` | Create GitLab issue (idempotent) | GITLAB_API_RETRY |
| `create_mr_node` | Create GitLab MR (idempotent) | GITLAB_API_RETRY |
| `human_approval_node` | HITL approval gate | None |
| `add_to_merge_train_node` | Add MR to merge train | GITLAB_API_RETRY |
| `report_summary_node` | Generate final summary | None |
| `notify_failure_node` | Handle failure notification | None |

### Workflow Diagram

```
                              ┌─────────────────┐
                              │  validate_role  │ ◀── ENTRY
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌──────────────┐
                              │ analyze_deps │
                              └──────┬───────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │check_reverse_deps │
                           └─────────┬─────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ create_worktree │
                            └────────┬────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     │      (Send API fan-out)       │
                     ▼                               ▼
            ┌──────────────┐               ┌──────────────┐
            │ run_molecule │               │  run_pytest  │
            └──────┬───────┘               └───────┬──────┘
                   │                               │
                   └───────────┬───────────────────┘
                               ▼
                     ┌───────────────────┐
                     │merge_test_results │
                     └─────────┬─────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ validate_deploy │
                      └────────┬────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │create_commit│
                        └──────┬──────┘
                               │
                               ▼
                         ┌───────────┐
                         │push_branch│
                         └─────┬─────┘
                               │
                               ▼
                        ┌─────────────┐
                        │create_issue │
                        └──────┬──────┘
                               │
                               ▼
                          ┌─────────┐
                          │create_mr│
                          └────┬────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ human_approval   │ ◀── interrupt()
                     └────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼ (approved)                  ▼ (rejected)
      ┌───────────────────┐         ┌───────────────┐
      │add_to_merge_train │         │notify_failure │──▶ END
      └─────────┬─────────┘         └───────────────┘
                │
                ▼
         ┌───────────────┐
         │report_summary │
         └───────┬───────┘
                 │
                 ▼
                END
```

### Retry Policies

| Policy | Max Attempts | Initial Interval | Backoff |
|--------|--------------|------------------|---------|
| `GITLAB_API_RETRY_POLICY` | 3 | 1.0s | 2.0x |
| `SUBPROCESS_RETRY_POLICY` | 2 | 5.0s | 1.0x |
| `GIT_RETRY_POLICY` | 3 | 2.0s | 2.0x |

---

## Database Schema and Models

### Tables (21 total)

**Core**: `roles`, `role_dependencies`, `credentials`

**Git**: `worktrees`

**GitLab**: `iterations`, `issues`, `merge_requests`, `merge_train_entries`

**Workflow**: `workflow_definitions`, `workflow_executions`, `node_executions`

**Testing**: `test_runs`, `test_cases`, `test_regressions`

**Context**: `execution_contexts`, `context_capabilities`, `tool_invocations`

**HOTL**: `agent_sessions`, `agent_file_changes`

**Tracking**: `audit_log`, `token_usage`

### Wave Configuration

| Wave | Name | Example Roles |
|------|------|---------------|
| 0 | Foundation | common |
| 1 | Infrastructure Foundation | windows_prerequisites, ems_registry_urls, iis-config |
| 2 | Core Platform | ems_platform_services, ems_web_app, database_clone |
| 3 | Web Applications | ems_master_calendar, ems_campus_webservice |
| 4 | Supporting Services | grafana_alloy_windows, email_infrastructure |

---

## Identified Gaps vs SOTA

### 1. Dual Persistence Complexity
**Gap**: LangGraph checkpoints and StateDB maintain separate state.
**Recommendation**: Create `CheckpointerWithStateDB` wrapper that coordinates both systems.

### 2. No LangGraph Store Interface
**Gap**: No cross-thread memory for sharing state between executions.
**Recommendation**: Implement `HarnessStore` using StateDB as backing store.

### 3. No Static Breakpoints
**Gap**: Only dynamic `interrupt()` is used; no `interrupt_before` compile option.
**Recommendation**: Add environment variable `HARNESS_BREAKPOINTS=true` to enable static breakpoints.

### 4. Wave Parallelism Uses asyncio, Not Send
**Gap**: Wave-level parallelism uses `asyncio.gather()` instead of LangGraph Send API.
**Recommendation**: Consider Send API for native checkpoint integration (lower priority - current implementation works).

---

## Files Analyzed

| File | Lines | Purpose |
|------|-------|---------|
| `harness/dag/langgraph_engine.py` | ~2049 | Main workflow engine |
| `harness/dag/checkpointer.py` | ~300 | Checkpointer factory |
| `harness/dag/parallel.py` | ~250 | Parallel wave execution |
| `harness/db/state.py` | ~800 | StateDB implementation |
| `harness/db/schema.sql` | ~400 | Database schema |
| `harness/db/models.py` | ~500 | Pydantic models |
| `harness/hotl/supervisor.py` | ~900 | HOTL supervisor |
| `harness/cli.py` | ~600 | CLI with resume command |
| `harness.yml` | ~60 | Configuration |
