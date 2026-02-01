# V1 Harness Completion Plan

**Status**: Research Complete, Ready for Implementation
**Created**: 2026-01-30
**Goal**: Complete v1 before moving to v2 Zig/C++ rewrite

---

## Executive Summary

The v1 harness has excellent infrastructure (LangGraph, SQLite, MCP, HITL) but is missing critical business logic to execute the original vision. This plan addresses the gaps in priority order.

---

## Phase 1: Missing Scripts (CRITICAL - Blocks All Workflows)

The workflow nodes call external scripts that don't exist. These must be created first.

### 1.1 Create `scripts/create-role-worktree.sh`

**Purpose**: Create isolated git worktree for role development

```bash
#!/bin/bash
# Usage: create-role-worktree.sh <role-name> [base-branch]

ROLE_NAME="$1"
BASE_BRANCH="${2:-origin/main}"
WORKTREE_BASE="${WORKTREE_BASE:-$HOME/git/worktrees}"
BRANCH_PREFIX="${BRANCH_PREFIX:-sid/}"

# Create worktree directory
WORKTREE_PATH="${WORKTREE_BASE}/${ROLE_NAME}"
BRANCH_NAME="${BRANCH_PREFIX}${ROLE_NAME}"

# Create worktree from base branch
git worktree add -b "$BRANCH_NAME" "$WORKTREE_PATH" "$BASE_BRANCH"

# Copy essential files
cp -f .env.local "$WORKTREE_PATH/" 2>/dev/null || true
cp -f ems.kdbx "$WORKTREE_PATH/" 2>/dev/null || true

echo "WORKTREE_PATH=$WORKTREE_PATH"
echo "BRANCH_NAME=$BRANCH_NAME"
```

### 1.2 Create `scripts/create-gitlab-issues.sh`

**Purpose**: Create GitLab issue via glab CLI with proper labels

```bash
#!/bin/bash
# Usage: create-gitlab-issues.sh <role-name> <wave> <description-file>

ROLE_NAME="$1"
WAVE="$2"
DESC_FILE="$3"

# Build labels
LABELS="role,ansible,molecule,wave-${WAVE}"

# Create issue
ISSUE_URL=$(glab issue create \
  --title "feat(${ROLE_NAME}): Box up \`${ROLE_NAME}\` Ansible role" \
  --description "$(cat "$DESC_FILE")" \
  --label "$LABELS" \
  --assignee "${GITLAB_ASSIGNEE:-jsullivan2}" \
  --yes | grep -oP 'https://[^\s]+')

# Extract IID from URL
ISSUE_IID=$(echo "$ISSUE_URL" | grep -oP '\d+$')

echo "ISSUE_URL=$ISSUE_URL"
echo "ISSUE_IID=$ISSUE_IID"
```

### 1.3 Create `scripts/create-gitlab-mr.sh`

**Purpose**: Create merge request linked to issue

```bash
#!/bin/bash
# Usage: create-gitlab-mr.sh <role-name> <branch> <issue-iid> <description-file>

ROLE_NAME="$1"
BRANCH="$2"
ISSUE_IID="$3"
DESC_FILE="$4"

# Create MR
MR_URL=$(glab mr create \
  --source-branch "$BRANCH" \
  --title "feat(${ROLE_NAME}): Add \`${ROLE_NAME}\` Ansible role" \
  --description "$(cat "$DESC_FILE")" \
  --label "role,ansible,molecule" \
  --assignee "${GITLAB_ASSIGNEE:-jsullivan2}" \
  --remove-source-branch \
  --squash-before-merge \
  --yes | grep -oP 'https://[^\s]+')

MR_IID=$(echo "$MR_URL" | grep -oP '\d+$')

echo "MR_URL=$MR_URL"
echo "MR_IID=$MR_IID"
```

---

## Phase 2: Ansible Role Parser (Populates Database)

Create parser to extract credentials and dependencies from role code.

### 2.1 Create `harness/harness/parsers/ansible_parser.py`

**Extracts from:**
- `roles/<role>/meta/main.yml` - Role dependencies
- `roles/<role>/tasks/*.yml` - Task patterns, credentials
- `roles/<role>/defaults/main.yml` - Variables with credential hints
- Makefiles - Credential references

**Populates:**
- `role_dependencies` table with source_file, source_line
- `credentials` table with entry_name, purpose, source location

### 2.2 Credential Detection Patterns

```python
CREDENTIAL_PATTERNS = [
    # KeePassXC references
    r'keepass[_-]?xc|\.kdbx|kdbx_password',
    # Vault patterns
    r'ansible[_-]?vault|vault[_-]?password',
    # Direct credential references
    r'password|secret|token|api[_-]?key|private[_-]?key',
    # SQL Server specific
    r'sa_password|sql[_-]?password|mssql',
    # Environment variable patterns
    r'lookup\([\'"]env[\'"]',
]
```

---

## Phase 3: Workflow Integration (Wire It Together)

### 3.1 Add Notification Calls to Workflow

In `dag/langgraph_engine.py`, add notification calls:

```python
# On workflow failure
from harness.notifications import notify_workflow_failed

async def on_workflow_error(state, error):
    await notify_workflow_failed(
        role_name=state["role_name"],
        failed_node=state.get("current_node"),
        error=str(error),
        context=state.get("error_context")
    )
```

### 3.2 Add Merge Train to Workflow

After MR creation, add merge train step:

```python
async def add_to_merge_train_node(state: BoxUpRoleState) -> dict:
    """Add MR to merge train using full API."""
    mr_iid = state.get("mr_iid")
    if not mr_iid:
        return {"merge_train_status": "skipped"}

    client = GitLabClient(config)

    # Use comprehensive API instead of simplified glab
    if client.is_merge_train_enabled():
        result = client.add_to_merge_train(
            mr_iid=mr_iid,
            when_pipeline_succeeds=True,
            squash=True
        )
        return {
            "merge_train_status": "added",
            "merge_train_position": result.get("position")
        }
    else:
        # Fallback
        client.merge_when_pipeline_succeeds(mr_iid)
        return {"merge_train_status": "fallback"}
```

### 3.3 Fix Issue Weight

In `gitlab/api.py`, add weight via API after creation:

```python
def create_issue(self, ..., weight: Optional[int] = None):
    # ... existing glab create code ...

    # Set weight via API (glab doesn't support it)
    if weight is not None:
        self._api_put(
            f"projects/{self.config.project_path_encoded}/issues/{issue_iid}",
            {"weight": weight}
        )
```

---

## Phase 4: Claude Agent Integration (HOTL Extension)

This is the key extension to enable autonomous coding.

### 4.1 Architecture

```
HOTL Supervisor (LangGraph)
    ├── Research Phase → Query database/MCP
    ├── Plan Phase → Identify tasks
    ├── Gap Analysis → Find test failures
    ├── Agent Execution Phase (NEW)
    │   └── Spawn Claude Code subagent
    │       └── Agent writes code
    │       └── Agent runs tests
    │       └── Agent reports via MCP
    ├── Test Phase → Validate changes
    └── Notify Phase → Send status
```

### 4.2 New Files

1. **`harness/harness/hotl/claude_integration.py`**
   - `HOTLClaudeIntegration` class
   - Spawns Claude agents via SDK
   - Tracks agent sessions

2. **`harness/harness/hotl/agent_session.py`**
   - `AgentSessionManager` class
   - Session state tracking
   - Progress polling

3. **Database schema additions:**
   ```sql
   CREATE TABLE agent_sessions (...);
   CREATE TABLE human_interventions (...);
   CREATE TABLE agent_file_changes (...);
   ```

4. **MCP tools for agent feedback:**
   - `agent_report_progress`
   - `agent_request_intervention`
   - `agent_log_file_operation`

### 4.3 Claude Agent Skill

Create `.claude/skills/hotl-autonomous-coding/SKILL.md`:

```markdown
# HOTL Autonomous Coding Skill

Teaches Claude how to fix failing tests and address code gaps
as part of the HOTL autonomous workflow.

## Guidelines
- Use MCP tools to understand state
- Make targeted changes only
- Run tests after each change
- Report progress frequently
- Request intervention on ambiguity
```

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests

- Parser tests with sample role files
- GitLab API mock tests
- Workflow integration tests

### 5.2 Integration Tests

- End-to-end box-up-role with test role
- HOTL cycle with mock agent
- Notification delivery tests

### 5.3 Tinyland Validation

Per original requirement:
> "I'd prefer to be able to test this pattern out in my tinyland ems upstream before pushing to my bates-ils upstream"

Test against tinyland before bates-ils production.

---

## Implementation Order

| Priority | Phase | Effort | Blocks |
|----------|-------|--------|--------|
| P0 | 1. Missing Scripts | 2 hours | All workflows |
| P1 | 3.3 Fix Issue Weight | 30 min | Issue tracking |
| P1 | 3.1 Notification Calls | 1 hour | Alerting |
| P2 | 2. Ansible Parser | 4 hours | Credential tracking |
| P2 | 3.2 Merge Train | 1 hour | Full GitLab integration |
| P3 | 4. Claude Integration | 8 hours | HOTL autonomy |
| P4 | 5. Testing | 4 hours | Production readiness |

**Total Estimated Effort: ~20 hours for full v1 completion**

---

## Success Criteria

1. `harness box-up-role common` executes end-to-end
2. Issue created with correct labels, weight, iteration
3. MR created and added to merge train
4. Molecule tests run before commit
5. Notifications sent on failure
6. HOTL can spawn Claude agent for simple fix
7. All tests pass in tinyland before bates-ils

---

## Next Steps

1. [ ] Create missing shell scripts (Phase 1)
2. [ ] Fix issue weight in GitLab API
3. [ ] Wire notifications into workflow
4. [ ] Implement Ansible parser
5. [ ] Add Claude integration to HOTL
6. [ ] Test in tinyland
7. [ ] Deploy to bates-ils
