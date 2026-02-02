# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### "No module named 'harness'"

**Cause:** Package not installed or wrong Python environment.

**Solution:**
```bash
cd harness
uv sync
uv run harness --version
```

### "uv: command not found"

**Cause:** uv package manager not installed.

**Solution:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Bootstrap Issues

### "GITLAB_TOKEN not found"

**Cause:** GitLab API token not configured.

**Solution:**
```bash
# Create .env file
echo 'GITLAB_TOKEN="glpat-your-token-here"' > .env

# Or set environment variable
export GITLAB_TOKEN="glpat-your-token-here"

# Verify
harness credentials
```

### "MCP server not configured"

**Cause:** Bootstrap incomplete.

**Solution:**
```bash
harness bootstrap
# Follow interactive prompts
```

## Database Issues

### "No roles in database"

**Cause:** Database empty or not synced.

**Solution:**
```bash
harness sync --roles
harness list-roles
```

### "Database locked"

**Cause:** Multiple processes accessing SQLite.

**Solution:**
```bash
# Stop other harness processes
pkill -f "harness"

# Or use WAL mode (default in newer versions)
harness db stats
```

### "Schema mismatch"

**Cause:** Database from older version.

**Solution:**
```bash
# Backup first
harness db backup ./backup.db

# Reset (destructive)
harness db reset

# Re-sync
harness sync --roles --worktrees
```

## Workflow Issues

### "Role not found"

**Cause:** Role doesn't exist or not synced.

**Solution:**
```bash
# Sync roles
harness sync --roles

# List available roles
harness list-roles

# Check specific role
harness status <role-name>
```

### "Worktree already exists"

**Cause:** Previous workflow left worktree.

**Solution:**
```bash
# List worktrees
harness worktrees

# Clean up via git
git worktree remove /path/to/worktree

# Or force remove
git worktree remove --force /path/to/worktree
```

### "Human approval required"

**Cause:** Workflow paused at HITL gate.

**Solution:**
```bash
# Check pending executions
harness hotl status

# Approve
harness resume <execution-id> --approve

# Or reject
harness resume <execution-id> --reject --reason "Needs changes"
```

### "Workflow stuck"

**Cause:** Execution hung or crashed.

**Solution:**
```bash
# Check status
harness hotl status

# Cancel stuck executions
harness hotl cancel_executions

# Resume from checkpoint
harness resume <execution-id>
```

## GitLab Issues

### "401 Unauthorized"

**Cause:** Invalid or expired token.

**Solution:**
```bash
# Validate token
harness credentials --validate

# Update token in .env
# Ensure token has api, read_repository, write_repository scopes
```

### "403 Forbidden"

**Cause:** Token lacks required permissions.

**Solution:**
- Check token scopes in GitLab
- Required: `api`, `read_repository`, `write_repository`
- For merge trains: `write_merge_request`

### "Merge train not available"

**Cause:** Project doesn't have merge trains enabled.

**Solution:**
- Enable in GitLab: Settings → Merge requests → Merge trains
- Or workflow will fall back to `merge_when_pipeline_succeeds`

## Testing Issues

### "Molecule tests timeout"

**Cause:** Tests taking longer than timeout.

**Solution:**
```yaml
# In harness.yml
testing:
  molecule_timeout: 1200  # Increase from default 600
```

### "Pytest not found"

**Cause:** Tests directory doesn't exist for role.

**Solution:**
- Create `tests/` directory in role
- Or disable pytest requirement:
```yaml
testing:
  pytest_required: false
```

## Debug Commands

### Check System Health

```bash
# Full validation
harness check --schema --data --graph

# Database stats
harness db stats

# View workflow graph
harness graph --format mermaid
```

### View Logs

```bash
# HOTL supervisor logs
harness hotl status

# Recent executions
harness hotl get_recent_executions

# Specific execution
harness status <role-name>
```

### Cost Tracking

```bash
# Recent costs
harness costs report --days 7

# Session breakdown
harness costs session <session-id>
```

## Recovery Procedures

### Full Reset

```bash
# 1. Backup
harness db backup ./backup-$(date +%Y%m%d).db

# 2. Reset database
harness db reset

# 3. Re-sync
harness sync --roles --worktrees --gitlab

# 4. Verify
harness check
```

### Recover from Crashed Workflow

```bash
# 1. Find execution ID
harness hotl get_recent_executions

# 2. Check state
harness status <role-name>

# 3. Resume or cancel
harness resume <id>  # Try to continue
# or
harness hotl cancel_executions  # Start fresh
```

## Getting Help

- **Docs:** https://tinyland.gitlab.io/projects/dag-harness/
- **Issues:** https://gitlab.com/tinyland/projects/dag-harness/-/issues
- **LLM Docs:** See `docs/llms.md` for AI assistant reference
