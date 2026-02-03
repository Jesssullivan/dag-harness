# Claude Code Integration

dag-harness provides deep integration with Claude Code through MCP servers,
hooks, and skills. This guide covers setup, usage, and troubleshooting.

## Overview

The integration has three layers:

| Layer | Purpose | Location |
|-------|---------|----------|
| **MCP Server** | Exposes 20+ tools for role management, testing, GitLab ops | `harness/mcp/server.py` |
| **Hooks** | Pre/post tool validation, audit logging, file tracking | `harness/hooks/` |
| **Skills** | Slash commands like `/box-up-role` | `harness/skills/` |

## Installing for Claude Code

### Automatic Installation

The bootstrap command installs all Claude Code integration:

```bash
harness bootstrap
```

This creates or updates `.claude/settings.json` with:

- MCP server configuration
- Hook definitions (PreToolUse, PostToolUse)
- Skill registration

### Manual Installation

Install the Claude Code integration components individually:

```bash
harness install run              # Full installation
harness install run --skip-hooks # Skip hook scripts
harness install run --skip-mcp   # Skip MCP server config
```

Verify the installation:

```bash
harness install check
```

### MCP Server Configuration

The MCP server is configured in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "dag-harness": {
      "command": "uv",
      "args": [
        "run", "--directory", "./harness",
        "python", "-m", "harness.mcp.server"
      ],
      "env": {
        "HARNESS_DB_PATH": "./harness/harness.db"
      }
    }
  }
}
```

Once configured, Claude Code can access all dag-harness tools through the
MCP protocol. Tools include role status queries, test execution, GitLab
operations, and workflow management.

## The /box-up-role Skill

The primary skill for packaging Ansible roles into GitLab iterations.

### Usage

In Claude Code, invoke the skill:

```
/box-up-role common
```

This triggers the full DAG workflow:

1. Validates the role exists and has proper structure
2. Analyzes dependencies and checks for conflicts
3. Creates an isolated git worktree
4. Runs molecule and pytest tests (in parallel when possible)
5. Commits, pushes, and creates a GitLab issue + merge request
6. Queues the MR in the merge train

### Breakpoints

Add breakpoints to pause before critical operations:

```
/box-up-role common --breakpoints create_mr,add_to_merge_train
```

When paused, Claude Code receives the execution state and can:

- Inspect test results before creating the MR
- Review the commit before pushing
- Modify the issue description before creation

### Dry Run

Preview the workflow without side effects:

```
/box-up-role common --dry-run
```

## Hook Integration

### PreToolUse Hooks

Hooks that run before tool execution for validation and safety:

| Hook | Purpose |
|------|---------|
| `audit.py` | Log tool invocations to the audit trail |
| `verification.py` | Validate preconditions before destructive operations |
| `base.py` | Base hook infrastructure |

Example: The audit hook records every MCP tool call with arguments, timing,
and outcome to the `tool_invocations` table in StateDB.

### PostToolUse Hooks

Hooks that run after tool execution for tracking and notifications:

| Hook | Purpose |
|------|---------|
| `file_tracker.py` | Track file changes made by tools |
| `audit.py` | Record tool results and duration |

### Hook Configuration

Hooks are registered in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "command": "python3 harness/harness/hooks/audit.py pre",
        "event": "PreToolUse"
      }
    ],
    "PostToolUse": [
      {
        "command": "python3 harness/harness/hooks/audit.py post",
        "event": "PostToolUse"
      }
    ]
  }
}
```

## HOTL Mode with Claude Code

Human-on-the-loop (HOTL) mode enables autonomous operation where Claude Code
processes roles in wave order with minimal human intervention.

### Starting HOTL

```bash
harness hotl start --max-iterations 100
```

Or from Claude Code:

```
/hotl start
```

### How HOTL Works

1. The **supervisor** picks the next eligible role (respecting wave ordering)
2. A **worker** executes the `box-up-role` workflow
3. At checkpoint nodes, the system either:
   - Proceeds automatically (if confidence is high)
   - Pauses for human approval (if the operation is destructive)
4. **Notifications** are sent at configurable intervals (Discord, email)
5. The cycle repeats until all roles are processed or `max_iterations` is reached

### HOTL Commands

| Command | Description |
|---------|-------------|
| `harness hotl start` | Begin autonomous processing |
| `harness hotl status` | Check current HOTL state |
| `harness hotl stop` | Gracefully stop after current role |
| `harness hotl stop --force` | Immediately cancel |
| `harness hotl resume THREAD_ID` | Resume from a checkpoint |

### Notifications

Configure notifications to stay informed during HOTL operation:

```bash
# Discord
harness hotl start --discord https://discord.com/api/webhooks/...

# Email
harness hotl start --email ops@example.com
```

Or set via environment variables:

```bash
export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
export EMAIL_RECIPIENT=ops@example.com
```

### Agent Sessions

HOTL tracks each Claude Code session in the `agent_sessions` table:

- Session ID, status, and task description
- File changes made during the session
- Process ID for running sessions
- Intervention reasons when human input is needed

Query session history:

```bash
harness hotl status --json | jq '.sessions'
```

## Troubleshooting

### MCP Server Not Responding

Test the server manually:

```bash
uv run --directory ./harness python -m harness.mcp.server
```

Check that the `harness.db` file exists and is accessible:

```bash
ls -la harness/harness.db
```

### Hooks Not Firing

Verify hook registration:

```bash
harness install check
```

Ensure hook scripts are executable:

```bash
chmod +x harness/harness/hooks/*.py
```

### Skills Not Available

Re-install the skill definitions:

```bash
harness install run --force
```

Check that `.claude/settings.json` contains the skill configuration.

### HOTL Not Starting

Common issues:

1. **No eligible roles** -- Run `harness sync` first to populate the database
2. **Missing credentials** -- Ensure `GITLAB_TOKEN` is set
3. **Database locked** -- Another harness process may be running

Check the database for pending work:

```bash
harness status
harness hotl status
```

### Claude Code Cannot Find Tools

If Claude Code reports "no tools available":

1. Restart Claude Code to reload MCP server configuration
2. Verify `.claude/settings.json` is valid JSON
3. Check that `uv` is on the PATH in the Claude Code environment
4. Test the MCP server independently (see above)

### Audit Trail

View recent tool invocations:

```bash
sqlite3 harness/harness.db "SELECT tool_name, status, created_at FROM tool_invocations ORDER BY created_at DESC LIMIT 20;"
```
