#!/bin/bash
#
# create-gitlab-issues.sh - Create GitLab issue for role box-up
#
# Usage: create-gitlab-issues.sh <role-name> <wave> [description-file]
#
# Environment variables:
#   GITLAB_ASSIGNEE  - Default assignee (default: jsullivan2)
#   GITLAB_PROJECT   - GitLab project path
#   ITERATION_TITLE  - Iteration to assign issue to (optional)
#
# Outputs (for parsing):
#   ISSUE_URL=<url>
#   ISSUE_IID=<iid>
#   ITERATION_ASSIGNED=<true|false>
#

set -euo pipefail

# Arguments
ROLE_NAME="${1:?Usage: $0 <role-name> <wave> [description-file]}"
WAVE="${2:?Usage: $0 <role-name> <wave> [description-file]}"
DESC_FILE="${3:-}"

# Configuration
ASSIGNEE="${GITLAB_ASSIGNEE:-jsullivan2}"
PROJECT="${GITLAB_PROJECT:-}"

# Build labels
LABELS="role,ansible,molecule,wave-${WAVE}"

# Build title
TITLE="feat(${ROLE_NAME}): Box up \`${ROLE_NAME}\` Ansible role"

# Build description
if [ -n "$DESC_FILE" ] && [ -f "$DESC_FILE" ]; then
    DESCRIPTION=$(cat "$DESC_FILE")
else
    # Generate default description
    DESCRIPTION="## Summary

Box up the \`${ROLE_NAME}\` Ansible role into a testable, deployable module.

**Wave**: ${WAVE}

## Tasks

- [ ] Validate role structure
- [ ] Analyze dependencies
- [ ] Create molecule tests
- [ ] Run molecule converge/verify
- [ ] Create documentation

## Acceptance Criteria

- Role passes molecule idempotence tests
- Role has clear entry point
- Dependencies are documented
- Credentials are documented
"
fi

# Build glab command
CMD=(glab issue create
    --title "$TITLE"
    --description "$DESCRIPTION"
    --label "$LABELS"
    --assignee "$ASSIGNEE"
    --yes)

# Add project if specified
if [ -n "$PROJECT" ]; then
    CMD+=(--repo "$PROJECT")
fi

# Create the issue
echo "Creating GitLab issue for ${ROLE_NAME}..." >&2
OUTPUT=$("${CMD[@]}" 2>&1) || {
    echo "ERROR: Failed to create issue" >&2
    echo "$OUTPUT" >&2
    exit 1
}

# Extract URL from output (glab outputs: "Created issue #N: <url>")
ISSUE_URL=$(echo "$OUTPUT" | grep -oP 'https://[^\s]+' | head -1)
ISSUE_IID=$(echo "$ISSUE_URL" | grep -oP '\d+$')

if [ -z "$ISSUE_URL" ] || [ -z "$ISSUE_IID" ]; then
    echo "ERROR: Could not parse issue URL from output" >&2
    echo "Output was: $OUTPUT" >&2
    exit 1
fi

# Try to assign to iteration if specified
ITERATION_ASSIGNED="false"
if [ -n "${ITERATION_TITLE:-}" ]; then
    echo "Attempting to assign to iteration: $ITERATION_TITLE" >&2
    # This would require API call - leave for API layer
    ITERATION_ASSIGNED="pending"
fi

# Output for parsing
echo "ISSUE_URL=$ISSUE_URL"
echo "ISSUE_IID=$ISSUE_IID"
echo "ITERATION_ASSIGNED=$ITERATION_ASSIGNED"

echo "Issue created successfully: $ISSUE_URL" >&2
