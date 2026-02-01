#!/bin/bash
#
# create-gitlab-mr.sh - Create GitLab merge request for role box-up
#
# Usage: create-gitlab-mr.sh <role-name> <branch> <issue-iid> [description-file]
#
# Environment variables:
#   GITLAB_ASSIGNEE  - Default assignee (default: jsullivan2)
#   GITLAB_PROJECT   - GitLab project path
#   TARGET_BRANCH    - Target branch (default: main)
#
# Outputs (for parsing):
#   MR_URL=<url>
#   MR_IID=<iid>
#   DRAFT=<true|false>
#

set -euo pipefail

# Arguments
ROLE_NAME="${1:?Usage: $0 <role-name> <branch> <issue-iid> [description-file]}"
SOURCE_BRANCH="${2:?Usage: $0 <role-name> <branch> <issue-iid> [description-file]}"
ISSUE_IID="${3:?Usage: $0 <role-name> <branch> <issue-iid> [description-file]}"
DESC_FILE="${4:-}"

# Configuration
ASSIGNEE="${GITLAB_ASSIGNEE:-jsullivan2}"
PROJECT="${GITLAB_PROJECT:-}"
TARGET="${TARGET_BRANCH:-main}"

# Build title
TITLE="feat(${ROLE_NAME}): Add \`${ROLE_NAME}\` Ansible role"

# Build description
if [ -n "$DESC_FILE" ] && [ -f "$DESC_FILE" ]; then
    DESCRIPTION=$(cat "$DESC_FILE")
else
    # Generate default description
    DESCRIPTION="## Summary

Box up the \`${ROLE_NAME}\` Ansible role with molecule tests.

Closes #${ISSUE_IID}

## Changes

- Added molecule test configuration
- Validated role dependencies
- Documented entry points

## Test Plan

- [ ] Molecule converge passes
- [ ] Molecule verify passes
- [ ] Molecule idempotence passes
- [ ] CI pipeline passes

## Deployment

\`\`\`bash
# Test locally
npm run molecule:role -- --role=${ROLE_NAME}

# Deploy to dev
ansible-playbook -i inventory/dev site.yml --tags ${ROLE_NAME}
\`\`\`
"
fi

# Build glab command
CMD=(glab mr create
    --source-branch "$SOURCE_BRANCH"
    --target-branch "$TARGET"
    --title "$TITLE"
    --description "$DESCRIPTION"
    --label "role,ansible,molecule"
    --assignee "$ASSIGNEE"
    --remove-source-branch
    --squash-before-merge
    --yes)

# Add project if specified
if [ -n "$PROJECT" ]; then
    CMD+=(--repo "$PROJECT")
fi

# Create the MR
echo "Creating merge request for ${ROLE_NAME}..." >&2
OUTPUT=$("${CMD[@]}" 2>&1) || {
    echo "ERROR: Failed to create MR" >&2
    echo "$OUTPUT" >&2
    exit 1
}

# Extract URL from output
MR_URL=$(echo "$OUTPUT" | grep -oP 'https://[^\s]+' | head -1)
MR_IID=$(echo "$MR_URL" | grep -oP '\d+$')

if [ -z "$MR_URL" ] || [ -z "$MR_IID" ]; then
    echo "ERROR: Could not parse MR URL from output" >&2
    echo "Output was: $OUTPUT" >&2
    exit 1
fi

# Check if draft (glab doesn't have --draft flag in older versions)
DRAFT="false"

# Output for parsing
echo "MR_URL=$MR_URL"
echo "MR_IID=$MR_IID"
echo "DRAFT=$DRAFT"

echo "Merge request created successfully: $MR_URL" >&2
