#!/usr/bin/env bash
# Verify GitLab credentials
set -euo pipefail

echo "=== GitLab Credentials Verification ==="

# Check GITLAB_TOKEN is set
if [[ -z "${GITLAB_TOKEN:-}" ]]; then
    echo "ERROR: GITLAB_TOKEN not set"
    exit 1
fi

# Verify token with GitLab API
echo "Verifying token with GitLab API..."
RESPONSE=$(curl -s -w "\n%{http_code}" \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "https://gitlab.com/api/v4/user")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "ERROR: GitLab API returned HTTP $HTTP_CODE"
    echo "$BODY" | jq -r '.message // .error // "Unknown error"' 2>/dev/null || echo "$BODY"
    exit 1
fi

USERNAME=$(echo "$BODY" | jq -r '.username')
echo "SUCCESS: Token valid for user @${USERNAME}"

# Check project access (use numeric ID for reliability)
echo "Checking project access..."
PROJECT_ID="${GITLAB_PROJECT_ID:-78116258}"
PROJECT_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "https://gitlab.com/api/v4/projects/${PROJECT_ID}")

PROJECT_CODE=$(echo "$PROJECT_RESPONSE" | tail -n1)
if [[ "$PROJECT_CODE" == "200" ]]; then
    PROJECT_NAME=$(echo "$PROJECT_RESPONSE" | sed '$d' | jq -r '.path_with_namespace')
    echo "SUCCESS: Project access verified ($PROJECT_NAME)"
else
    echo "WARNING: Project not accessible (HTTP $PROJECT_CODE) - may need to create"
fi

echo "=== Credentials verification complete ==="
