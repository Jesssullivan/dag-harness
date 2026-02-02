#!/usr/bin/env bash
# Create GitLab project (idempotent)
set -euo pipefail

PROJECT_PATH="${GITLAB_PROJECT_PATH:-tinyland/projects/dag-harness}"
PROJECT_NAME="${GITLAB_PROJECT_NAME:-dag-harness}"
NAMESPACE_ID="${GITLAB_NAMESPACE_ID:-}"  # Group ID for tinyland/projects

echo "=== GitLab Project Creation ==="

# Check if project already exists (use numeric ID for reliability)
echo "Checking if project exists..."
PROJECT_ID="${GITLAB_PROJECT_ID:-78116258}"
RESPONSE=$(curl -s -w "\n%{http_code}" \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "https://gitlab.com/api/v4/projects/${PROJECT_ID}")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" == "200" ]]; then
    PROJECT_ID=$(echo "$BODY" | jq -r '.id')
    WEB_URL=$(echo "$BODY" | jq -r '.web_url')
    echo "SUCCESS: Project already exists"
    echo "  ID: $PROJECT_ID"
    echo "  URL: $WEB_URL"
    exit 0
fi

# Project doesn't exist, create it
echo "Project not found, creating..."

# Get namespace ID if not provided
if [[ -z "$NAMESPACE_ID" ]]; then
    echo "Looking up namespace 'tinyland/projects'..."
    NS_RESPONSE=$(curl -s \
        --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        "https://gitlab.com/api/v4/groups/tinyland%2Fprojects")
    NAMESPACE_ID=$(echo "$NS_RESPONSE" | jq -r '.id // empty')

    if [[ -z "$NAMESPACE_ID" ]]; then
        echo "ERROR: Could not find namespace 'tinyland/projects'"
        exit 1
    fi
    echo "  Namespace ID: $NAMESPACE_ID"
fi

# Create project
CREATE_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    --header "Content-Type: application/json" \
    --request POST \
    --data "{
        \"name\": \"${PROJECT_NAME}\",
        \"path\": \"${PROJECT_NAME}\",
        \"namespace_id\": ${NAMESPACE_ID},
        \"visibility\": \"private\",
        \"initialize_with_readme\": false
    }" \
    "https://gitlab.com/api/v4/projects")

CREATE_CODE=$(echo "$CREATE_RESPONSE" | tail -n1)
CREATE_BODY=$(echo "$CREATE_RESPONSE" | sed '$d')

if [[ "$CREATE_CODE" == "201" ]]; then
    PROJECT_ID=$(echo "$CREATE_BODY" | jq -r '.id')
    WEB_URL=$(echo "$CREATE_BODY" | jq -r '.web_url')
    echo "SUCCESS: Project created"
    echo "  ID: $PROJECT_ID"
    echo "  URL: $WEB_URL"
else
    echo "ERROR: Failed to create project (HTTP $CREATE_CODE)"
    echo "$CREATE_BODY" | jq '.' 2>/dev/null || echo "$CREATE_BODY"
    exit 1
fi

echo "=== GitLab project creation complete ==="
