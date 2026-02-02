#!/usr/bin/env bash
# Verify GitLab CI pipeline
set -euo pipefail

# Use numeric project ID for reliability
PROJECT_ID="${GITLAB_PROJECT_ID:-78116258}"

echo "=== GitLab CI Verification ==="

# Get latest pipeline
echo "Fetching latest pipeline..."
RESPONSE=$(curl -s \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "https://gitlab.com/api/v4/projects/${PROJECT_ID}/pipelines?per_page=1")

PIPELINE_ID=$(echo "$RESPONSE" | jq -r '.[0].id // empty')
PIPELINE_STATUS=$(echo "$RESPONSE" | jq -r '.[0].status // empty')
PIPELINE_REF=$(echo "$RESPONSE" | jq -r '.[0].ref // empty')
PIPELINE_SHA=$(echo "$RESPONSE" | jq -r '.[0].sha // empty' | head -c 8)
PIPELINE_URL=$(echo "$RESPONSE" | jq -r '.[0].web_url // empty')

if [[ -z "$PIPELINE_ID" ]]; then
    echo "ERROR: No pipelines found"
    exit 1
fi

echo "Pipeline #$PIPELINE_ID:"
echo "  Status: $PIPELINE_STATUS"
echo "  Branch: $PIPELINE_REF"
echo "  Commit: $PIPELINE_SHA"
echo "  URL: $PIPELINE_URL"

# Check status
case "$PIPELINE_STATUS" in
    "success")
        echo "SUCCESS: Pipeline passed!"
        ;;
    "running"|"pending"|"created")
        echo "INFO: Pipeline is $PIPELINE_STATUS - waiting..."
        # Could add polling logic here
        ;;
    "failed"|"canceled")
        echo "ERROR: Pipeline $PIPELINE_STATUS"
        # Get failed jobs
        JOBS=$(curl -s \
            --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
            "https://gitlab.com/api/v4/projects/${PROJECT_ID}/pipelines/${PIPELINE_ID}/jobs")
        echo "Failed jobs:"
        echo "$JOBS" | jq -r '.[] | select(.status == "failed") | "  - \(.name): \(.web_url)"'
        exit 1
        ;;
    *)
        echo "WARNING: Unknown status: $PIPELINE_STATUS"
        ;;
esac

echo "=== CI verification complete ==="
