#!/usr/bin/env bash
# Push to GitLab
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

REMOTE="${GITLAB_REMOTE:-origin}"

echo "=== Push to GitLab ==="

# Check if remote exists
if ! git remote get-url "$REMOTE" &>/dev/null; then
    echo "ERROR: Remote '$REMOTE' not found"
    exit 1
fi

REMOTE_URL=$(git remote get-url "$REMOTE")
echo "Remote: $REMOTE -> $REMOTE_URL"

# Get current branch
BRANCH=$(git branch --show-current)
echo "Branch: $BRANCH"

# Check if we have commits to push
LOCAL_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git ls-remote "$REMOTE" "refs/heads/$BRANCH" 2>/dev/null | cut -f1 || echo "")

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
    echo "SUCCESS: Already up to date"
    exit 0
fi

# Push
echo "Pushing to $REMOTE..."
if git push -u "$REMOTE" "$BRANCH" 2>&1; then
    echo "SUCCESS: Pushed $BRANCH to $REMOTE"
else
    echo "ERROR: Push failed"
    exit 1
fi

# Push tags
echo "Pushing tags..."
git push "$REMOTE" --tags 2>&1 || echo "  No tags to push"

echo "=== Push complete ==="
