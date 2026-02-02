#!/usr/bin/env bash
# Sync to GitHub (optional)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

echo "=== GitHub Sync ==="

# Check for GitHub remote
GITHUB_REMOTE=""
for remote in github gh upstream; do
    if git remote get-url "$remote" 2>/dev/null | grep -q "github.com"; then
        GITHUB_REMOTE="$remote"
        break
    fi
done

if [[ -z "$GITHUB_REMOTE" ]]; then
    echo "INFO: No GitHub remote configured"
    echo "  To enable dual-push, add: git remote add github git@github.com:owner/repo.git"
    exit 0
fi

REMOTE_URL=$(git remote get-url "$GITHUB_REMOTE")
echo "GitHub remote: $GITHUB_REMOTE -> $REMOTE_URL"

BRANCH=$(git branch --show-current)
echo "Branch: $BRANCH"

# Push to GitHub
echo "Pushing to GitHub..."
if git push "$GITHUB_REMOTE" "$BRANCH" 2>&1; then
    echo "SUCCESS: Synced to GitHub"
else
    echo "WARNING: GitHub push failed (may require SSH key or token)"
fi

echo "=== GitHub sync complete ==="
