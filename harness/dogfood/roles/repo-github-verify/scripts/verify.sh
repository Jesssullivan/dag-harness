#!/usr/bin/env bash
# Verify GitHub repository
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

echo "=== GitHub Repository Verification ==="

# Check for GitHub remote
echo "Checking git remotes..."
if git remote -v | grep -q "github.com"; then
    GITHUB_REMOTE=$(git remote -v | grep "github.com" | head -1 | awk '{print $1}')
    GITHUB_URL=$(git remote get-url "$GITHUB_REMOTE" 2>/dev/null || echo "")
    echo "SUCCESS: GitHub remote found: $GITHUB_REMOTE -> $GITHUB_URL"
else
    echo "INFO: No GitHub remote configured"
    echo "  To add: git remote add github git@github.com:owner/repo.git"
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Check if we can fetch from any remote
echo "Checking remote connectivity..."
for remote in $(git remote); do
    if git ls-remote --exit-code "$remote" HEAD &>/dev/null; then
        echo "  SUCCESS: $remote is accessible"
    else
        echo "  WARNING: $remote is not accessible"
    fi
done

echo "=== GitHub verification complete ==="
