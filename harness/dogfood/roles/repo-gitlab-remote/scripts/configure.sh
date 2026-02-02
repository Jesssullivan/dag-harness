#!/usr/bin/env bash
# Configure GitLab remote
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

GITLAB_URL="${GITLAB_REMOTE_URL:-https://gitlab.com/tinyland/projects/dag-harness.git}"
REMOTE_NAME="${GITLAB_REMOTE_NAME:-origin}"

echo "=== GitLab Remote Configuration ==="

# Check if remote already exists
if git remote get-url "$REMOTE_NAME" &>/dev/null; then
    CURRENT_URL=$(git remote get-url "$REMOTE_NAME")
    if [[ "$CURRENT_URL" == *"gitlab.com"* ]]; then
        echo "SUCCESS: GitLab remote already configured"
        echo "  $REMOTE_NAME -> $CURRENT_URL"
        exit 0
    else
        # Remote exists but points elsewhere, add gitlab as separate remote
        REMOTE_NAME="gitlab"
        if git remote get-url "$REMOTE_NAME" &>/dev/null; then
            echo "SUCCESS: GitLab remote 'gitlab' already exists"
            exit 0
        fi
    fi
fi

# Add remote
echo "Adding GitLab remote '$REMOTE_NAME'..."
git remote add "$REMOTE_NAME" "$GITLAB_URL" || {
    # Remote may already exist
    git remote set-url "$REMOTE_NAME" "$GITLAB_URL"
}

echo "SUCCESS: Remote configured"
git remote -v | grep "$REMOTE_NAME"

echo "=== GitLab remote configuration complete ==="
