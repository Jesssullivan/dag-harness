#!/bin/bash
#
# create-role-worktree.sh - Create isolated git worktree for role development
#
# Usage: create-role-worktree.sh <role-name> [base-branch]
#
# Environment variables:
#   WORKTREE_BASE    - Base directory for worktrees (default: ~/git/worktrees)
#   BRANCH_PREFIX    - Prefix for branch names (default: sid/)
#   REPO_ROOT        - Repository root (default: current directory)
#
# Outputs (for parsing):
#   WORKTREE_PATH=<path>
#   BRANCH_NAME=<branch>
#   BASE_COMMIT=<commit>
#

set -euo pipefail

# Arguments
ROLE_NAME="${1:?Usage: $0 <role-name> [base-branch]}"
BASE_BRANCH="${2:-origin/main}"

# Configuration from environment
WORKTREE_BASE="${WORKTREE_BASE:-$HOME/git/worktrees}"
BRANCH_PREFIX="${BRANCH_PREFIX:-sid/}"
REPO_ROOT="${REPO_ROOT:-.}"

# Derived paths
BRANCH_NAME="${BRANCH_PREFIX}${ROLE_NAME}"
WORKTREE_PATH="${WORKTREE_BASE}/${ROLE_NAME}"

# Ensure worktree base directory exists
mkdir -p "$WORKTREE_BASE"

# Check if worktree already exists
if [ -d "$WORKTREE_PATH" ]; then
    echo "ERROR: Worktree already exists at $WORKTREE_PATH" >&2
    echo "Remove with: git worktree remove $WORKTREE_PATH" >&2
    exit 1
fi

# Check if branch already exists (locally or remote)
if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH_NAME" 2>/dev/null; then
    echo "ERROR: Branch $BRANCH_NAME already exists locally" >&2
    exit 1
fi

# Fetch latest from origin
echo "Fetching latest from origin..." >&2
git -C "$REPO_ROOT" fetch origin --quiet

# Get base commit for tracking
BASE_COMMIT=$(git -C "$REPO_ROOT" rev-parse "$BASE_BRANCH")

# Create worktree with new branch
echo "Creating worktree at $WORKTREE_PATH..." >&2
git -C "$REPO_ROOT" worktree add -b "$BRANCH_NAME" "$WORKTREE_PATH" "$BASE_BRANCH"

# Copy essential files if they exist
for file in .env.local .env ems.kdbx; do
    if [ -f "$REPO_ROOT/$file" ]; then
        cp -f "$REPO_ROOT/$file" "$WORKTREE_PATH/" 2>/dev/null || true
        echo "Copied $file to worktree" >&2
    fi
done

# Verify worktree was created
if [ ! -d "$WORKTREE_PATH/.git" ] && [ ! -f "$WORKTREE_PATH/.git" ]; then
    echo "ERROR: Worktree creation failed" >&2
    exit 1
fi

# Output for parsing
echo "WORKTREE_PATH=$WORKTREE_PATH"
echo "BRANCH_NAME=$BRANCH_NAME"
echo "BASE_COMMIT=$BASE_COMMIT"

echo "Worktree created successfully" >&2
