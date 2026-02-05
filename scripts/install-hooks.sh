#!/usr/bin/env bash
# install-hooks.sh - Install git hooks for dag-harness
#
# Usage:
#   ./scripts/install-hooks.sh           Install to current repo
#   ./scripts/install-hooks.sh --symlink Use symlinks instead of copies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$REPO_ROOT/hooks/templates"

# Parse arguments
USE_SYMLINK=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --symlink) USE_SYMLINK=1; shift ;;
        *) shift ;;
    esac
done

# Verify we're in a git repo
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "ERROR: Not in a git repository"
    exit 1
fi

DEST_HOOKS="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."
echo "Source: $HOOKS_DIR"
echo "Destination: $DEST_HOOKS"
echo ""

# List of hooks to install
HOOKS=(
    pre-commit
    commit-msg
)

for hook in "${HOOKS[@]}"; do
    src="$HOOKS_DIR/$hook"
    dest="$DEST_HOOKS/$hook"

    if [ ! -f "$src" ]; then
        echo "  SKIP: $hook (not found)"
        continue
    fi

    # Backup existing hook
    if [ -f "$dest" ] && [ ! -L "$dest" ]; then
        echo "  BACKUP: $hook -> ${hook}.backup"
        mv "$dest" "${dest}.backup"
    fi

    if [ "$USE_SYMLINK" = "1" ]; then
        echo "  LINK: $hook"
        ln -sf "$src" "$dest"
    else
        echo "  COPY: $hook"
        cp "$src" "$dest"
        chmod +x "$dest"
    fi
done

echo ""
echo "Hooks installed successfully!"
echo ""
echo "These hooks will block:"
echo "  - AI attribution in commits"
echo "  - AI artifact files (CLAUDE*, .claude/, .mcp.json, etc)"
echo "  - Potential secrets"
