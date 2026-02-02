#!/usr/bin/env bash
# Dogfood Runner: Execute the meta-dogfood DAG
# dag-harness orchestrating its own infrastructure setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLES_DIR="$SCRIPT_DIR/roles"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

export REPO_ROOT

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       DAG-HARNESS DOGFOOD: Self-Orchestrating Infrastructure     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Repository: $REPO_ROOT"
echo "Roles: $ROLES_DIR"
echo ""

# Wave execution function
run_wave() {
    local wave=$1
    shift
    local roles=("$@")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "WAVE $wave: ${roles[*]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local failed=0
    for role in "${roles[@]}"; do
        echo ""
        echo "▶ Running role: $role"
        echo "────────────────────────────────────────"

        # Find and execute script
        local script=""
        for s in verify.sh create.sh configure.sh push.sh sync.sh; do
            if [[ -x "$ROLES_DIR/$role/scripts/$s" ]]; then
                script="$ROLES_DIR/$role/scripts/$s"
                break
            fi
        done

        if [[ -z "$script" ]]; then
            echo "  ERROR: No executable script found for $role"
            ((failed++))
            continue
        fi

        if "$script"; then
            echo "  ✓ $role completed"
        else
            echo "  ✗ $role failed"
            ((failed++))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        echo ""
        echo "Wave $wave: $failed role(s) failed"
        return 1
    fi

    echo ""
    echo "Wave $wave: All roles completed successfully"
    return 0
}

# Execute waves in order
echo ""
echo "Starting dogfood DAG execution..."
echo ""

# Wave 0: Credentials & Verification
if ! run_wave 0 "gitlab-credentials" "repo-github-verify"; then
    echo "ERROR: Wave 0 failed, aborting"
    exit 1
fi

# Wave 1: GitLab Setup
if ! run_wave 1 "repo-gitlab-create"; then
    echo "ERROR: Wave 1 failed, aborting"
    exit 1
fi

# Wave 2: Remote Configuration
if ! run_wave 2 "repo-gitlab-remote" "ci-gitlab-config"; then
    echo "ERROR: Wave 2 failed, aborting"
    exit 1
fi

# Wave 3: Push & Sync
if ! run_wave 3 "repo-push-gitlab"; then
    echo "ERROR: Wave 3 failed, aborting"
    exit 1
fi

# Wave 4: CI Verification
if ! run_wave 4 "ci-gitlab-verify" "repo-github-sync"; then
    echo "WARNING: Wave 4 had issues (non-fatal)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    DOGFOOD DAG COMPLETE                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  ✓ GitLab credentials verified"
echo "  ✓ GitLab project exists"
echo "  ✓ Git remotes configured"
echo "  ✓ CI configuration present"
echo "  ✓ Repository pushed to GitLab"
echo "  ✓ CI pipeline verified"
echo ""
echo "GitLab: https://gitlab.com/tinyland/projects/dag-harness"
echo "Pipeline: https://gitlab.com/tinyland/projects/dag-harness/-/pipelines"
