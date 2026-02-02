#!/usr/bin/env bash
# Configure GitLab CI
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

CI_FILE=".gitlab-ci.yml"

echo "=== GitLab CI Configuration ==="

if [[ -f "$CI_FILE" ]]; then
    echo "SUCCESS: .gitlab-ci.yml already exists"

    # Validate syntax
    echo "Validating CI configuration..."
    if command -v python3 &>/dev/null; then
        python3 -c "import yaml; yaml.safe_load(open('$CI_FILE'))" && \
            echo "  Syntax: valid YAML" || \
            echo "  WARNING: Invalid YAML syntax"
    fi

    # Show stages
    echo "  Stages: $(grep -E '^stages:' -A 10 "$CI_FILE" | grep '^\s*-' | wc -l | tr -d ' ') defined"
else
    echo "ERROR: .gitlab-ci.yml not found"
    echo "  Create it manually or use 'harness init' to generate"
    exit 1
fi

echo "=== GitLab CI configuration complete ==="
