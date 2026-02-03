#!/usr/bin/env bash
# Setup dag-harness in the EMS project
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up dag-harness..."

# 1. Install harness in dev mode
cd "$HARNESS_DIR"
uv pip install -e ".[dev]"

# 2. Initialize harness
uv run harness init --repo-root "$(git rev-parse --show-toplevel)"

# 3. Run migrations
uv run harness migrate

echo "dag-harness setup complete!"
echo "Run 'npm run harness:status' to verify."
