#!/bin/bash
# dag-harness bootstrap script
# Usage: curl -sSL https://raw.githubusercontent.com/Jesssullivan/dag-harness/main/scripts/bootstrap.sh | bash
# Or:    curl -sSL https://disposable-ansible-dag.ephemera.xoxd.ai/scripts/bootstrap.sh | bash
#
# This script:
# 1. Detects platform (darwin-arm64, darwin-x86_64, rocky-x86_64, linux-x86_64)
# 2. Selects installation method (git+https → wheel URL → binary fallback)
# 3. Performs greedy credential discovery
# 4. Runs self-tests to verify installation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="dag-harness"
VERSION="0.2.0"
MIN_PYTHON_VERSION="3.11"
GITHUB_REPO="Jesssullivan/dag-harness"
GITLAB_REPO="tinyland/projects/dag-harness"
GITHUB_RELEASES_URL="https://github.com/${GITHUB_REPO}/releases"
DOCS_URL="https://disposable-ansible-dag.ephemera.xoxd.ai"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step() { echo -e "${CYAN}==>${NC} $*"; }

# Detect platform
detect_platform() {
    local os arch
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    case "$os" in
        darwin)
            case "$arch" in
                arm64|aarch64) echo "darwin-arm64" ;;
                x86_64) echo "darwin-x86_64" ;;
                *) log_error "Unsupported macOS architecture: $arch"; exit 1 ;;
            esac
            ;;
        linux)
            # Check for Rocky Linux specifically
            if [ -f /etc/rocky-release ]; then
                echo "rocky-x86_64"
            else
                case "$arch" in
                    x86_64) echo "linux-x86_64" ;;
                    aarch64|arm64) echo "linux-arm64" ;;
                    *) log_error "Unsupported Linux architecture: $arch"; exit 1 ;;
                esac
            fi
            ;;
        *)
            log_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get Python version as comparable integer (e.g., 3.11.5 -> 311)
python_version_int() {
    local version
    version=$("$1" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor:02d}')" 2>/dev/null) || echo "0"
    echo "$version"
}

# Find suitable Python interpreter
find_python() {
    local candidates=("python3.12" "python3.11" "python3" "python")
    local min_version_int=311  # 3.11

    for cmd in "${candidates[@]}"; do
        if command_exists "$cmd"; then
            local ver_int
            ver_int=$(python_version_int "$cmd")
            if [ "$ver_int" -ge "$min_version_int" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done

    return 1
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    local issues=()

    # Check Python
    local python_cmd
    if python_cmd=$(find_python); then
        local version
        version=$("$python_cmd" --version 2>&1)
        log_success "Python: $version ($python_cmd)"
        export HARNESS_PYTHON="$python_cmd"
    else
        issues+=("Python ${MIN_PYTHON_VERSION}+ not found")
    fi

    # Check git
    if command_exists git; then
        log_success "Git: $(git --version)"
    else
        issues+=("git not found")
    fi

    # Check uv (preferred)
    if command_exists uv; then
        log_success "uv: $(uv --version)"
        export HARNESS_INSTALLER="uv"
    elif command_exists pip || command_exists pip3; then
        local pip_cmd
        pip_cmd=$(command_exists pip3 && echo "pip3" || echo "pip")
        log_success "pip: $($pip_cmd --version | head -1)"
        export HARNESS_INSTALLER="pip"
    else
        issues+=("Neither uv nor pip found")
    fi

    if [ ${#issues[@]} -gt 0 ]; then
        log_error "Prerequisites not met:"
        for issue in "${issues[@]}"; do
            echo "  - $issue"
        done
        return 1
    fi

    return 0
}

# Install via uv from GitHub (preferred)
install_via_uv_github() {
    log_step "Installing via uv from GitHub..."
    local git_url="git+https://github.com/${GITHUB_REPO}.git@v${VERSION}"

    if uv tool install "$git_url" 2>/dev/null; then
        log_success "Installed $PACKAGE_NAME via uv from GitHub"
        return 0
    fi

    # Try without version tag (latest main)
    git_url="git+https://github.com/${GITHUB_REPO}.git"
    if uv tool install "$git_url" 2>/dev/null; then
        log_success "Installed $PACKAGE_NAME via uv from GitHub (main branch)"
        return 0
    fi

    # Fallback: try installing from current directory if it exists
    if [ -f "harness/pyproject.toml" ]; then
        log_info "Attempting local installation..."
        if (cd harness && uv tool install .); then
            log_success "Installed $PACKAGE_NAME from local source"
            return 0
        fi
    fi

    return 1
}

# Install via pip from GitHub
install_via_pip_github() {
    log_step "Installing via pip from GitHub..."
    local pip_cmd="${HARNESS_PYTHON:-python3} -m pip"
    local git_url="git+https://github.com/${GITHUB_REPO}.git@v${VERSION}"

    if $pip_cmd install --user "$git_url" 2>/dev/null; then
        log_success "Installed $PACKAGE_NAME via pip from GitHub"
        return 0
    fi

    # Try without version tag (latest main)
    git_url="git+https://github.com/${GITHUB_REPO}.git"
    if $pip_cmd install --user "$git_url" 2>/dev/null; then
        log_success "Installed $PACKAGE_NAME via pip from GitHub (main branch)"
        return 0
    fi

    # Fallback: try installing from current directory
    if [ -f "harness/pyproject.toml" ]; then
        log_info "Attempting local installation..."
        if $pip_cmd install --user ./harness; then
            log_success "Installed $PACKAGE_NAME from local source"
            return 0
        fi
    fi

    return 1
}

# Install via direct wheel URL (if GitHub release exists)
install_via_wheel_url() {
    log_step "Attempting wheel download from GitHub release..."
    local pip_cmd="${HARNESS_PYTHON:-python3} -m pip"
    local wheel_url="${GITHUB_RELEASES_URL}/download/v${VERSION}/dag_harness-${VERSION}-py3-none-any.whl"

    # Check if wheel exists
    if curl -fsS --head "$wheel_url" >/dev/null 2>&1; then
        if $pip_cmd install --user "$wheel_url" 2>/dev/null; then
            log_success "Installed $PACKAGE_NAME from GitHub release wheel"
            return 0
        fi
    else
        log_info "No wheel found at release URL, skipping..."
    fi

    return 1
}

# Download and install binary
install_via_binary() {
    local platform="$1"
    log_step "Installing pre-built binary for $platform..."

    local binary_name="harness-${platform}"
    local download_url="${GITHUB_RELEASES_URL}/latest/download/${binary_name}"
    local install_dir="${HOME}/.local/bin"

    mkdir -p "$install_dir"

    log_info "Downloading from $download_url..."
    if curl -fsSL "$download_url" -o "${install_dir}/harness"; then
        chmod +x "${install_dir}/harness"
        log_success "Binary installed to ${install_dir}/harness"

        # Add to PATH if needed
        if [[ ":$PATH:" != *":${install_dir}:"* ]]; then
            log_warn "Add ${install_dir} to your PATH:"
            echo "  export PATH=\"\$PATH:${install_dir}\""
        fi
        return 0
    fi

    return 1
}

# Select and execute installation strategy
install_harness() {
    local platform="$1"
    log_step "Installing dag-harness v${VERSION}..."

    # Strategy 1: uv from git+https (preferred - always works, builds from source)
    if [ "${HARNESS_INSTALLER:-}" = "uv" ]; then
        if install_via_uv_github; then
            return 0
        fi
        log_warn "uv installation failed, trying pip..."
    fi

    # Strategy 2: pip from git+https
    if [ -n "${HARNESS_PYTHON:-}" ]; then
        if install_via_pip_github; then
            return 0
        fi
        log_warn "pip from GitHub failed, trying wheel URL..."
    fi

    # Strategy 3: Direct wheel URL (faster if release exists)
    if [ -n "${HARNESS_PYTHON:-}" ]; then
        if install_via_wheel_url; then
            return 0
        fi
        log_warn "Wheel download failed, trying binary..."
    fi

    # Strategy 4: Binary fallback (platform-specific pre-built binary)
    if install_via_binary "$platform"; then
        return 0
    fi

    log_error "All installation methods failed"
    log_info "Try manual installation: pip install git+https://github.com/${GITHUB_REPO}.git"
    return 1
}

# Discover credentials from common locations
discover_credentials() {
    log_step "Discovering credentials..."
    local found=0

    # Environment variables
    for var in GITLAB_TOKEN GL_TOKEN GLAB_TOKEN; do
        if [ -n "${!var:-}" ]; then
            log_success "Found \$$var in environment"
            found=$((found + 1))
            break
        fi
    done

    # Check .env files
    local env_files=(
        ".env"
        ".env.local"
        "${HOME}/.env"
        "${HOME}/.config/harness/.env"
        "${HOME}/.harness/.env"
    )

    # Add parent directories
    local dir="$PWD"
    for _ in {1..5}; do
        dir=$(dirname "$dir")
        env_files+=("${dir}/.env")
    done

    for env_file in "${env_files[@]}"; do
        if [ -f "$env_file" ]; then
            if grep -qE '^(GITLAB_TOKEN|GL_TOKEN|GLAB_TOKEN)=' "$env_file" 2>/dev/null; then
                log_success "Found GitLab token in $env_file"
                found=$((found + 1))
                break
            fi
        fi
    done

    # Check glab CLI config
    local glab_config="${HOME}/.config/glab-cli/config.yml"
    if [ -f "$glab_config" ]; then
        if grep -q "token:" "$glab_config" 2>/dev/null; then
            log_success "Found glab CLI authentication"
            found=$((found + 1))
        fi
    fi

    # Check macOS Keychain
    if [ "$(uname -s)" = "Darwin" ]; then
        if security find-generic-password -s "GITLAB_TOKEN" >/dev/null 2>&1; then
            log_success "Found GITLAB_TOKEN in macOS Keychain"
            found=$((found + 1))
        fi
    fi

    # Note: ANTHROPIC_API_KEY and DISCORD_WEBHOOK_URL checks removed
    # - ANTHROPIC_API_KEY: Users authenticated via Claude Code (Max subscription)
    # - DISCORD_WEBHOOK_URL: Simplified to email-only notifications

    if [ $found -eq 0 ]; then
        log_warn "No credentials discovered. Run 'harness credentials --prompt' to configure."
    else
        log_info "Discovered $found credential source(s)"
    fi
}

# Run self-tests
run_selftest() {
    log_step "Running self-tests..."

    if command_exists harness; then
        if harness bootstrap --check-only; then
            log_success "Self-tests passed"
            return 0
        else
            log_warn "Some self-tests failed"
            return 1
        fi
    else
        log_warn "harness command not found in PATH. Try: source ~/.bashrc"
        return 1
    fi
}

# Run harness init if inside a git repository
run_harness_init() {
    if ! command_exists harness; then
        log_warn "harness command not found in PATH after installation"
        return 1
    fi

    # Check if we're inside a git repository
    if git rev-parse --show-toplevel &>/dev/null; then
        log_step "Detected git repository, running harness init..."
        if harness init; then
            log_success "harness init completed"
            return 0
        else
            log_warn "harness init failed (non-fatal)"
            return 1
        fi
    else
        log_info "Not inside a git repository, skipping harness init"
        log_info "Run 'harness init' manually inside your project"
        return 0
    fi
}

# Show post-install instructions
show_instructions() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN} dag-harness v${VERSION} installed successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify installation:    harness --version"
    echo "  2. Initialize a project:   cd /path/to/repo && harness init"
    echo "  3. Configure credentials:  harness credentials --prompt"
    echo "  4. Check system status:    harness check"
    echo ""
    echo "Alternative installation methods:"
    echo "  # Via git+https (build from source)"
    echo "  uv tool install git+https://github.com/${GITHUB_REPO}.git@v${VERSION}"
    echo ""
    echo "  # Via wheel URL (faster)"
    echo "  pip install ${GITHUB_RELEASES_URL}/download/v${VERSION}/dag_harness-${VERSION}-py3-none-any.whl"
    echo ""
    echo "Documentation: ${DOCS_URL}"
    echo "GitHub:        https://github.com/${GITHUB_REPO}"
    echo "GitLab:        https://gitlab.com/${GITLAB_REPO}"
    echo ""
}

# Main function
main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     dag-harness Bootstrap Installer      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
    echo ""

    # Detect platform
    local platform
    platform=$(detect_platform)
    log_info "Detected platform: $platform"

    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Please install missing prerequisites and try again."
        exit 1
    fi

    # Install harness
    if ! install_harness "$platform"; then
        log_error "Installation failed."
        exit 1
    fi

    # Discover credentials
    discover_credentials

    # Run harness init if inside a git repo
    run_harness_init || true  # Don't fail on init issues

    # Run self-tests
    run_selftest || true  # Don't fail on selftest issues

    # Show instructions
    show_instructions
}

# Run main function
main "$@"
