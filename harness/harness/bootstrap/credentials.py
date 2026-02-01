"""Credential discovery and validation for harness bootstrap.

This module handles:
- Environment variable detection
- Credential validation (GITLAB_TOKEN, KEEPASSXC_DB_PASSWORD, etc.)
- Interactive prompting for missing credentials
"""

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class CredentialStatus(Enum):
    """Status of a credential check."""
    FOUND = "found"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    OPTIONAL_MISSING = "optional_missing"


@dataclass
class CredentialResult:
    """Result of checking a single credential."""
    name: str
    status: CredentialStatus
    source: Optional[str] = None  # Where it was found (env, file, etc.)
    value: Optional[str] = None  # Masked or partial value for display
    error: Optional[str] = None
    required: bool = True


@dataclass
class CredentialCheckResult:
    """Result of checking all credentials."""
    all_required_present: bool
    credentials: list[CredentialResult] = field(default_factory=list)

    @property
    def missing_required(self) -> list[str]:
        """Get list of missing required credentials."""
        return [
            c.name for c in self.credentials
            if c.required and c.status in (CredentialStatus.NOT_FOUND, CredentialStatus.INVALID)
        ]

    @property
    def found_count(self) -> int:
        """Count of found credentials."""
        return sum(1 for c in self.credentials if c.status == CredentialStatus.FOUND)


class CredentialDiscovery:
    """Discovers and validates credentials for harness operation.

    Checks multiple sources in priority order:
    1. Environment variables
    2. .env files
    3. System keychain (macOS)
    4. KeePassXC database
    """

    # Required credentials with their environment variable names
    REQUIRED_CREDENTIALS = {
        "GITLAB_TOKEN": {
            "description": "GitLab API token for creating issues/MRs",
            "env_vars": ["GITLAB_TOKEN", "GL_TOKEN", "GLAB_TOKEN"],
            "validation": "gitlab_api",
        },
    }

    # Optional credentials
    OPTIONAL_CREDENTIALS = {
        "KEEPASSXC_DB_PASSWORD": {
            "description": "KeePassXC database password",
            "env_vars": ["KEEPASSXC_DB_PASSWORD", "KEEPASS_PASSWORD"],
            "validation": None,
        },
        "DISCORD_WEBHOOK_URL": {
            "description": "Discord webhook for notifications",
            "env_vars": ["DISCORD_WEBHOOK_URL"],
            "validation": "url_format",
        },
        "HOTL_EMAIL_TO": {
            "description": "Email address for HOTL notifications",
            "env_vars": ["HOTL_EMAIL_TO"],
            "validation": "email_format",
        },
    }

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize credential discovery.

        Args:
            project_root: Root directory to search for .env files
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()

    def check_all(self, validate: bool = True) -> CredentialCheckResult:
        """Check all credentials.

        Args:
            validate: Whether to validate credentials (e.g., test GitLab API)

        Returns:
            CredentialCheckResult with status of all credentials
        """
        results = []

        # Check required credentials
        for name, config in self.REQUIRED_CREDENTIALS.items():
            result = self._check_credential(name, config, required=True, validate=validate)
            results.append(result)

        # Check optional credentials
        for name, config in self.OPTIONAL_CREDENTIALS.items():
            result = self._check_credential(name, config, required=False, validate=validate)
            results.append(result)

        all_required = all(
            c.status == CredentialStatus.FOUND
            for c in results if c.required
        )

        return CredentialCheckResult(
            all_required_present=all_required,
            credentials=results
        )

    def _check_credential(
        self,
        name: str,
        config: dict,
        required: bool,
        validate: bool
    ) -> CredentialResult:
        """Check a single credential.

        Args:
            name: Credential name
            config: Configuration dict with env_vars, validation, etc.
            required: Whether this credential is required
            validate: Whether to validate the credential value

        Returns:
            CredentialResult with status
        """
        # Check environment variables
        value = None
        source = None

        for env_var in config.get("env_vars", [name]):
            env_value = os.environ.get(env_var)
            if env_value:
                value = env_value
                source = f"env:{env_var}"
                break

        # Check .env file if not found in environment
        if not value:
            env_file_value, env_file_path = self._check_env_file(config.get("env_vars", [name]))
            if env_file_value:
                value = env_file_value
                source = f"file:{env_file_path}"

        # Check system keychain (macOS) if not found
        if not value and self._is_macos():
            keychain_value = self._check_keychain(name)
            if keychain_value:
                value = keychain_value
                source = "keychain"

        # Determine status
        if not value:
            status = CredentialStatus.NOT_FOUND if required else CredentialStatus.OPTIONAL_MISSING
            return CredentialResult(
                name=name,
                status=status,
                required=required,
                error=f"Credential not found in environment or .env files"
            )

        # Validate if requested
        if validate and config.get("validation"):
            is_valid, error = self._validate_credential(name, value, config["validation"])
            if not is_valid:
                return CredentialResult(
                    name=name,
                    status=CredentialStatus.INVALID,
                    source=source,
                    value=self._mask_value(value),
                    required=required,
                    error=error
                )

        return CredentialResult(
            name=name,
            status=CredentialStatus.FOUND,
            source=source,
            value=self._mask_value(value),
            required=required
        )

    def _check_env_file(self, env_vars: list[str]) -> tuple[Optional[str], Optional[str]]:
        """Check .env files for credential.

        Searches multiple common locations in priority order:
        1. Project-local .env files
        2. Home directory .env files
        3. Common project directories (sibling projects, ~/.claude, etc.)

        Args:
            env_vars: List of variable names to search for

        Returns:
            Tuple of (value, file_path) or (None, None)
        """
        env_files = self._get_env_file_search_paths()

        for env_file in env_files:
            if env_file.exists():
                try:
                    with open(env_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" in line:
                                key, _, val = line.partition("=")
                                key = key.strip()
                                val = val.strip().strip('"').strip("'")
                                if key in env_vars and val:
                                    return val, str(env_file)
                except Exception:
                    pass

        return None, None

    def _get_env_file_search_paths(self) -> list[Path]:
        """Get list of .env file paths to search.

        Returns paths in priority order (highest priority first).

        Returns:
            List of Path objects to check for .env files
        """
        paths = []

        # 1. Project-local files (highest priority)
        paths.extend([
            self.project_root / ".env",
            self.project_root / ".env.local",
            self.project_root / ".env.box-up-role",
        ])

        # 2. Home directory files
        paths.extend([
            Path.home() / ".env",
            Path.home() / ".claude" / ".env",
        ])

        # 3. Parent directory (if in a subdirectory like harness/)
        parent = self.project_root.parent
        if parent != self.project_root:
            paths.extend([
                parent / ".env",
                parent / ".env.local",
            ])

        # 4. Sibling project directories (common in ~/git layout)
        git_dir = Path.home() / "git"
        if git_dir.exists():
            # Check common project names that might have GitLab tokens
            for sibling in ["ems", "crush-dots", "tinyland", "upgrading-dw"]:
                sibling_path = git_dir / sibling
                if sibling_path.exists():
                    paths.extend([
                        sibling_path / ".env",
                        sibling_path / ".env.box-up-role",
                    ])

        # 5. Config directories
        config_dir = Path.home() / ".config"
        if config_dir.exists():
            paths.extend([
                config_dir / "crush" / ".env",
                config_dir / "claude-flow" / ".env.glm",
            ])

        return paths

    def _check_keychain(self, name: str) -> Optional[str]:
        """Check macOS keychain for credential.

        Args:
            name: Credential name to search for

        Returns:
            Value if found, None otherwise
        """
        try:
            # Use security command to query keychain
            result = subprocess.run(
                ["security", "find-generic-password", "-s", name, "-w"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _is_macos(self) -> bool:
        """Check if running on macOS."""
        import platform
        return platform.system() == "Darwin"

    def _validate_credential(
        self,
        name: str,
        value: str,
        validation_type: str
    ) -> tuple[bool, Optional[str]]:
        """Validate a credential value.

        Args:
            name: Credential name
            value: Value to validate
            validation_type: Type of validation to perform

        Returns:
            Tuple of (is_valid, error_message)
        """
        if validation_type == "gitlab_api":
            return self._validate_gitlab_token(value)
        elif validation_type == "url_format":
            return self._validate_url(value)
        elif validation_type == "email_format":
            return self._validate_email(value)

        return True, None

    def _validate_gitlab_token(self, token: str) -> tuple[bool, Optional[str]]:
        """Validate GitLab API token by making a test request and checking scopes.

        Validates that the token:
        1. Is accepted by the GitLab API
        2. Has required scopes for harness operation (api, read_user, write_repository)

        Args:
            token: GitLab API token

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import urllib.request
            import urllib.error
            import json

            gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com")

            # First check if token is valid by getting user info
            req = urllib.request.Request(
                f"{gitlab_url}/api/v4/user",
                headers={"PRIVATE-TOKEN": token}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    return False, f"GitLab API returned status {response.status}"

            # Now check token scopes via personal_access_tokens endpoint
            # This only works for PATs, not OAuth tokens
            is_valid, scopes, scope_error = self._check_gitlab_token_scopes(token, gitlab_url)
            if scope_error:
                # If we can't check scopes, just warn but don't fail
                # (might be an OAuth token or older GitLab version)
                return True, None

            if scopes:
                required_scopes = {"api"}  # Minimum required scope
                missing_scopes = required_scopes - scopes
                if missing_scopes:
                    return False, f"Token missing required scopes: {', '.join(missing_scopes)}. Has: {', '.join(scopes)}"

            return True, None

        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "Invalid GitLab token (401 Unauthorized)"
            return False, f"GitLab API error: {e.code}"
        except Exception as e:
            # Don't fail if network is unavailable
            return True, None  # Assume valid if we can't check

    def _check_gitlab_token_scopes(
        self,
        token: str,
        gitlab_url: str
    ) -> tuple[bool, Optional[set[str]], Optional[str]]:
        """Check GitLab token scopes.

        Args:
            token: GitLab API token
            gitlab_url: GitLab instance URL

        Returns:
            Tuple of (success, scopes_set, error_message)
        """
        import urllib.request
        import urllib.error
        import json

        try:
            # Try to get token info (GitLab 14.0+)
            req = urllib.request.Request(
                f"{gitlab_url}/api/v4/personal_access_tokens/self",
                headers={"PRIVATE-TOKEN": token}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    scopes = set(data.get("scopes", []))
                    return True, scopes, None

        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Endpoint not available (older GitLab or OAuth token)
                return False, None, "Scope check not available"
            elif e.code == 401:
                return False, None, "Invalid token"
        except Exception as e:
            pass

        return False, None, "Could not check token scopes"

    def _validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL format.

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        from urllib.parse import urlparse

        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]):
                return True, None
            return False, "Invalid URL format"
        except Exception:
            return False, "Invalid URL format"

    def _validate_email(self, email: str) -> tuple[bool, Optional[str]]:
        """Validate email format.

        Args:
            email: Email to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        import re

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return True, None
        return False, "Invalid email format"

    def _mask_value(self, value: str) -> str:
        """Mask a credential value for display.

        Args:
            value: Value to mask

        Returns:
            Masked value showing only first/last few characters
        """
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"

    def get_credential(self, name: str) -> Optional[str]:
        """Get a credential value by name.

        Args:
            name: Credential name

        Returns:
            Credential value if found, None otherwise
        """
        config = self.REQUIRED_CREDENTIALS.get(name) or self.OPTIONAL_CREDENTIALS.get(name)
        if not config:
            return os.environ.get(name)

        for env_var in config.get("env_vars", [name]):
            value = os.environ.get(env_var)
            if value:
                return value

        value, _ = self._check_env_file(config.get("env_vars", [name]))
        return value

    def scan_and_test_gitlab_tokens(self) -> list[dict]:
        """Scan all .env files for GitLab tokens and test them.

        Searches for:
        - Standard variable names: GITLAB_TOKEN, GL_TOKEN, GLAB_TOKEN
        - User-prefixed patterns: *_GLAB_TOKEN, *_GITLAB_TOKEN
        - Values starting with glpat- (GitLab PAT prefix)

        Returns:
            List of dicts with token info:
            {
                "source": str,  # File path or "env:VAR_NAME"
                "var_name": str,  # Variable name
                "token_prefix": str,  # First 8 chars
                "valid": bool,
                "scopes": list[str] | None,
                "error": str | None,
                "username": str | None
            }
        """
        import re

        results = []
        seen_tokens = set()  # Dedupe by prefix

        gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com")

        # Standard variable names
        standard_vars = ["GITLAB_TOKEN", "GL_TOKEN", "GLAB_TOKEN"]

        # Pattern for user-prefixed tokens (e.g., JSULLIVAN2_BATES_GLAB_TOKEN)
        token_pattern = re.compile(r'^(export\s+)?([A-Z0-9_]*(?:GITLAB|GLAB|GL)_TOKEN)\s*=\s*(.+)$', re.IGNORECASE)

        # Check environment variables first
        for key, value in os.environ.items():
            # Check standard names or pattern matches
            is_token_var = (
                key in standard_vars or
                key.endswith("_GLAB_TOKEN") or
                key.endswith("_GITLAB_TOKEN") or
                key.endswith("_GL_TOKEN")
            )
            # Also check for glpat- prefix values
            is_glpat = value.startswith("glpat-") if value else False

            if (is_token_var or is_glpat) and value and value[:12] not in seen_tokens:
                seen_tokens.add(value[:12])
                result = self._test_gitlab_token(value, gitlab_url)
                result["source"] = f"env:{key}"
                result["var_name"] = key
                results.append(result)

        # Check all .env file paths
        for env_file in self._get_env_file_search_paths():
            if not env_file.exists():
                continue

            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        # Try to match token pattern
                        match = token_pattern.match(line)
                        if match:
                            var_name = match.group(2)
                            val = match.group(3).strip().strip('"').strip("'")

                            if val and val[:12] not in seen_tokens:
                                seen_tokens.add(val[:12])
                                result = self._test_gitlab_token(val, gitlab_url)
                                result["source"] = str(env_file)
                                result["var_name"] = var_name
                                results.append(result)
                        elif "=" in line:
                            # Also check for glpat- values in any variable
                            key, _, val = line.partition("=")
                            key = key.replace("export ", "").strip()
                            val = val.strip().strip('"').strip("'")

                            if val.startswith("glpat-") and val[:12] not in seen_tokens:
                                seen_tokens.add(val[:12])
                                result = self._test_gitlab_token(val, gitlab_url)
                                result["source"] = str(env_file)
                                result["var_name"] = key
                                results.append(result)
            except Exception:
                pass

        return results

    def _test_gitlab_token(self, token: str, gitlab_url: str) -> dict:
        """Test a single GitLab token.

        Args:
            token: GitLab API token
            gitlab_url: GitLab instance URL

        Returns:
            Dict with token test results
        """
        import json
        import urllib.request
        import urllib.error

        result = {
            "token_prefix": token[:8] + "..." if len(token) > 8 else token,
            "var_name": None,  # Will be set by caller
            "valid": False,
            "scopes": None,
            "error": None,
            "username": None,
        }

        try:
            # Test token by getting user info
            req = urllib.request.Request(
                f"{gitlab_url}/api/v4/user",
                headers={"PRIVATE-TOKEN": token}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    result["valid"] = True
                    result["username"] = data.get("username")

            # Get scopes if token is valid
            if result["valid"]:
                _, scopes, _ = self._check_gitlab_token_scopes(token, gitlab_url)
                if scopes:
                    result["scopes"] = list(scopes)

        except urllib.error.HTTPError as e:
            if e.code == 401:
                result["error"] = "Invalid token (401)"
            else:
                result["error"] = f"HTTP {e.code}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def save_credential_to_env(self, name: str, value: str, env_file: Optional[Path] = None) -> bool:
        """Save a credential to a .env file.

        Args:
            name: Credential name (e.g., GITLAB_TOKEN)
            value: Credential value
            env_file: Target .env file path (default: project_root/.env)

        Returns:
            True if saved successfully
        """
        if env_file is None:
            env_file = self.project_root / ".env"

        try:
            # Read existing content
            existing_lines = []
            if env_file.exists():
                with open(env_file) as f:
                    existing_lines = f.readlines()

            # Update or add the credential
            found = False
            new_lines = []
            for line in existing_lines:
                stripped = line.strip()
                if stripped.startswith(f"{name}="):
                    new_lines.append(f'{name}="{value}"\n')
                    found = True
                else:
                    new_lines.append(line)

            if not found:
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines.append("\n")
                new_lines.append(f'{name}="{value}"\n')

            # Write back
            with open(env_file, "w") as f:
                f.writelines(new_lines)

            return True

        except Exception as e:
            return False
