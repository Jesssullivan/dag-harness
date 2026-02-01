"""
Ansible role parser for extracting dependencies and credentials.

Parses Ansible role structures to extract:
- Role dependencies from meta/main.yml
- Credential references from tasks, defaults, and Makefiles
- Variable patterns that hint at credential usage

Phase 2 of the harness project.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from harness.db.models import Credential, DependencyType, Role, RoleDependency
from harness.db.state import StateDB

logger = logging.getLogger(__name__)


# Credential detection patterns
CREDENTIAL_PATTERNS = [
    r"keepass[_-]?xc|\.kdbx|kdbx_password",
    r"ansible[_-]?vault|vault[_-]?password",
    r"password|secret|token|api[_-]?key|private[_-]?key",
    r"sa_password|sql[_-]?password|mssql",
    r"lookup\(['\"]env['\"]",
]

# Compiled regex for performance
CREDENTIAL_REGEX = re.compile(
    "|".join(f"({p})" for p in CREDENTIAL_PATTERNS),
    re.IGNORECASE
)

# Pattern to extract KeePassXC entry names
# Matches both: lookup('keepassxc_password', 'Entry') and keepassxc_password('Entry')
KEEPASS_ENTRY_PATTERN = re.compile(
    r"(?:lookup\s*\(\s*['\"])?keepassxc_password['\"]?\s*,?\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE
)

# Pattern to extract vault variable names
VAULT_VAR_PATTERN = re.compile(
    r"vault_(\w+)",
    re.IGNORECASE
)

# Pattern for Makefile credential references
MAKEFILE_CRED_PATTERN = re.compile(
    r"(?:KEEPASS|VAULT|SECRET|PASSWORD|TOKEN|API_KEY)[\w_]*\s*[:?]?=",
    re.IGNORECASE
)


@dataclass
class CredentialRef:
    """Reference to a credential found in source files."""

    name: str
    source_file: str
    source_line: int
    pattern_matched: str
    purpose: Optional[str] = None
    is_base58: bool = False
    attribute: Optional[str] = None

    def __hash__(self) -> int:
        """Enable use in sets for deduplication."""
        return hash((self.name, self.source_file, self.source_line))

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, CredentialRef):
            return NotImplemented
        return (
            self.name == other.name
            and self.source_file == other.source_file
            and self.source_line == other.source_line
        )


@dataclass
class RoleInfo:
    """Parsed information about an Ansible role."""

    name: str
    dependencies: list[str] = field(default_factory=list)
    credentials: list[CredentialRef] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    description: Optional[str] = None
    has_molecule_tests: bool = False
    molecule_path: Optional[str] = None
    wave: Optional[int] = None
    wave_name: Optional[str] = None

    def to_role(self) -> Role:
        """Convert to a Role database model."""
        return Role(
            name=self.name,
            wave=self.wave or 0,
            wave_name=self.wave_name,
            description=self.description,
            molecule_path=self.molecule_path,
            has_molecule_tests=self.has_molecule_tests,
        )


class AnsibleRoleParser:
    """
    Parser for Ansible role directories.

    Extracts dependencies, credentials, and metadata from:
    - roles/<role>/meta/main.yml
    - roles/<role>/tasks/*.yml
    - roles/<role>/defaults/main.yml
    - roles/<role>/Makefile or Makefile
    """

    def __init__(self, db: Optional[StateDB] = None):
        """
        Initialize the parser.

        Args:
            db: Optional StateDB instance for persisting parsed data
        """
        self.db = db
        self._credential_cache: dict[str, list[CredentialRef]] = {}

    def parse_role(self, role_path: Path) -> RoleInfo:
        """
        Parse a single Ansible role directory.

        Args:
            role_path: Path to the role directory (e.g., roles/common)

        Returns:
            RoleInfo with extracted metadata

        Raises:
            FileNotFoundError: If the role path doesn't exist
            ValueError: If the role path is not a valid role directory
        """
        role_path = Path(role_path)

        if not role_path.exists():
            raise FileNotFoundError(f"Role path does not exist: {role_path}")

        if not role_path.is_dir():
            raise ValueError(f"Role path is not a directory: {role_path}")

        role_name = role_path.name
        logger.info(f"Parsing role: {role_name}")

        # Initialize role info
        role_info = RoleInfo(name=role_name)

        # Parse meta/main.yml for dependencies
        meta_file = role_path / "meta" / "main.yml"
        if meta_file.exists():
            role_info.dependencies = self.extract_dependencies(meta_file)
            role_info.source_files.append(str(meta_file))

            # Try to get description from meta
            role_info.description = self._extract_description(meta_file)

        # Parse tasks for credentials
        tasks_dir = role_path / "tasks"
        if tasks_dir.exists() and tasks_dir.is_dir():
            task_credentials = self.extract_credentials(tasks_dir)
            role_info.credentials.extend(task_credentials)

            # Add task files to source files
            for task_file in tasks_dir.glob("*.yml"):
                role_info.source_files.append(str(task_file))

        # Parse defaults/main.yml for credential hints
        defaults_file = role_path / "defaults" / "main.yml"
        if defaults_file.exists():
            default_credentials = self._scan_yaml_file(defaults_file)
            role_info.credentials.extend(default_credentials)
            role_info.source_files.append(str(defaults_file))

        # Parse vars/main.yml for credential hints
        vars_file = role_path / "vars" / "main.yml"
        if vars_file.exists():
            var_credentials = self._scan_yaml_file(vars_file)
            role_info.credentials.extend(var_credentials)
            role_info.source_files.append(str(vars_file))

        # Scan Makefile for credentials
        makefile = role_path / "Makefile"
        if makefile.exists():
            makefile_credentials = self.scan_makefile(makefile)
            role_info.credentials.extend(makefile_credentials)
            role_info.source_files.append(str(makefile))

        # Check for molecule tests
        molecule_dir = role_path / "molecule"
        if molecule_dir.exists() and molecule_dir.is_dir():
            role_info.has_molecule_tests = True
            role_info.molecule_path = str(molecule_dir)

            # Also check default scenario
            default_scenario = molecule_dir / "default"
            if default_scenario.exists():
                role_info.molecule_path = str(default_scenario)

        # Deduplicate credentials
        role_info.credentials = self._deduplicate_credentials(role_info.credentials)

        logger.info(
            f"Parsed role {role_name}: "
            f"{len(role_info.dependencies)} deps, "
            f"{len(role_info.credentials)} credentials"
        )

        return role_info

    def extract_dependencies(self, meta_file: Path) -> list[str]:
        """
        Extract role dependencies from meta/main.yml.

        Args:
            meta_file: Path to meta/main.yml

        Returns:
            List of dependency role names
        """
        meta_file = Path(meta_file)

        if not meta_file.exists():
            logger.warning(f"Meta file does not exist: {meta_file}")
            return []

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML in {meta_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to read {meta_file}: {e}")
            return []

        if not content:
            return []

        dependencies: list[str] = []

        # Handle galaxy_info.dependencies or just dependencies
        deps = content.get("dependencies", [])

        if not deps:
            galaxy_info = content.get("galaxy_info", {})
            if galaxy_info:
                deps = galaxy_info.get("dependencies", [])

        if not deps:
            return []

        for dep in deps:
            if isinstance(dep, str):
                # Simple string dependency
                dependencies.append(dep)
            elif isinstance(dep, dict):
                # Dict-style dependency with role key
                role_name = dep.get("role") or dep.get("name")
                if role_name:
                    dependencies.append(role_name)

        logger.debug(f"Found {len(dependencies)} dependencies in {meta_file}")
        return dependencies

    def extract_credentials(self, tasks_dir: Path) -> list[CredentialRef]:
        """
        Extract credential references from task files.

        Args:
            tasks_dir: Path to the tasks directory

        Returns:
            List of CredentialRef objects
        """
        tasks_dir = Path(tasks_dir)

        if not tasks_dir.exists() or not tasks_dir.is_dir():
            logger.warning(f"Tasks directory does not exist: {tasks_dir}")
            return []

        credentials: list[CredentialRef] = []

        # Scan all YAML files in tasks directory
        for task_file in tasks_dir.glob("*.yml"):
            file_credentials = self._scan_yaml_file(task_file)
            credentials.extend(file_credentials)

        # Also check included task files
        for task_file in tasks_dir.glob("*.yaml"):
            file_credentials = self._scan_yaml_file(task_file)
            credentials.extend(file_credentials)

        return credentials

    def scan_makefile(self, makefile: Path) -> list[CredentialRef]:
        """
        Scan a Makefile for credential references.

        Args:
            makefile: Path to the Makefile

        Returns:
            List of CredentialRef objects
        """
        makefile = Path(makefile)

        if not makefile.exists():
            logger.warning(f"Makefile does not exist: {makefile}")
            return []

        credentials: list[CredentialRef] = []

        try:
            with open(makefile, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read Makefile {makefile}: {e}")
            return []

        for line_num, line in enumerate(lines, start=1):
            # Check for Makefile-specific credential patterns
            makefile_match = MAKEFILE_CRED_PATTERN.search(line)
            if makefile_match:
                cred_name = makefile_match.group(0).rstrip(":?=").strip()
                credentials.append(CredentialRef(
                    name=cred_name,
                    source_file=str(makefile),
                    source_line=line_num,
                    pattern_matched="makefile_variable",
                    purpose="Makefile credential variable",
                ))

            # Check for general credential patterns
            general_match = CREDENTIAL_REGEX.search(line)
            if general_match:
                matched_text = general_match.group(0)

                # Skip if already captured as Makefile variable
                if makefile_match and matched_text in makefile_match.group(0):
                    continue

                # Try to extract a meaningful name
                cred_name = self._extract_credential_name(line, matched_text)

                credentials.append(CredentialRef(
                    name=cred_name,
                    source_file=str(makefile),
                    source_line=line_num,
                    pattern_matched=matched_text,
                    purpose="Credential reference in Makefile",
                ))

        return credentials

    def _scan_yaml_file(self, yaml_file: Path) -> list[CredentialRef]:
        """
        Scan a YAML file for credential references.

        Args:
            yaml_file: Path to the YAML file

        Returns:
            List of CredentialRef objects
        """
        yaml_file = Path(yaml_file)

        if not yaml_file.exists():
            return []

        credentials: list[CredentialRef] = []

        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read {yaml_file}: {e}")
            return []

        for line_num, line in enumerate(lines, start=1):
            # Check for KeePassXC entry references
            keepass_match = KEEPASS_ENTRY_PATTERN.search(line)
            if keepass_match:
                entry_name = keepass_match.group(1)

                # Check for attribute specification
                attribute = None
                is_base58 = False

                if "attribute=" in line.lower():
                    attr_match = re.search(r"attribute\s*=\s*['\"]?(\w+)['\"]?", line, re.IGNORECASE)
                    if attr_match:
                        attribute = attr_match.group(1)

                if "base58" in line.lower():
                    is_base58 = True

                credentials.append(CredentialRef(
                    name=entry_name,
                    source_file=str(yaml_file),
                    source_line=line_num,
                    pattern_matched="keepassxc_password",
                    purpose=f"KeePassXC entry: {entry_name}",
                    is_base58=is_base58,
                    attribute=attribute,
                ))
                continue

            # Check for vault variable patterns
            vault_match = VAULT_VAR_PATTERN.search(line)
            if vault_match:
                var_name = vault_match.group(1)
                credentials.append(CredentialRef(
                    name=f"vault_{var_name}",
                    source_file=str(yaml_file),
                    source_line=line_num,
                    pattern_matched="vault_variable",
                    purpose=f"Ansible Vault variable: {var_name}",
                ))
                continue

            # Check for general credential patterns
            general_match = CREDENTIAL_REGEX.search(line)
            if general_match:
                matched_text = general_match.group(0)
                cred_name = self._extract_credential_name(line, matched_text)

                # Determine purpose from context
                purpose = self._infer_purpose(line, matched_text)

                # Check for base58 encoding
                is_base58 = "base58" in line.lower()

                credentials.append(CredentialRef(
                    name=cred_name,
                    source_file=str(yaml_file),
                    source_line=line_num,
                    pattern_matched=matched_text,
                    purpose=purpose,
                    is_base58=is_base58,
                ))

        return credentials

    def _extract_credential_name(self, line: str, matched_text: str) -> str:
        """
        Extract a meaningful credential name from a line.

        Args:
            line: The full line of text
            matched_text: The text that matched the credential pattern

        Returns:
            A meaningful credential name
        """
        # Try to find a variable assignment pattern
        var_pattern = re.compile(r"(\w+)\s*[:=]")
        var_match = var_pattern.search(line)

        if var_match:
            return var_match.group(1)

        # Try to find a lookup pattern with entry name
        lookup_pattern = re.compile(r"lookup\s*\(\s*['\"](\w+)['\"]")
        lookup_match = lookup_pattern.search(line)

        if lookup_match:
            return f"lookup_{lookup_match.group(1)}"

        # Fall back to the matched text, cleaned up
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", matched_text)
        return clean_name.lower()

    def _infer_purpose(self, line: str, matched_text: str) -> str:
        """
        Infer the purpose of a credential from context.

        Args:
            line: The full line of text
            matched_text: The matched pattern

        Returns:
            A description of the credential's purpose
        """
        line_lower = line.lower()

        if "sql" in line_lower or "mssql" in line_lower or "sa_" in line_lower:
            return "SQL Server authentication"

        if "api" in line_lower:
            return "API authentication"

        if "keepass" in line_lower:
            return "KeePassXC credential lookup"

        if "vault" in line_lower:
            return "Ansible Vault encrypted value"

        if "ssh" in line_lower or "private_key" in line_lower:
            return "SSH key or authentication"

        if "token" in line_lower:
            return "Authentication token"

        if "password" in line_lower:
            return "Password credential"

        if "secret" in line_lower:
            return "Secret value"

        return f"Credential reference ({matched_text})"

    def _extract_description(self, meta_file: Path) -> Optional[str]:
        """
        Extract role description from meta/main.yml.

        Args:
            meta_file: Path to meta/main.yml

        Returns:
            Role description or None
        """
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except Exception:
            return None

        if not content:
            return None

        # Try galaxy_info.description first
        galaxy_info = content.get("galaxy_info", {})
        if galaxy_info and "description" in galaxy_info:
            return galaxy_info["description"]

        # Try top-level description
        return content.get("description")

    def _deduplicate_credentials(self, credentials: list[CredentialRef]) -> list[CredentialRef]:
        """
        Remove duplicate credential references.

        Args:
            credentials: List of credentials, possibly with duplicates

        Returns:
            Deduplicated list
        """
        seen: set[tuple[str, str, int]] = set()
        unique: list[CredentialRef] = []

        for cred in credentials:
            key = (cred.name, cred.source_file, cred.source_line)
            if key not in seen:
                seen.add(key)
                unique.append(cred)

        return unique

    def parse_roles_directory(self, roles_dir: Path) -> list[RoleInfo]:
        """
        Parse all roles in a directory.

        Args:
            roles_dir: Path to the roles directory

        Returns:
            List of RoleInfo objects
        """
        roles_dir = Path(roles_dir)

        if not roles_dir.exists() or not roles_dir.is_dir():
            raise ValueError(f"Roles directory does not exist: {roles_dir}")

        roles: list[RoleInfo] = []

        for role_path in sorted(roles_dir.iterdir()):
            if not role_path.is_dir():
                continue

            # Skip hidden directories and common non-role directories
            if role_path.name.startswith(".") or role_path.name in {"__pycache__", "node_modules"}:
                continue

            # Check if it looks like an Ansible role
            if not self._is_valid_role(role_path):
                logger.debug(f"Skipping non-role directory: {role_path.name}")
                continue

            try:
                role_info = self.parse_role(role_path)
                roles.append(role_info)
            except Exception as e:
                logger.error(f"Failed to parse role {role_path.name}: {e}")

        return roles

    def _is_valid_role(self, role_path: Path) -> bool:
        """
        Check if a directory looks like a valid Ansible role.

        Args:
            role_path: Path to check

        Returns:
            True if it appears to be a valid role
        """
        # An Ansible role should have at least one of these directories
        role_indicators = ["tasks", "handlers", "vars", "defaults", "meta", "templates", "files"]

        for indicator in role_indicators:
            if (role_path / indicator).is_dir():
                return True

        return False

    # =========================================================================
    # DATABASE INTEGRATION
    # =========================================================================

    def populate_database(
        self,
        roles_dir: Path,
        wave_config: Optional[dict[str, tuple[int, str]]] = None
    ) -> dict[str, int]:
        """
        Parse roles and populate the database.

        Args:
            roles_dir: Path to the roles directory
            wave_config: Optional mapping of role name to (wave_number, wave_name)

        Returns:
            Dict with counts: roles, dependencies, credentials

        Raises:
            RuntimeError: If no database is configured
        """
        if not self.db:
            raise RuntimeError("No database configured for parser")

        roles_dir = Path(roles_dir)
        role_infos = self.parse_roles_directory(roles_dir)

        # First pass: create all roles
        role_id_map: dict[str, int] = {}

        for role_info in role_infos:
            # Apply wave configuration if provided
            if wave_config and role_info.name in wave_config:
                role_info.wave, role_info.wave_name = wave_config[role_info.name]

            role = role_info.to_role()
            role_id = self.db.upsert_role(role)
            role_id_map[role_info.name] = role_id

            logger.debug(f"Upserted role {role_info.name} with id {role_id}")

        # Second pass: create dependencies
        dependency_count = 0

        for role_info in role_infos:
            role_id = role_id_map.get(role_info.name)
            if not role_id:
                continue

            for dep_name in role_info.dependencies:
                dep_id = role_id_map.get(dep_name)
                if not dep_id:
                    # Dependency references an unknown role
                    logger.warning(
                        f"Role {role_info.name} depends on unknown role: {dep_name}"
                    )
                    continue

                # Find the source file for this dependency
                source_file = None
                for sf in role_info.source_files:
                    if "meta" in sf:
                        source_file = sf
                        break

                dependency = RoleDependency(
                    role_id=role_id,
                    depends_on_id=dep_id,
                    dependency_type=DependencyType.EXPLICIT,
                    source_file=source_file,
                )

                try:
                    self.db.add_dependency(dependency)
                    dependency_count += 1
                except Exception as e:
                    logger.error(f"Failed to add dependency {role_info.name} -> {dep_name}: {e}")

        # Third pass: create credentials
        credential_count = 0

        for role_info in role_infos:
            role_id = role_id_map.get(role_info.name)
            if not role_id:
                continue

            for cred_ref in role_info.credentials:
                credential = Credential(
                    role_id=role_id,
                    entry_name=cred_ref.name,
                    purpose=cred_ref.purpose,
                    is_base58=cred_ref.is_base58,
                    attribute=cred_ref.attribute,
                    source_file=cred_ref.source_file,
                    source_line=cred_ref.source_line,
                )

                try:
                    self.db.add_credential(credential)
                    credential_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to add credential {cred_ref.name} for {role_info.name}: {e}"
                    )

        logger.info(
            f"Database populated: {len(role_infos)} roles, "
            f"{dependency_count} dependencies, {credential_count} credentials"
        )

        return {
            "roles": len(role_infos),
            "dependencies": dependency_count,
            "credentials": credential_count,
        }

    def sync_role(self, role_path: Path) -> RoleInfo:
        """
        Parse a single role and sync it to the database.

        Args:
            role_path: Path to the role directory

        Returns:
            Parsed RoleInfo

        Raises:
            RuntimeError: If no database is configured
        """
        if not self.db:
            raise RuntimeError("No database configured for parser")

        role_info = self.parse_role(role_path)

        # Upsert the role
        role = role_info.to_role()
        role_id = self.db.upsert_role(role)

        # Clear existing credentials for this role and re-add
        # Note: Dependencies are handled via upsert which handles conflicts

        for cred_ref in role_info.credentials:
            credential = Credential(
                role_id=role_id,
                entry_name=cred_ref.name,
                purpose=cred_ref.purpose,
                is_base58=cred_ref.is_base58,
                attribute=cred_ref.attribute,
                source_file=cred_ref.source_file,
                source_line=cred_ref.source_line,
            )

            try:
                self.db.add_credential(credential)
            except Exception as e:
                logger.error(f"Failed to add credential {cred_ref.name}: {e}")

        logger.info(f"Synced role {role_info.name} to database")
        return role_info

    def add_implicit_dependency(
        self,
        role_name: str,
        depends_on: str,
        source_file: Optional[str] = None
    ) -> bool:
        """
        Add an implicit dependency between roles.

        Implicit dependencies are those discovered through credential
        sharing or other indirect relationships.

        Args:
            role_name: Name of the dependent role
            depends_on: Name of the role being depended on
            source_file: Optional source file where dependency was discovered

        Returns:
            True if dependency was added, False if failed

        Raises:
            RuntimeError: If no database is configured
        """
        if not self.db:
            raise RuntimeError("No database configured for parser")

        role = self.db.get_role(role_name)
        depends_on_role = self.db.get_role(depends_on)

        if not role or not role.id:
            logger.error(f"Role not found: {role_name}")
            return False

        if not depends_on_role or not depends_on_role.id:
            logger.error(f"Dependency role not found: {depends_on}")
            return False

        dependency = RoleDependency(
            role_id=role.id,
            depends_on_id=depends_on_role.id,
            dependency_type=DependencyType.IMPLICIT,
            source_file=source_file,
        )

        try:
            self.db.add_dependency(dependency)
            logger.info(f"Added implicit dependency: {role_name} -> {depends_on}")
            return True
        except Exception as e:
            logger.error(f"Failed to add implicit dependency: {e}")
            return False

    def add_credential_dependency(
        self,
        role_name: str,
        depends_on: str,
        credential_name: str
    ) -> bool:
        """
        Add a credential-based dependency between roles.

        Credential dependencies are those where one role uses credentials
        that are managed/provided by another role.

        Args:
            role_name: Name of the dependent role
            depends_on: Name of the role that provides the credential
            credential_name: Name of the shared credential

        Returns:
            True if dependency was added, False if failed

        Raises:
            RuntimeError: If no database is configured
        """
        if not self.db:
            raise RuntimeError("No database configured for parser")

        role = self.db.get_role(role_name)
        depends_on_role = self.db.get_role(depends_on)

        if not role or not role.id:
            logger.error(f"Role not found: {role_name}")
            return False

        if not depends_on_role or not depends_on_role.id:
            logger.error(f"Dependency role not found: {depends_on}")
            return False

        dependency = RoleDependency(
            role_id=role.id,
            depends_on_id=depends_on_role.id,
            dependency_type=DependencyType.CREDENTIAL,
            source_file=f"credential:{credential_name}",
        )

        try:
            self.db.add_dependency(dependency)
            logger.info(
                f"Added credential dependency: {role_name} -> {depends_on} "
                f"(via {credential_name})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add credential dependency: {e}")
            return False


def create_parser(db: Optional[StateDB] = None) -> AnsibleRoleParser:
    """
    Factory function to create a parser instance.

    Args:
        db: Optional StateDB instance

    Returns:
        Configured AnsibleRoleParser
    """
    return AnsibleRoleParser(db=db)
