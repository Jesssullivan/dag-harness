"""
Tests for the Ansible role parser.
"""

import tempfile
from pathlib import Path

import pytest

from harness.db.state import StateDB
from harness.parsers import AnsibleRoleParser, CredentialRef, RoleInfo


class TestCredentialRef:
    """Tests for CredentialRef dataclass."""

    def test_create_credential_ref(self):
        """Test creating a CredentialRef."""
        cred = CredentialRef(
            name="test_password",
            source_file="/path/to/tasks/main.yml",
            source_line=42,
            pattern_matched="password",
        )
        assert cred.name == "test_password"
        assert cred.source_line == 42
        assert cred.is_base58 is False

    def test_credential_ref_with_attribute(self):
        """Test CredentialRef with attribute."""
        cred = CredentialRef(
            name="keepass_entry",
            source_file="/path/to/tasks/main.yml",
            source_line=10,
            pattern_matched="keepassxc_password",
            attribute="username",
            is_base58=True,
        )
        assert cred.attribute == "username"
        assert cred.is_base58 is True

    def test_credential_ref_hash(self):
        """Test CredentialRef hash for set operations."""
        cred1 = CredentialRef(
            name="test",
            source_file="/a/b.yml",
            source_line=1,
            pattern_matched="password",
        )
        cred2 = CredentialRef(
            name="test",
            source_file="/a/b.yml",
            source_line=1,
            pattern_matched="secret",
        )
        # Same name, file, line = same hash
        assert hash(cred1) == hash(cred2)

    def test_credential_ref_equality(self):
        """Test CredentialRef equality."""
        cred1 = CredentialRef(
            name="test",
            source_file="/a/b.yml",
            source_line=1,
            pattern_matched="password",
        )
        cred2 = CredentialRef(
            name="test",
            source_file="/a/b.yml",
            source_line=1,
            pattern_matched="secret",
        )
        assert cred1 == cred2


class TestRoleInfo:
    """Tests for RoleInfo dataclass."""

    def test_create_role_info(self):
        """Test creating a RoleInfo."""
        role_info = RoleInfo(name="common")
        assert role_info.name == "common"
        assert role_info.dependencies == []
        assert role_info.credentials == []
        assert role_info.has_molecule_tests is False

    def test_role_info_to_role(self):
        """Test converting RoleInfo to Role model."""
        role_info = RoleInfo(
            name="test_role",
            wave=2,
            wave_name="Core Platform",
            description="A test role",
            has_molecule_tests=True,
            molecule_path="/path/to/molecule",
        )
        role = role_info.to_role()
        assert role.name == "test_role"
        assert role.wave == 2
        assert role.wave_name == "Core Platform"
        assert role.description == "A test role"
        assert role.has_molecule_tests is True


class TestAnsibleRoleParser:
    """Tests for AnsibleRoleParser."""

    @pytest.fixture
    def temp_role_dir(self):
        """Create a temporary role directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            role_dir = Path(tmpdir) / "test_role"
            role_dir.mkdir()

            # Create standard role directories
            (role_dir / "tasks").mkdir()
            (role_dir / "defaults").mkdir()
            (role_dir / "meta").mkdir()
            (role_dir / "vars").mkdir()

            yield role_dir

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return AnsibleRoleParser()

    def test_parse_empty_role(self, temp_role_dir, parser):
        """Test parsing an empty role."""
        role_info = parser.parse_role(temp_role_dir)
        assert role_info.name == "test_role"
        assert role_info.dependencies == []
        assert role_info.credentials == []

    def test_parse_role_with_dependencies(self, temp_role_dir, parser):
        """Test parsing role with meta/main.yml dependencies."""
        meta_file = temp_role_dir / "meta" / "main.yml"
        meta_file.write_text("""
dependencies:
  - common
  - role: windows_prerequisites
  - name: iis-config
""")
        role_info = parser.parse_role(temp_role_dir)
        assert len(role_info.dependencies) == 3
        assert "common" in role_info.dependencies
        assert "windows_prerequisites" in role_info.dependencies
        assert "iis-config" in role_info.dependencies

    def test_parse_role_with_galaxy_dependencies(self, temp_role_dir, parser):
        """Test parsing role with galaxy_info dependencies."""
        meta_file = temp_role_dir / "meta" / "main.yml"
        meta_file.write_text("""
galaxy_info:
  author: test
  description: Test role
  dependencies:
    - common
    - infrastructure
""")
        role_info = parser.parse_role(temp_role_dir)
        assert len(role_info.dependencies) == 2
        assert role_info.description == "Test role"

    def test_parse_role_with_credentials_in_tasks(self, temp_role_dir, parser):
        """Test parsing role with credential patterns in tasks."""
        tasks_file = temp_role_dir / "tasks" / "main.yml"
        tasks_file.write_text("""
- name: Set password
  set_fact:
    db_password: "{{ vault_db_password }}"

- name: Get keepass entry
  set_fact:
    api_key: "{{ lookup('keepassxc_password', 'API/MyService') }}"
""")
        role_info = parser.parse_role(temp_role_dir)
        assert len(role_info.credentials) >= 2

    def test_parse_role_with_keepass_attribute(self, temp_role_dir, parser):
        """Test parsing KeePassXC with attribute specification."""
        tasks_file = temp_role_dir / "tasks" / "main.yml"
        tasks_file.write_text("""
- name: Get username
  set_fact:
    username: "{{ lookup('keepassxc_password', 'DB/Admin', attribute='username') }}"
""")
        role_info = parser.parse_role(temp_role_dir)
        assert len(role_info.credentials) >= 1
        keepass_cred = next(
            (c for c in role_info.credentials if c.pattern_matched == "keepassxc_password"),
            None
        )
        assert keepass_cred is not None
        assert keepass_cred.attribute == "username"

    def test_parse_role_with_molecule(self, temp_role_dir, parser):
        """Test detecting molecule tests."""
        (temp_role_dir / "molecule").mkdir()
        (temp_role_dir / "molecule" / "default").mkdir()

        role_info = parser.parse_role(temp_role_dir)
        assert role_info.has_molecule_tests is True
        assert "molecule" in role_info.molecule_path

    def test_extract_dependencies_nonexistent_file(self, parser):
        """Test extracting dependencies from nonexistent file."""
        deps = parser.extract_dependencies(Path("/nonexistent/meta/main.yml"))
        assert deps == []

    def test_extract_credentials_nonexistent_dir(self, parser):
        """Test extracting credentials from nonexistent directory."""
        creds = parser.extract_credentials(Path("/nonexistent/tasks"))
        assert creds == []

    def test_scan_makefile(self, temp_role_dir, parser):
        """Test scanning Makefile for credentials."""
        makefile = temp_role_dir / "Makefile"
        makefile.write_text("""
VAULT_PASSWORD ?= $(HOME)/.vault_pass
KEEPASS_DB := /path/to/db.kdbx
SECRET_KEY = very_secret

test:
\tmolecule test
""")
        creds = parser.scan_makefile(makefile)
        assert len(creds) >= 2

    def test_is_valid_role(self, temp_role_dir, parser):
        """Test role validation."""
        assert parser._is_valid_role(temp_role_dir) is True

        # Create a non-role directory
        non_role = temp_role_dir.parent / "not_a_role"
        non_role.mkdir()
        assert parser._is_valid_role(non_role) is False

    def test_parse_roles_directory(self, parser):
        """Test parsing multiple roles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            roles_dir = Path(tmpdir)

            # Create two roles
            role1 = roles_dir / "role_a"
            role1.mkdir()
            (role1 / "tasks").mkdir()

            role2 = roles_dir / "role_b"
            role2.mkdir()
            (role2 / "tasks").mkdir()

            roles = parser.parse_roles_directory(roles_dir)
            assert len(roles) == 2
            role_names = [r.name for r in roles]
            assert "role_a" in role_names
            assert "role_b" in role_names

    def test_deduplicate_credentials(self, parser):
        """Test credential deduplication."""
        creds = [
            CredentialRef("a", "/b", 1, "p1"),
            CredentialRef("a", "/b", 1, "p2"),  # Duplicate
            CredentialRef("a", "/b", 2, "p1"),  # Different line
        ]
        unique = parser._deduplicate_credentials(creds)
        assert len(unique) == 2


class TestAnsibleRoleParserWithDatabase:
    """Tests for AnsibleRoleParser with database integration."""

    @pytest.fixture
    def db(self):
        """Create an in-memory database."""
        return StateDB(":memory:")

    @pytest.fixture
    def parser_with_db(self, db):
        """Create a parser with database."""
        return AnsibleRoleParser(db=db)

    @pytest.fixture
    def roles_dir(self):
        """Create a temporary roles directory with test roles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            roles_dir = Path(tmpdir)

            # Role A depends on Role B
            role_a = roles_dir / "role_a"
            role_a.mkdir()
            (role_a / "tasks").mkdir()
            (role_a / "meta").mkdir()
            (role_a / "meta" / "main.yml").write_text("dependencies:\n  - role_b\n")
            (role_a / "tasks" / "main.yml").write_text("- set_fact:\n    password: secret\n")

            # Role B has no dependencies
            role_b = roles_dir / "role_b"
            role_b.mkdir()
            (role_b / "tasks").mkdir()
            (role_b / "meta").mkdir()
            (role_b / "meta" / "main.yml").write_text("dependencies: []\n")

            yield roles_dir

    def test_populate_database(self, parser_with_db, roles_dir, db):
        """Test populating database with parsed roles."""
        result = parser_with_db.populate_database(roles_dir)

        assert result["roles"] == 2
        assert result["dependencies"] >= 1
        assert result["credentials"] >= 1

        # Verify roles exist in database
        role_a = db.get_role("role_a")
        role_b = db.get_role("role_b")
        assert role_a is not None
        assert role_b is not None

        # Verify dependencies
        deps = db.get_dependencies("role_a")
        assert len(deps) == 1
        assert deps[0][0] == "role_b"

    def test_populate_database_with_wave_config(self, parser_with_db, roles_dir, db):
        """Test populating database with wave configuration."""
        wave_config = {
            "role_a": (1, "Infrastructure"),
            "role_b": (0, "Foundation"),
        }
        parser_with_db.populate_database(roles_dir, wave_config=wave_config)

        role_a = db.get_role("role_a")
        role_b = db.get_role("role_b")

        assert role_a.wave == 1
        assert role_a.wave_name == "Infrastructure"
        assert role_b.wave == 0
        assert role_b.wave_name == "Foundation"

    def test_sync_role(self, parser_with_db, roles_dir, db):
        """Test syncing a single role."""
        role_path = roles_dir / "role_a"
        role_info = parser_with_db.sync_role(role_path)

        assert role_info.name == "role_a"

        # Verify in database
        role = db.get_role("role_a")
        assert role is not None

    def test_add_implicit_dependency(self, parser_with_db, roles_dir, db):
        """Test adding implicit dependency."""
        # First populate roles
        parser_with_db.populate_database(roles_dir)

        # Add implicit dependency
        result = parser_with_db.add_implicit_dependency(
            "role_b", "role_a", source_file="discovered.yml"
        )
        assert result is True

        # Verify dependency exists
        deps = db.get_dependencies("role_b")
        assert len(deps) == 1
        assert deps[0][0] == "role_a"

    def test_add_credential_dependency(self, parser_with_db, roles_dir, db):
        """Test adding credential-based dependency."""
        # First populate roles
        parser_with_db.populate_database(roles_dir)

        # Add credential dependency
        result = parser_with_db.add_credential_dependency(
            "role_b", "role_a", "shared_secret"
        )
        assert result is True

    def test_no_database_raises_error(self):
        """Test that methods requiring database raise error without one."""
        parser = AnsibleRoleParser()  # No database

        with pytest.raises(RuntimeError, match="No database configured"):
            parser.populate_database(Path("/tmp"))

        with pytest.raises(RuntimeError, match="No database configured"):
            parser.sync_role(Path("/tmp/role"))

        with pytest.raises(RuntimeError, match="No database configured"):
            parser.add_implicit_dependency("a", "b")

        with pytest.raises(RuntimeError, match="No database configured"):
            parser.add_credential_dependency("a", "b", "c")


class TestCredentialPatterns:
    """Tests for credential detection patterns."""

    @pytest.fixture
    def parser(self):
        return AnsibleRoleParser()

    def test_detect_keepass_patterns(self, parser):
        """Test detecting KeePassXC patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yml"
            yaml_file.write_text("""
- name: Test
  set_fact:
    value: "{{ lookup('keepassxc_password', 'MyEntry') }}"
""")
            creds = parser._scan_yaml_file(yaml_file)
            assert len(creds) >= 1
            assert any(c.name == "MyEntry" for c in creds)

    def test_detect_vault_patterns(self, parser):
        """Test detecting vault patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yml"
            yaml_file.write_text("""
db_password: "{{ vault_db_password }}"
api_key: "{{ vault_api_key }}"
""")
            creds = parser._scan_yaml_file(yaml_file)
            assert len(creds) >= 2

    def test_detect_sql_password(self, parser):
        """Test detecting SQL password patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yml"
            yaml_file.write_text("""
mssql_sa_password: "{{ vault_mssql_sa }}"
sql_password: secret
""")
            creds = parser._scan_yaml_file(yaml_file)
            assert len(creds) >= 1
            assert any("SQL" in c.purpose for c in creds)

    def test_detect_api_key(self, parser):
        """Test detecting API key patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yml"
            yaml_file.write_text("""
grafana_api_key: "{{ vault_grafana_api_key }}"
""")
            creds = parser._scan_yaml_file(yaml_file)
            assert len(creds) >= 1

    def test_detect_env_lookup(self, parser):
        """Test detecting environment lookups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yml"
            yaml_file.write_text("""
password: "{{ lookup('env', 'MY_SECRET') }}"
""")
            creds = parser._scan_yaml_file(yaml_file)
            assert len(creds) >= 1
