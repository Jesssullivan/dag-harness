"""Bootstrap initialization for harness projects.

Provides the `harness init` command implementation that sets up a new
harness project in a git repository:
  - Detects git repository root
  - Creates .harness/ directory
  - Initializes SQLite database with schema
  - Generates harness.yml configuration
  - Detects ansible roles and populates the database
  - Updates .gitignore with .harness/ entries
"""

import subprocess
from pathlib import Path
from typing import Optional

import yaml

from harness.config import HarnessConfig
from harness.db.state import StateDB


# Lines to add to .gitignore for .harness/ artifacts
GITIGNORE_ENTRIES = [
    "# harness state (auto-generated)",
    ".harness/*.db",
    ".harness/*.db-journal",
    ".harness/*.db-wal",
    ".harness/*.db-shm",
    ".harness/*.log",
]


def _detect_git_root(repo_root: Optional[Path] = None) -> Path:
    """Detect the git repository root directory.

    Args:
        repo_root: Explicit repo root to use. If None, detect via git.

    Returns:
        Resolved Path to the git repository root.

    Raises:
        RuntimeError: If not inside a git repository and no repo_root given.
    """
    if repo_root is not None:
        resolved = repo_root.resolve()
        if not (resolved / ".git").exists() and not resolved.joinpath(".git").is_file():
            raise RuntimeError(
                f"Not a git repository: {resolved}\n"
                "Run 'git init' first or pass a valid --repo-root."
            )
        return resolved

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Not inside a git repository. "
            "Run 'git init' first or use --repo-root to specify the path."
        ) from exc


def _create_harness_dir(repo_root: Path) -> Path:
    """Create the .harness/ directory if it does not exist.

    Returns:
        Path to the .harness/ directory.
    """
    harness_dir = repo_root / ".harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    return harness_dir


def _init_database(harness_dir: Path) -> StateDB:
    """Initialize the harness database.

    The StateDB constructor auto-creates the schema when the database
    file does not already exist.

    Returns:
        Initialized StateDB instance.
    """
    db_path = harness_dir / "harness.db"
    return StateDB(db_path)


def _generate_config(
    repo_root: Path,
    harness_dir: Path,
    force: bool = False,
    config_path: Optional[str] = None,
) -> Path:
    """Generate harness.yml from detected values.

    If a config file already exists and ``force`` is False, the existing
    file is left untouched.

    Returns:
        Path to the config file.
    """
    target = Path(config_path) if config_path else repo_root / "harness.yml"

    if target.exists() and not force:
        return target

    config = HarnessConfig(
        db_path=str(harness_dir / "harness.db"),
        repo_root=str(repo_root),
    )

    # Detect waves from ansible/roles if present
    roles_path = repo_root / "ansible" / "roles"
    if roles_path.is_dir():
        detected_roles = _scan_roles(roles_path)
        if detected_roles:
            # Build waves dict from detected roles that match defaults
            for wave_num, wave_info in config.waves.items():
                matched = [r for r in wave_info.get("roles", []) if r in detected_roles]
                wave_info["roles"] = matched

    config.save(str(target))
    return target


def _scan_roles(roles_path: Path) -> list[str]:
    """Scan ansible/roles/ directory for role names.

    Returns:
        Sorted list of role directory names.
    """
    if not roles_path.is_dir():
        return []

    roles = []
    for entry in roles_path.iterdir():
        if entry.is_dir() and not entry.name.startswith(("_", ".")):
            roles.append(entry.name)

    return sorted(roles)


def _detect_and_populate_roles(
    db: StateDB,
    repo_root: Path,
    config: HarnessConfig,
) -> list[str]:
    """Detect roles in ansible/roles/ and populate the database.

    Args:
        db: StateDB instance.
        repo_root: Repository root.
        config: HarnessConfig with wave definitions.

    Returns:
        List of detected role names.
    """
    from harness.db.models import Role

    roles_path = repo_root / "ansible" / "roles"
    if not roles_path.is_dir():
        return []

    role_names = _scan_roles(roles_path)
    for name in role_names:
        role_dir = roles_path / name
        wave_num, wave_name = config.get_wave_for_role(name)
        has_molecule = (role_dir / "molecule").is_dir()

        description = ""
        meta_path = role_dir / "meta" / "main.yml"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = yaml.safe_load(f)
                    if meta and isinstance(meta, dict) and "galaxy_info" in meta:
                        description = meta["galaxy_info"].get("description", "")
            except Exception:
                pass

        role = Role(
            name=name,
            wave=wave_num,
            wave_name=wave_name if wave_name != f"Wave {wave_num}" else None,
            description=description or None,
            molecule_path=str(role_dir / "molecule") if has_molecule else None,
            has_molecule_tests=has_molecule,
        )
        db.upsert_role(role)

    return role_names


def _update_gitignore(repo_root: Path) -> bool:
    """Append .harness/ entries to .gitignore if not already present.

    Returns:
        True if the file was modified, False if already up to date.
    """
    gitignore_path = repo_root / ".gitignore"

    existing_content = ""
    if gitignore_path.exists():
        existing_content = gitignore_path.read_text()

    # Check whether the sentinel line is already present
    if ".harness/*.db" in existing_content:
        return False

    lines_to_add = "\n".join(GITIGNORE_ENTRIES)
    separator = "\n" if existing_content and not existing_content.endswith("\n") else ""
    with open(gitignore_path, "a") as f:
        f.write(f"{separator}\n{lines_to_add}\n")

    return True


def init_harness(
    repo_root: Optional[Path] = None,
    force: bool = False,
    no_detect_roles: bool = False,
    config_path: Optional[str] = None,
) -> dict:
    """Initialize harness in a repository.

    Steps:
      1. Detect git repository root (or use ``repo_root``)
      2. Create ``.harness/`` directory
      3. Initialize database with schema
      4. Generate ``harness.yml`` from template (if not exists or ``--force``)
      5. Detect ``ansible/roles/`` and populate roles table
      6. Update ``.gitignore`` with ``.harness/`` entries

    Args:
        repo_root: Explicit repository root path. Detected via git if None.
        force: Overwrite existing configuration files.
        no_detect_roles: Skip automatic role detection.
        config_path: Custom path for the config file.

    Returns:
        Dict with initialization results::

            {
                "repo_root": str,
                "harness_dir": str,
                "db_path": str,
                "config_path": str,
                "roles_detected": list[str],
                "gitignore_updated": bool,
                "config_created": bool,
            }

    Raises:
        RuntimeError: If not inside a git repository.
    """
    # 1. Detect git repo root
    detected_root = _detect_git_root(repo_root)

    # 2. Create .harness/ directory
    harness_dir = _create_harness_dir(detected_root)

    # 3. Initialize database
    db = _init_database(harness_dir)

    # 4. Generate config
    config_target = Path(config_path) if config_path else detected_root / "harness.yml"
    config_existed = config_target.exists()
    config_file = _generate_config(
        detected_root,
        harness_dir,
        force=force,
        config_path=config_path,
    )
    config_created = not config_existed or force

    # Load the config for role detection
    config = HarnessConfig.load(str(config_file))

    # 5. Detect roles
    roles_detected: list[str] = []
    if not no_detect_roles:
        roles_detected = _detect_and_populate_roles(db, detected_root, config)

    # 6. Update .gitignore
    gitignore_updated = _update_gitignore(detected_root)

    return {
        "repo_root": str(detected_root),
        "harness_dir": str(harness_dir),
        "db_path": str(harness_dir / "harness.db"),
        "config_path": str(config_file),
        "roles_detected": roles_detected,
        "gitignore_updated": gitignore_updated,
        "config_created": config_created,
    }
