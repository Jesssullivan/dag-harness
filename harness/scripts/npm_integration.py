"""Generate npm script entries for harness integration.

Provides utilities for generating the npm script entries that EMS projects
use to invoke dag-harness commands. The generated scripts use `uv run` for
consistent virtual environment management.
"""

import json


def get_npm_scripts() -> dict[str, str]:
    """Return dict of npm script name -> command for harness integration.

    All scripts are prefixed with 'harness:' and use 'uv run' to ensure
    the correct virtual environment is activated automatically.
    """
    return {
        "harness:init": "cd harness && uv run harness init",
        "harness:status": "cd harness && uv run harness status",
        "harness:box-up": "cd harness && uv run harness box-up-role",
        "harness:list-roles": "cd harness && uv run harness list-roles",
        "harness:deps": "cd harness && uv run harness deps",
        "harness:graph": "cd harness && uv run harness graph",
        "harness:costs": "cd harness && uv run harness costs",
        "harness:hotl:start": "cd harness && uv run harness hotl start",
        "harness:hotl:stop": "cd harness && uv run harness hotl stop",
        "harness:resume": "cd harness && uv run harness resume",
        "harness:migrate": "cd harness && uv run harness migrate",
        "harness:config:validate": "cd harness && uv run harness config --validate",
        "harness:test": "cd harness && uv run pytest tests/ -m unit",
        "harness:test:all": "cd harness && uv run pytest tests/ -v",
    }


def generate_package_json_snippet() -> str:
    """Generate a JSON snippet to paste into package.json.

    Returns a formatted JSON string containing the harness script entries,
    ready to be merged into an existing package.json scripts section.
    """
    scripts = get_npm_scripts()
    # Build the snippet with comment header and script entries
    output = {
        "// ========================================": "",
        "// ANSIBLE-DAG-HARNESS INTEGRATION": "",
        "// ========================================": "",
    }
    output.update(scripts)
    return json.dumps(output, indent=4)


if __name__ == "__main__":
    print(generate_package_json_snippet())
