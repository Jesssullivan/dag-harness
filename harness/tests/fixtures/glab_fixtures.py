"""GitLab CLI/API mock fixtures for harness tests.

Provides comprehensive mocking for:
- glab CLI commands (subprocess.run)
- GitLab API responses
- Rate limiting scenarios
- GraphQL responses for iteration assignment
"""

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# PREDEFINED API RESPONSES
# =============================================================================


@pytest.fixture
def mock_glab_api_responses() -> dict[str, Any]:
    """
    Predefined API response dicts for common GitLab API endpoints.

    Returns a dictionary with responses for:
    - iterations: List of sprint/iteration objects
    - issue_created: Newly created issue response
    - issue_existing: Existing issue lookup
    - mr_created: Newly created merge request
    - mr_existing: Existing MR lookup
    - merge_train: Merge train status
    - project: Project metadata
    - user: Current user info
    """
    return {
        "iterations": [
            {
                "id": 12345,
                "title": "Sprint 42 - Week 5",
                "state": "opened",
                "start_date": "2026-01-27",
                "due_date": "2026-02-07",
                "group": {"id": 100, "name": "bates-ils"},
            },
            {
                "id": 12346,
                "title": "Sprint 43 - Week 6",
                "state": "upcoming",
                "start_date": "2026-02-10",
                "due_date": "2026-02-21",
                "group": {"id": 100, "name": "bates-ils"},
            },
        ],
        "issue_created": {
            "id": 67890,
            "iid": 123,
            "project_id": 456,
            "title": "Box up `common` role",
            "description": "Automated box-up workflow for common role",
            "state": "opened",
            "web_url": "https://gitlab.example.com/project/-/issues/123",
            "labels": ["role", "ansible", "box-up"],
            "assignees": [{"id": 1, "username": "testuser", "name": "Test User"}],
            "weight": 3,
            "iteration": {"id": 12345, "title": "Sprint 42 - Week 5"},
            "created_at": "2026-02-03T10:00:00Z",
            "updated_at": "2026-02-03T10:00:00Z",
        },
        "issue_existing": {
            "id": 67891,
            "iid": 124,
            "project_id": 456,
            "title": "Box up `sql_server_2022` role",
            "state": "opened",
            "web_url": "https://gitlab.example.com/project/-/issues/124",
            "labels": ["role", "ansible"],
        },
        "mr_created": {
            "id": 11111,
            "iid": 456,
            "project_id": 456,
            "source_branch": "sid/common",
            "target_branch": "main",
            "title": "Box up common role",
            "description": "## Summary\n- Molecule tests passing\n- Dependencies validated",
            "state": "opened",
            "web_url": "https://gitlab.example.com/project/-/merge_requests/456",
            "merge_status": "can_be_merged",
            "squash_on_merge": True,
            "force_remove_source_branch": True,
            "draft": False,
            "work_in_progress": False,
            "created_at": "2026-02-03T10:05:00Z",
            "updated_at": "2026-02-03T10:05:00Z",
        },
        "mr_existing": {
            "id": 11112,
            "iid": 457,
            "project_id": 456,
            "source_branch": "sid/sql_server_2022",
            "target_branch": "main",
            "title": "Box up sql_server_2022 role",
            "state": "opened",
            "merge_status": "can_be_merged",
            "web_url": "https://gitlab.example.com/project/-/merge_requests/457",
        },
        "merge_train": [
            {
                "id": 1,
                "merge_request": {"iid": 456, "title": "Box up common role"},
                "pipeline": {"id": 999, "status": "running", "web_url": "https://gitlab.example.com/pipelines/999"},
                "status": "merging",
                "created_at": "2026-02-03T10:10:00Z",
            }
        ],
        "merge_train_empty": [],
        "project": {
            "id": 456,
            "name": "ems-upgrade",
            "path_with_namespace": "bates-ils/ems-upgrade",
            "default_branch": "main",
            "web_url": "https://gitlab.example.com/bates-ils/ems-upgrade",
            "merge_trains_enabled": True,
        },
        "user": {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "email": "testuser@example.com",
            "state": "active",
        },
        "branches": [
            {"name": "main", "protected": True},
            {"name": "sid/common", "protected": False},
            {"name": "sid/sql_server_2022", "protected": False},
        ],
        "pipelines": [
            {
                "id": 999,
                "status": "running",
                "ref": "sid/common",
                "sha": "abc123def456",
                "web_url": "https://gitlab.example.com/pipelines/999",
            }
        ],
    }


# =============================================================================
# GLAB CLI MOCK
# =============================================================================


@pytest.fixture
def mock_glab_cli(mock_glab_api_responses: dict[str, Any]):
    """
    Mock subprocess.run for glab CLI commands.

    Handles common glab commands:
    - glab auth status
    - glab issue create/view/list
    - glab mr create/view/list/merge
    - glab api (REST API calls)

    Usage:
        def test_something(mock_glab_cli):
            # glab commands will be mocked
            result = subprocess.run(["glab", "issue", "create", ...])
    """

    def _mock_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""

        cmd_str = " ".join(cmd)

        # Auth commands
        if "glab auth status" in cmd_str:
            result.stdout = "Logged in to gitlab.example.com as testuser (glpat-testtoken123)"
            return result

        # Issue commands
        if "glab issue create" in cmd_str:
            result.stdout = mock_glab_api_responses["issue_created"]["web_url"]
            return result

        if "glab issue view" in cmd_str:
            result.stdout = json.dumps(mock_glab_api_responses["issue_created"])
            return result

        if "glab issue list" in cmd_str:
            result.stdout = json.dumps([mock_glab_api_responses["issue_created"]])
            return result

        # MR commands
        if "glab mr create" in cmd_str:
            result.stdout = mock_glab_api_responses["mr_created"]["web_url"]
            return result

        if "glab mr view" in cmd_str:
            result.stdout = json.dumps(mock_glab_api_responses["mr_created"])
            return result

        if "glab mr list" in cmd_str:
            result.stdout = json.dumps([mock_glab_api_responses["mr_created"]])
            return result

        if "glab mr merge" in cmd_str:
            result.stdout = "Merge request merged successfully"
            return result

        # API commands
        if "glab api" in cmd_str:
            if "iterations" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["iterations"])
            elif "/issues/" in cmd_str and "POST" not in cmd_str.upper():
                result.stdout = json.dumps(mock_glab_api_responses["issue_created"])
            elif "/issues" in cmd_str and "POST" in cmd_str.upper():
                result.stdout = json.dumps(mock_glab_api_responses["issue_created"])
            elif "/merge_requests/" in cmd_str and "POST" not in cmd_str.upper():
                result.stdout = json.dumps(mock_glab_api_responses["mr_created"])
            elif "/merge_requests" in cmd_str and "POST" in cmd_str.upper():
                result.stdout = json.dumps(mock_glab_api_responses["mr_created"])
            elif "merge_trains" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["merge_train"])
            elif "/projects/" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["project"])
            elif "/user" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["user"])
            elif "/repository/branches" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["branches"])
            elif "/pipelines" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["pipelines"])
            else:
                result.stdout = "[]"
            return result

        # Default: empty response
        return result

    with patch("subprocess.run", side_effect=_mock_run) as mock_run:
        yield mock_run


# =============================================================================
# FAILURE SCENARIOS
# =============================================================================


@pytest.fixture
def mock_glab_failure():
    """
    Fixture that simulates glab API failures.

    Useful for testing error handling and retry logic.

    Usage:
        def test_api_failure(mock_glab_failure):
            # All glab commands will fail
            result = subprocess.run(["glab", "issue", "create", ...])
            assert result.returncode != 0
    """

    def _mock_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 1
        result.stdout = ""
        result.stderr = "Error: API request failed with status 500: Internal Server Error"
        return result

    with patch("subprocess.run", side_effect=_mock_run) as mock_run:
        yield mock_run


@pytest.fixture
def mock_glab_rate_limit():
    """
    Simulates GitLab 429 rate limit responses.

    The mock returns rate limit errors for the first 2 calls,
    then succeeds on subsequent calls. This is useful for testing
    retry logic with exponential backoff.

    Usage:
        def test_rate_limit_retry(mock_glab_rate_limit):
            # First 2 calls fail with 429, then succeed
            ...
    """
    call_count = {"count": 0}

    def _mock_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        result = MagicMock(spec=subprocess.CompletedProcess)
        call_count["count"] += 1

        if call_count["count"] <= 2:
            # Rate limited
            result.returncode = 1
            result.stdout = ""
            result.stderr = json.dumps({
                "message": "429 Too Many Requests",
                "error": "Rate limit exceeded. Retry after 60 seconds.",
            })
        else:
            # Success after retries
            result.returncode = 0
            result.stdout = "https://gitlab.example.com/project/-/issues/123"
            result.stderr = ""

        return result

    with patch("subprocess.run", side_effect=_mock_run) as mock_run:
        mock_run.call_count_tracker = call_count
        yield mock_run


# =============================================================================
# GRAPHQL MOCK FOR ITERATION ASSIGNMENT
# =============================================================================


@pytest.fixture
def mock_glab_graphql():
    """
    Mock GraphQL responses for iteration assignment operations.

    GitLab's iteration assignment often requires GraphQL mutations,
    especially for group-level iterations.

    Returns mock for:
    - Query current iteration
    - Assign issue to iteration
    - Query iteration cadences

    Usage:
        def test_iteration_assignment(mock_glab_graphql):
            # GraphQL queries will be mocked
            ...
    """
    graphql_responses = {
        "currentIteration": {
            "data": {
                "group": {
                    "iterations": {
                        "nodes": [
                            {
                                "id": "gid://gitlab/Iteration/12345",
                                "iid": "42",
                                "title": "Sprint 42 - Week 5",
                                "state": "current",
                                "startDate": "2026-01-27",
                                "dueDate": "2026-02-07",
                                "webPath": "/groups/bates-ils/-/iterations/42",
                            }
                        ]
                    }
                }
            }
        },
        "assignIteration": {
            "data": {
                "issueSetIteration": {
                    "issue": {
                        "id": "gid://gitlab/Issue/67890",
                        "iid": "123",
                        "iteration": {
                            "id": "gid://gitlab/Iteration/12345",
                            "title": "Sprint 42 - Week 5",
                        }
                    },
                    "errors": []
                }
            }
        },
        "iterationCadences": {
            "data": {
                "group": {
                    "iterationCadences": {
                        "nodes": [
                            {
                                "id": "gid://gitlab/Iterations::Cadence/1",
                                "title": "Weekly Sprints",
                                "automatic": True,
                                "active": True,
                                "durationInWeeks": 2,
                            }
                        ]
                    }
                }
            }
        },
    }

    def _mock_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stderr = ""

        cmd_str = " ".join(cmd)

        # Check for GraphQL queries in glab api calls
        if "glab api graphql" in cmd_str:
            # Determine which query based on the query content
            if "currentIteration" in cmd_str or "iterations" in cmd_str.lower():
                result.stdout = json.dumps(graphql_responses["currentIteration"])
            elif "issueSetIteration" in cmd_str or "setIteration" in cmd_str:
                result.stdout = json.dumps(graphql_responses["assignIteration"])
            elif "iterationCadences" in cmd_str:
                result.stdout = json.dumps(graphql_responses["iterationCadences"])
            else:
                result.stdout = json.dumps({"data": {}})
        else:
            result.stdout = "{}"

        return result

    with patch("subprocess.run", side_effect=_mock_run) as mock_run:
        mock_run.graphql_responses = graphql_responses
        yield mock_run


# =============================================================================
# COMBINED/ADVANCED FIXTURES
# =============================================================================


@pytest.fixture
def mock_glab_full_workflow(mock_glab_api_responses: dict[str, Any]):
    """
    Complete glab mock for full workflow testing.

    Combines CLI and API mocking with stateful behavior:
    - Tracks created issues and MRs
    - Updates state on mutations
    - Supports checking call history

    Usage:
        def test_full_workflow(mock_glab_full_workflow):
            # Full workflow testing with state tracking
            ...
    """
    state = {
        "issues_created": [],
        "mrs_created": [],
        "iterations_assigned": [],
        "calls": [],
    }

    def _mock_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stderr = ""

        cmd_str = " ".join(cmd)
        state["calls"].append(cmd_str)

        # Issue creation
        if "glab issue create" in cmd_str:
            issue = mock_glab_api_responses["issue_created"].copy()
            issue["iid"] = len(state["issues_created"]) + 100
            state["issues_created"].append(issue)
            result.stdout = issue["web_url"]
            return result

        # MR creation
        if "glab mr create" in cmd_str:
            mr = mock_glab_api_responses["mr_created"].copy()
            mr["iid"] = len(state["mrs_created"]) + 400
            state["mrs_created"].append(mr)
            result.stdout = mr["web_url"]
            return result

        # API calls
        if "glab api" in cmd_str:
            if "iterations" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["iterations"])
            elif "merge_trains" in cmd_str:
                result.stdout = json.dumps(mock_glab_api_responses["merge_train"])
            else:
                result.stdout = "[]"
            return result

        # Auth status
        if "glab auth status" in cmd_str:
            result.stdout = "Logged in to gitlab.example.com as testuser"
            return result

        result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=_mock_run) as mock_run:
        mock_run.state = state
        yield mock_run


# =============================================================================
# EXPORTED FIXTURES
# =============================================================================

__all__ = [
    "mock_glab_api_responses",
    "mock_glab_cli",
    "mock_glab_failure",
    "mock_glab_rate_limit",
    "mock_glab_graphql",
    "mock_glab_full_workflow",
]
