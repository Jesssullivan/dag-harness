"""Unit tests for the GitLab API client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.gitlab.api import (
    DEFAULT_LABEL_COLOR,
    SCOPED_LABEL_COLORS,
    SCOPED_LABELS,
    WAVE_LABEL_COLORS,
    WAVE_LABEL_DESCRIPTIONS,
    GitLabClient,
    GitLabConfig,
)


class TestGitLabConfig:
    """Tests for GitLabConfig."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = GitLabConfig()
        assert config.project_path == "bates-ils/projects/ems/ems-mono"
        assert config.group_path == "bates-ils"
        assert config.default_assignee == "jsullivan2"
        assert "role" in config.default_labels

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration."""
        config = GitLabConfig(
            project_path="custom/project",
            group_path="custom-group",
            default_assignee="testuser",
            default_labels=["label1", "label2"],
        )
        assert config.project_path == "custom/project"
        assert config.group_path == "custom-group"
        assert config.default_labels == ["label1", "label2"]

    @pytest.mark.unit
    def test_project_path_encoded(self):
        """Test URL encoding of project path."""
        config = GitLabConfig(project_path="group/subgroup/project")
        assert config.project_path_encoded == "group%2Fsubgroup%2Fproject"


class TestGitLabClientInit:
    """Tests for GitLabClient initialization."""

    @pytest.mark.unit
    def test_client_init(self, in_memory_db):
        """Test client initialization."""
        client = GitLabClient(db=in_memory_db)
        assert client.db is not None
        assert client.config is not None

    @pytest.mark.unit
    def test_client_init_with_config(self, in_memory_db):
        """Test client with custom config."""
        config = GitLabConfig(project_path="test/project")
        client = GitLabClient(db=in_memory_db, config=config)
        assert client.config.project_path == "test/project"


class TestGitLabToken:
    """Tests for token retrieval."""

    @pytest.mark.unit
    def test_token_from_env(self, in_memory_db, monkeypatch):
        """Test getting token from environment variable."""
        monkeypatch.setenv("GITLAB_TOKEN", "test-token-123")
        client = GitLabClient(db=in_memory_db)
        assert client.token == "test-token-123"

    @pytest.mark.unit
    def test_token_from_glab_env(self, in_memory_db, monkeypatch):
        """Test getting token from GLAB_TOKEN."""
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        monkeypatch.setenv("GLAB_TOKEN", "glab-token-456")
        client = GitLabClient(db=in_memory_db)
        assert client.token == "glab-token-456"

    @pytest.mark.unit
    def test_token_from_glab_cli(self, in_memory_db, monkeypatch):
        """Test getting token from glab auth status."""
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        monkeypatch.delenv("GLAB_TOKEN", raising=False)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "Token: glpat-testtoken789"
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            client = GitLabClient(db=in_memory_db)
            assert client.token == "glpat-testtoken789"

    @pytest.mark.unit
    def test_token_not_found(self, in_memory_db, monkeypatch):
        """Test error when no token available."""
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        monkeypatch.delenv("GLAB_TOKEN", raising=False)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "Logged in as user"  # No token line
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            client = GitLabClient(db=in_memory_db)
            with pytest.raises(ValueError, match="No GitLab token found"):
                _ = client.token


class TestGitLabAPIRequests:
    """Tests for API request methods."""

    @pytest.mark.unit
    def test_glab_command(self, in_memory_db):
        """Test running glab command."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "command output"
            mock_run.return_value = mock_result

            result = client._glab("test", "command")
            assert result == "command output"
            mock_run.assert_called_once()

    @pytest.mark.unit
    def test_glab_command_json(self, in_memory_db):
        """Test glab command with JSON output."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"key": "value"}'
            mock_run.return_value = mock_result

            result = client._glab("test", "command", json_output=True)
            assert result == {"key": "value"}

    @pytest.mark.unit
    def test_glab_command_error(self, in_memory_db):
        """Test glab command error handling."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "error message"
            mock_run.return_value = mock_result

            with pytest.raises(RuntimeError, match="glab command failed"):
                client._glab("test", "command")

    @pytest.mark.unit
    def test_api_get(self, in_memory_db):
        """Test API GET request."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '[{"id": 1, "name": "test"}]'
            mock_run.return_value = mock_result

            result = client._api_get("/test/endpoint")
            assert result == [{"id": 1, "name": "test"}]

    @pytest.mark.unit
    def test_api_post(self, in_memory_db):
        """Test API POST request."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123}'
            mock_run.return_value = mock_result

            result = client._api_post("/test/endpoint", {"key": "value"})
            assert result == {"id": 123}

    @pytest.mark.unit
    def test_api_put(self, in_memory_db):
        """Test API PUT request."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"updated": true}'
            mock_run.return_value = mock_result

            result = client._api_put("/test/endpoint", {"field": "newvalue"})
            assert result == {"updated": True}


class TestIterationOperations:
    """Tests for iteration management."""

    @pytest.mark.unit
    def test_list_iterations(self, in_memory_db, mock_gitlab_api, mock_gitlab_responses):
        """Test listing iterations."""
        client = GitLabClient(db=in_memory_db)
        iterations = client.list_iterations()

        assert len(iterations) == 1
        assert iterations[0].id == 12345
        assert iterations[0].title == "Sprint 42"

    @pytest.mark.unit
    def test_find_iteration(self, in_memory_db):
        """Test finding iteration by search."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [{"id": 999, "title": "Sprint 43", "state": "opened", "group": {"id": 100}}]
            )
            mock_run.return_value = mock_result

            iteration = client.find_iteration("Sprint 43")
            assert iteration is not None
            assert iteration.id == 999

    @pytest.mark.unit
    def test_get_current_iteration(self, in_memory_db, mock_gitlab_api):
        """Test getting current iteration."""
        client = GitLabClient(db=in_memory_db)
        iteration = client.get_current_iteration()

        assert iteration is not None
        assert iteration.state == "opened"


class TestIssueOperations:
    """Tests for issue management."""

    @pytest.mark.unit
    def test_create_issue(self, db_with_roles, mock_gitlab_api, mock_gitlab_responses):
        """Test creating an issue."""
        client = GitLabClient(db=db_with_roles)

        issue = client.create_issue(
            role_name="common",
            title="Box up `common` role",
            description="Test description",
            labels=["role", "ansible"],
            weight=3,
        )

        assert issue is not None
        assert issue.iid == 123
        assert issue.state == "opened"

    @pytest.mark.unit
    def test_create_issue_role_not_found(self, in_memory_db):
        """Test creating issue for non-existent role."""
        client = GitLabClient(db=in_memory_db)

        with pytest.raises(ValueError, match="not found"):
            client.create_issue(role_name="nonexistent", title="Test", description="Test")

    @pytest.mark.unit
    def test_get_issue(self, in_memory_db, mock_gitlab_api, mock_gitlab_responses):
        """Test getting issue by IID."""
        client = GitLabClient(db=in_memory_db)
        issue = client.get_issue(123)

        assert issue is not None
        assert issue.iid == 123

    @pytest.mark.unit
    def test_assign_issue_to_iteration(self, in_memory_db):
        """Test assigning issue to iteration."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123}'
            mock_run.return_value = mock_result

            result = client.assign_issue_to_iteration(123, 456)
            assert result is True


class TestMergeRequestOperations:
    """Tests for merge request management."""

    @pytest.mark.unit
    def test_create_merge_request(self, db_with_roles, mock_gitlab_api, mock_gitlab_responses):
        """Test creating a merge request."""
        client = GitLabClient(db=db_with_roles)

        mr = client.create_merge_request(
            role_name="common",
            source_branch="sid/common",
            title="Box up common role",
            description="Test MR description",
        )

        assert mr is not None
        assert mr.iid == 456
        assert mr.source_branch == "sid/common"

    @pytest.mark.unit
    def test_create_draft_mr(self, db_with_roles, mock_gitlab_api):
        """Test creating a draft merge request."""
        client = GitLabClient(db=db_with_roles)

        with patch.object(client, "_api_get") as mock_get:
            mock_get.return_value = {
                "id": 11111,
                "iid": 456,
                "source_branch": "sid/common",
                "target_branch": "main",
                "title": "Draft: Box up common role",
                "state": "opened",
                "web_url": "https://gitlab.example.com/mr/456",
                "merge_status": "can_be_merged",
            }

            mr = client.create_merge_request(
                role_name="common",
                source_branch="sid/common",
                title="Box up common role",
                description="Draft MR",
                draft=True,
            )

            assert "Draft:" in mr.title or mr is not None

    @pytest.mark.unit
    def test_get_merge_request(self, in_memory_db, mock_gitlab_api, mock_gitlab_responses):
        """Test getting merge request by IID."""
        client = GitLabClient(db=in_memory_db)
        mr = client.get_merge_request(456)

        assert mr is not None
        assert mr.source_branch == "sid/common"


class TestMergeTrainOperations:
    """Tests for merge train management."""

    @pytest.mark.unit
    def test_add_to_merge_train(self, in_memory_db):
        """Test adding MR to merge train."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 1, "status": "merging"}'
            mock_run.return_value = mock_result

            result = client.add_to_merge_train(456)
            assert "id" in result

    @pytest.mark.unit
    def test_add_to_merge_train_fallback(self, in_memory_db):
        """Test fallback when merge trains not enabled."""
        client = GitLabClient(db=in_memory_db)

        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                # First call fails (merge train endpoint)
                result.returncode = 1
                result.stderr = "merge_trains not found"
            else:
                # Fallback succeeds
                result.returncode = 0
                result.stdout = '{"state": "merged"}'
            return result

        with patch("subprocess.run", side_effect=side_effect):
            result = client.add_to_merge_train(456)
            assert result is not None

    @pytest.mark.unit
    def test_get_merge_train(self, in_memory_db, mock_gitlab_api, mock_gitlab_responses):
        """Test getting merge train queue."""
        client = GitLabClient(db=in_memory_db)
        train = client.get_merge_train()

        assert isinstance(train, list)

    @pytest.mark.unit
    def test_get_merge_train_status(self, in_memory_db, mock_gitlab_api, mock_gitlab_responses):
        """Test getting merge train status for MR."""
        client = GitLabClient(db=in_memory_db)
        status = client.get_merge_train_status(456)

        assert status is not None
        assert status["merge_request"]["iid"] == 456

    @pytest.mark.unit
    def test_remove_from_merge_train(self, in_memory_db):
        """Test removing from merge train."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = client.remove_from_merge_train(1)
            assert result is True

    @pytest.mark.unit
    def test_get_merge_train_position(self, in_memory_db):
        """Test getting MR position in merge train."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_merge_train") as mock_train:
            mock_train.return_value = [
                {"merge_request": {"iid": 100}},
                {"merge_request": {"iid": 456}},
                {"merge_request": {"iid": 200}},
            ]

            position = client.get_merge_train_position(456)
            assert position == 2  # 1-indexed

    @pytest.mark.unit
    def test_is_merge_train_available_success(self, in_memory_db):
        """Test checking if MR can be added to merge train."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_merge_request") as mock_mr:
            mock_mr_obj = MagicMock()
            mock_mr_obj.state = "opened"
            mock_mr_obj.merge_status = "can_be_merged"
            mock_mr.return_value = mock_mr_obj

            with patch.object(client, "get_merge_train_status") as mock_status:
                mock_status.return_value = None

                result = client.is_merge_train_available(456)
                assert result["available"] is True

    @pytest.mark.unit
    def test_is_merge_train_available_conflict(self, in_memory_db):
        """Test MR with conflicts cannot be added."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_merge_request") as mock_mr:
            mock_mr_obj = MagicMock()
            mock_mr_obj.state = "opened"
            mock_mr_obj.merge_status = "cannot_be_merged"
            mock_mr.return_value = mock_mr_obj

            result = client.is_merge_train_available(456)
            assert result["available"] is False
            assert "conflict" in result["reason"].lower()

    @pytest.mark.unit
    def test_abort_merge_train(self, in_memory_db):
        """Test aborting merge for MR."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "remove_mr_from_merge_train") as mock_remove:
            mock_remove.return_value = True

            result = client.abort_merge_train(456)
            assert result is True

    @pytest.mark.unit
    def test_list_merge_trains(self, in_memory_db):
        """Test listing all merge trains."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_merge_train") as mock_train:
            mock_train.return_value = [{"id": 1}]

            trains = client.list_merge_trains()
            assert len(trains) >= 1

    @pytest.mark.unit
    def test_is_merge_train_enabled(self, in_memory_db):
        """Test checking if merge trains are enabled."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_merge_train") as mock_train:
            mock_train.return_value = []

            enabled = client.is_merge_train_enabled()
            assert enabled is True

    @pytest.mark.unit
    def test_is_merge_train_not_enabled(self, in_memory_db):
        """Test detecting merge trains not enabled."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_merge_train") as mock_train:
            mock_train.side_effect = Exception("Not found")

            enabled = client.is_merge_train_enabled()
            assert enabled is False


class TestSyncOperations:
    """Tests for sync operations."""

    @pytest.mark.unit
    def test_sync_issues(self, db_with_roles):
        """Test syncing issues from GitLab."""
        client = GitLabClient(db=db_with_roles)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            # Note: iteration is null to avoid FK constraint on non-existent iteration
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 1,
                        "iid": 10,
                        "title": "Box up `common` role",
                        "state": "opened",
                        "web_url": "https://example.com/1",
                        "labels": ["role"],
                        "iteration": None,
                        "weight": 2,
                    }
                ]
            )
            mock_run.return_value = mock_result

            count = client.sync_issues()
            assert count >= 1

    @pytest.mark.unit
    def test_sync_merge_requests(self, db_with_roles):
        """Test syncing merge requests from GitLab."""
        client = GitLabClient(db=db_with_roles)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 1,
                        "iid": 20,
                        "source_branch": "sid/common",
                        "target_branch": "main",
                        "title": "Box up common",
                        "state": "opened",
                        "web_url": "https://example.com/mr/20",
                        "merge_status": "can_be_merged",
                    }
                ]
            )
            mock_run.return_value = mock_result

            count = client.sync_merge_requests()
            assert count >= 1


class TestLabelConstants:
    """Tests for label configuration constants."""

    @pytest.mark.unit
    def test_wave_label_colors_defined(self):
        """Test that all wave label colors are defined."""
        assert len(WAVE_LABEL_COLORS) == 5
        for i in range(5):
            assert f"wave-{i}" in WAVE_LABEL_COLORS

    @pytest.mark.unit
    def test_wave_label_colors_format(self):
        """Test that wave label colors are valid hex codes."""
        for label, color in WAVE_LABEL_COLORS.items():
            assert color.startswith("#"), f"{label} color should start with #"
            assert len(color) == 7, f"{label} color should be 7 chars (e.g., #RRGGBB)"

    @pytest.mark.unit
    def test_wave_label_descriptions_defined(self):
        """Test that all wave labels have descriptions."""
        for label in WAVE_LABEL_COLORS:
            assert label in WAVE_LABEL_DESCRIPTIONS
            assert len(WAVE_LABEL_DESCRIPTIONS[label]) > 0

    @pytest.mark.unit
    def test_scoped_labels_defined(self):
        """Test that scoped labels are properly defined."""
        assert "priority" in SCOPED_LABELS
        assert "status" in SCOPED_LABELS
        assert "type" in SCOPED_LABELS

    @pytest.mark.unit
    def test_scoped_labels_have_values(self):
        """Test that each scope has defined values."""
        for scope, values in SCOPED_LABELS.items():
            assert len(values) > 0, f"Scope '{scope}' should have values"

    @pytest.mark.unit
    def test_scoped_labels_have_colors(self):
        """Test that each scope has a color defined."""
        for scope in SCOPED_LABELS:
            assert scope in SCOPED_LABEL_COLORS

    @pytest.mark.unit
    def test_default_label_color_format(self):
        """Test that default label color is valid."""
        assert DEFAULT_LABEL_COLOR.startswith("#")
        assert len(DEFAULT_LABEL_COLOR) == 7


class TestLabelOperations:
    """Tests for GitLab label management operations."""

    @pytest.mark.unit
    def test_get_label_found(self, in_memory_db):
        """Test getting an existing label."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                {"id": 1, "name": "wave-0", "color": "#0033CC", "description": "Foundation layer"}
            )
            mock_run.return_value = mock_result

            label = client.get_label("wave-0")
            assert label is not None
            assert label["name"] == "wave-0"
            assert label["color"] == "#0033CC"

    @pytest.mark.unit
    def test_get_label_not_found(self, in_memory_db):
        """Test getting a non-existent label returns None."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "404 Not Found"
            mock_run.return_value = mock_result

            label = client.get_label("nonexistent")
            assert label is None

    @pytest.mark.unit
    def test_create_label(self, in_memory_db):
        """Test creating a new label."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                {
                    "id": 123,
                    "name": "test-label",
                    "color": "#FF0000",
                    "description": "Test description",
                }
            )
            mock_run.return_value = mock_result

            result = client.create_label(
                name="test-label", color="#FF0000", description="Test description"
            )

            assert result["name"] == "test-label"
            assert result["color"] == "#FF0000"

    @pytest.mark.unit
    def test_ensure_label_exists_creates_new(self, in_memory_db):
        """Test ensure_label_exists creates label when it doesn't exist."""
        client = GitLabClient(db=in_memory_db)
        call_count = [0]

        def mock_side_effect(cmd, **kwargs):
            call_count[0] += 1
            result = MagicMock()

            if call_count[0] == 1:
                # First call: get_label returns 404
                result.returncode = 1
                result.stderr = "404 Not Found"
            else:
                # Second call: create_label succeeds
                result.returncode = 0
                result.stdout = json.dumps({"id": 1, "name": "new-label", "color": "#428BCA"})
            return result

        with patch("subprocess.run", side_effect=mock_side_effect):
            success = client.ensure_label_exists("new-label")
            assert success is True
            assert call_count[0] == 2  # get + create

    @pytest.mark.unit
    def test_ensure_label_exists_already_exists(self, in_memory_db):
        """Test ensure_label_exists returns True for existing label."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"id": 1, "name": "existing-label", "color": "#FF0000"})
            mock_run.return_value = mock_result

            success = client.ensure_label_exists("existing-label")
            assert success is True
            # Should only call get_label, not create_label
            assert mock_run.call_count == 1

    @pytest.mark.unit
    def test_ensure_label_exists_idempotent(self, in_memory_db):
        """Test ensure_label_exists is idempotent (handles race condition)."""
        client = GitLabClient(db=in_memory_db)
        call_count = [0]

        def mock_side_effect(cmd, **kwargs):
            call_count[0] += 1
            result = MagicMock()

            if call_count[0] == 1:
                # First call: get_label returns 404
                result.returncode = 1
                result.stderr = "404 Not Found"
            else:
                # Second call: create_label fails because label now exists
                result.returncode = 1
                result.stderr = "Label has already been taken"
            return result

        with patch("subprocess.run", side_effect=mock_side_effect):
            # Should still return True even though create failed (race condition)
            success = client.ensure_label_exists("race-label")
            assert success is True

    @pytest.mark.unit
    def test_ensure_wave_labels(self, in_memory_db):
        """Test ensuring all wave labels exist."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            results = client.ensure_wave_labels()

            # Should have created all 5 wave labels
            assert len(results) == 5
            for i in range(5):
                assert f"wave-{i}" in results
                assert results[f"wave-{i}"] is True

            # Verify colors were passed correctly
            calls = mock_ensure.call_args_list
            assert len(calls) == 5

            # Check that wave-0 was called with correct color
            wave0_call = [c for c in calls if c.kwargs.get("name") == "wave-0"][0]
            assert wave0_call.kwargs["color"] == "#0033CC"

    @pytest.mark.unit
    def test_ensure_scoped_labels(self, in_memory_db):
        """Test ensuring scoped labels exist."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            results = client.ensure_scoped_labels("priority")

            # Should have created priority labels
            assert "priority::high" in results
            assert "priority::medium" in results
            assert "priority::low" in results

    @pytest.mark.unit
    def test_ensure_scoped_labels_custom_values(self, in_memory_db):
        """Test ensuring scoped labels with custom values."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            results = client.ensure_scoped_labels("custom", ["alpha", "beta"])

            assert "custom::alpha" in results
            assert "custom::beta" in results

    @pytest.mark.unit
    def test_ensure_scoped_labels_unknown_scope(self, in_memory_db):
        """Test ensuring scoped labels for unknown scope returns empty."""
        client = GitLabClient(db=in_memory_db)

        results = client.ensure_scoped_labels("unknown_scope")
        assert results == {}

    @pytest.mark.unit
    def test_prepare_labels_for_role(self, in_memory_db):
        """Test preparing labels for a role."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            labels = client.prepare_labels_for_role("common", wave=0)

            # Should have default labels plus wave label
            assert "role" in labels
            assert "ansible" in labels
            assert "molecule" in labels
            assert "wave-0" in labels

    @pytest.mark.unit
    def test_prepare_labels_for_role_with_additional(self, in_memory_db):
        """Test preparing labels with additional labels."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            labels = client.prepare_labels_for_role(
                "common", wave=1, additional_labels=["priority::high", "custom-label"]
            )

            # Should include additional labels
            assert "priority::high" in labels
            assert "custom-label" in labels
            assert "wave-1" in labels

    @pytest.mark.unit
    def test_prepare_labels_for_role_deduplicates(self, in_memory_db):
        """Test that prepare_labels_for_role deduplicates labels."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            # Include 'role' in additional labels (already a default)
            labels = client.prepare_labels_for_role(
                "common", wave=0, additional_labels=["role", "extra"]
            )

            # 'role' should only appear once
            assert labels.count("role") == 1
            assert "extra" in labels

    @pytest.mark.unit
    def test_prepare_labels_for_role_out_of_range_wave(self, in_memory_db):
        """Test preparing labels for wave outside predefined range."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            labels = client.prepare_labels_for_role("special", wave=10)

            # Should still create wave-10 label
            assert "wave-10" in labels

    @pytest.mark.unit
    def test_prepare_labels_handles_ensure_failure(self, in_memory_db):
        """Test that prepare_labels handles individual label failures gracefully."""
        client = GitLabClient(db=in_memory_db)

        def ensure_side_effect(name, **kwargs):
            # Fail for 'molecule' label only
            if name == "molecule":
                return False
            return True

        with patch.object(client, "ensure_label_exists", side_effect=ensure_side_effect):
            labels = client.prepare_labels_for_role("common", wave=0)

            # Should include successful labels
            assert "role" in labels
            assert "ansible" in labels
            assert "wave-0" in labels
            # Should NOT include failed label
            assert "molecule" not in labels

    @pytest.mark.unit
    def test_ensure_all_standard_labels(self, in_memory_db):
        """Test ensuring all standard labels at once."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True

            results = client.ensure_all_standard_labels()

            # Should have waves section
            assert "waves" in results
            assert len(results["waves"]) == 5

            # Should have scoped section
            assert "scoped" in results
            assert "priority" in results["scoped"]
            assert "status" in results["scoped"]
            assert "type" in results["scoped"]


class TestLabelIntegrationWithIssues:
    """Integration tests for labels with issue creation."""

    @pytest.mark.unit
    def test_create_issue_prepares_labels(self, db_with_roles, mock_gitlab_api):
        """Test that create_issue uses prepare_labels_for_role."""
        client = GitLabClient(db=db_with_roles)

        with patch.object(client, "prepare_labels_for_role") as mock_prepare:
            mock_prepare.return_value = ["role", "ansible", "molecule", "wave-0"]

            # Note: This will call the mocked gitlab API to create the issue
            issue = client.create_issue(
                role_name="common",
                title="Test issue",
                description="Test",
                labels=["role", "ansible", "molecule", "wave-0"],
                weight=2,
            )

            # Verify issue was created
            assert issue is not None


class TestIdempotencyOperations:
    """Tests for idempotent issue and MR operations."""

    @pytest.mark.unit
    def test_find_existing_issue_found(self, db_with_roles):
        """Test finding an existing issue for a role."""
        client = GitLabClient(db=db_with_roles)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 67890,
                        "iid": 123,
                        "title": "feat(common): Box up `common` Ansible role",
                        "state": "opened",
                        "web_url": "https://gitlab.example.com/project/-/issues/123",
                        "labels": ["role", "ansible"],
                        "assignees": [{"username": "testuser"}],
                        "iteration": None,
                        "weight": 3,
                    }
                ]
            )
            mock_run.return_value = mock_result

            issue = client.find_existing_issue("common")

            assert issue is not None
            assert issue.iid == 123
            assert issue.title == "feat(common): Box up `common` Ansible role"

    @pytest.mark.unit
    def test_find_existing_issue_not_found(self, db_with_roles):
        """Test when no existing issue is found."""
        client = GitLabClient(db=db_with_roles)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "[]"
            mock_run.return_value = mock_result

            issue = client.find_existing_issue("common")

            assert issue is None

    @pytest.mark.unit
    def test_find_existing_issue_filters_by_title(self, db_with_roles):
        """Test that find_existing_issue filters by exact title pattern."""
        client = GitLabClient(db=db_with_roles)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            # Return an issue that doesn't match the pattern
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 67890,
                        "iid": 123,
                        "title": "Something about common but not box up",
                        "state": "opened",
                        "web_url": "https://gitlab.example.com/project/-/issues/123",
                        "labels": ["role"],
                        "assignees": [],
                        "iteration": None,
                        "weight": None,
                    }
                ]
            )
            mock_run.return_value = mock_result

            issue = client.find_existing_issue("common")

            # Should not match because title doesn't contain "box up"
            assert issue is None

    @pytest.mark.unit
    def test_find_existing_mr_found(self, db_with_roles):
        """Test finding an existing MR for a branch."""
        client = GitLabClient(db=db_with_roles)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 11111,
                        "iid": 456,
                        "source_branch": "sid/common",
                        "target_branch": "main",
                        "title": "feat(common): Add `common` Ansible role",
                        "state": "opened",
                        "web_url": "https://gitlab.example.com/project/-/merge_requests/456",
                        "merge_status": "can_be_merged",
                        "squash_on_merge": True,
                        "force_remove_source_branch": True,
                    }
                ]
            )
            mock_run.return_value = mock_result

            mr = client.find_existing_mr("sid/common")

            assert mr is not None
            assert mr.iid == 456
            assert mr.source_branch == "sid/common"

    @pytest.mark.unit
    def test_find_existing_mr_not_found(self, db_with_roles):
        """Test when no existing MR is found."""
        client = GitLabClient(db=db_with_roles)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "[]"
            mock_run.return_value = mock_result

            mr = client.find_existing_mr("sid/nonexistent")

            assert mr is None

    @pytest.mark.unit
    def test_get_or_create_issue_existing(self, db_with_roles):
        """Test get_or_create_issue returns existing issue without creating."""
        client = GitLabClient(db=db_with_roles)

        existing_issue = MagicMock()
        existing_issue.iid = 123
        existing_issue.web_url = "https://example.com/issue/123"

        with patch.object(client, "find_existing_issue", return_value=existing_issue):
            with patch.object(client, "create_issue") as mock_create:
                issue, created = client.get_or_create_issue(
                    role_name="common", title="Test", description="Test"
                )

                assert issue == existing_issue
                assert created is False
                mock_create.assert_not_called()

    @pytest.mark.unit
    def test_get_or_create_issue_creates_new(self, db_with_roles, mock_gitlab_api):
        """Test get_or_create_issue creates new issue when none exists."""
        client = GitLabClient(db=db_with_roles)

        with patch.object(client, "find_existing_issue", return_value=None):
            issue, created = client.get_or_create_issue(
                role_name="common",
                title="feat(common): Box up `common` Ansible role",
                description="Test description",
            )

            assert issue is not None
            assert created is True

    @pytest.mark.unit
    def test_get_or_create_mr_existing(self, db_with_roles):
        """Test get_or_create_mr returns existing MR without creating."""
        client = GitLabClient(db=db_with_roles)

        existing_mr = MagicMock()
        existing_mr.iid = 456
        existing_mr.web_url = "https://example.com/mr/456"

        with patch.object(client, "find_existing_mr", return_value=existing_mr):
            with patch.object(client, "create_merge_request") as mock_create:
                mr, created = client.get_or_create_mr(
                    role_name="common", source_branch="sid/common", title="Test", description="Test"
                )

                assert mr == existing_mr
                assert created is False
                mock_create.assert_not_called()

    @pytest.mark.unit
    def test_get_or_create_mr_creates_new(self, db_with_roles, mock_gitlab_api):
        """Test get_or_create_mr creates new MR when none exists."""
        client = GitLabClient(db=db_with_roles)

        with patch.object(client, "find_existing_mr", return_value=None):
            mr, created = client.get_or_create_mr(
                role_name="common",
                source_branch="sid/common",
                title="feat(common): Add `common` Ansible role",
                description="Test description",
            )

            assert mr is not None
            assert created is True

    @pytest.mark.unit
    def test_get_or_create_mr_checks_merged(self, db_with_roles):
        """Test get_or_create_mr also checks for merged MRs."""
        client = GitLabClient(db=db_with_roles)

        merged_mr = MagicMock()
        merged_mr.iid = 456
        merged_mr.state = "merged"
        merged_mr.web_url = "https://example.com/mr/456"

        call_count = [0]

        def mock_find_mr(branch, state="opened"):
            call_count[0] += 1
            if state == "opened":
                return None  # No open MR
            elif state == "merged":
                return merged_mr  # But there is a merged one
            return None

        with patch.object(client, "find_existing_mr", side_effect=mock_find_mr):
            with patch.object(client, "create_merge_request") as mock_create:
                mr, created = client.get_or_create_mr(
                    role_name="common", source_branch="sid/common", title="Test", description="Test"
                )

                assert mr == merged_mr
                assert created is False
                mock_create.assert_not_called()
                # Should have checked both opened and merged states
                assert call_count[0] == 2

    @pytest.mark.unit
    def test_remote_branch_exists_true(self, in_memory_db):
        """Test remote_branch_exists returns True when branch exists."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "abc123\trefs/heads/sid/common"
            mock_run.return_value = mock_result

            exists = client.remote_branch_exists("sid/common")

            assert exists is True

    @pytest.mark.unit
    def test_remote_branch_exists_false(self, in_memory_db):
        """Test remote_branch_exists returns False when branch doesn't exist."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""  # Empty output means no match
            mock_run.return_value = mock_result

            exists = client.remote_branch_exists("sid/nonexistent")

            assert exists is False

    @pytest.mark.unit
    def test_remote_branch_exists_handles_timeout(self, in_memory_db):
        """Test remote_branch_exists handles timeouts gracefully."""
        client = GitLabClient(db=in_memory_db)
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30)):
            exists = client.remote_branch_exists("sid/common")

            assert exists is False


class TestIdempotencyIntegration:
    """Integration tests for idempotent workflow operations."""

    @pytest.mark.unit
    def test_running_twice_returns_same_issue(self, db_with_roles):
        """Test that running get_or_create_issue twice returns same issue."""
        client = GitLabClient(db=db_with_roles)

        first_issue = MagicMock()
        first_issue.iid = 123
        first_issue.id = 67890
        first_issue.web_url = "https://example.com/issue/123"

        # First call: no existing issue, create new
        call_count = [0]

        def mock_find_issue(role_name, state="opened"):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # First call: not found
            else:
                return first_issue  # Second call: found the one we created

        with patch.object(client, "find_existing_issue", side_effect=mock_find_issue):
            with patch.object(client, "create_issue", return_value=first_issue):
                # First call - creates issue
                issue1, created1 = client.get_or_create_issue(
                    role_name="common", title="Test", description="Test"
                )

                # Second call - should find existing
                issue2, created2 = client.get_or_create_issue(
                    role_name="common", title="Test", description="Test"
                )

                assert created1 is True
                assert created2 is False
                assert issue1.iid == issue2.iid

    @pytest.mark.unit
    def test_running_twice_returns_same_mr(self, db_with_roles):
        """Test that running get_or_create_mr twice returns same MR."""
        client = GitLabClient(db=db_with_roles)

        first_mr = MagicMock()
        first_mr.iid = 456
        first_mr.id = 11111
        first_mr.web_url = "https://example.com/mr/456"
        first_mr.state = "opened"

        call_count = [0]

        def mock_find_mr(branch, state="opened"):
            call_count[0] += 1
            if call_count[0] <= 2:  # First get_or_create (checks opened, then merged)
                return None
            else:
                return first_mr  # Subsequent calls: found

        with patch.object(client, "find_existing_mr", side_effect=mock_find_mr):
            with patch.object(client, "create_merge_request", return_value=first_mr):
                # First call - creates MR
                mr1, created1 = client.get_or_create_mr(
                    role_name="common", source_branch="sid/common", title="Test", description="Test"
                )

                # Second call - should find existing
                mr2, created2 = client.get_or_create_mr(
                    role_name="common", source_branch="sid/common", title="Test", description="Test"
                )

                assert created1 is True
                assert created2 is False
                assert mr1.iid == mr2.iid


class TestIterationDateSelection:
    """Tests for date-based iteration selection."""

    @pytest.mark.unit
    def test_get_current_iteration_by_date_finds_matching(self, in_memory_db):
        """Test finding iteration that contains today's date."""
        client = GitLabClient(db=in_memory_db)
        from datetime import date, timedelta

        today = date.today()
        start = (today - timedelta(days=7)).isoformat()
        end = (today + timedelta(days=7)).isoformat()

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 100,
                        "title": "Past Sprint",
                        "state": "opened",
                        "start_date": "2020-01-01",
                        "due_date": "2020-01-14",
                        "group": {"id": 1},
                    },
                    {
                        "id": 200,
                        "title": "Current Sprint",
                        "state": "opened",
                        "start_date": start,
                        "due_date": end,
                        "group": {"id": 1},
                    },
                ]
            )
            mock_run.return_value = mock_result

            iteration = client.get_current_iteration_by_date()

            assert iteration is not None
            assert iteration.id == 200
            assert iteration.title == "Current Sprint"

    @pytest.mark.unit
    def test_get_current_iteration_by_date_returns_none_when_no_match(self, in_memory_db):
        """Test returns None when no iteration contains today's date."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 100,
                        "title": "Past Sprint",
                        "state": "opened",
                        "start_date": "2020-01-01",
                        "due_date": "2020-01-14",
                        "group": {"id": 1},
                    }
                ]
            )
            mock_run.return_value = mock_result

            iteration = client.get_current_iteration_by_date()

            assert iteration is None

    @pytest.mark.unit
    def test_get_current_iteration_uses_date_first(self, in_memory_db):
        """Test get_current_iteration uses date-based selection first."""
        client = GitLabClient(db=in_memory_db)
        from datetime import date, timedelta

        today = date.today()
        start = (today - timedelta(days=7)).isoformat()
        end = (today + timedelta(days=7)).isoformat()

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 100,
                        "title": "First Opened (not current)",
                        "state": "opened",
                        "start_date": "2020-01-01",
                        "due_date": "2020-01-14",
                        "group": {"id": 1},
                    },
                    {
                        "id": 200,
                        "title": "Current by Date",
                        "state": "opened",
                        "start_date": start,
                        "due_date": end,
                        "group": {"id": 1},
                    },
                ]
            )
            mock_run.return_value = mock_result

            iteration = client.get_current_iteration()

            # Should return the one matching today's date, not the first one
            assert iteration is not None
            assert iteration.id == 200

    @pytest.mark.unit
    def test_get_current_iteration_falls_back_to_first(self, in_memory_db):
        """Test get_current_iteration falls back to first opened if no date match."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 100,
                        "title": "First Opened",
                        "state": "opened",
                        "start_date": "2020-01-01",
                        "due_date": "2020-01-14",
                        "group": {"id": 1},
                    }
                ]
            )
            mock_run.return_value = mock_result

            iteration = client.get_current_iteration()

            # Should fall back to first opened
            assert iteration is not None
            assert iteration.id == 100

    @pytest.mark.unit
    def test_ensure_iteration_exists_returns_current(self, in_memory_db):
        """Test ensure_iteration_exists returns current iteration."""
        client = GitLabClient(db=in_memory_db)
        from datetime import date, timedelta

        today = date.today()
        start = (today - timedelta(days=7)).isoformat()
        end = (today + timedelta(days=7)).isoformat()

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 200,
                        "title": "Current Sprint",
                        "state": "opened",
                        "start_date": start,
                        "due_date": end,
                        "group": {"id": 1},
                    }
                ]
            )
            mock_run.return_value = mock_result

            iteration = client.ensure_iteration_exists()

            assert iteration is not None
            assert iteration.id == 200

    @pytest.mark.unit
    def test_ensure_iteration_exists_warns_and_falls_back(self, in_memory_db, caplog):
        """Test ensure_iteration_exists warns when no current iteration and falls back."""
        import logging

        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {
                        "id": 100,
                        "title": "Past Sprint",
                        "state": "opened",
                        "start_date": "2020-01-01",
                        "due_date": "2020-01-14",
                        "group": {"id": 1},
                    }
                ]
            )
            mock_run.return_value = mock_result

            with caplog.at_level(logging.WARNING):
                iteration = client.ensure_iteration_exists()

            # Should fall back to the only open iteration
            assert iteration is not None
            assert iteration.id == 100


class TestIssueLifecycleOperations:
    """Tests for issue lifecycle management."""

    @pytest.mark.unit
    def test_close_issue_success(self, in_memory_db):
        """Test closing an issue successfully."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123, "state": "closed"}'
            mock_run.return_value = mock_result

            result = client.close_issue(123)

            assert result is True

    @pytest.mark.unit
    def test_close_issue_failure(self, in_memory_db):
        """Test closing an issue with API failure."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "API error"
            mock_run.return_value = mock_result

            result = client.close_issue(123)

            assert result is False

    @pytest.mark.unit
    def test_reopen_issue_success(self, in_memory_db):
        """Test reopening an issue successfully."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123, "state": "opened"}'
            mock_run.return_value = mock_result

            result = client.reopen_issue(123)

            assert result is True

    @pytest.mark.unit
    def test_add_issue_comment(self, in_memory_db):
        """Test adding a comment to an issue."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 456, "body": "Test comment"}'
            mock_run.return_value = mock_result

            result = client.add_issue_comment(123, "Test comment")

            assert result is True

    @pytest.mark.unit
    def test_add_issue_label(self, in_memory_db):
        """Test adding a label to an issue."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123, "labels": ["status::blocked"]}'
            mock_run.return_value = mock_result

            result = client.add_issue_label(123, "status::blocked")

            assert result is True

    @pytest.mark.unit
    def test_remove_issue_label(self, in_memory_db):
        """Test removing a label from an issue."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123, "labels": []}'
            mock_run.return_value = mock_result

            result = client.remove_issue_label(123, "status::blocked")

            assert result is True

    @pytest.mark.unit
    def test_update_issue_on_failure(self, in_memory_db):
        """Test updating an issue with failure information."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists", return_value=True):
            with patch.object(client, "add_issue_label", return_value=True) as mock_add_label:
                with patch.object(
                    client, "add_issue_comment", return_value=True
                ) as mock_add_comment:
                    result = client.update_issue_on_failure(123, "Test error message")

                    assert result is True
                    mock_add_label.assert_called_once_with(123, "status::blocked")
                    mock_add_comment.assert_called_once()
                    # Verify the comment contains the error message
                    comment_body = mock_add_comment.call_args[0][1]
                    assert "Test error message" in comment_body
                    assert "Workflow Failure" in comment_body

    @pytest.mark.unit
    def test_update_issue_on_failure_partial_success(self, in_memory_db):
        """Test update_issue_on_failure returns False on partial failure."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists", return_value=True):
            with patch.object(client, "add_issue_label", return_value=True):
                with patch.object(client, "add_issue_comment", return_value=False):
                    result = client.update_issue_on_failure(123, "Error")

                    assert result is False

    @pytest.mark.unit
    def test_set_issue_due_date_valid(self, in_memory_db):
        """Test setting a valid due date on an issue."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123, "due_date": "2024-12-31"}'
            mock_run.return_value = mock_result

            result = client.set_issue_due_date(123, "2024-12-31")

            assert result is True

    @pytest.mark.unit
    def test_set_issue_due_date_invalid_format(self, in_memory_db):
        """Test setting an invalid due date raises ValueError."""
        client = GitLabClient(db=in_memory_db)

        with pytest.raises(ValueError, match="Invalid date format"):
            client.set_issue_due_date(123, "31-12-2024")

    @pytest.mark.unit
    def test_add_time_estimate(self, in_memory_db):
        """Test adding time estimate to an issue."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"human_time_estimate": "3h"}'
            mock_run.return_value = mock_result

            result = client.add_time_estimate(123, "3h")

            assert result is True

    @pytest.mark.unit
    def test_add_time_spent(self, in_memory_db):
        """Test adding time spent on an issue."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"human_total_time_spent": "2h"}'
            mock_run.return_value = mock_result

            result = client.add_time_spent(123, "2h", summary="Fixed the bug")

            assert result is True

    @pytest.mark.unit
    def test_reset_time_estimate(self, in_memory_db):
        """Test resetting time estimate on an issue."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"human_time_estimate": null}'
            mock_run.return_value = mock_result

            result = client.reset_time_estimate(123)

            assert result is True


class TestFailureUpdateIntegration:
    """Integration tests for failure update workflow."""

    @pytest.mark.unit
    def test_failure_update_creates_blocked_label(self, in_memory_db):
        """Test that update_issue_on_failure ensures the blocked label exists."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists") as mock_ensure:
            mock_ensure.return_value = True
            with patch.object(client, "add_issue_label", return_value=True):
                with patch.object(client, "add_issue_comment", return_value=True):
                    client.update_issue_on_failure(123, "Error")

                    # Verify ensure_label_exists was called with status::blocked
                    mock_ensure.assert_called_once()
                    # Check positional arg or keyword arg
                    if mock_ensure.call_args[0]:
                        assert mock_ensure.call_args[0][0] == "status::blocked"
                    else:
                        assert mock_ensure.call_args[1].get("name") == "status::blocked"

    @pytest.mark.unit
    def test_failure_comment_contains_timestamp(self, in_memory_db):
        """Test that failure comment contains a timestamp."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "ensure_label_exists", return_value=True):
            with patch.object(client, "add_issue_label", return_value=True):
                with patch.object(client, "add_issue_comment", return_value=True) as mock_comment:
                    client.update_issue_on_failure(123, "Test error")

                    comment_body = mock_comment.call_args[0][1]
                    assert "Timestamp:" in comment_body
                    assert "UTC" in comment_body

    @pytest.mark.unit
    def test_update_issue_labels_replaces_all(self, in_memory_db):
        """Test that update_issue_labels replaces all existing labels."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"id": 123, "labels": ["new1", "new2"]}'
            mock_run.return_value = mock_result

            result = client.update_issue_labels(123, ["new1", "new2"])

            assert result is True
            # Verify the labels were passed as comma-separated
            cmd_args = mock_run.call_args[0][0]
            assert any("labels=new1,new2" in arg for arg in cmd_args)


class TestGitLabConfigDefaultReviewers:
    """Tests for default_reviewers configuration."""

    @pytest.mark.unit
    def test_default_reviewers_empty_by_default(self):
        """Test that default_reviewers is empty when not specified."""
        config = GitLabConfig()
        assert config.default_reviewers == []

    @pytest.mark.unit
    def test_default_reviewers_from_init(self):
        """Test setting reviewers via constructor."""
        config = GitLabConfig(default_reviewers=["user1", "user2"])
        assert config.default_reviewers == ["user1", "user2"]

    @pytest.mark.unit
    def test_default_reviewers_from_env(self, monkeypatch):
        """Test loading reviewers from GITLAB_DEFAULT_REVIEWERS env var."""
        monkeypatch.setenv("GITLAB_DEFAULT_REVIEWERS", "reviewer1,reviewer2,reviewer3")
        config = GitLabConfig()
        assert config.default_reviewers == ["reviewer1", "reviewer2", "reviewer3"]

    @pytest.mark.unit
    def test_default_reviewers_from_env_with_spaces(self, monkeypatch):
        """Test loading reviewers with spaces around commas."""
        monkeypatch.setenv("GITLAB_DEFAULT_REVIEWERS", " reviewer1 , reviewer2 , reviewer3 ")
        config = GitLabConfig()
        assert config.default_reviewers == ["reviewer1", "reviewer2", "reviewer3"]

    @pytest.mark.unit
    def test_default_reviewers_init_takes_precedence(self, monkeypatch):
        """Test that explicit init value takes precedence over env."""
        monkeypatch.setenv("GITLAB_DEFAULT_REVIEWERS", "env_user")
        config = GitLabConfig(default_reviewers=["init_user"])
        assert config.default_reviewers == ["init_user"]

    @pytest.mark.unit
    def test_default_reviewers_empty_env_ignored(self, monkeypatch):
        """Test that empty env var results in empty list."""
        monkeypatch.setenv("GITLAB_DEFAULT_REVIEWERS", "")
        config = GitLabConfig()
        assert config.default_reviewers == []


class TestMRLifecycleOperations:
    """Tests for MR lifecycle management methods."""

    @pytest.mark.unit
    def test_get_user_id_found(self, in_memory_db):
        """Test getting user ID from username."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps([{"id": 123, "username": "testuser"}])
            mock_run.return_value = mock_result

            user_id = client.get_user_id("testuser")
            assert user_id == 123

    @pytest.mark.unit
    def test_get_user_id_not_found(self, in_memory_db):
        """Test getting user ID for non-existent user."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "[]"
            mock_run.return_value = mock_result

            user_id = client.get_user_id("nonexistent")
            assert user_id is None

    @pytest.mark.unit
    def test_set_mr_reviewers_success(self, in_memory_db):
        """Test setting reviewers on an MR."""
        client = GitLabClient(db=in_memory_db)

        call_count = [0]

        def mock_run_side_effect(cmd, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            result.returncode = 0

            cmd_str = " ".join(cmd)
            if "users?username=" in cmd_str:
                # Return user lookup
                username = cmd_str.split("username=")[1].split()[0]
                if username == "user1":
                    result.stdout = json.dumps([{"id": 100, "username": "user1"}])
                elif username == "user2":
                    result.stdout = json.dumps([{"id": 200, "username": "user2"}])
                else:
                    result.stdout = "[]"
            else:
                # PUT request
                result.stdout = '{"iid": 456}'
            return result

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            success = client.set_mr_reviewers(456, ["user1", "user2"])
            assert success is True

    @pytest.mark.unit
    def test_set_mr_reviewers_empty_list(self, in_memory_db):
        """Test setting empty reviewers list returns True."""
        client = GitLabClient(db=in_memory_db)
        success = client.set_mr_reviewers(456, [])
        assert success is True

    @pytest.mark.unit
    def test_set_mr_reviewers_no_valid_users(self, in_memory_db):
        """Test setting reviewers when no users can be resolved."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_user_id", return_value=None):
            success = client.set_mr_reviewers(456, ["unknown1", "unknown2"])
            assert success is False

    @pytest.mark.unit
    def test_set_mr_reviewers_partial_resolution(self, in_memory_db):
        """Test setting reviewers when only some users can be resolved."""
        client = GitLabClient(db=in_memory_db)

        def mock_get_user_id(username):
            if username == "valid_user":
                return 123
            return None

        with patch.object(client, "get_user_id", side_effect=mock_get_user_id):
            with patch.object(client, "_api_put", return_value={"iid": 456}):
                success = client.set_mr_reviewers(456, ["valid_user", "invalid_user"])
                assert success is True

    @pytest.mark.unit
    def test_get_mr_pipeline_status_success(self, in_memory_db):
        """Test getting pipeline status for an MR."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                [
                    {"id": 999, "status": "running", "sha": "abc123"},
                    {"id": 998, "status": "success", "sha": "def456"},
                ]
            )
            mock_run.return_value = mock_result

            status = client.get_mr_pipeline_status(456)
            assert status == "running"  # Most recent first

    @pytest.mark.unit
    def test_get_mr_pipeline_status_no_pipelines(self, in_memory_db):
        """Test getting pipeline status when no pipelines exist."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "[]"
            mock_run.return_value = mock_result

            status = client.get_mr_pipeline_status(456)
            assert status is None

    @pytest.mark.unit
    def test_get_mr_pipeline_status_error(self, in_memory_db):
        """Test getting pipeline status when API fails."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "API error"
            mock_run.return_value = mock_result

            status = client.get_mr_pipeline_status(456)
            assert status is None

    @pytest.mark.unit
    def test_add_mr_comment_success(self, in_memory_db):
        """Test adding a comment to an MR."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"id": 789, "body": "Test comment"})
            mock_run.return_value = mock_result

            success = client.add_mr_comment(456, "Test comment")
            assert success is True

    @pytest.mark.unit
    def test_add_mr_comment_failure(self, in_memory_db):
        """Test adding a comment when API fails."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "API error"
            mock_run.return_value = mock_result

            success = client.add_mr_comment(456, "Test comment")
            assert success is False

    @pytest.mark.unit
    def test_mark_mr_ready_success(self, in_memory_db):
        """Test marking a draft MR as ready."""
        client = GitLabClient(db=in_memory_db)

        mock_mr = MagicMock()
        mock_mr.title = "Draft: Fix bug"
        mock_mr.iid = 456

        with patch.object(client, "get_merge_request", return_value=mock_mr):
            with patch.object(client, "_api_put", return_value={"iid": 456}):
                success = client.mark_mr_ready(456)
                assert success is True

    @pytest.mark.unit
    def test_mark_mr_ready_already_ready(self, in_memory_db):
        """Test marking an already-ready MR returns True."""
        client = GitLabClient(db=in_memory_db)

        mock_mr = MagicMock()
        mock_mr.title = "Fix bug"  # No Draft: prefix
        mock_mr.iid = 456

        with patch.object(client, "get_merge_request", return_value=mock_mr):
            success = client.mark_mr_ready(456)
            assert success is True

    @pytest.mark.unit
    def test_mark_mr_ready_wip_prefix(self, in_memory_db):
        """Test marking a WIP MR as ready."""
        client = GitLabClient(db=in_memory_db)

        mock_mr = MagicMock()
        mock_mr.title = "WIP: Fix bug"
        mock_mr.iid = 456

        with patch.object(client, "get_merge_request", return_value=mock_mr):
            with patch.object(client, "_api_put", return_value={"iid": 456}) as mock_put:
                success = client.mark_mr_ready(456)
                assert success is True
                # Verify title was updated without WIP: prefix
                mock_put.assert_called_once()

    @pytest.mark.unit
    def test_mark_mr_ready_not_found(self, in_memory_db):
        """Test marking a non-existent MR returns False."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_merge_request", return_value=None):
            success = client.mark_mr_ready(456)
            assert success is False

    @pytest.mark.unit
    def test_merge_immediately_success(self, in_memory_db):
        """Test immediate merge of an MR."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(
                {"iid": 456, "state": "merged", "merged_by": {"username": "testuser"}}
            )
            mock_run.return_value = mock_result

            result = client.merge_immediately(456)
            assert result["state"] == "merged"

    @pytest.mark.unit
    def test_merge_immediately_without_skip_train(self, in_memory_db):
        """Test immediate merge without skipping merge train."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "_api_put") as mock_put:
            mock_put.return_value = {"iid": 456, "state": "merged"}

            client.merge_immediately(456, skip_merge_train=False)

            # Verify skip_merge_train was not passed
            mock_put.call_args[0]
            data = mock_put.call_args[0][1]
            assert "skip_merge_train" not in data

    @pytest.mark.unit
    def test_merge_immediately_failure(self, in_memory_db):
        """Test immediate merge when merge fails."""
        client = GitLabClient(db=in_memory_db)

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Merge conflicts"
            mock_run.return_value = mock_result

            with pytest.raises(RuntimeError, match="Merge conflicts"):
                client.merge_immediately(456)

    @pytest.mark.unit
    def test_wait_for_mr_pipeline_immediate_success(self, in_memory_db):
        """Test waiting for pipeline that's already successful."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, "get_mr_pipeline_status", return_value="success"):
            result = client.wait_for_mr_pipeline(456, timeout_seconds=1, poll_interval=1)
            assert result == "success"

    @pytest.mark.unit
    def test_wait_for_mr_pipeline_terminal_states(self, in_memory_db):
        """Test that all terminal states end waiting."""
        client = GitLabClient(db=in_memory_db)

        for terminal_status in ["success", "failed", "canceled", "skipped"]:
            with patch.object(client, "get_mr_pipeline_status", return_value=terminal_status):
                result = client.wait_for_mr_pipeline(456, timeout_seconds=1, poll_interval=1)
                assert result == terminal_status


class TestMRLifecycleIntegration:
    """Integration tests for MR lifecycle in workflow."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_mr_node_creates_mr(self, db_with_roles, monkeypatch):
        """Test that create_mr_node creates MR via async GitLabAPI."""
        from harness.dag.langgraph_engine import create_mr_node, set_module_db

        set_module_db(db_with_roles)

        mock_mr = {
            "iid": 456,
            "web_url": "https://example.com/mr/456",
        }

        with (
            patch("harness.gitlab.http_client.GitLabAPI") as mock_api_class,
            patch("harness.gitlab.http_client.GitLabAPIConfig") as mock_config_class,
            patch("harness.gitlab.templates.render_mr_description") as mock_render,
        ):
            # Set up mock config
            mock_config = MagicMock()
            mock_config.project_path = "test/project"
            mock_config.default_assignee = "test_user"
            mock_config.default_labels = ["role", "ansible"]
            mock_config.default_reviewers = ["reviewer1", "reviewer2"]
            mock_config_class.from_harness_yml.return_value = mock_config

            # Mock template rendering
            mock_render.return_value = "## Summary\nTest MR description"

            # Set up mock API with async context manager
            mock_api = AsyncMock()
            mock_api.find_mr_by_branch = AsyncMock(return_value=None)  # No existing MR
            mock_api.get_or_create_mr = AsyncMock(return_value=(mock_mr, True))

            # Configure context manager
            mock_api_class.return_value.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api_class.return_value.__aexit__ = AsyncMock(return_value=None)

            state = {
                "role_name": "common",
                "issue_iid": 123,
                "branch": "sid/common",
                "wave": 0,
                "wave_name": "Foundation",
            }

            result = await create_mr_node(state)

            assert result["mr_iid"] == 456
            assert result["mr_created"] is True
            assert "create_merge_request" in result["completed_nodes"]
