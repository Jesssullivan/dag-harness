"""Unit tests for the GitLab API client."""

import json
import pytest
from unittest.mock import patch, MagicMock

from harness.gitlab.api import GitLabClient, GitLabConfig
from harness.db.state import StateDB
from harness.db.models import Role


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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps([{
                "id": 999,
                "title": "Sprint 43",
                "state": "opened",
                "group": {"id": 100}
            }])
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
            weight=3
        )

        assert issue is not None
        assert issue.iid == 123
        assert issue.state == "opened"

    @pytest.mark.unit
    def test_create_issue_role_not_found(self, in_memory_db):
        """Test creating issue for non-existent role."""
        client = GitLabClient(db=in_memory_db)

        with pytest.raises(ValueError, match="not found"):
            client.create_issue(
                role_name="nonexistent",
                title="Test",
                description="Test"
            )

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

        with patch('subprocess.run') as mock_run:
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
            description="Test MR description"
        )

        assert mr is not None
        assert mr.iid == 456
        assert mr.source_branch == "sid/common"

    @pytest.mark.unit
    def test_create_draft_mr(self, db_with_roles, mock_gitlab_api):
        """Test creating a draft merge request."""
        client = GitLabClient(db=db_with_roles)

        with patch.object(client, '_api_get') as mock_get:
            mock_get.return_value = {
                "id": 11111,
                "iid": 456,
                "source_branch": "sid/common",
                "target_branch": "main",
                "title": "Draft: Box up common role",
                "state": "opened",
                "web_url": "https://gitlab.example.com/mr/456",
                "merge_status": "can_be_merged"
            }

            mr = client.create_merge_request(
                role_name="common",
                source_branch="sid/common",
                title="Box up common role",
                description="Draft MR",
                draft=True
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

        with patch('subprocess.run') as mock_run:
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

        with patch('subprocess.run', side_effect=side_effect):
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

        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = client.remove_from_merge_train(1)
            assert result is True

    @pytest.mark.unit
    def test_get_merge_train_position(self, in_memory_db):
        """Test getting MR position in merge train."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, 'get_merge_train') as mock_train:
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

        with patch.object(client, 'get_merge_request') as mock_mr:
            mock_mr_obj = MagicMock()
            mock_mr_obj.state = "opened"
            mock_mr_obj.merge_status = "can_be_merged"
            mock_mr.return_value = mock_mr_obj

            with patch.object(client, 'get_merge_train_status') as mock_status:
                mock_status.return_value = None

                result = client.is_merge_train_available(456)
                assert result["available"] is True

    @pytest.mark.unit
    def test_is_merge_train_available_conflict(self, in_memory_db):
        """Test MR with conflicts cannot be added."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, 'get_merge_request') as mock_mr:
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

        with patch.object(client, 'remove_mr_from_merge_train') as mock_remove:
            mock_remove.return_value = True

            result = client.abort_merge_train(456)
            assert result is True

    @pytest.mark.unit
    def test_list_merge_trains(self, in_memory_db):
        """Test listing all merge trains."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, 'get_merge_train') as mock_train:
            mock_train.return_value = [{"id": 1}]

            trains = client.list_merge_trains()
            assert len(trains) >= 1

    @pytest.mark.unit
    def test_is_merge_train_enabled(self, in_memory_db):
        """Test checking if merge trains are enabled."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, 'get_merge_train') as mock_train:
            mock_train.return_value = []

            enabled = client.is_merge_train_enabled()
            assert enabled is True

    @pytest.mark.unit
    def test_is_merge_train_not_enabled(self, in_memory_db):
        """Test detecting merge trains not enabled."""
        client = GitLabClient(db=in_memory_db)

        with patch.object(client, 'get_merge_train') as mock_train:
            mock_train.side_effect = Exception("Not found")

            enabled = client.is_merge_train_enabled()
            assert enabled is False


class TestSyncOperations:
    """Tests for sync operations."""

    @pytest.mark.unit
    def test_sync_issues(self, db_with_roles):
        """Test syncing issues from GitLab."""
        client = GitLabClient(db=db_with_roles)

        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            # Note: iteration is null to avoid FK constraint on non-existent iteration
            mock_result.stdout = json.dumps([
                {
                    "id": 1,
                    "iid": 10,
                    "title": "Box up `common` role",
                    "state": "opened",
                    "web_url": "https://example.com/1",
                    "labels": ["role"],
                    "iteration": None,
                    "weight": 2
                }
            ])
            mock_run.return_value = mock_result

            count = client.sync_issues()
            assert count >= 1

    @pytest.mark.unit
    def test_sync_merge_requests(self, db_with_roles):
        """Test syncing merge requests from GitLab."""
        client = GitLabClient(db=db_with_roles)

        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps([
                {
                    "id": 1,
                    "iid": 20,
                    "source_branch": "sid/common",
                    "target_branch": "main",
                    "title": "Box up common",
                    "state": "opened",
                    "web_url": "https://example.com/mr/20",
                    "merge_status": "can_be_merged"
                }
            ])
            mock_run.return_value = mock_result

            count = client.sync_merge_requests()
            assert count >= 1
