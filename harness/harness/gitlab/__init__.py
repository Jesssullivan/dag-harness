"""GitLab API integration."""

from harness.gitlab.api import GitLabClient
from harness.gitlab.errors import (
    GitLabAPIError,
    GitLabAuthenticationError,
    GitLabConflictError,
    GitLabNotFoundError,
    GitLabRateLimitError,
    GitLabServerError,
    GitLabTimeoutError,
    parse_gitlab_error,
)
from harness.gitlab.idempotency import (
    IdempotencyHelper,
    RoleArtifacts,
    cache_result,
    clear_cache,
    find_all_role_artifacts,
    find_existing_issue,
    find_existing_mr,
    get_cache_stats,
)
from harness.gitlab.labels import (
    DEFAULT_LABEL_COLOR,
    WAVE_LABEL_COLORS,
    WAVE_LABEL_DESCRIPTIONS,
    WAVE_LABELS,
    LabelManager,
    MilestoneManager,
)
from harness.gitlab.retry import (
    check_response,
    check_response_async,
    gitlab_retry,
    gitlab_retry_async,
)
from harness.gitlab.advanced import (
    GitLabProjectManager,
    GitLabIterationManager,
    GitLabMergeTrainManager,
    encode_project_path,
    parse_gid,
    build_gid,
    # GraphQL queries
    ITERATION_QUERY,
    PROJECT_MERGE_REQUESTS_QUERY,
    MERGE_TRAIN_QUERY,
    ITERATION_CADENCE_QUERY,
    # Convenience functions
    create_project_manager,
    create_iteration_manager,
    create_merge_train_manager,
)
from harness.gitlab.graphql import (
    GitLabGraphQLClient,
    GitLabGraphQLError,
    GraphQLErrorLocation,
)
from harness.gitlab.merge_ordering import (
    MergeOrder,
    MergeOrderingService,
    PlacementResult,
)
from harness.gitlab.merge_train import (
    MergeReadinessResult,
    MergeTrainHelper,
    get_mr_merge_readiness,
    wait_for_merge_train_position,
    preflight_merge_train_check,
    create_merge_train_helper,
)
from harness.gitlab.status import (
    FullStatus,
    GitLabStatusReporter,
    MergeTrainHealth,
    RoleGitLabStatus,
    WaveStatus,
)

__all__ = [
    # Client
    "GitLabClient",
    # Idempotency helpers
    "IdempotencyHelper",
    "RoleArtifacts",
    "cache_result",
    "clear_cache",
    "find_all_role_artifacts",
    "find_existing_issue",
    "find_existing_mr",
    "get_cache_stats",
    # Label management
    "LabelManager",
    "MilestoneManager",
    "WAVE_LABELS",
    "WAVE_LABEL_COLORS",
    "WAVE_LABEL_DESCRIPTIONS",
    "DEFAULT_LABEL_COLOR",
    # Errors
    "GitLabAPIError",
    "GitLabAuthenticationError",
    "GitLabConflictError",
    "GitLabNotFoundError",
    "GitLabRateLimitError",
    "GitLabServerError",
    "GitLabTimeoutError",
    "parse_gitlab_error",
    # Retry decorators
    "gitlab_retry",
    "gitlab_retry_async",
    "check_response",
    "check_response_async",
    # Advanced managers
    "GitLabProjectManager",
    "GitLabIterationManager",
    "GitLabMergeTrainManager",
    # Utility functions
    "encode_project_path",
    "parse_gid",
    "build_gid",
    # GraphQL queries
    "ITERATION_QUERY",
    "PROJECT_MERGE_REQUESTS_QUERY",
    "MERGE_TRAIN_QUERY",
    "ITERATION_CADENCE_QUERY",
    # Factory functions
    "create_project_manager",
    "create_iteration_manager",
    "create_merge_train_manager",
    # GraphQL client
    "GitLabGraphQLClient",
    "GitLabGraphQLError",
    "GraphQLErrorLocation",
    # Enhanced merge train
    "MergeReadinessResult",
    "MergeTrainHelper",
    "get_mr_merge_readiness",
    "wait_for_merge_train_position",
    "preflight_merge_train_check",
    "create_merge_train_helper",
    # Merge ordering
    "MergeOrder",
    "MergeOrderingService",
    "PlacementResult",
    # Status reporting
    "GitLabStatusReporter",
    "RoleGitLabStatus",
    "WaveStatus",
    "MergeTrainHealth",
    "FullStatus",
]
