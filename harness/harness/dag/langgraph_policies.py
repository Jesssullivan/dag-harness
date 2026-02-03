"""
Retry policies for LangGraph workflow nodes.

Defines retry behavior for different types of operations:
- GitLab API calls: Network resilience with exponential backoff
- Subprocess execution: Timeout recovery
- Git operations: Push reliability
"""

import subprocess

import httpx
from langgraph.types import RetryPolicy


# Retry policy for GitLab API calls (create_issue, create_mr, add_to_merge_train)
# Retries on network errors, timeouts, and 5xx server errors
# Does NOT retry on 4xx client errors (bad request, auth, not found)
GITLAB_API_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,  # Start with 1 second delay
    backoff_factor=2.0,  # Exponential backoff: 1s, 2s, 4s
    max_interval=10.0,  # Cap at 10 seconds between retries
    jitter=True,  # Add randomness to prevent thundering herd
    retry_on=(
        httpx.RequestError,  # Network errors (connection, DNS, etc.)
        httpx.TimeoutException,  # Request timeouts
        RuntimeError,  # Generic runtime errors from API client
    ),
)

# Retry policy for subprocess execution (molecule, pytest)
# Retries on timeout only - test failures are NOT retried
# This handles transient infrastructure issues (VM spin-up, network)
SUBPROCESS_RETRY_POLICY = RetryPolicy(
    max_attempts=2,  # Only retry once for subprocess timeouts
    initial_interval=5.0,  # Wait 5 seconds before retry
    backoff_factor=1.0,  # No backoff for subprocess retries
    max_interval=5.0,
    jitter=False,
    retry_on=(
        subprocess.TimeoutExpired,  # Only retry on timeout
    ),
)

# Retry policy for git operations (push_branch)
# Retries on network errors during push
GIT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=15.0,
    jitter=True,
    retry_on=(
        subprocess.CalledProcessError,  # Git command failures
        RuntimeError,
    ),
)

__all__ = [
    "GITLAB_API_RETRY_POLICY",
    "SUBPROCESS_RETRY_POLICY",
    "GIT_RETRY_POLICY",
]
