"""
GitLab API error classes and exception hierarchy.

This module provides a comprehensive exception hierarchy for GitLab API errors,
enabling proper error handling, retry logic, and debugging.
"""

from typing import Any


class GitLabAPIError(Exception):
    """
    Base exception for all GitLab API errors.

    Attributes:
        status_code: HTTP status code from the response
        message: Human-readable error message
        response_body: Raw response body for debugging
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict[str, Any] | str | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the exception message with status code if available."""
        if self.status_code:
            return f"GitLab API Error ({self.status_code}): {self.message}"
        return f"GitLab API Error: {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"response_body={self.response_body!r})"
        )


class GitLabRateLimitError(GitLabAPIError):
    """
    Raised when GitLab returns a 429 rate limit response.

    Attributes:
        retry_after_seconds: Number of seconds to wait before retrying,
            parsed from the Retry-After header
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        status_code: int = 429,
        response_body: dict[str, Any] | str | None = None,
        retry_after_seconds: int | None = None,
    ):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message, status_code, response_body)

    def _format_message(self) -> str:
        base = super()._format_message()
        if self.retry_after_seconds:
            return f"{base} (retry after {self.retry_after_seconds}s)"
        return base

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"retry_after_seconds={self.retry_after_seconds})"
        )


class GitLabAuthenticationError(GitLabAPIError):
    """
    Raised when authentication fails (401) or authorization is denied (403).

    This error indicates either:
    - Invalid or expired token (401)
    - Insufficient permissions for the requested operation (403)
    """

    def __init__(
        self,
        message: str = "Authentication or authorization failed",
        status_code: int = 401,
        response_body: dict[str, Any] | str | None = None,
    ):
        super().__init__(message, status_code, response_body)


class GitLabNotFoundError(GitLabAPIError):
    """
    Raised when a requested resource is not found (404).

    This typically indicates:
    - Invalid project/group path
    - Non-existent issue, MR, or other resource
    - Resource deleted or moved
    """

    def __init__(
        self,
        message: str = "Resource not found",
        status_code: int = 404,
        response_body: dict[str, Any] | str | None = None,
    ):
        super().__init__(message, status_code, response_body)


class GitLabConflictError(GitLabAPIError):
    """
    Raised when a conflict occurs (409).

    This typically indicates:
    - Resource already exists
    - Concurrent modification conflict
    - State conflict (e.g., merging an already merged MR)
    """

    def __init__(
        self,
        message: str = "Conflict occurred",
        status_code: int = 409,
        response_body: dict[str, Any] | str | None = None,
    ):
        super().__init__(message, status_code, response_body)


class GitLabServerError(GitLabAPIError):
    """
    Raised when GitLab returns a server error (5xx).

    These errors are typically transient and may be resolved by retrying.
    Common causes:
    - GitLab service overload
    - Temporary infrastructure issues
    - Database timeouts
    """

    def __init__(
        self,
        message: str = "GitLab server error",
        status_code: int = 500,
        response_body: dict[str, Any] | str | None = None,
    ):
        super().__init__(message, status_code, response_body)


class GitLabTimeoutError(GitLabAPIError):
    """
    Raised when a request to GitLab times out.

    This is a client-side timeout, not a GitLab 504 response.
    """

    def __init__(
        self,
        message: str = "Request timed out",
        status_code: int | None = None,
        response_body: dict[str, Any] | str | None = None,
    ):
        super().__init__(message, status_code, response_body)


def parse_gitlab_error(response: "httpx.Response") -> GitLabAPIError:
    """
    Parse an httpx response and return the appropriate GitLabAPIError subclass.

    Args:
        response: The httpx response object

    Returns:
        An appropriate GitLabAPIError subclass based on the status code

    Example:
        >>> import httpx
        >>> # Assuming a 404 response
        >>> error = parse_gitlab_error(response)
        >>> isinstance(error, GitLabNotFoundError)
        True
    """
    import httpx  # Import here to avoid circular imports

    status_code = response.status_code

    # Try to parse the response body as JSON
    try:
        response_body = response.json()
        # GitLab typically returns errors in "message" or "error" fields
        if isinstance(response_body, dict):
            message = response_body.get("message") or response_body.get("error") or str(response_body)
        else:
            message = str(response_body)
    except Exception:
        response_body = response.text
        message = response.text if response.text else f"HTTP {status_code}"

    # Map status codes to exception classes
    if status_code == 429:
        # Parse Retry-After header
        retry_after = response.headers.get("Retry-After")
        retry_after_seconds = None
        if retry_after:
            try:
                retry_after_seconds = int(retry_after)
            except ValueError:
                pass
        return GitLabRateLimitError(
            message=message,
            status_code=status_code,
            response_body=response_body,
            retry_after_seconds=retry_after_seconds,
        )
    elif status_code == 401 or status_code == 403:
        return GitLabAuthenticationError(
            message=message,
            status_code=status_code,
            response_body=response_body,
        )
    elif status_code == 404:
        return GitLabNotFoundError(
            message=message,
            status_code=status_code,
            response_body=response_body,
        )
    elif status_code == 409:
        return GitLabConflictError(
            message=message,
            status_code=status_code,
            response_body=response_body,
        )
    elif 500 <= status_code < 600:
        return GitLabServerError(
            message=message,
            status_code=status_code,
            response_body=response_body,
        )
    else:
        return GitLabAPIError(
            message=message,
            status_code=status_code,
            response_body=response_body,
        )


# Export all exception classes for easy importing
__all__ = [
    "GitLabAPIError",
    "GitLabRateLimitError",
    "GitLabAuthenticationError",
    "GitLabNotFoundError",
    "GitLabConflictError",
    "GitLabServerError",
    "GitLabTimeoutError",
    "parse_gitlab_error",
]
