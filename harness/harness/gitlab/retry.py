"""
GitLab API retry decorator with exponential backoff.

This module provides a retry decorator for GitLab API calls that handles:
- Server errors (5xx) with exponential backoff
- Rate limiting (429) with Retry-After header support
- Network timeouts
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import httpx

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

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def gitlab_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retrying GitLab API calls with exponential backoff.

    Retry behavior:
    - 5xx errors: Retry with exponential backoff (1s, 2s, 4s by default)
    - 429 rate limit: Retry using Retry-After header value
    - Network timeouts: Retry with exponential backoff
    - 4xx errors (except 429): DO NOT retry, raise immediately

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        exponential_base: Base for exponential backoff (default: 2.0)

    Returns:
        Decorated function with retry logic

    Example:
        @gitlab_retry(max_attempts=3)
        def fetch_project(client: httpx.Client, project_id: str):
            response = client.get(f"/projects/{project_id}")
            if response.status_code >= 400:
                raise parse_gitlab_error(response)
            return response.json()
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except GitLabRateLimitError as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "Rate limit exceeded after %d attempts: %s",
                            max_attempts,
                            e.message,
                        )
                        raise

                    # Use Retry-After header if available, otherwise use exponential backoff
                    if e.retry_after_seconds:
                        delay = min(e.retry_after_seconds, max_delay)
                    else:
                        delay = min(
                            base_delay * (exponential_base ** (attempt - 1)),
                            max_delay,
                        )

                    logger.warning(
                        "Rate limit hit (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        delay,
                        e.message,
                    )
                    time.sleep(delay)

                except GitLabServerError as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "Server error after %d attempts: %s",
                            max_attempts,
                            e.message,
                        )
                        raise

                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %.1fs: %s",
                        e.status_code,
                        attempt,
                        max_attempts,
                        delay,
                        e.message,
                    )
                    time.sleep(delay)

                except (httpx.TimeoutException, GitLabTimeoutError) as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "Timeout after %d attempts: %s",
                            max_attempts,
                            str(e),
                        )
                        if isinstance(e, httpx.TimeoutException):
                            raise GitLabTimeoutError(f"Request timed out: {e}") from e
                        raise

                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )
                    logger.warning(
                        "Timeout (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        delay,
                        str(e),
                    )
                    time.sleep(delay)

                except (
                    GitLabAuthenticationError,
                    GitLabNotFoundError,
                    GitLabConflictError,
                ) as e:
                    # Do NOT retry 4xx errors (except 429 handled above)
                    logger.error(
                        "Non-retryable error (%d): %s",
                        e.status_code,
                        e.message,
                    )
                    raise

                except GitLabAPIError as e:
                    # Generic GitLab errors - check if retryable
                    if e.status_code and 400 <= e.status_code < 500:
                        # 4xx errors are not retryable
                        logger.error(
                            "Non-retryable client error (%d): %s",
                            e.status_code,
                            e.message,
                        )
                        raise

                    # Unknown error, try retrying
                    last_exception = e
                    if attempt == max_attempts:
                        raise

                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )
                    logger.warning(
                        "Unknown error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        delay,
                        str(e),
                    )
                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop exited unexpectedly")

        return sync_wrapper

    return decorator


def gitlab_retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Async version of gitlab_retry decorator.

    Same behavior as gitlab_retry but for async functions.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        exponential_base: Base for exponential backoff (default: 2.0)

    Returns:
        Decorated async function with retry logic

    Example:
        @gitlab_retry_async(max_attempts=3)
        async def fetch_project(client: httpx.AsyncClient, project_id: str):
            response = await client.get(f"/projects/{project_id}")
            if response.status_code >= 400:
                raise parse_gitlab_error(response)
            return response.json()
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except GitLabRateLimitError as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "Rate limit exceeded after %d attempts: %s",
                            max_attempts,
                            e.message,
                        )
                        raise

                    if e.retry_after_seconds:
                        delay = min(e.retry_after_seconds, max_delay)
                    else:
                        delay = min(
                            base_delay * (exponential_base ** (attempt - 1)),
                            max_delay,
                        )

                    logger.warning(
                        "Rate limit hit (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        delay,
                        e.message,
                    )
                    await asyncio.sleep(delay)

                except GitLabServerError as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "Server error after %d attempts: %s",
                            max_attempts,
                            e.message,
                        )
                        raise

                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %.1fs: %s",
                        e.status_code,
                        attempt,
                        max_attempts,
                        delay,
                        e.message,
                    )
                    await asyncio.sleep(delay)

                except (httpx.TimeoutException, GitLabTimeoutError) as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            "Timeout after %d attempts: %s",
                            max_attempts,
                            str(e),
                        )
                        if isinstance(e, httpx.TimeoutException):
                            raise GitLabTimeoutError(f"Request timed out: {e}") from e
                        raise

                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )
                    logger.warning(
                        "Timeout (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        delay,
                        str(e),
                    )
                    await asyncio.sleep(delay)

                except (
                    GitLabAuthenticationError,
                    GitLabNotFoundError,
                    GitLabConflictError,
                ) as e:
                    logger.error(
                        "Non-retryable error (%d): %s",
                        e.status_code,
                        e.message,
                    )
                    raise

                except GitLabAPIError as e:
                    if e.status_code and 400 <= e.status_code < 500:
                        logger.error(
                            "Non-retryable client error (%d): %s",
                            e.status_code,
                            e.message,
                        )
                        raise

                    last_exception = e
                    if attempt == max_attempts:
                        raise

                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )
                    logger.warning(
                        "Unknown error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        delay,
                        str(e),
                    )
                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop exited unexpectedly")

        return async_wrapper

    return decorator


def check_response(response: httpx.Response) -> None:
    """
    Check an httpx response and raise appropriate exception if error.

    This is a convenience function for use with the retry decorators.

    Args:
        response: The httpx response to check

    Raises:
        GitLabAPIError: Appropriate subclass based on status code

    Example:
        @gitlab_retry()
        def get_project(client: httpx.Client, project_id: str):
            response = client.get(f"/projects/{project_id}")
            check_response(response)
            return response.json()
    """
    if response.status_code >= 400:
        raise parse_gitlab_error(response)


async def check_response_async(response: httpx.Response) -> None:
    """
    Async version of check_response (same behavior, included for API consistency).

    Args:
        response: The httpx response to check

    Raises:
        GitLabAPIError: Appropriate subclass based on status code
    """
    check_response(response)


__all__ = [
    "gitlab_retry",
    "gitlab_retry_async",
    "check_response",
    "check_response_async",
]
