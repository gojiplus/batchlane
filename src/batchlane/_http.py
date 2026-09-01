"""Shared HTTP transport.

Plain httpx plus tenacity, deliberately. LiteLLM's own ``AsyncHTTPHandler``
exists to feed its cost-logging and callback pipeline, which this package
excludes; importing it for retry semantics alone would couple us to that
lifecycle for no benefit.

Everything is synchronous. A batch is one upload, one submit, then a poll every
few minutes -- async concurrency buys nothing at that cadence, and a sync API
avoids the "cannot be called from a running event loop" trap.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

__all__ = ["HttpError", "request"]

_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class HttpError(RuntimeError):
    """A provider returned a response we cannot use."""

    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        """Build the error.

        Args:
            method: HTTP method used.
            url: URL requested.
            status: HTTP status returned.
            body: Response body, truncated by the caller if large.
        """
        self.status = status
        self.body = body
        super().__init__(f"{method} {url} -> {status}: {body[:500]}")


def _is_retryable(exc: BaseException) -> bool:
    """Whether a failure is worth retrying.

    Args:
        exc: The raised exception.

    Returns:
        True for transport errors and retryable status codes.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, HttpError) and exc.status in _RETRY_STATUS


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    files: Any = None,
    data: Any = None,
    params: dict[str, Any] | None = None,
    timeout: httpx.Timeout | None = None,
) -> httpx.Response:
    """Make one HTTP call, retrying transport errors and 429/5xx.

    Args:
        method: HTTP method.
        url: Full URL.
        headers: Request headers.
        json_body: JSON payload, if any.
        files: Multipart file payload, if any.
        data: Form fields accompanying a multipart upload.
        params: Query string parameters.
        timeout: Override the default timeout.

    Returns:
        The successful response.

    Raises:
        HttpError: If the provider returns a non-2xx status after retries.
    """
    with httpx.Client(timeout=timeout or DEFAULT_TIMEOUT) as client:
        response = client.request(
            method,
            url,
            headers=headers,
            json=json_body,
            files=files,
            data=data,
            params=params,
        )
    if response.status_code >= 400:
        raise HttpError(method, url, response.status_code, response.text)
    return response
