"""Retry behavior. A batch upload that dies on one 429 wastes the whole run."""

import httpx
import pytest
import respx
from tenacity import wait_none

from batchlane._http import HttpError, request

# Same retry policy, without the real backoff sleeps. Keeps the unit suite fast
# while still exercising the retry/stop logic.
fast_request = request.retry_with(wait=wait_none())


@respx.mock
def test_429_is_retried_then_succeeds():
    route = respx.get("https://x.test/a").mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    assert fast_request("GET", "https://x.test/a").json() == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_transport_errors_are_retried():
    route = respx.get("https://x.test/b").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={})]
    )
    fast_request("GET", "https://x.test/b")
    assert route.call_count == 2


@respx.mock
def test_400_is_not_retried_because_retrying_cannot_help():
    route = respx.get("https://x.test/c").mock(
        return_value=httpx.Response(400, text="bad input_file_id")
    )
    with pytest.raises(HttpError) as exc:
        fast_request("GET", "https://x.test/c")
    assert route.call_count == 1
    assert exc.value.status == 400
    # The provider's own message must survive; it is usually the only clue.
    assert "bad input_file_id" in str(exc.value)


@respx.mock
def test_retries_are_bounded():
    route = respx.get("https://x.test/d").mock(return_value=httpx.Response(503))
    with pytest.raises(HttpError):
        fast_request("GET", "https://x.test/d")
    assert route.call_count == 5
