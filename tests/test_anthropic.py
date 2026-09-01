"""Anthropic's lane, whose shape differs from the OpenAI-shaped providers."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.anthropic import BASE_URL, AnthropicAdapter

ADAPTER = AnthropicAdapter()
MODEL = "claude-haiku-4-5-20251001"


def _line(cid="row-1", text="hi", **params):
    return bl.BatchLine(cid, MODEL, [{"role": "user", "content": text}], params)


def _handle():
    return bl.BatchHandle(
        provider="anthropic",
        job_id="msgbatch_1",
        endpoint="chat.completions",
        lane="batch_inline",
        created_at=bl.handle.utcnow(),
    )


def test_requests_are_inline_and_carry_max_tokens():
    # Anthropic 400s without max_tokens, so a missing one must not bounce the
    # whole batch.
    built = ADAPTER.build_requests([_line(), _line("row-2", max_tokens=8)])
    assert [r["custom_id"] for r in built] == ["row-1", "row-2"]
    assert built[0]["params"]["max_tokens"] > 0
    assert built[1]["params"]["max_tokens"] == 8
    assert built[0]["params"]["model"] == MODEL


def test_system_message_is_hoisted_out_of_messages():
    line = bl.BatchLine(
        "row-1",
        MODEL,
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}],
        {},
    )
    params = ADAPTER.build_requests([line])[0]["params"]
    assert "system" in params
    assert all(m["role"] != "system" for m in params["messages"])


@respx.mock
def test_submit_posts_requests_inline_with_no_upload_step():
    route = respx.post(BASE_URL).mock(
        return_value=httpx.Response(
            200, json={"id": "msgbatch_1", "processing_status": "in_progress"}
        )
    )
    handle = ADAPTER.submit(
        [_line()], endpoint="chat.completions", window=None, api_key="k"
    )
    assert handle.lane == "batch_inline"
    assert handle.job_id == "msgbatch_1"
    body = json.loads(route.calls[0].request.content)
    assert "requests" in body
    assert "input_file_id" not in body
    assert route.calls[0].request.headers["anthropic-version"]
    assert route.calls[0].request.headers["x-api-key"] == "k"


def test_a_window_is_refused_because_anthropic_accepts_none():
    with pytest.raises(bl.CapabilityNotSupportedError, match="completion_window"):
        ADAPTER.check([_line()], endpoint="chat.completions", window="24h")


@pytest.mark.parametrize(
    ("status", "counts", "expected"),
    [
        ("in_progress", {"processing": 2}, "running"),
        ("canceling", {"processing": 2}, "running"),
        ("ended", {"succeeded": 2}, "succeeded"),
        # Per-line failures do not fail the job, matching OpenAI's semantics.
        ("ended", {"succeeded": 1, "errored": 1}, "succeeded"),
        ("ended", {"canceled": 2}, "cancelled"),
        ("ended", {"expired": 2}, "expired"),
        ("ended", {"errored": 2}, "failed"),
    ],
)
def test_terminality_comes_from_counts_not_the_status_string(status, counts, expected):
    # processing_status is only in_progress/canceling/ended, so "ended" alone
    # cannot distinguish success from cancellation or expiry.
    got = ADAPTER.parse_status({"processing_status": status, "request_counts": counts})
    assert got.state == expected
    assert got.raw_state == status


@respx.mock
def test_results_refuse_clearly_while_results_url_is_still_null():
    respx.get(f"{BASE_URL}/msgbatch_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msgbatch_1",
                "processing_status": "in_progress",
                "results_url": None,
            },
        )
    )
    with pytest.raises(RuntimeError, match="results_url stays null"):
        list(ADAPTER.results(_handle(), api_key="k"))


@respx.mock
def test_results_decode_to_openai_shape_and_join_out_of_order():
    respx.get(f"{BASE_URL}/msgbatch_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msgbatch_1",
                "processing_status": "ended",
                "results_url": "https://api.anthropic.com/r/1",
            },
        )
    )
    # Deliberately reversed: the live API returns results out of submission
    # order, so anything positional is wrong.
    lines = [
        {
            "custom_id": "row-2",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "m2",
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": [{"type": "text", "text": "4"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        },
        {
            "custom_id": "row-1",
            "result": {"type": "errored", "error": {"type": "overloaded"}},
        },
    ]
    respx.get("https://api.anthropic.com/r/1").mock(
        return_value=httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))
    )

    got = {r.custom_id: r for r in ADAPTER.results(_handle(), api_key="k")}
    assert set(got) == {"row-1", "row-2"}
    assert got["row-2"].ok
    # Decoded out of Anthropic's native shape into OpenAI's.
    assert got["row-2"].response["choices"][0]["message"]["content"] == "4"
    assert not got["row-1"].ok
    assert got["row-1"].error["type"] == "errored"


@respx.mock
def test_cancel_reaches_the_provider():
    route = respx.post(f"{BASE_URL}/msgbatch_1/cancel").mock(
        return_value=httpx.Response(200, json={"id": "msgbatch_1"})
    )
    ADAPTER.cancel(_handle(), api_key="k")
    assert route.called


@respx.mock
def test_list_jobs_returns_usable_handles():
    # Needed for real: recovering a job id after the submitting process died.
    respx.get(BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "msgbatch_1", "processing_status": "ended"},
                    {"id": "msgbatch_2", "processing_status": "in_progress"},
                ]
            },
        )
    )
    handles = list(ADAPTER.list_jobs(api_key="k"))
    assert [h.job_id for h in handles] == ["msgbatch_1", "msgbatch_2"]
    assert all(h.provider == "anthropic" for h in handles)
    # A listed handle must be usable directly, not a stub.
    assert bl.BatchHandle.from_json(handles[0].to_json()) == handles[0]
