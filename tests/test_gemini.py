"""Gemini AI Studio's lane, and the result-joining hazard it carries."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.gemini import BASE_URL, GeminiAdapter

ADAPTER = GeminiAdapter()
MODEL = "gemini-2.5-flash"


def _line(cid, text):
    return bl.BatchLine(
        cid, MODEL, [{"role": "user", "content": text}], {"max_tokens": 8}
    )


def _handle(keys=("r1", "r2")):
    return bl.BatchHandle(
        provider="gemini",
        job_id="batches/abc",
        endpoint="chat.completions",
        lane="batch_inline",
        created_at=bl.handle.utcnow(),
        model=MODEL,
        extra={"keys": json.dumps(list(keys))},
    )


def _reply(text):
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6,
        },
    }


def _job(inlined, state="JOB_STATE_SUCCEEDED"):
    return {
        "name": "batches/abc",
        "metadata": {"state": state},
        "response": {"inlinedResponses": inlined},
    }


def test_requests_are_inline_and_carry_the_key_we_supplied():
    body = ADAPTER.build_batch([_line("r1", "hi"), _line("r2", "yo")], display_name="d")
    reqs = body["batch"]["input_config"]["requests"]["requests"]
    assert [r["metadata"]["key"] for r in reqs] == ["r1", "r2"]
    # Gemini's own schema, not OpenAI chat.
    assert "contents" in reqs[0]["request"]
    assert "messages" not in reqs[0]["request"]


@respx.mock
def test_submit_puts_the_model_in_the_url_and_records_submission_order():
    route = respx.post(f"{BASE_URL}/models/{MODEL}:batchGenerateContent").mock(
        return_value=httpx.Response(200, json={"name": "batches/abc"})
    )
    handle = ADAPTER.submit(
        [_line("r1", "hi"), _line("r2", "yo")],
        endpoint="chat.completions",
        window=None,
        api_key="k",
    )
    assert route.called
    assert route.calls[0].request.headers["x-goog-api-key"] == "k"
    assert handle.job_id == "batches/abc"
    # Needed to label results if the provider echoes no key.
    assert json.loads(handle.extra["keys"]) == ["r1", "r2"]


def test_a_window_is_refused_because_gemini_accepts_none():
    with pytest.raises(bl.CapabilityNotSupportedError, match="completion_window"):
        ADAPTER.check([_line("r1", "hi")], endpoint="chat.completions", window="24h")


def test_mixed_model_batch_refused_because_the_model_is_in_the_url():
    lines = [_line("r1", "hi"), bl.BatchLine("r2", "gemini-2.5-pro", [], {})]
    with pytest.raises(bl.MixedModelBatchError):
        ADAPTER.check(lines, endpoint="chat.completions", window=None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("JOB_STATE_PENDING", "pending"),
        ("JOB_STATE_RUNNING", "running"),
        ("JOB_STATE_SUCCEEDED", "succeeded"),
        ("JOB_STATE_FAILED", "failed"),
        ("JOB_STATE_CANCELLED", "cancelled"),
        ("JOB_STATE_EXPIRED", "expired"),
    ],
)
def test_state_is_read_from_the_nested_metadata_field(raw, expected):
    got = ADAPTER.parse_status(
        {"metadata": {"state": raw, "batchStats": {"requestCount": 2}}}
    )
    assert got.state == expected
    assert got.raw_state == raw
    assert got.total == 2


# --- the hazard: Gemini documents index-based matching for inline results ---


@respx.mock
def test_an_echoed_key_wins_over_position_even_when_order_is_reversed():
    # If the provider echoes the key, ordering cannot hurt us.
    respx.get(f"{BASE_URL}/batches/abc").mock(
        return_value=httpx.Response(
            200,
            json=_job(
                [
                    {"metadata": {"key": "r2"}, "response": _reply("second")},
                    {"metadata": {"key": "r1"}, "response": _reply("first")},
                ]
            ),
        )
    )
    got = {r.custom_id: r for r in ADAPTER.results(_handle(), api_key="k")}
    assert got["r1"].response["choices"][0]["message"]["content"] == "first"
    assert got["r2"].response["choices"][0]["message"]["content"] == "second"


@respx.mock
def test_without_an_echoed_key_results_fall_back_to_submission_order():
    respx.get(f"{BASE_URL}/batches/abc").mock(
        return_value=httpx.Response(
            200,
            json=_job([{"response": _reply("first")}, {"response": _reply("second")}]),
        )
    )
    got = list(ADAPTER.results(_handle(), api_key="k"))
    assert [r.custom_id for r in got] == ["r1", "r2"]
    assert got[0].response["choices"][0]["message"]["content"] == "first"


@respx.mock
def test_a_count_mismatch_with_no_echoed_key_refuses_rather_than_mislabels():
    # The worst failure this package could produce: plausible answers attached
    # to the wrong rows, no error, nothing to notice. Refuse instead.
    respx.get(f"{BASE_URL}/batches/abc").mock(
        return_value=httpx.Response(200, json=_job([{"response": _reply("only one")}]))
    )
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        list(ADAPTER.results(_handle(keys=("r1", "r2", "r3")), api_key="k"))


@respx.mock
def test_a_per_row_error_is_reported_against_its_own_row():
    respx.get(f"{BASE_URL}/batches/abc").mock(
        return_value=httpx.Response(
            200,
            json=_job(
                [
                    {
                        "metadata": {"key": "r1"},
                        "error": {"code": 429, "message": "quota"},
                    },
                    {"metadata": {"key": "r2"}, "response": _reply("ok")},
                ]
            ),
        )
    )
    got = {r.custom_id: r for r in ADAPTER.results(_handle(), api_key="k")}
    assert not got["r1"].ok
    assert got["r1"].error["code"] == 429
    assert got["r2"].ok


@respx.mock
def test_results_refuse_clearly_before_the_job_finishes():
    respx.get(f"{BASE_URL}/batches/abc").mock(
        return_value=httpx.Response(
            200,
            json={"name": "batches/abc", "metadata": {"state": "JOB_STATE_RUNNING"}},
        )
    )
    with pytest.raises(RuntimeError, match="no inline results yet"):
        list(ADAPTER.results(_handle(), api_key="k"))


@respx.mock
def test_cancel_uses_the_rpc_colon_suffix_not_a_subpath():
    route = respx.post(f"{BASE_URL}/batches/abc:cancel").mock(
        return_value=httpx.Response(200, json={})
    )
    ADAPTER.cancel(_handle(), api_key="k")
    assert route.called
