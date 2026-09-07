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


@respx.mock
def test_large_batch_uploads_jsonl_then_collects_keyed_file_results(monkeypatch):
    from batchlane.adapters import gemini

    monkeypatch.setattr(gemini, "INLINE_LIMIT", 1)
    start = respx.post(gemini.UPLOAD_URL).mock(
        return_value=httpx.Response(
            200, headers={"x-goog-upload-url": "https://upload.invalid/session"}
        )
    )
    upload = respx.post("https://upload.invalid/session").mock(
        return_value=httpx.Response(200, json={"file": {"name": "files/input"}})
    )
    submit = respx.post(f"{BASE_URL}/models/{MODEL}:batchGenerateContent").mock(
        return_value=httpx.Response(200, json={"name": "batches/file-job"})
    )
    handle = ADAPTER.submit(
        [_line("r1", "hi"), _line("r2", "yo")],
        endpoint="chat.completions",
        window=None,
        api_key="k",
    )
    assert handle.lane == "batch_file"
    records = [
        json.loads(line) for line in upload.calls[0].request.content.splitlines()
    ]
    assert [r["key"] for r in records] == ["r1", "r2"]
    assert all("contents" in r["request"] for r in records)
    assert int(
        start.calls[0].request.headers["x-goog-upload-header-content-length"]
    ) == len(upload.calls[0].request.content)
    assert "x-goog-api-key" not in upload.calls[0].request.headers
    assert json.loads(submit.calls[0].request.content)["batch"]["input_config"] == {
        "file_name": "files/input"
    }
    respx.get(f"{BASE_URL}/batches/file-job").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": {"responsesFile": "files/output"},
                "metadata": {"state": "JOB_STATE_SUCCEEDED"},
            },
        )
    )
    respx.get(
        f"{gemini.DOWNLOAD_URL}/files/output:download", params={"alt": "media"}
    ).mock(
        return_value=httpx.Response(
            200,
            text="\n".join(
                [
                    json.dumps({"key": "r2", "error": {"code": 429}}),
                    json.dumps({"key": "r1", "response": _reply("first")}),
                ]
            ),
        )
    )
    got = list(ADAPTER.results(bl.BatchHandle.from_json(handle.to_json()), api_key="k"))
    assert [r.custom_id for r in got] == ["r2", "r1"]
    assert not got[0].ok
    assert bl.answer_text(got[1]) == "first"


@respx.mock
@pytest.mark.parametrize(
    "rows",
    [[{"response": {}}], [{"key": "r1", "error": {}}, {"key": "r1", "error": {}}]],
)
def test_file_results_refuse_missing_or_duplicate_keys(rows):
    from batchlane.adapters.gemini import DOWNLOAD_URL

    respx.get(f"{BASE_URL}/batches/abc").mock(
        return_value=httpx.Response(
            200, json={"response": {"responsesFile": "files/output"}}
        )
    )
    respx.get(f"{DOWNLOAD_URL}/files/output:download", params={"alt": "media"}).mock(
        return_value=httpx.Response(200, text="\n".join(json.dumps(r) for r in rows))
    )
    with pytest.raises(RuntimeError, match="missing or duplicate key"):
        list(ADAPTER.results(_handle(), api_key="k"))


@respx.mock
@pytest.mark.parametrize("file_input", [False, True])
def test_crash_recovery_finds_nested_label_and_restores_result_identity(
    tmp_path, monkeypatch, file_input
):
    from batchlane import runner

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    saved = []

    def submit(request):
        saved.append(json.loads(request.content)["batch"]["display_name"])
        return httpx.Response(200, json={"name": "batches/paid"})

    submits = respx.post(f"{BASE_URL}/models/{MODEL}:batchGenerateContent").mock(
        side_effect=submit
    )

    def listing(request):
        meta = {"displayName": saved[0], "model": f"models/{MODEL}"}
        if file_input:
            meta["inputConfig"] = {"fileName": "files/input"}
        return httpx.Response(
            200, json={"operations": [{"name": "batches/paid", "metadata": meta}]}
        )

    respx.get(f"{BASE_URL}/batches").mock(side_effect=listing)
    real_append = runner._append_checkpoint

    def crash(*args):
        raise RuntimeError("stopped before receipt")

    monkeypatch.setattr(runner, "_append_checkpoint", crash)
    lines = [bl.BatchLine("r1", f"gemini/{MODEL}", [{"role": "user", "content": "hi"}])]
    checkpoint = tmp_path / "job.jsonl"
    with pytest.raises(RuntimeError, match="stopped"):
        bl.submit_all(lines, checkpoint=checkpoint)
    monkeypatch.setattr(runner, "_append_checkpoint", real_append)
    handles = bl.submit_all(lines, checkpoint=checkpoint)
    assert submits.call_count == 1
    assert handles[0].model == MODEL
    assert handles[0].lane == ("batch_file" if file_input else "batch_inline")
    assert json.loads(handles[0].extra["keys"]) == ["r1"]
    respx.get(f"{BASE_URL}/batches/paid").mock(
        return_value=httpx.Response(200, json=_job([{"response": _reply("first")}]))
    )
    got = list(bl.results(handles[0]))
    assert [(r.custom_id, bl.answer_text(r)) for r in got] == [("r1", "first")]
