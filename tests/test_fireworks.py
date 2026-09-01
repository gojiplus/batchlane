"""Fireworks' lane: account-scoped, dataset-based, and with no cancel."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.base import KEY_FIELD
from batchlane.adapters.fireworks import ACCOUNT_ENV, BASE_URL, FireworksAdapter

ADAPTER = FireworksAdapter()
MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
ROOT = f"{BASE_URL}/accounts/test-account"


@pytest.fixture(autouse=True)
def account(monkeypatch):
    monkeypatch.setenv(ACCOUNT_ENV, "test-account")


def _line(cid="r0", text="hi"):
    return bl.BatchLine(
        cid, MODEL, [{"role": "user", "content": text}], {"max_tokens": 8}
    )


def _handle(job="bl-abc-0"):
    return bl.BatchHandle(
        provider="fireworks_ai",
        job_id=job,
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
        model=MODEL,
        extra={"output_dataset": f"{job}-out", KEY_FIELD: job},
    )


def test_a_missing_account_names_the_variable_to_set(monkeypatch):
    # Every Fireworks URL is account-scoped and the docs never say where to
    # find the id, so the error has to.
    monkeypatch.delenv(ACCOUNT_ENV, raising=False)
    with pytest.raises(bl.BatchlaneError, match=ACCOUNT_ENV):
        ADAPTER.submit([_line()], endpoint="chat.completions", window=None, api_key="k")


@respx.mock
def test_submit_registers_two_datasets_uploads_then_starts_the_job():
    datasets = respx.post(f"{ROOT}/datasets").mock(
        return_value=httpx.Response(200, json={})
    )
    upload = respx.post(url__regex=rf"{ROOT}/datasets/.*:upload").mock(
        return_value=httpx.Response(200, json={})
    )
    create = respx.post(f"{ROOT}/batchInferenceJobs").mock(
        return_value=httpx.Response(
            200, json={"name": "accounts/test-account/batchInferenceJobs/bl-abc-0"}
        )
    )
    handle = ADAPTER.submit(
        [_line()], endpoint="chat.completions", window=None, api_key="k", key="bl-abc-0"
    )
    # Input and output datasets both have to exist before the job starts.
    assert datasets.call_count == 2
    assert upload.called
    body = json.loads(create.calls[0].request.content)
    assert body["inputDatasetId"] == "accounts/test-account/datasets/bl-abc-0-in"
    assert body["outputDatasetId"] == "accounts/test-account/datasets/bl-abc-0-out"
    assert handle.extra["output_dataset"] == "bl-abc-0-out"


@respx.mock
def test_the_submission_key_becomes_the_job_id_not_a_label():
    # Fireworks lets the client set the job id, so resubmitting the same work
    # collides server-side. That is closer to real idempotency than any other
    # lane offers.
    respx.post(f"{ROOT}/datasets").mock(return_value=httpx.Response(200, json={}))
    respx.post(url__regex=rf"{ROOT}/datasets/.*:upload").mock(
        return_value=httpx.Response(200, json={})
    )
    create = respx.post(f"{ROOT}/batchInferenceJobs").mock(
        return_value=httpx.Response(200, json={"name": "a/b/bl-abc-0"})
    )
    ADAPTER.submit(
        [_line()], endpoint="chat.completions", window=None, api_key="k", key="bl-abc-0"
    )
    assert "batchInferenceJobId=bl-abc-0" in str(create.calls[0].request.url)


@respx.mock
def test_no_model_on_the_line_because_the_job_owns_it():
    respx.post(f"{ROOT}/datasets").mock(return_value=httpx.Response(200, json={}))
    upload = respx.post(url__regex=rf"{ROOT}/datasets/.*:upload").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.post(f"{ROOT}/batchInferenceJobs").mock(
        return_value=httpx.Response(200, json={"name": "a/b/j"})
    )
    ADAPTER.submit([_line()], endpoint="chat.completions", window=None, api_key="k")
    sent = upload.calls[0].request.content.decode(errors="ignore")
    assert '"custom_id"' in sent
    assert '"model"' not in sent


def test_cancel_refuses_because_fireworks_documents_no_endpoint():
    # The descriptor says supports_cancel=False, and the ABC's refusing
    # default must stand rather than silently doing nothing.
    assert ADAPTER.capabilities.supports_cancel is False
    with pytest.raises(bl.CapabilityNotSupportedError, match="cancel"):
        ADAPTER.cancel(_handle(), api_key="k")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("JOB_STATE_PENDING", "pending"),
        ("JOB_STATE_RUNNING", "running"),
        ("JOB_STATE_WRITING_RESULTS", "running"),
        ("JOB_STATE_COMPLETED", "succeeded"),
        ("JOB_STATE_FAILED", "failed"),
        ("JOB_STATE_EXPIRED", "expired"),
        ("COMPLETED", "succeeded"),
    ],
)
def test_status_mapping_covers_both_documented_spellings(raw, expected):
    got = ADAPTER.parse_status(
        {"state": raw, "jobProgress": {"totalInputRequests": 4, "failedRequests": 1}}
    )
    assert got.state == expected
    assert got.raw_state == raw
    assert got.total == 4


@pytest.mark.parametrize(
    ("window", "expected"), [("24h", "24h"), ("2h", "12h"), ("500h", "72h")]
)
@respx.mock
def test_the_window_is_clamped_to_the_documented_range(window, expected):
    respx.post(f"{ROOT}/datasets").mock(return_value=httpx.Response(200, json={}))
    respx.post(url__regex=rf"{ROOT}/datasets/.*:upload").mock(
        return_value=httpx.Response(200, json={})
    )
    create = respx.post(f"{ROOT}/batchInferenceJobs").mock(
        return_value=httpx.Response(200, json={"name": "a/b/j"})
    )
    ADAPTER.submit([_line()], endpoint="chat.completions", window=window, api_key="k")
    assert json.loads(create.calls[0].request.content)["maxJobDuration"] == expected


@respx.mock
def test_results_come_through_the_signed_url_indirection():
    respx.get(f"{ROOT}/datasets/bl-abc-0-out:getDownloadEndpoint").mock(
        return_value=httpx.Response(
            200,
            json={
                "filenameToSignedUrls": {
                    "results.jsonl": "https://signed-url.invalid/results.jsonl"
                }
            },
        )
    )
    respx.get("https://signed-url.invalid/results.jsonl").mock(
        return_value=httpx.Response(
            200,
            text="\n".join(
                json.dumps(
                    {
                        "custom_id": cid,
                        "response": {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"content": t}}]},
                        },
                    }
                )
                # Out of submission order, as providers do.
                for cid, t in (("r1", "second"), ("r0", "first"))
            ),
        )
    )
    got = {r.custom_id: r for r in ADAPTER.results(_handle(), api_key="k")}
    assert got["r0"].response["choices"][0]["message"]["content"] == "first"
    assert got["r1"].response["choices"][0]["message"]["content"] == "second"


@respx.mock
def test_results_refuse_clearly_before_the_output_dataset_is_written():
    respx.get(f"{ROOT}/datasets/bl-abc-0-out:getDownloadEndpoint").mock(
        return_value=httpx.Response(200, json={"filenameToSignedUrls": {}})
    )
    with pytest.raises(RuntimeError, match="no files yet"):
        list(ADAPTER.results(_handle(), api_key="k"))


@respx.mock
def test_recovery_is_an_exact_id_lookup_not_a_scan():
    respx.get(f"{ROOT}/batchInferenceJobs/bl-abc-0").mock(
        return_value=httpx.Response(200, json={"name": "a/b/bl-abc-0", "model": MODEL})
    )
    found = ADAPTER.find_submitted(
        "bl-abc-0", api_key="k", expected_rows=1, since=bl.handle.utcnow()
    )
    assert found is not None
    assert found.job_id == "bl-abc-0"


@respx.mock
def test_recovery_returns_none_when_the_job_was_never_created():
    # A 404 means the submission never landed, so resubmitting is safe.
    respx.get(f"{ROOT}/batchInferenceJobs/bl-missing-0").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    assert (
        ADAPTER.find_submitted(
            "bl-missing-0", api_key="k", expected_rows=1, since=bl.handle.utcnow()
        )
        is None
    )
