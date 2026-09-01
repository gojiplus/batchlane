"""Mistral's lane: model on the job, integer-hour timeout, its own path."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.base import KEY_FIELD
from batchlane.adapters.mistral import BASE_URL, JOBS_PATH, MistralAdapter

ADAPTER = MistralAdapter()
MODEL = "mistral-small-latest"


def _line(cid="r0", text="hi", model=MODEL):
    return bl.BatchLine(
        cid, model, [{"role": "user", "content": text}], {"max_tokens": 8}
    )


def _handle():
    return bl.BatchHandle(
        provider="mistral",
        job_id="job_1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
        model=MODEL,
    )


def _reply(text):
    return {
        "status_code": 200,
        "body": {"choices": [{"message": {"content": text}}]},
    }


def test_the_model_goes_on_the_job_not_only_the_line():
    # This is what makes a mixed-model batch inexpressible here.
    with respx.mock:
        respx.post(f"{BASE_URL}/files").mock(
            return_value=httpx.Response(200, json={"id": "f1"})
        )
        create = respx.post(f"{BASE_URL}{JOBS_PATH}").mock(
            return_value=httpx.Response(200, json={"id": "job_1", "status": "QUEUED"})
        )
        ADAPTER.submit([_line()], endpoint="chat.completions", window=None, api_key="k")
    body = json.loads(create.calls[0].request.content)
    assert body["model"] == MODEL
    assert body["input_files"] == ["f1"]
    assert body["endpoint"] == "/v1/chat/completions"


def test_a_mixed_model_batch_is_refused():
    with pytest.raises(bl.MixedModelBatchError):
        ADAPTER.check(
            [_line("r0"), _line("r1", model="mistral-large-latest")],
            endpoint="chat.completions",
            window=None,
        )


@pytest.mark.parametrize(
    ("window", "expected"), [(None, 24), ("24h", 24), ("72h", 72), ("6h", 6)]
)
def test_the_window_becomes_an_integer_hour_count(window, expected):
    # Mistral takes timeout_hours, not a window enum, so any Nh is valid.
    with respx.mock:
        respx.post(f"{BASE_URL}/files").mock(
            return_value=httpx.Response(200, json={"id": "f1"})
        )
        create = respx.post(f"{BASE_URL}{JOBS_PATH}").mock(
            return_value=httpx.Response(200, json={"id": "job_1", "status": "QUEUED"})
        )
        ADAPTER.submit(
            [_line()], endpoint="chat.completions", window=window, api_key="k"
        )
    assert json.loads(create.calls[0].request.content)["timeout_hours"] == expected


@respx.mock
def test_the_submission_key_is_stamped_in_metadata():
    respx.post(f"{BASE_URL}/files").mock(
        return_value=httpx.Response(200, json={"id": "f"})
    )
    create = respx.post(f"{BASE_URL}{JOBS_PATH}").mock(
        return_value=httpx.Response(200, json={"id": "job_1", "status": "QUEUED"})
    )
    ADAPTER.submit(
        [_line()], endpoint="chat.completions", window=None, api_key="k", key="bl-abc-0"
    )
    assert (
        json.loads(create.calls[0].request.content)["metadata"][KEY_FIELD] == "bl-abc-0"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("QUEUED", "pending"),
        ("RUNNING", "running"),
        ("CANCELLATION_REQUESTED", "running"),
        ("SUCCESS", "succeeded"),
        ("FAILED", "failed"),
        ("TIMEOUT_EXCEEDED", "expired"),
        ("CANCELLED", "cancelled"),
    ],
)
def test_status_mapping(raw, expected):
    got = ADAPTER.parse_status(
        {
            "status": raw,
            "total_requests": 3,
            "succeeded_requests": 1,
            "failed_requests": 0,
        }
    )
    assert got.state == expected
    assert got.raw_state == raw
    assert (got.total, got.succeeded) == (3, 1)


@respx.mock
def test_results_returned_out_of_order_still_join_to_the_right_row():
    # The failure mode Anthropic and Gemini both exhibit in real life.
    respx.get(f"{BASE_URL}{JOBS_PATH}/job_1").mock(
        return_value=httpx.Response(
            200, json={"id": "job_1", "status": "SUCCESS", "output_file": "out_1"}
        )
    )
    respx.get(f"{BASE_URL}/files/out_1/content").mock(
        return_value=httpx.Response(
            200,
            text="\n".join(
                json.dumps({"custom_id": cid, "response": _reply(text)})
                for cid, text in (("r2", "third"), ("r0", "first"), ("r1", "second"))
            ),
        )
    )
    got = {r.custom_id: r for r in ADAPTER.results(_handle(), api_key="k")}
    assert got["r0"].response["choices"][0]["message"]["content"] == "first"
    assert got["r2"].response["choices"][0]["message"]["content"] == "third"


@respx.mock
def test_an_error_file_row_is_reported_against_its_own_row():
    respx.get(f"{BASE_URL}{JOBS_PATH}/job_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "job_1",
                "status": "SUCCESS",
                "output_file": "out_1",
                "error_file": "err_1",
            },
        )
    )
    respx.get(f"{BASE_URL}/files/out_1/content").mock(
        return_value=httpx.Response(
            200, text=json.dumps({"custom_id": "r0", "response": _reply("ok")})
        )
    )
    respx.get(f"{BASE_URL}/files/err_1/content").mock(
        return_value=httpx.Response(
            200, text=json.dumps({"custom_id": "r1", "error": {"message": "too long"}})
        )
    )
    got = {r.custom_id: r for r in ADAPTER.results(_handle(), api_key="k")}
    assert got["r0"].ok
    assert not got["r1"].ok
    assert got["r1"].error["message"] == "too long"


@respx.mock
def test_cancel_and_list_use_mistrals_own_paths():
    cancel = respx.post(f"{BASE_URL}{JOBS_PATH}/job_1/cancel").mock(
        return_value=httpx.Response(200, json={})
    )
    ADAPTER.cancel(_handle(), api_key="k")
    assert cancel.called

    respx.get(f"{BASE_URL}{JOBS_PATH}").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "job_1", "model": MODEL, "metadata": {KEY_FIELD: "bl-x-0"}}
                ]
            },
        )
    )
    handles = list(ADAPTER.list_jobs(api_key="k"))
    assert handles[0].job_id == "job_1"
    # The stamped key must survive listing or crash recovery cannot find it.
    assert handles[0].extra[KEY_FIELD] == "bl-x-0"
