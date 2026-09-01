"""The top-level functions, which is what a user actually touches."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.openai_shaped import ROWS
from batchlane.registry import resolve_api_key

GROQ = ROWS["groq"]
MODEL = "groq/llama-3.3-70b-versatile"


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def _line(cid="q1"):
    return bl.BatchLine(cid, MODEL, [{"role": "user", "content": "2+2?"}])


@respx.mock
def test_submit_strips_the_provider_prefix_before_it_reaches_the_wire(groq_key):
    upload = respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f1"})
    )
    respx.post(f"{GROQ.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "validating"})
    )

    handle = bl.submit([_line()])

    assert handle.provider == "groq"
    sent = upload.calls[0].request.content.decode(errors="ignore")
    # The "groq/" prefix is batchlane's routing syntax, not Groq's model name.
    assert '"model": "llama-3.3-70b-versatile"' in sent
    assert "groq/llama" not in sent


@respx.mock
def test_status_and_results_round_trip_through_the_public_api(groq_key):
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "b1",
                "status": "completed",
                "output_file_id": "o1",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            },
        )
    )
    respx.get(f"{GROQ.base_url}/files/o1/content").mock(
        return_value=httpx.Response(
            200,
            text=json.dumps(
                {
                    "custom_id": "q1",
                    "response": {"status_code": 200, "body": {"answer": 4}},
                }
            ),
        )
    )
    handle = bl.BatchHandle(
        provider="groq",
        job_id="b1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )

    status = bl.status(handle)
    assert status.state == "succeeded"
    assert status.is_terminal

    out = list(bl.results(handle))
    assert [r.custom_id for r in out] == ["q1"]
    assert out[0].response == {"answer": 4}


@respx.mock
def test_cancel_reaches_the_provider(groq_key):
    route = respx.post(f"{GROQ.base_url}/batches/b1/cancel").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "cancelling"})
    )
    handle = bl.BatchHandle(
        provider="groq",
        job_id="b1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )
    bl.cancel(handle)
    assert route.called


def test_missing_credential_names_the_env_var_to_set(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(bl.BatchlaneError, match="GROQ_API_KEY"):
        resolve_api_key("groq")


def test_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert resolve_api_key("groq", "explicit") == "explicit"


def test_together_accepts_either_documented_env_var(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("TOGETHERAI_API_KEY", "alt")
    assert resolve_api_key("together_ai") == "alt"
