"""The one-line entry point: a model, some prompts, answers back in order."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.openai_shaped import ROWS

GROQ = ROWS["groq"]
MODEL = "groq/llama-3.3-70b-versatile"


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def _mock(answers, job="b1"):
    """Mock a finished batch returning `answers`, a custom_id -> text mapping."""
    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f1"})
    )
    submit = respx.post(f"{GROQ.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": job, "status": "validating"})
    )
    respx.get(f"{GROQ.base_url}/batches/{job}").mock(
        return_value=httpx.Response(
            200, json={"id": job, "status": "completed", "output_file_id": "o1"}
        )
    )
    body = "\n".join(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": text}}]},
                },
            }
        )
        for cid, text in answers.items()
    )
    respx.get(f"{GROQ.base_url}/files/o1/content").mock(
        return_value=httpx.Response(200, text=body)
    )
    return submit


@respx.mock
def test_answers_come_back_in_input_order(groq_key):
    # Deliberately answered out of order: the provider does this in real life,
    # so map() must realign rather than trust arrival order.
    _mock({"row-2": "third", "row-0": "first", "row-1": "second"})
    assert bl.map(MODEL, ["a", "b", "c"], poll_interval=0) == [
        "first",
        "second",
        "third",
    ]


@respx.mock
def test_a_failed_row_is_none_and_does_not_shift_the_others(groq_key):
    # The list must stay aligned with the input, so a hole is None rather than
    # a missing element that silently renumbers everything after it.
    _mock({"row-0": "first", "row-2": "third"})
    assert bl.map(MODEL, ["a", "b", "c"], poll_interval=0) == ["first", None, "third"]


@respx.mock
def test_system_prompt_is_applied_to_every_row(groq_key):
    submit = _mock({"row-0": "x", "row-1": "y"})
    bl.map(MODEL, ["a", "b"], system="Be terse.", poll_interval=0)
    uploaded = respx.calls[0].request.content.decode(errors="ignore")
    assert uploaded.count('"Be terse."') == 2
    assert submit.called


@respx.mock
def test_model_params_reach_the_request(groq_key):
    _mock({"row-0": "x"})
    bl.map(MODEL, ["a"], max_tokens=7, poll_interval=0)
    uploaded = respx.calls[0].request.content.decode(errors="ignore")
    assert '"max_tokens": 7' in uploaded


@respx.mock
def test_an_empty_prompt_list_is_refused_rather_than_silently_returning_nothing(
    groq_key,
):
    with pytest.raises(bl.BatchlaneError, match="empty"):
        bl.map(MODEL, [])


def test_answer_text_handles_every_shape_it_can_receive():
    ok = bl.RequestResult("a", response={"choices": [{"message": {"content": "hi"}}]})
    assert bl.answer_text(ok) == "hi"
    assert bl.answer_text(bl.RequestResult("a", error={"m": "boom"})) is None
    assert bl.answer_text(bl.RequestResult("a")) is None
    # A well-formed response that simply carries no choices.
    assert bl.answer_text(bl.RequestResult("a", response={"choices": []})) is None
