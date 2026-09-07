"""Native Responses payloads survive submission, recovery, and collection."""

import json
from dataclasses import replace

import pytest
import respx

import batchlane as bl
from batchlane.adapters.openai_shaped import ROWS, OpenAIShapedAdapter

BASE = ROWS["openai"].base_url
INPUT = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Is this legible?"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,YQ==",
                "detail": "high",
            },
        ],
    }
]
LINE = bl.BatchLine(
    "image-1",
    "openai/gpt-4o-mini",
    input=INPUT,
    params={"max_output_tokens": 80, "reasoning": {"effort": "low"}},
)
BODY = {
    "object": "response",
    "model": "gpt-4o-mini",
    "status": "completed",
    "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "yes"}]},
        {"type": "function_call", "name": "audit", "arguments": "{}"},
    ],
    "usage": {
        "input_tokens": 12,
        "output_tokens": 8,
        "output_tokens_details": {"reasoning_tokens": 3},
    },
}


@respx.mock
def test_responses_submit_resume_and_collect_preserve_native_bodies(tmp_path):
    upload = respx.post(f"{BASE}/files").respond(200, json={"id": "file-1"})
    create = respx.post(f"{BASE}/batches").respond(200, json={"id": "batch-1"})
    respx.get(f"{BASE}/batches/batch-1").respond(
        200, json={"id": "batch-1", "status": "completed", "output_file_id": "out-1"}
    )
    respx.get(f"{BASE}/files/out-1/content").respond(
        200,
        text=json.dumps(
            {
                "custom_id": LINE.custom_id,
                "response": {"status_code": 200, "body": BODY},
            }
        ),
    )
    checkpoint = tmp_path / "run.jsonl"
    handles = bl.submit_all(
        [LINE], endpoint="responses", checkpoint=checkpoint, api_key="k"
    )
    assert handles[0].endpoint == "responses"
    uploaded = upload.calls[0].request.content.decode()
    assert '"input_image"' in uploaded
    assert '"max_output_tokens": 80' in uploaded
    assert '"messages"' not in uploaded
    assert json.loads(create.calls[0].request.content)["endpoint"] == "/v1/responses"
    pairs = list(
        bl.run([LINE], endpoint="responses", checkpoint=checkpoint, api_key="k")
    )
    assert create.call_count == 1
    assert pairs[0][0].input == INPUT
    result = pairs[0][1]
    assert result.response == BODY
    assert bl.answer_text(result) == "yes"
    assert bl.actual_cost([result], "openai").input_tokens == 12
    with pytest.raises(bl.BatchlaneError, match=r"different|match|changed"):
        bl.submit_all(
            [replace(LINE, input="changed")],
            endpoint="responses",
            checkpoint=checkpoint,
            api_key="k",
        )
    assert create.call_count == 1


@pytest.mark.parametrize("provider", ["groq", "anthropic", "gemini"])
def test_other_lanes_refuse_responses_before_io(provider):
    with respx.mock as routes, pytest.raises(bl.CapabilityNotSupportedError):
        bl.plan([replace(LINE, model=f"{provider}/some-model")], endpoint="responses")
    assert not routes.calls


@pytest.mark.parametrize(
    "line",
    [
        replace(LINE, input=None),
        replace(LINE, messages=[{"role": "user", "content": "ambiguous"}]),
        replace(LINE, params={"input": "override"}),
        replace(LINE, params={"stream": True}),
        replace(LINE, params={"background": True}),
    ],
)
def test_invalid_responses_payload_refused_before_upload(line):
    with respx.mock as routes, pytest.raises(bl.BatchlaneError):
        bl.submit([line], endpoint="responses", api_key="k")
    assert not routes.calls


def test_native_input_cannot_be_silently_dropped_by_chat_lane():
    with pytest.raises(bl.CapabilityNotSupportedError, match="input"):
        bl.plan([LINE])


def test_response_text_does_not_treat_refusal_or_tool_call_as_answer():
    result = bl.RequestResult(
        "r",
        response={
            "output": [
                {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]},
                {"type": "function_call", "arguments": "{}"},
            ]
        },
    )
    assert bl.answer_text(result) is None
    assert result.response["output"][0]["content"][0]["refusal"] == "no"


def test_responses_cost_is_explicitly_unknown_before_collection():
    cost = bl.plan([LINE], endpoint="responses").cost
    assert cost.input_tokens is None
    assert cost.batch_usd is None
    assert "unknown" in str(cost)


def test_native_string_input_and_parameters_round_trip():
    line = bl.BatchLine("text", "gpt-4o-mini", input="hello", params={"store": False})
    body = json.loads(
        OpenAIShapedAdapter(ROWS["openai"]).build_jsonl([line], endpoint="responses")
    )["body"]
    assert body == {"model": "gpt-4o-mini", "input": "hello", "store": False}
