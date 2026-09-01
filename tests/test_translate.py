"""Golden-output pinning for the LiteLLM seam.

batchlane depends on LiteLLM internals -- ProviderConfigManager and the
per-provider transform_request -- which are not published API. These tests
diff their real output against committed fixtures, so a LiteLLM version bump
that reshapes a request body fails here rather than silently emitting wrong
JSONL into a 50,000-row job.

To refresh after a deliberate upgrade: BATCHLANE_REGEN=1 pytest tests/test_translate.py
"""

import json
import os
from pathlib import Path

import pytest

from batchlane.translate import decode_response, encode_body, resolve

FIXTURES = Path(__file__).parent / "fixtures"

MESSAGES = [
    {"role": "system", "content": "You classify text."},
    {"role": "user", "content": "The product exceeded my expectations."},
]
PARAMS = {"temperature": 0.2, "max_tokens": 64}

CASES = [
    ("anthropic", "anthropic/claude-haiku-4-5-20251001"),
    ("groq", "groq/llama-3.3-70b-versatile"),
    ("together_ai", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("deepinfra", "deepinfra/deepseek-ai/DeepSeek-V3"),
    ("mistral", "mistral/mistral-small-latest"),
    ("gemini", "gemini/gemini-2.5-flash"),
]


@pytest.mark.parametrize(("provider", "spec"), CASES, ids=[c[0] for c in CASES])
def test_request_body_matches_committed_fixture(provider, spec):
    resolved_provider, bare, _base = resolve(spec)
    assert resolved_provider == provider

    body = encode_body(resolved_provider, bare, MESSAGES, dict(PARAMS))
    path = FIXTURES / f"request_{provider}.json"

    if os.environ.get("BATCHLANE_REGEN"):
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")

    expected = json.loads(path.read_text())
    assert body == expected, (
        f"litellm's request body for {provider} changed. If the upgrade was "
        f"deliberate, re-run with BATCHLANE_REGEN=1 and review the diff."
    )


def test_openai_shaped_bodies_carry_the_model_on_the_line():
    # This is what makes a mixed-model batch expressible on these providers.
    for provider, spec in [
        c for c in CASES if c[0] in {"groq", "together_ai", "deepinfra"}
    ]:
        _p, bare, _b = resolve(spec)
        assert encode_body(provider, bare, MESSAGES, dict(PARAMS))["model"] == bare


def test_empty_extra_body_is_stripped():
    # litellm injects extra_body={} on OpenAI-shaped providers. Harmless live,
    # but some batch validators reject unknown keys and it is noise in a file
    # a human may have to read.
    body = encode_body("groq", "llama-3.3-70b-versatile", MESSAGES, dict(PARAMS))
    assert "extra_body" not in body


def test_gemini_uses_its_own_schema_not_openai_chat():
    body = encode_body("gemini", "gemini-2.5-flash", MESSAGES, dict(PARAMS))
    assert "contents" in body
    assert "messages" not in body


def test_gemini_response_decodes_back_to_openai_shape():
    decoded = decode_response(
        "gemini",
        "gemini-2.5-flash",
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "positive"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 1,
                "totalTokenCount": 13,
            },
        },
    )
    assert decoded.choices[0].message.content == "positive"
    assert decoded.choices[0].finish_reason == "stop"
    assert decoded.usage.total_tokens == 13


def test_anthropic_hoists_system_and_is_not_openai_chat_shaped():
    body = encode_body(
        "anthropic", "claude-haiku-4-5-20251001", MESSAGES, {"max_tokens": 8}
    )
    assert "system" in body
    assert all(m["role"] != "system" for m in body["messages"])


def test_decode_refuses_for_providers_that_need_no_decoding():
    with pytest.raises(Exception, match="no decode is needed"):
        decode_response("groq", "llama-3.3-70b-versatile", {})


def test_encode_body_does_not_mutate_the_caller_s_messages():
    # litellm's AnthropicConfig.transform_request pops the system message out of
    # the list it is given. Callers reuse one list across lines all the time, so
    # without a defensive copy every line after the first loses its system
    # prompt -- silently, and wrongly, across a whole job.
    shared = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ]
    first = encode_body(
        "anthropic", "claude-haiku-4-5-20251001", shared, {"max_tokens": 8}
    )
    second = encode_body(
        "anthropic", "claude-haiku-4-5-20251001", shared, {"max_tokens": 8}
    )
    assert [m["role"] for m in shared] == ["system", "user"]
    assert "system" in first
    assert "system" in second
