"""Live provider tests.

The check that can actually fail: run the same prompts through the synchronous
endpoint and through the batch lane, then compare. Schema validity and a full
result count are both compatible with an adapter that mis-maps custom_ids or
silently returns another model's output; only comparing against an independent
route catches that.

Gated on BATCHLANE_LIVE=1 plus the provider's key, so it never runs by accident.
"""

import os
import time

import pytest

import batchlane as bl

pytestmark = pytest.mark.live

PROMPTS = {
    "cap-france": "What is the capital of France? Answer with one word.",
    "two-plus-two": "What is 2+2? Answer with one digit.",
}

PROVIDERS = [
    ("gemini", "gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
    ("openai", "openai/gpt-4o-mini", "OPENAI_API_KEY"),
    ("anthropic", "anthropic/claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
    ("groq", "groq/llama-3.3-70b-versatile", "GROQ_API_KEY"),
    (
        "together_ai",
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "TOGETHER_API_KEY",
    ),
    ("deepinfra", "deepinfra/deepseek-ai/DeepSeek-V3", "DEEPINFRA_TOKEN"),
]

POLL_SECONDS = 20
MAX_WAIT_SECONDS = 60 * 45


def _skip_unless_live(env_var: str) -> None:
    if os.environ.get("BATCHLANE_LIVE") != "1":
        pytest.skip("set BATCHLANE_LIVE=1 to run live provider tests")
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} is not set")


@pytest.mark.parametrize(
    ("provider", "model", "env_var"), PROVIDERS, ids=[p[0] for p in PROVIDERS]
)
def test_batch_lane_agrees_with_the_sync_endpoint(provider, model, env_var):
    _skip_unless_live(env_var)
    import litellm

    lines = [
        bl.BatchLine(cid, model, [{"role": "user", "content": text}], {"max_tokens": 8})
        for cid, text in PROMPTS.items()
    ]

    handle = bl.submit(lines, window=None)
    assert handle.provider == provider

    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        status = bl.status(handle)
        if status.is_terminal:
            break
        time.sleep(POLL_SECONDS)
    else:
        pytest.fail(f"{provider} batch did not finish within {MAX_WAIT_SECONDS}s")

    assert status.state == "succeeded", f"{provider}: {status.raw_state} {status.error}"

    got = {r.custom_id: r for r in bl.results(handle)}
    assert set(got) == set(PROMPTS), "custom_ids did not round-trip"
    assert all(r.ok for r in got.values())

    for cid, text in PROMPTS.items():
        batch_text = got[cid].response["choices"][0]["message"]["content"]
        sync_text = (
            litellm.completion(
                model=model, messages=[{"role": "user", "content": text}], max_tokens=8
            )
            .choices[0]
            .message.content
        )
        assert batch_text.strip(), f"{provider}/{cid}: batch returned empty content"
        tier = (got[cid].response.get("usage") or {}).get("service_tier")
        if tier is not None:
            # Proof the discount lane actually applied, not just that a
            # job-shaped API answered.
            assert tier == "batch", f"{provider}/{cid}: served at tier {tier!r}"
        # Not equality -- models are not deterministic. But a mis-joined id or a
        # wrong-model response will not share a token with the sync answer.
        assert _overlaps(batch_text, sync_text), (
            f"{provider}/{cid}: batch said {batch_text!r}, sync said {sync_text!r}"
        )


def _overlaps(a: str, b: str) -> bool:
    tokens_a = {t.strip(".,!?").lower() for t in a.split() if t.strip(".,!?")}
    tokens_b = {t.strip(".,!?").lower() for t in b.split() if t.strip(".,!?")}
    return bool(tokens_a & tokens_b)


@pytest.mark.parametrize(
    ("provider", "model", "env_var"), PROVIDERS, ids=[p[0] for p in PROVIDERS]
)
def test_submitted_job_can_be_cancelled(provider, model, env_var):
    _skip_unless_live(env_var)
    handle = bl.submit(
        [
            bl.BatchLine(
                "c1", model, [{"role": "user", "content": "hi"}], {"max_tokens": 8}
            )
        ]
    )
    bl.cancel(handle)
    assert bl.status(handle).raw_state


@pytest.mark.parametrize(
    ("provider", "model", "env_var"), PROVIDERS, ids=[p[0] for p in PROVIDERS]
)
def test_list_works_where_the_descriptor_claims_it(provider, model, env_var):
    # Only a live call can check this. A mocked list endpoint answers 200
    # whether or not the provider actually has one, so the unit contract test
    # verifies adapter consistency and nothing about the provider.
    _skip_unless_live(env_var)
    caps = bl.capabilities_for(provider)
    if not caps.supports_list:
        pytest.skip(f"{provider} documents no list endpoint")
    handles = list(bl.list_jobs(provider, limit=3))
    assert all(h.provider == provider for h in handles)


@pytest.mark.parametrize(
    ("provider", "model", "env_var"), PROVIDERS, ids=[p[0] for p in PROVIDERS]
)
def test_results_are_matched_to_rows_by_content_not_by_count(provider, model, env_var):
    """The mis-join check.

    Every row gets a prompt whose correct answer is unique to it, so a result
    attached to the wrong row is visible rather than merely possible. A batch
    that returns the right number of well-formed answers against the wrong
    inputs would pass a count or schema check and fail this one.
    """
    _skip_unless_live(env_var)
    distinct = {
        "capital": ("What is the capital of France? One word.", "paris"),
        "arith": ("What is 40 plus 2? Digits only.", "42"),
        "color": ("What colour is a ripe banana? One word.", "yellow"),
    }
    lines = [
        bl.BatchLine(cid, model, [{"role": "user", "content": q}], {"max_tokens": 8})
        for cid, (q, _) in distinct.items()
    ]
    handle = bl.submit(lines)

    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        status = bl.status(handle)
        if status.is_terminal:
            break
        time.sleep(POLL_SECONDS)
    else:
        pytest.fail(f"{provider} batch did not finish within {MAX_WAIT_SECONDS}s")
    assert status.state == "succeeded", f"{provider}: {status.raw_state}"

    got = {r.custom_id: r for r in bl.results(handle)}
    assert set(got) == set(distinct), f"custom_ids did not round-trip: {sorted(got)}"
    for cid, (_q, expected) in distinct.items():
        answer = got[cid].response["choices"][0]["message"]["content"].strip().lower()
        assert expected in answer, (
            f"{provider}: row {cid!r} came back as {answer!r}, expected {expected!r}. "
            f"Results are mis-joined."
        )
