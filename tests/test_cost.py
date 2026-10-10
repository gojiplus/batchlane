"""Cost estimation, including how much of the estimate is actually known."""

import os
from unittest.mock import Mock, call

import pytest

import batchlane as bl
from batchlane.capabilities import CAPABILITIES


@pytest.fixture(autouse=True)
def _fixed_cost_inputs(monkeypatch):
    """Keep arithmetic independent of tokenizer downloads and online prices."""
    import litellm

    monkeypatch.setattr(litellm, "token_counter", Mock(return_value=7))
    prices = {
        "openai/gpt-4o-mini": {
            "input_cost_per_token": 1.5e-7,
            "output_cost_per_token": 6e-7,
            "input_cost_per_token_batches": 7.5e-8,
            "output_cost_per_token_batches": 3e-7,
        },
        "groq/llama-3.3-70b-versatile": {
            "input_cost_per_token": 5.9e-7,
            "output_cost_per_token": 7.9e-7,
        },
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo": {
            "input_cost_per_token": 8.8e-7,
            "output_cost_per_token": 8.8e-7,
        },
        "anthropic/claude-haiku-4-5-20251001": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 5e-6,
        },
    }
    for model, rates in prices.items():
        provider, bare = model.split("/", 1)
        info = {"litellm_provider": provider, "mode": "chat", **rates}
        monkeypatch.setitem(litellm.model_cost, model, info)
        monkeypatch.setitem(litellm.model_cost, bare, info)


def _rows(model, n=10, max_tokens=None, text="hello world"):
    params = {"max_tokens": max_tokens} if max_tokens is not None else {}
    return [
        bl.BatchLine(f"r{i}", model, [{"role": "user", "content": text}], params)
        for i in range(n)
    ]


def test_a_published_batch_rate_is_used_verbatim():
    # OpenAI is the only lane whose batch rate litellm actually carries, so it
    # is the one case where the estimate is not derived.
    import litellm

    info = litellm.get_model_info("openai/gpt-4o-mini")
    rows = _rows("openai/gpt-4o-mini", n=5, max_tokens=20)
    est = bl.plan(rows).cost

    assert est.rate_source == "published"
    assert est.caveat is None
    assert est.input_tokens == 35
    assert est.output_tokens == 100
    expected = (
        35 * info["input_cost_per_token_batches"]
        + 100 * info["output_cost_per_token_batches"]
    )
    assert est.batch_usd == pytest.approx(expected)


def test_a_derived_rate_is_exactly_the_documented_discount_off_sync():
    # Hand-computed rather than "a number came back": a broken multiplier
    # would still return a plausible float.
    import litellm

    model = "groq/llama-3.3-70b-versatile"
    info = litellm.get_model_info(model)
    discount = CAPABILITIES["groq"].discount
    rows = _rows(model, n=5, max_tokens=20)
    est = bl.plan(rows).cost

    assert est.rate_source == "derived"
    assert est.input_tokens == 35
    assert est.output_tokens == 100
    expected = 35 * info["input_cost_per_token"] * (1 - discount) + (
        100 * info["output_cost_per_token"] * (1 - discount)
    )
    assert est.batch_usd == pytest.approx(expected)
    assert est.sync_usd == pytest.approx(est.batch_usd / (1 - discount))


@pytest.mark.parametrize(
    "model", ["groq/llama-3.3-70b-versatile", "openai/gpt-4o-mini"]
)
def test_batch_is_strictly_cheaper_than_sync_where_a_discount_exists(model):
    est = bl.plan(_rows(model, max_tokens=10)).cost
    assert est.batch_usd < est.sync_usd
    assert est.saving_usd > 0


def test_a_lane_whose_saving_varies_by_model_claims_none():
    # Together documents "up to 50%" and excludes some models outright. A flat
    # multiplier would overstate the saving on an unknown share of a job, so
    # the estimate refuses to imply one.
    assert CAPABILITIES["together_ai"].discount is None
    est = bl.plan(
        _rows("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", max_tokens=10)
    ).cost
    assert est.rate_source == "unknown"
    assert est.saving_usd == 0
    assert "varies by model" in est.caveat


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o-mini",
        "groq/llama-3.3-70b-versatile",
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "anthropic/claude-haiku-4-5-20251001",
    ],
)
def test_cost_arithmetic_needs_neither_tokenizers_nor_symlinks(
    model, monkeypatch, tmp_path
):
    import litellm.utils
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "huggingface"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "huggingface"))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tiktoken"))
    denied = PermissionError("A required privilege is not held by the client")
    denied.winerror = 1314
    symlink = Mock(side_effect=denied)
    tokenizer = Mock(side_effect=AssertionError("real tokenizer used in arithmetic"))
    monkeypatch.setattr(os, "symlink", symlink)
    monkeypatch.setattr(litellm.utils, "_select_tokenizer", tokenizer)

    with pytest.raises(PermissionError) as exc:
        (tmp_path / "link").symlink_to(tmp_path / "source")
    assert exc.value.winerror == 1314
    symlink.reset_mock()

    est = bl.plan(_rows(model, n=2, max_tokens=10)).cost

    # estimate_cost catches tokenizer exceptions, so raising alone cannot guard
    # against an accidental download being hidden by its character-count fallback.
    tokenizer.assert_not_called()
    symlink.assert_not_called()
    assert est.input_tokens == 14
    assert est.output_tokens == 20
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "error", [RuntimeError("unavailable"), PermissionError("denied")]
)
def test_tokenizer_failure_prices_the_character_count_fallback(monkeypatch, error):
    import litellm

    counter = Mock(side_effect=error)
    monkeypatch.setattr(litellm, "token_counter", counter)
    rows = _rows("openai/gpt-4o-mini", n=2, max_tokens=5, text="twelve chars")
    rows[0].messages.append({"role": "system", "content": "be terse"})

    est = bl.plan(rows).cost

    assert counter.call_args_list == [
        call(model="gpt-4o-mini", messages=row.messages) for row in rows
    ]
    assert est.input_tokens == 8
    assert est.output_tokens == 10
    assert est.batch_usd == pytest.approx(8 * 7.5e-8 + 10 * 3e-7)
    assert est.sync_usd == pytest.approx(8 * 1.5e-7 + 10 * 6e-7)


def test_tokenizer_failure_does_not_replace_successful_counts(monkeypatch):
    import litellm

    counter = Mock(side_effect=[11, RuntimeError("unavailable"), 17])
    monkeypatch.setattr(litellm, "token_counter", counter)

    est = bl.plan(_rows("openai/gpt-4o-mini", n=3, text="twelve chars")).cost

    assert counter.call_count == 3
    assert est.input_tokens == 31
    assert est.output_tokens is None
    assert est.batch_usd == pytest.approx(31 * 7.5e-8)


def test_without_max_tokens_only_the_input_side_is_priced():
    # Output length is not knowable before the job runs, and inventing a
    # number would make the estimate confidently wrong.
    est = bl.plan(_rows("openai/gpt-4o-mini", max_tokens=None)).cost
    assert est.output_tokens is None
    assert "input only" in str(est)
    assert est.batch_usd > 0


def test_max_tokens_makes_the_estimate_an_upper_bound_and_says_so():
    est = bl.plan(_rows("openai/gpt-4o-mini", n=4, max_tokens=25)).cost
    assert est.output_tokens == 100
    assert "upper bound" in str(est)


def test_a_derived_estimate_is_marked_approximate_and_a_published_one_is_not():
    derived = bl.plan(_rows("groq/llama-3.3-70b-versatile", max_tokens=5)).cost
    published = bl.plan(_rows("openai/gpt-4o-mini", max_tokens=5)).cost
    assert str(derived).startswith("~")
    assert not str(published).startswith("~")


def test_more_rows_cost_more():
    small = bl.plan(_rows("openai/gpt-4o-mini", n=5, max_tokens=10)).cost
    large = bl.plan(_rows("openai/gpt-4o-mini", n=50, max_tokens=10)).cost
    assert large.batch_usd > small.batch_usd
    assert large.input_tokens > small.input_tokens


def test_an_unknown_model_does_not_crash_the_estimate():
    rows = [
        bl.BatchLine(
            "r0",
            "groq/some-model-that-does-not-exist",
            [{"role": "user", "content": "x"}],
        )
    ]
    est = bl.plan(rows).cost
    assert est.input_tokens > 0
    assert est.caveat is not None


# Each figure below is what the provider's own documentation states, recorded
# as a literal. The tests above read the discount from CAPABILITIES, which
# only proves the code applies whatever number is in the table -- changing the
# table to 60% left them all green. This is the test that catches that.
DOCUMENTED_DISCOUNTS = {
    "openai": 0.5,
    "anthropic": 0.5,
    "gemini": 0.5,
    "groq": 0.5,
    "mistral": 0.5,
    "fireworks_ai": 0.5,
    "deepinfra": 0.2,
    "together_ai": None,  # "up to 50%", several models excluded outright
}


@pytest.mark.parametrize(("provider", "documented"), DOCUMENTED_DISCOUNTS.items())
def test_the_table_matches_what_the_provider_documents(provider, documented):
    assert CAPABILITIES[provider].discount == documented, (
        f"{provider}'s discount in the capability table does not match its "
        f"documented rate. An estimate is only as honest as this number."
    )


def test_every_shipped_lane_has_a_documented_discount_recorded():
    # A lane added without one would silently price at full rate.
    missing = set(bl.supported_providers()) - set(DOCUMENTED_DISCOUNTS)
    assert not missing, f"no documented discount recorded for {sorted(missing)}"


# --- what a finished job actually cost, not what it was projected to ---


def _done(model, prompt_tokens, completion_tokens, tier=None):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if tier:
        usage["service_tier"] = tier
    return bl.RequestResult("r0", response={"model": model, "usage": usage})


def test_actual_cost_is_priced_from_reported_usage_not_a_bound():
    import litellm

    info = litellm.get_model_info("openai/gpt-4o-mini")
    got = bl.actual_cost([_done("gpt-4o-mini", 100, 40)], "openai")
    assert got.measured is True
    assert (got.input_tokens, got.output_tokens) == (100, 40)
    assert got.batch_usd == pytest.approx(
        100 * info["input_cost_per_token_batches"]
        + 40 * info["output_cost_per_token_batches"]
    )
    assert "(actual)" in str(got)


def test_a_tier_other_than_batch_says_the_discount_did_not_apply():
    # The number alone would look like a saving. The provider's own tier field
    # is the only thing that says whether one actually happened.
    got = bl.actual_cost(
        [_done("claude-haiku-4-5-20251001", 10, 5, tier="standard")], "anthropic"
    )
    assert got.service_tier == "standard"
    assert "did not apply" in got.caveat


def test_a_batch_tier_is_reported_without_a_complaint():
    got = bl.actual_cost(
        [_done("claude-haiku-4-5-20251001", 10, 5, tier="batch")], "anthropic"
    )
    assert got.service_tier == "batch"
    assert "did not apply" not in (got.caveat or "")


def test_usage_totals_add_up_across_rows():
    rows = [_done("gpt-4o-mini", 10, 4) for _ in range(5)]
    got = bl.actual_cost(rows, "openai")
    assert (got.input_tokens, got.output_tokens) == (50, 20)
