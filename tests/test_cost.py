"""Cost estimation, including how much of the estimate is actually known."""

import pytest

import batchlane as bl
from batchlane.capabilities import CAPABILITIES


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
    expected = (
        est.input_tokens * info["input_cost_per_token_batches"]
        + est.output_tokens * info["output_cost_per_token_batches"]
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
    expected = est.input_tokens * info["input_cost_per_token"] * (1 - discount) + (
        est.output_tokens * info["output_cost_per_token"] * (1 - discount)
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
