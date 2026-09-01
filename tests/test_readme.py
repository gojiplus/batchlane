"""The README's examples are program output, so check they still are.

The refusal transcript in the README went stale the moment OpenAI and Gemini
were added: it still listed three providers when six shipped. Nothing caught
it, because nothing ran it. A README that lies in its examples is worse than a
thin one, since a reader trusts a transcript.

Rather than fight doctest -- one example depends on a binding defined in an
earlier prompt-less block, another expects a wrapped multi-line traceback --
compute each expected value from live code and assert it appears in the file.
Offline, no keys.
"""

import pathlib

import pytest

import batchlane as bl

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
RAW = README.read_text()
# A README wraps its lines, so an expected value can straddle a newline.
# Compare against a whitespace-collapsed copy.
TEXT = " ".join(RAW.split())

STALE = (
    "README.md is stale. Regenerate this example from live code rather than "
    "editing the expected value by hand."
)


def test_capability_examples_still_hold():
    assert repr(bl.capabilities_for("groq").window.allowed) in TEXT, STALE
    assert repr(bl.capabilities_for("groq").result_retention) in TEXT, STALE


def test_the_refusal_transcript_lists_every_shipped_provider():
    # This is the exact example that went stale. It names the providers that
    # do have a lane, so it changes every time an adapter is added.
    assert ", ".join(bl.supported_providers()) in TEXT, STALE


def test_the_plan_example_matches_what_plan_returns():
    rows = [
        bl.BatchLine(
            "row-1",
            "groq/llama-3.3-70b-versatile",
            [{"role": "user", "content": "Classify: the product was great"}],
        ),
        bl.BatchLine(
            "row-2",
            "groq/llama-3.3-70b-versatile",
            [{"role": "user", "content": "Classify: it broke in a week"}],
        ),
    ]
    plan = bl.plan(rows)
    assert repr((plan.n_chunks, plan.total_bytes, plan.limit_bytes)) in TEXT, STALE


def test_the_support_table_lists_exactly_what_ships():
    table = RAW[RAW.index("| Provider |") : RAW.index("## What is missing")]
    for provider in bl.supported_providers():
        # The table uses display names, so match on something stable per row.
        assert provider.split("_")[0] in table.lower(), (
            f"{provider} ships but has no row in the support table"
        )


@pytest.mark.parametrize("shipped", sorted(bl.supported_providers()))
def test_no_shipped_provider_is_described_as_refused(shipped):
    from batchlane.capabilities import LOCAL_RUNTIME, NO_LANE, NOT_SHIPPED

    assert shipped not in set(NO_LANE) | set(NOT_SHIPPED) | set(LOCAL_RUNTIME)


def test_install_instruction_is_not_a_pypi_claim_while_unpublished():
    # batchlane is not on PyPI. `pip install batchlane` would simply fail.
    assert "pip install batchlane\n" not in RAW
    assert "pip install git+https://github.com/gojiplus/batchlane" in TEXT


def test_dev_instructions_use_the_grouping_the_project_actually_has():
    # The project declares [dependency-groups], not [project.optional-dependencies],
    # so --all-extras installs no dev tooling at all.
    assert "uv sync --all-extras" not in TEXT
    assert "uv sync --all-groups" in TEXT
