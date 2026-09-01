import pytest

import batchlane as bl
from batchlane.errors import AdapterNotShippedError


def test_no_lane_provider_explains_why_structurally():
    with pytest.raises(bl.NoBatchLaneError) as exc:
        bl.get_adapter("openrouter")
    message = str(exc.value)
    assert "openrouter" in message
    assert "retail" in message
    # A refusal that does not point somewhere useful is a dead end.
    assert "groq" in message


def test_unknown_provider_refuses_without_claiming_knowledge():
    with pytest.raises(bl.NoBatchLaneError) as exc:
        bl.get_adapter("some-startup-inference")
    assert "no record of one" in str(exc.value)


def test_unshipped_adapter_is_not_reported_as_a_missing_lane():
    # xAI genuinely has a lane. Saying otherwise would be false.
    with pytest.raises(AdapterNotShippedError) as exc:
        bl.get_adapter("xai")
    message = str(exc.value)
    assert "runs a batch lane" in message
    assert "no asynchronous batch lane" not in message


def test_planned_provider_says_planned_not_absent():
    with pytest.raises(AdapterNotShippedError):
        bl.get_adapter("mistral")


def test_empty_batch_refused():
    with pytest.raises(bl.BatchlaneError, match="empty batch"):
        bl.submit([])


def test_batch_spanning_two_providers_refused():
    lines = [
        bl.BatchLine(
            "a", "groq/llama-3.3-70b-versatile", [{"role": "user", "content": "x"}]
        ),
        bl.BatchLine(
            "b", "deepinfra/deepseek-ai/DeepSeek-V3", [{"role": "user", "content": "y"}]
        ),
    ]
    with pytest.raises(bl.BatchlaneError, match="one provider"):
        bl.submit(lines)


# Providers that demonstrably run a real asynchronous batch lane. Claiming any
# of these has "no batch lane" is a factual error, whether or not batchlane
# implements it.
PROVIDERS_WITH_A_REAL_LANE = [
    "openai",
    "azure",
    "vertex_ai",
    "bedrock",
    "anthropic",
    "groq",
    "together_ai",
    "deepinfra",
    "mistral",
    "gemini",
    "fireworks_ai",
    "xai",
]


@pytest.mark.parametrize("provider", PROVIDERS_WITH_A_REAL_LANE)
def test_a_provider_with_a_real_lane_is_never_reported_as_having_none(provider):
    # The bug this guards: openai/azure/vertex_ai/bedrock fell through to the
    # catch-all branch and reported "no asynchronous batch lane", which is
    # flatly false -- OpenAI's is the canonical batch API. Parametrized over
    # the class rather than the four instances that happened to be wrong.
    try:
        bl.get_adapter(provider)
    except bl.NoBatchLaneError as exc:  # pragma: no cover - only on regression
        pytest.fail(f"{provider} has a real batch lane but was denied: {exc}")
    except AdapterNotShippedError:
        pass  # honest: the lane exists, we just have not built it


@pytest.mark.parametrize("provider", ["azure", "vertex_ai", "bedrock"])
def test_litellm_covered_providers_point_somewhere_useful(provider):
    # A refusal that names no alternative is a dead end. litellm reaches these.
    with pytest.raises(AdapterNotShippedError, match="litellm"):
        bl.get_adapter(provider)
