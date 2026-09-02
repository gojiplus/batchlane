"""What each provider's batch lane can and cannot do.

Every field here corresponds to a real asymmetry between providers that
silently corrupts results or wastes a run if ignored -- a join key that moves,
a cancel endpoint that does not exist, a window the caller cannot set.

Values are sourced from official provider documentation; ``notes`` records
anything the docs left ambiguous so a live run can settle it rather than a
guess being baked in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

__all__ = ["CAPABILITIES", "LaneCapabilities", "WindowSpec", "capabilities_for"]

ModelScope = Literal["line", "job", "url"]
InputMode = Literal["file", "inline"]

CHAT = "chat.completions"
EMBEDDINGS = "embeddings"

#: What batchlane can actually build a request body for today. `translate`
#: builds chat-completion bodies only, so advertising any other endpoint would
#: mean posting a chat payload to an endpoint that wants a different schema --
#: rejected at best, silently wrong at worst. Widen this only alongside a
#: translate path that produces the matching body; test_capabilities.py
#: enforces that no descriptor claims more than this.
IMPLEMENTED_ENDPOINTS = frozenset({CHAT})


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """How a provider lets the caller express turnaround time."""

    allowed: tuple[str, ...]
    default: str
    hint_only: bool = False


@dataclass(frozen=True, slots=True)
class LaneCapabilities:
    """One provider's batch lane, described precisely enough to gate on."""

    provider: str
    endpoints: frozenset[str]
    model_scope: ModelScope
    input_modes: frozenset[str]
    window: WindowSpec | None
    max_requests: int | None = None
    max_input_bytes: int | None = None
    supports_cancel: bool = True
    supports_list: bool = True
    result_retention: timedelta | None = None
    discount_note: str | None = None
    #: Fraction off the synchronous rate, for deriving a batch price where the
    #: provider publishes none. None means the saving is too model-dependent
    #: to express as one number, and no estimate should claim otherwise.
    discount: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


# The reference implementation of the shape every other OpenAI-compatible lane
# copied. litellm covers it too, but batchlane ships it anyway: without it a
# caller with mixed providers needs two code paths, which defeats the point of
# a single interface.
_OPENAI = LaneCapabilities(
    provider="openai",
    # The lane also covers embeddings and completions; batchlane does not build
    # those bodies yet, so it does not claim them. See IMPLEMENTED_ENDPOINTS.
    endpoints=frozenset({CHAT}),
    model_scope="line",
    input_modes=frozenset({"file"}),
    window=WindowSpec(allowed=("24h",), default="24h"),
    max_requests=50_000,
    max_input_bytes=200 * 1024 * 1024,
    discount_note="50%",
    discount=0.5,
    notes=("Output files persist until deleted; no documented auto-expiry.",),
)

_GROQ = LaneCapabilities(
    provider="groq",
    endpoints=frozenset({CHAT}),
    model_scope="line",
    input_modes=frozenset({"file"}),
    window=WindowSpec(allowed=("24h", "7d"), default="24h"),
    max_requests=50_000,
    # Groq's own docs disagree: the batch guide says 200MB, the API reference
    # says 100MB. Taking the smaller, because over-sending is a rejected
    # upload while under-sending only costs an extra chunk.
    max_input_bytes=100 * 1024 * 1024,
    result_retention=timedelta(days=30),
    discount_note="50%, restricted to a documented model allowlist",
    discount=0.5,
    notes=(
        "Groq documents 'durations from 24h to 7d', so intermediate values "
        "are likely valid, but the accepted format for them is not given. "
        "Only the two documented endpoints are allowed here rather than "
        "guessing at a grammar and earning a 400.",
        "Its guide and API reference disagree on max file size, 200MB against "
        "100MB; the smaller is used.",
    ),
)

_TOGETHER = LaneCapabilities(
    provider="together_ai",
    endpoints=frozenset({CHAT}),
    model_scope="line",
    input_modes=frozenset({"file"}),
    window=WindowSpec(allowed=("24h",), default="24h", hint_only=True),
    max_requests=50_000,
    max_input_bytes=100 * 1024 * 1024,
    result_retention=None,
    discount_note="up to 50%; several models excluded from batch entirely",
    # 'up to 50%' with several models excluded outright, so a single
    # multiplier would overstate the saving on an unknown share of a job.
    discount=None,
    notes=(
        "Upload purpose is 'batch-api', not the OpenAI-standard 'batch'.",
        "Reports a single progress float rather than a request_counts object.",
        "Result retention is not documented; confirm on a live run.",
    ),
)

_DEEPINFRA = LaneCapabilities(
    provider="deepinfra",
    endpoints=frozenset({CHAT}),
    model_scope="line",
    input_modes=frozenset({"file"}),
    window=WindowSpec(allowed=("24h",), default="24h"),
    max_requests=50_000,
    max_input_bytes=200 * 1024 * 1024,
    result_retention=None,
    discount_note="20%",
    discount=0.2,
    notes=(
        "Model must be uniform across the file even though it is carried per line.",
        "GET /batches and POST /batches/{id}/cancel are both documented "
        "(docs.deepinfra.com/batch/batch-endpoints).",
        "Result TTL is caller-set via output_expires_after; no documented default.",
    ),
)

_MISTRAL = LaneCapabilities(
    provider="mistral",
    endpoints=frozenset({CHAT}),
    model_scope="job",
    input_modes=frozenset({"file", "inline"}),
    window=WindowSpec(allowed=(), default="24h"),
    max_requests=1_000_000,
    max_input_bytes=512 * 1024 * 1024,
    result_retention=None,
    discount_note="50%",
    discount=0.5,
    notes=(
        "Window is timeout_hours, a bare integer, so any Nh value is valid "
        "and the allowed tuple is left empty to mean unconstrained.",
        "Batch path is /v1/batch/jobs, not /v1/batches.",
        "Result retention and discount exclusions are undocumented.",
    ),
)

_GEMINI = LaneCapabilities(
    provider="gemini",
    endpoints=frozenset({CHAT}),
    model_scope="url",
    # v1 implements inline submission only, so the descriptor advertises the
    # 20MB inline ceiling rather than the 2GB one that needs the File API.
    input_modes=frozenset({"inline"}),
    window=None,
    max_requests=None,
    max_input_bytes=20 * 1024 * 1024,
    result_retention=timedelta(weeks=6),
    discount_note="50%",
    discount=0.5,
    notes=(
        "No caller-set window; the platform enforces a fixed 48h expiry.",
        "Docs say inline results map to requests by ARRAY INDEX, not by the "
        "key supplied. The adapter joins on an echoed key where one is "
        "present and otherwise falls back to position, refusing outright if "
        "the counts disagree.",
        "State is nested at metadata.state with JOB_STATE_* values; a "
        "BATCH_STATE_* enum in the REST reference appears to be stale.",
    ),
)

_FIREWORKS = LaneCapabilities(
    provider="fireworks_ai",
    endpoints=frozenset({CHAT}),
    model_scope="job",
    input_modes=frozenset({"file"}),
    window=WindowSpec(allowed=(), default="24h"),
    max_requests=None,
    max_input_bytes=80 * 1024 * 1024 * 1024,
    supports_cancel=False,
    result_retention=None,
    discount_note="50%",
    discount=0.5,
    notes=(
        "No cancel endpoint is documented; cancel() refuses rather than no-ops.",
        "Input is a registered dataset, not a file upload.",
        "Window is maxJobDuration, an integer bounded to [12h, 72h].",
    ),
)

# Verified against the live API on 2026-09-01: submit -> poll -> results.
# Anthropic is the odd one out -- no file upload at all, requests go inline in
# the create call and results stream from a URL that stays null until the job
# ends. litellm can *retrieve* an Anthropic batch but not create one
# ("LiteLLM doesn't support custom_llm_provider=anthropic for 'create_batch'"),
# so this is a genuine gap despite anthropic appearing in its retrieve path.
_ANTHROPIC = LaneCapabilities(
    provider="anthropic",
    endpoints=frozenset({CHAT}),
    model_scope="line",
    input_modes=frozenset({"inline"}),
    window=None,
    max_requests=100_000,
    max_input_bytes=256 * 1024 * 1024,
    result_retention=timedelta(days=29),
    discount_note="50%",
    discount=0.5,
    notes=(
        "No file upload; requests are inline under a 'requests' key on create.",
        "processing_status is only in_progress/ended -- terminality lives in "
        "request_counts, not the status string.",
        "request_counts keys are processing/succeeded/errored/canceled/expired.",
        "results_url is null until processing_status == 'ended'.",
        "Results are JSONL of {custom_id, result:{type, message}} and arrive "
        "out of submission order, so they must be joined on custom_id.",
        "usage.service_tier == 'batch' confirms the discount lane was used.",
    ),
)

CAPABILITIES: dict[str, LaneCapabilities] = {
    c.provider: c
    for c in (
        _OPENAI,
        _GROQ,
        _TOGETHER,
        _DEEPINFRA,
        _MISTRAL,
        _GEMINI,
        _FIREWORKS,
        _ANTHROPIC,
    )
}

#: Providers that genuinely run no asynchronous lane, and why. A batch discount
#: requires owning a fleet with an idle trough to backfill, so for these it is a
#: structural fact rather than a gap that will close on its own.
NO_LANE: dict[str, str] = {
    "openrouter": (
        "It resells upstream capacity at retail, so it has no idle fleet to "
        "backfill and cannot price a discount lane."
    ),
}

#: Runtimes you host yourself. There is no batch lane because there is no
#: per-token price to discount: the hardware is already yours. Someone here who
#: wants a job to finish sooner needs throughput, not a discount lane, so the
#: refusal says that rather than implying a gap that might close.
LOCAL_RUNTIME: dict[str, str] = {
    "hosted_vllm": (
        "you host it, so there is no per-token price to discount. Note that "
        "litellm lists hosted_vllm as batch-capable and routes it to the "
        "OpenAI batches handler, but a stock `vllm serve` exposes no "
        "/v1/batches; vLLM's batch support is the offline `vllm run-batch` "
        "CLI, which reads the same JSONL and writes results to a file"
    ),
    "vllm": (
        "you host it, so there is no per-token price to discount. Use "
        "`vllm run-batch -i input.jsonl -o output.jsonl`, which loads the "
        "model once and streams the whole file through it"
    ),
}
LOCAL_RUNTIME.update(
    dict.fromkeys(
        (
            "ollama",
            "ollama_chat",
            "lm_studio",
            "llamafile",
            "oobabooga",
            "triton",
            "xinference",
        ),
        "you host it, so there is no per-token price to discount. What helps "
        "here is throughput -- keeping the server busy with concurrent "
        "requests -- not a batch lane",
    )
)

#: Providers that do run a lane which batchlane has deliberately not shipped.
#: Kept distinct from NO_LANE so the refusal never claims a lane is absent when
#: it is merely unimplemented.
#: Providers litellm's own /batches already reaches. batchlane does not
#: duplicate them, but must never report their lane as absent -- pointing a
#: caller at litellm is a useful answer; "no record of one" is a dead end and
#: a false one.
_LITELLM_COVERED = (
    "litellm already reaches {name} batches, so use litellm.create_batch "
    "until batchlane ships an adapter"
)

NOT_SHIPPED: dict[str, str] = {
    "azure": _LITELLM_COVERED.format(name="Azure OpenAI"),
    "vertex_ai": _LITELLM_COVERED.format(name="Vertex AI"),
    "bedrock": _LITELLM_COVERED.format(name="Bedrock"),
    "xai": (
        "its lane discounts only 20%, and its own docs exclude the flagship "
        "models (grok-4.6, grok-4.5, grok-build-0.1), so it was not worth the "
        "adapter for v1"
    ),
}


def capabilities_for(provider: str) -> LaneCapabilities | None:
    """Look up a provider's lane capabilities.

    Args:
        provider: A LiteLLM provider key, e.g. ``"groq"``.

    Returns:
        The descriptor, or None when batchlane has no adapter for it.
    """
    return CAPABILITIES.get(provider)
