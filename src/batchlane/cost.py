"""What a job will cost, and how much of that is actually known.

A package whose premise is "you are paying double" should be able to show the
number. Two things get in the way, and both are stated rather than smoothed
over.

Providers mostly do not publish a batch rate. LiteLLM's model registry carries
``input_cost_per_token_batches`` keys for every model, but the values are
populated for OpenAI and null for the other seven lanes here. Where a rate is
published it is used; otherwise the batch price is derived from the lane's
documented discount, and the estimate says so.

Output length is unknowable before the job runs. With ``max_tokens`` set the
estimate is an upper bound; without it, only the input side can be priced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .handle import BatchLine, RequestResult

__all__ = ["CostEstimate", "actual_cost", "estimate_cost"]

RateSource = Literal["published", "derived", "unknown"]


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """An estimate, carrying its own uncertainty."""

    input_tokens: int
    output_tokens: int | None
    batch_usd: float | None
    sync_usd: float | None
    rate_source: RateSource
    caveat: str | None = None
    #: True when the token counts came from the provider's own usage report
    #: rather than from counting the prompts and bounding the output.
    measured: bool = False
    #: What the provider says it served this at, where it says anything. A
    #: value other than "batch" means the discount did not apply, which is
    #: worth more than the number itself.
    service_tier: str | None = None

    @property
    def saving_usd(self) -> float | None:
        """How much the batch lane saves against synchronous pricing.

        Returns:
            The difference, or None when either side could not be priced.
        """
        if self.batch_usd is None or self.sync_usd is None:
            return None
        return self.sync_usd - self.batch_usd

    def __str__(self) -> str:
        """Render the estimate for a human.

        Returns:
            A one-line summary, flagging a derived rate as approximate.
        """
        if self.batch_usd is None or self.sync_usd is None:
            return f"{self.input_tokens:,} input tokens; cost not estimable"
        mark = "~" if self.rate_source != "published" else ""
        if self.measured:
            bound = " (actual)"
        elif self.output_tokens is not None:
            bound = " (upper bound)"
        else:
            bound = " (input only)"
        return (
            f"{mark}{_money(self.batch_usd)} at batch rates vs "
            f"{mark}{_money(self.sync_usd)} sync{bound}"
        )


def _money(amount: float) -> str:
    """Format a dollar amount without rounding a small job away to zero.

    Args:
        amount: The amount in USD.

    Returns:
        A string with enough precision to be informative at any scale.
    """
    if amount >= 0.01:
        return f"${amount:,.2f}"
    if amount >= 0.000001:
        return f"${amount:.6f}"
    return f"${amount:.2e}"


def _rates(
    model: str, provider: str
) -> tuple[float, float, float, float, RateSource, str | None]:
    """Find per-token input and output rates for sync and batch.

    Args:
        model: Bare model name.
        provider: LiteLLM provider key.

    Returns:
        ``(sync_in, sync_out, batch_in, batch_out, source, caveat)``. Rates are
        zero when the registry knows nothing about the model.
    """
    import litellm

    from .capabilities import capabilities_for

    try:
        info = litellm.get_model_info(f"{provider}/{model}")
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, "unknown", "model not in litellm's price registry"

    sync_in = info.get("input_cost_per_token") or 0.0
    sync_out = info.get("output_cost_per_token") or 0.0
    batch_in = info.get("input_cost_per_token_batches")
    batch_out = info.get("output_cost_per_token_batches")
    if batch_in is not None and batch_out is not None:
        return sync_in, sync_out, batch_in, batch_out, "published", None

    caps = capabilities_for(provider)
    discount = caps.discount if caps else None
    if discount is None:
        note = caps.discount_note if caps else None
        return (
            sync_in,
            sync_out,
            sync_in,
            sync_out,
            "unknown",
            f"{provider} publishes no batch rate and its saving varies by model"
            + (f" ({note})" if note else ""),
        )
    return (
        sync_in,
        sync_out,
        sync_in * (1 - discount),
        sync_out * (1 - discount),
        "derived",
        f"{provider} publishes no batch rate; derived from its documented "
        f"{discount:.0%} discount"
        + (f" ({caps.discount_note})" if caps and caps.discount_note else ""),
    )


def estimate_cost(
    lines: Sequence[BatchLine], provider: str, model: str
) -> CostEstimate:
    """Price a job before running it.

    Args:
        lines: The requests that would be submitted.
        provider: LiteLLM provider key.
        model: Bare model name.

    Returns:
        The estimate, with its rate source and any caveat attached.
    """
    import litellm

    input_tokens = 0
    for line in lines:
        try:
            input_tokens += litellm.token_counter(model=model, messages=line.messages)
        except Exception:
            input_tokens += sum(
                len(str(m.get("content", ""))) // 4 for m in line.messages
            )

    # Output length is not knowable in advance. max_tokens makes it an upper
    # bound; without one, only the input side can honestly be priced.
    per_line = [line.params.get("max_tokens") for line in lines]
    output_tokens = (
        sum(v for v in per_line if isinstance(v, int))
        if per_line and all(isinstance(v, int) for v in per_line)
        else None
    )

    sync_in, sync_out, batch_in, batch_out, source, caveat = _rates(model, provider)
    if source == "unknown" and not sync_in:
        return CostEstimate(input_tokens, output_tokens, None, None, source, caveat)

    def total(rate_in: float, rate_out: float) -> float:
        cost = input_tokens * rate_in
        if output_tokens is not None:
            cost += output_tokens * rate_out
        return cost

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batch_usd=total(batch_in, batch_out),
        sync_usd=total(sync_in, sync_out),
        rate_source=source,
        caveat=caveat,
    )


def actual_cost(results: Iterable[RequestResult], provider: str) -> CostEstimate:
    """Price a finished job from the usage the provider itself reported.

    The estimate before a run bounds the output; this reads what was actually
    served. It also carries the service tier back where a provider states one,
    because "you paid batch rates" is a claim worth checking rather than
    assuming: Anthropic reports ``usage.service_tier``, and a value other than
    ``batch`` means the discount did not apply.

    Args:
        results: The finished job's results.
        provider: LiteLLM provider key.

    Returns:
        The cost, with ``measured`` set so a reader can tell it apart from an
        estimate.
    """
    input_tokens = output_tokens = 0
    model = ""
    tiers: set[str] = set()
    for result in results:
        body = result.response if isinstance(result.response, dict) else None
        if not body:
            continue
        model = model or str(body.get("model", ""))
        usage = body.get("usage") or {}
        input_tokens += int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        )
        output_tokens += int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        if tier := usage.get("service_tier"):
            tiers.add(str(tier))

    sync_in, sync_out, batch_in, batch_out, source, caveat = _rates(model, provider)
    served = ", ".join(sorted(tiers)) if tiers else None
    if served and served != "batch":
        caveat = (
            f"the provider served this at tier {served!r}, not 'batch', so the "
            f"discount did not apply"
        )
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batch_usd=input_tokens * batch_in + output_tokens * batch_out,
        sync_usd=input_tokens * sync_in + output_tokens * sync_out,
        rate_source=source,
        caveat=caveat,
        measured=True,
        service_tier=served,
    )
