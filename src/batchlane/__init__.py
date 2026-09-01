"""Submit async batch jobs to any LLM provider's discount lane.

Where a provider runs an asynchronous lane, batchlane reaches it. Where none
exists, it refuses and says why, rather than emulating a batch over the
synchronous endpoint -- which would save nothing while implying half price.

Example:
    >>> import batchlane
    >>> handle = batchlane.submit([
    ...     batchlane.BatchLine("q1", "groq/llama-3.3-70b-versatile",
    ...                         [{"role": "user", "content": "hi"}]),
    ... ])
    >>> batchlane.status(handle).state
    'pending'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .capabilities import CAPABILITIES, LaneCapabilities, capabilities_for
from .errors import (
    AdapterNotShippedError,
    BatchlaneError,
    CapabilityNotSupportedError,
    MixedModelBatchError,
    NoBatchLaneError,
)
from .handle import BatchHandle, BatchLine, JobStatus, RequestResult
from .registry import (
    adapter_for_model,
    get_adapter,
    resolve_api_key,
    supported_providers,
)
from .runner import ChunkPlan, answer_text, map, plan, run, wait  # noqa: A004

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "CAPABILITIES",
    "AdapterNotShippedError",
    "BatchHandle",
    "BatchLine",
    "BatchlaneError",
    "CapabilityNotSupportedError",
    "ChunkPlan",
    "JobStatus",
    "LaneCapabilities",
    "MixedModelBatchError",
    "NoBatchLaneError",
    "RequestResult",
    "answer_text",
    "cancel",
    "capabilities_for",
    "get_adapter",
    "list_jobs",
    "map",
    "plan",
    "results",
    "run",
    "status",
    "submit",
    "supported_providers",
    "wait",
]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("batchlane")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.0.0"


def submit(
    lines: Sequence[BatchLine],
    *,
    endpoint: str = "chat.completions",
    window: str | None = None,
    api_key: str | None = None,
) -> BatchHandle:
    """Submit a batch to the provider's discount lane.

    Args:
        lines: The requests to run. Every line must name a model from the same
            provider; batchlane does not split a batch across providers.
        endpoint: Which endpoint the lines target.
        window: Requested turnaround, or None for the provider's default.
        api_key: Credential, or None to read it from the environment.

    Returns:
        A handle for polling and collecting the job.

    Raises:
        BatchlaneError: If ``lines`` is empty or spans multiple providers.
    """
    if not lines:
        raise BatchlaneError("Cannot submit an empty batch.")

    resolved = [adapter_for_model(line.model) for line in lines]
    providers = {provider for _adapter, provider, _bare in resolved}
    if len(providers) > 1:
        raise BatchlaneError(
            f"A batch must target one provider; got {sorted(providers)}. "
            f"Split it and submit one batch per provider."
        )

    adapter, provider, _bare = resolved[0]
    # Adapters work in bare model names; the provider prefix is ours, not theirs.
    bare_lines = [
        BatchLine(line.custom_id, bare, line.messages, line.params)
        for line, (_a, _p, bare) in zip(lines, resolved, strict=True)
    ]
    return adapter.submit(
        bare_lines,
        endpoint=endpoint,
        window=window,
        api_key=resolve_api_key(provider, api_key),
    )


def status(handle: BatchHandle, *, api_key: str | None = None) -> JobStatus:
    """Poll a submitted job.

    Args:
        handle: The receipt from :func:`submit`.
        api_key: Credential, or None to read it from the environment.

    Returns:
        The job's normalized status.
    """
    adapter = get_adapter(handle.provider)
    return adapter.status(handle, api_key=resolve_api_key(handle.provider, api_key))


def results(
    handle: BatchHandle, *, api_key: str | None = None
) -> Iterator[RequestResult]:
    """Stream a completed job's results.

    Args:
        handle: The receipt from :func:`submit`.
        api_key: Credential, or None to read it from the environment.

    Returns:
        An iterator of results, joined on ``custom_id``.
    """
    adapter = get_adapter(handle.provider)
    return adapter.results(handle, api_key=resolve_api_key(handle.provider, api_key))


def list_jobs(
    provider: str, *, limit: int = 20, api_key: str | None = None
) -> Iterator[BatchHandle]:
    """List recent jobs on a provider.

    Args:
        provider: A LiteLLM provider key.
        limit: Maximum jobs to return.
        api_key: Credential, or None to read it from the environment.

    Returns:
        An iterator of handles, which can be passed straight to :func:`status`.
    """
    adapter = get_adapter(provider)
    return adapter.list_jobs(limit=limit, api_key=resolve_api_key(provider, api_key))


def cancel(handle: BatchHandle, *, api_key: str | None = None) -> None:
    """Cancel a running job.

    Args:
        handle: The receipt from :func:`submit`.
        api_key: Credential, or None to read it from the environment.
    """
    adapter = get_adapter(handle.provider)
    adapter.cancel(handle, api_key=resolve_api_key(handle.provider, api_key))
