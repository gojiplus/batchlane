"""Turn a list of rows into answers, without the caller writing the plumbing.

The transport layer (``submit``/``status``/``results``) is deliberately thin.
This module is the workflow on top of it: split a job to fit the provider's
caps, submit the pieces, wait, and hand back each input row beside its answer.

The design decision worth knowing: **the checkpoint stores handles, not
results.** Providers retain batch output for weeks (Anthropic 29 days, Gemini
6, Groq 30), so a resume re-attaches to the same jobs and re-reads them.
Fetching is free; inference is not. That is what makes resuming save money
rather than merely save typing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import BatchlaneError
from .handle import BatchHandle, RequestResult
from .registry import adapter_for_model, resolve_api_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .adapters.base import BatchAdapter
    from .handle import BatchLine, JobStatus

__all__ = ["ChunkPlan", "plan", "run", "wait"]

#: Headroom under a provider's byte cap. Per-line measurement is a slight
#: overestimate of marginal cost, but envelope overhead and any provider-side
#: re-encoding are not something to discover at upload time.
_SAFETY = 0.9

DEFAULT_POLL_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """How a job would be split to fit one provider's lane."""

    provider: str
    chunks: tuple[tuple[BatchLine, ...], ...]
    total_bytes: int
    limit_requests: int | None
    limit_bytes: int | None

    @property
    def n_chunks(self) -> int:
        """Number of provider jobs this would become.

        Returns:
            The chunk count.
        """
        return len(self.chunks)

    @property
    def fits_in_one(self) -> bool:
        """Whether the job can be submitted as a single batch.

        Returns:
            True if no splitting is needed.
        """
        return self.n_chunks <= 1


def _prepare(
    lines: Sequence[BatchLine],
) -> tuple[BatchAdapter, str, list[BatchLine]]:
    """Resolve one provider for a job and strip provider prefixes from models.

    Args:
        lines: The requests to run.

    Returns:
        A ``(adapter, provider, bare_lines)`` triple.

    Raises:
        BatchlaneError: If ``lines`` is empty or spans multiple providers.
    """
    from .handle import BatchLine as _BatchLine

    if not lines:
        raise BatchlaneError("Cannot submit an empty batch.")
    resolved = [adapter_for_model(line.model) for line in lines]
    providers = {provider for _a, provider, _b in resolved}
    if len(providers) > 1:
        raise BatchlaneError(
            f"A batch must target one provider; got {sorted(providers)}. "
            f"Split it and run one job per provider."
        )
    adapter, provider, _bare = resolved[0]
    bare_lines = [
        _BatchLine(line.custom_id, bare, line.messages, line.params)
        for line, (_a, _p, bare) in zip(lines, resolved, strict=True)
    ]
    return adapter, provider, bare_lines


def plan(
    lines: Sequence[BatchLine], *, endpoint: str = "chat.completions"
) -> ChunkPlan:
    """Work out how a job would be split, without submitting anything.

    Answers "will this fit?" before spending. Gemini's inline lane caps at
    20MB, an order of magnitude under the file-based providers, so a job that
    is one batch on Groq may be a dozen on Gemini.

    Args:
        lines: The requests to run.
        endpoint: Which endpoint the lines target.

    Returns:
        The chunking that :func:`run` would use.

    Raises:
        BatchlaneError: If a single line is too large to submit on its own.
    """
    adapter, provider, bare = _prepare(lines)
    caps = adapter.capabilities

    sizes = [adapter.payload_bytes([line], endpoint=endpoint) for line in bare]
    budget = int(caps.max_input_bytes * _SAFETY) if caps.max_input_bytes else None
    max_n = caps.max_requests

    for line, size in zip(bare, sizes, strict=True):
        if budget is not None and size > budget:
            raise BatchlaneError(
                f"Row {line.custom_id!r} is {size} bytes on its own, over "
                f"{provider}'s {caps.max_input_bytes}-byte cap. Shorten it; "
                f"batchlane cannot split a single request."
            )

    chunks: list[tuple[BatchLine, ...]] = []
    current: list[BatchLine] = []
    current_bytes = 0
    for line, size in zip(bare, sizes, strict=True):
        over_bytes = budget is not None and current and current_bytes + size > budget
        over_count = max_n is not None and len(current) >= max_n
        if over_bytes or over_count:
            chunks.append(tuple(current))
            current, current_bytes = [], 0
        current.append(line)
        current_bytes += size
    if current:
        chunks.append(tuple(current))

    return ChunkPlan(
        provider=provider,
        chunks=tuple(chunks),
        total_bytes=sum(sizes),
        limit_requests=max_n,
        limit_bytes=caps.max_input_bytes,
    )


def wait(
    handle: BatchHandle,
    *,
    poll_interval: float = DEFAULT_POLL_SECONDS,
    timeout: float | None = None,
    api_key: str | None = None,
) -> JobStatus:
    """Poll a job until it reaches a terminal state.

    The loop nobody should have to write twice.

    Args:
        handle: The receipt from :func:`~batchlane.submit`.
        poll_interval: Seconds between polls.
        timeout: Give up after this many seconds, or None to wait indefinitely.
        api_key: Credential, or None to read it from the environment.

    Returns:
        The terminal status.

    Raises:
        TimeoutError: If the job is still running when ``timeout`` elapses.
    """
    from . import status as _status

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        current = _status(handle, api_key=api_key)
        if current.is_terminal:
            return current
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"{handle.provider} batch {handle.job_id} still "
                f"{current.raw_state!r} after {timeout}s. The job is not lost; "
                f"poll it again with this handle."
            )
        time.sleep(poll_interval)


def _read_checkpoint(path: Path) -> dict[int, BatchHandle]:
    """Load previously submitted chunk handles.

    Args:
        path: Checkpoint file, which may not exist yet.

    Returns:
        A mapping of chunk index to its handle.
    """
    if not path.exists():
        return {}
    handles: dict[int, BatchHandle] = {}
    for raw in path.read_text().splitlines():
        if raw.strip():
            record = json.loads(raw)
            handles[record["chunk"]] = BatchHandle.from_json(
                json.dumps(record["handle"])
            )
    return handles


def _append_checkpoint(
    path: Path, index: int, handle: BatchHandle, ids: list[str]
) -> None:
    """Record a submitted chunk.

    Called immediately after submit and before anything that can fail, so a
    crash can never lose a job that has already been paid for.

    Args:
        path: Checkpoint file.
        index: Which chunk this is.
        handle: The receipt for it.
        ids: The custom_ids the chunk covers.
    """
    record = {"chunk": index, "custom_ids": ids, "handle": json.loads(handle.to_json())}
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def run(
    lines: Sequence[BatchLine],
    *,
    endpoint: str = "chat.completions",
    window: str | None = None,
    api_key: str | None = None,
    checkpoint: str | Path | None = None,
    poll_interval: float = DEFAULT_POLL_SECONDS,
    timeout: float | None = None,
) -> Iterator[tuple[BatchLine, RequestResult]]:
    """Run a job end to end and stream each input row beside its answer.

    Splits to fit the provider's caps, submits the pieces, waits, and joins
    results back to the rows that produced them. With ``checkpoint`` set, a
    crash or timeout resumes against the same provider jobs rather than
    re-running inference.

    Args:
        lines: The requests to run. All must target one provider.
        endpoint: Which endpoint the lines target.
        window: Requested turnaround, or None for the provider default.
        api_key: Credential, or None to read it from the environment.
        checkpoint: Path to record submitted jobs, enabling resume.
        poll_interval: Seconds between polls.
        timeout: Give up on a chunk after this long, or None to wait.

    Yields:
        tuple[BatchLine, RequestResult]: pairs in chunk-completion order.
            The line is the caller's own, so no joining is needed.
    """
    adapter, provider, _bare = _prepare(lines)
    key = resolve_api_key(provider, api_key)
    by_id = {line.custom_id: line for line in lines}

    chunking = plan(lines, endpoint=endpoint)
    path = Path(checkpoint) if checkpoint else None
    known = _read_checkpoint(path) if path else {}

    for index, chunk in enumerate(chunking.chunks):
        handle = known.get(index)
        if handle is None:
            handle = adapter.submit(
                list(chunk), endpoint=endpoint, window=window, api_key=key
            )
            if path is not None:
                _append_checkpoint(path, index, handle, [ln.custom_id for ln in chunk])

        final = wait(handle, poll_interval=poll_interval, timeout=timeout, api_key=key)
        yield from _pairs(adapter, handle, final, chunk, by_id, key)


def _pairs(
    adapter: BatchAdapter,
    handle: BatchHandle,
    final: JobStatus,
    chunk: tuple[BatchLine, ...],
    by_id: dict[str, BatchLine],
    api_key: str,
) -> Iterator[tuple[BatchLine, RequestResult]]:
    """Join one finished chunk's results back to the caller's rows.

    Args:
        adapter: The provider adapter.
        handle: The chunk's handle.
        final: Its terminal status.
        chunk: The lines submitted in it.
        by_id: The caller's original lines, keyed by custom_id.
        api_key: Credential for this provider.

    Yields:
        tuple[BatchLine, RequestResult]: pairs, including a synthesised
            error for any row the provider never answered -- silence is not
            success.
    """
    seen: set[str] = set()
    if final.state == "succeeded":
        for result in adapter.results(handle, api_key=api_key):
            seen.add(result.custom_id)
            line = by_id.get(result.custom_id)
            if line is not None:
                yield line, result

    for line in chunk:
        if line.custom_id not in seen:
            yield (
                by_id.get(line.custom_id, line),
                RequestResult(
                    custom_id=line.custom_id,
                    error={
                        "reason": "no result returned",
                        "job_state": final.state,
                        "provider_state": final.raw_state,
                    },
                ),
            )
