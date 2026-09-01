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

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import BatchlaneError
from .handle import BatchHandle, BatchLine, RequestResult
from .registry import adapter_for_model, resolve_api_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from .adapters.base import BatchAdapter
    from .handle import JobStatus

__all__ = ["ChunkPlan", "answer_text", "map", "plan", "run", "wait"]

#: Headroom under a provider's byte cap. Per-line measurement is a slight
#: overestimate of marginal cost, but envelope overhead and any provider-side
#: re-encoding are not something to discover at upload time.
_SAFETY = 0.9

DEFAULT_POLL_SECONDS = 30.0

#: Batch jobs run for hours, and nothing changes in the first few minutes, so
#: a fixed interval spends thousands of polls learning nothing. Escalate and
#: cap. A six-hour job costs about 30 polls here instead of 720.
_BACKOFF_CAP_SECONDS = 900.0
_BACKOFF_FACTOR = 2.0


def _chunk_key(lines: Sequence[BatchLine], index: int) -> str:
    """Derive a stable submission key from a chunk's content.

    Deterministic, so resubmitting the same work produces the same key and a
    job left behind by a crash can be recognised. No provider offers an
    idempotency key for batch, so this is the client-side stand-in.

    Args:
        lines: The chunk's requests.
        index: Which chunk this is within the job.

    Returns:
        A key short enough for a provider label field.
    """
    material = "|".join(f"{line.custom_id}:{line.model}" for line in lines)
    digest = hashlib.sha256(material.encode()).hexdigest()[:16]
    return f"bl-{digest}-{index}"


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
        poll_interval: Seconds before the first re-poll. The interval then
            doubles up to a fifteen-minute cap, since a job measured in hours
            reveals nothing by being asked every thirty seconds.
        timeout: Give up after this many seconds, or None to wait indefinitely.
        api_key: Credential, or None to read it from the environment.

    Returns:
        The terminal status.

    Raises:
        TimeoutError: If the job is still running when ``timeout`` elapses.
    """
    from . import status as _status

    deadline = None if timeout is None else time.monotonic() + timeout
    interval = poll_interval
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
        time.sleep(interval)
        interval = min(interval * _BACKOFF_FACTOR, _BACKOFF_CAP_SECONDS)


def _read_checkpoint(
    path: Path,
) -> tuple[dict[int, BatchHandle], dict[int, tuple[str, datetime]]]:
    """Load recorded handles and any unfinished submission intents.

    Args:
        path: Checkpoint file, which may not exist yet.

    Returns:
        A ``(handles, intents)`` pair keyed by chunk index. An intent with no
        matching handle means a submission may have reached the provider
        before the process stopped, so that chunk must be looked for rather
        than resubmitted.
    """
    if not path.exists():
        return {}, {}
    handles: dict[int, BatchHandle] = {}
    intents: dict[int, tuple[str, datetime]] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        if "handle" in record:
            handles[record["chunk"]] = BatchHandle.from_json(
                json.dumps(record["handle"])
            )
        elif "key" in record:
            intents[record["chunk"]] = (
                record["key"],
                datetime.fromisoformat(record["at"]),
            )
    return handles, {i: v for i, v in intents.items() if i not in handles}


def _append_intent(path: Path, index: int, key: str) -> None:
    """Record that a chunk is about to be submitted.

    Written and fsynced *before* the provider is called. No provider offers an
    idempotency key for batch, so this record is the only evidence that a job
    may exist after a crash mid-submission.

    Args:
        path: Checkpoint file.
        index: Which chunk this is.
        key: The submission key stamped on the provider's job.
    """
    record = {"chunk": index, "key": key, "at": datetime.now(UTC).isoformat()}
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


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
    known, intents = _read_checkpoint(path) if path else ({}, {})

    for index, chunk in enumerate(chunking.chunks):
        handle = known.get(index)
        if handle is None:
            chunk_key = _chunk_key(chunk, index)

            # An intent with no handle means the previous run may have
            # submitted this chunk and stopped before recording it. The
            # provider has no idempotency key to dedupe on, so look for the
            # job before spending on it again.
            prior = intents.get(index)
            if prior is not None:
                handle = adapter.find_submitted(
                    prior[0],
                    api_key=key,
                    expected_rows=len(chunk),
                    since=prior[1],
                )
                if handle is not None and path is not None:
                    _append_checkpoint(
                        path, index, handle, [ln.custom_id for ln in chunk]
                    )

            if handle is None:
                if path is not None:
                    _append_intent(path, index, chunk_key)
                handle = adapter.submit(
                    list(chunk),
                    endpoint=endpoint,
                    window=window,
                    api_key=key,
                    key=chunk_key,
                )
                if path is not None:
                    _append_checkpoint(
                        path, index, handle, [ln.custom_id for ln in chunk]
                    )

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


def answer_text(result: RequestResult) -> str | None:
    """Pull the assistant's text out of a result, if there is one.

    Every adapter normalizes to OpenAI chat-completion shape, so one accessor
    works across providers.

    Args:
        result: One row's outcome.

    Returns:
        The message content, or None if the row failed or carried no text.
    """
    if not result.ok or not isinstance(result.response, dict):
        return None
    choices = result.response.get("choices") or []
    if not choices:
        return None
    return (choices[0].get("message") or {}).get("content")


def map(  # noqa: A001 - deliberately mirrors the builtin's shape
    model: str,
    prompts: Iterable[str],
    *,
    system: str | None = None,
    checkpoint: str | Path | None = None,
    poll_interval: float = DEFAULT_POLL_SECONDS,
    timeout: float | None = None,
    api_key: str | None = None,
    **params: object,
) -> list[str | None]:
    """Run one prompt over many inputs and get the answers back in order.

    The common case, without building request objects by hand:

        answers = batchlane.map("groq/llama-3.3-70b-versatile", prompts)

    A thin wrapper over :func:`run`, so it inherits chunking to provider caps,
    resumption from a checkpoint, and joining results back to their rows.

    Args:
        model: A provider-prefixed model, e.g. ``"groq/llama-3.3-70b-versatile"``.
        prompts: The user prompts, one per row.
        system: An optional system prompt applied to every row.
        checkpoint: Path to record submitted jobs, enabling resume.
        poll_interval: Seconds between polls.
        timeout: Give up on a chunk after this long, or None to wait.
        api_key: Credential, or None to read it from the environment.
        **params: Passed through to the model, e.g. ``max_tokens=64``.

    Returns:
        One answer per prompt, in input order. An entry is None where that row
        failed, so the list always lines up with the input.
    """
    rows = list(prompts)
    lines = []
    for index, prompt in enumerate(rows):
        messages: list[dict[str, object]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        lines.append(BatchLine(f"row-{index}", model, messages, dict(params)))

    answers: dict[str, str | None] = {}
    for line, result in run(
        lines,
        checkpoint=checkpoint,
        poll_interval=poll_interval,
        timeout=timeout,
        api_key=api_key,
    ):
        answers[line.custom_id] = answer_text(result)
    return [answers.get(f"row-{i}") for i in range(len(rows))]
