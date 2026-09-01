"""Anthropic's Message Batches lane.

Structurally unlike the OpenAI-shaped providers, which is why it gets its own
module rather than a data row:

* There is no file upload. Requests go inline under a ``requests`` key on the
  create call, so there is no two-step upload-then-reference dance.
* ``processing_status`` has only two useful values, ``in_progress`` and
  ``ended``. Whether an ended job actually succeeded lives in
  ``request_counts``, not in the status string.
* ``results_url`` is null until the job ends, and results stream from it as
  JSONL rather than being fetched by an output file id.
* Results arrive out of submission order, so ``custom_id`` is the only valid
  join. This is observed behavior, not a docs claim.

LiteLLM can retrieve an Anthropic batch but cannot create one -- it raises
``BadRequestError: LiteLLM doesn't support custom_llm_provider=anthropic for
'create_batch'`` -- so this lane is unreachable through the gateway.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from .._http import request
from ..capabilities import CAPABILITIES
from ..errors import BatchlaneError
from ..handle import BatchHandle, JobStatus, RequestResult, State, utcnow
from ..translate import decode_response, encode_body
from .base import BatchAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..handle import BatchLine

__all__ = ["AnthropicAdapter"]

BASE_URL = "https://api.anthropic.com/v1/messages/batches"
API_VERSION = "2023-06-01"

#: Anthropic requires max_tokens on every request. Left implicit it is a 400,
#: so supply a floor rather than let a whole batch bounce on one missing field.
DEFAULT_MAX_TOKENS = 1024


class AnthropicAdapter(BatchAdapter):
    """Batch lane for Anthropic's Message Batches API."""

    #: Anthropic's create body accepts only ``requests``: no label, no
    #: metadata, nothing to stamp. See find_submitted for what replaces it.
    stamps_key: ClassVar[bool] = False

    def __init__(self) -> None:
        """Bind the adapter to Anthropic's capability descriptor."""
        self.capabilities = CAPABILITIES["anthropic"]

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers.

        Args:
            api_key: Anthropic API key.

        Returns:
            Headers to send.
        """
        return {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def build_requests(self, lines: Sequence[BatchLine]) -> list[dict[str, Any]]:
        """Render batch lines as Anthropic's inline ``requests`` array.

        Split out from :meth:`submit` so it can be tested without a network.

        Args:
            lines: The requests to run.

        Returns:
            One ``{custom_id, params}`` object per line.
        """
        built = []
        for line in lines:
            params = dict(line.params)
            params.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
            built.append(
                {
                    "custom_id": line.custom_id,
                    "params": encode_body(
                        "anthropic", line.model, line.messages, params
                    ),
                }
            )
        return built

    def payload_bytes(self, lines: Sequence[BatchLine], *, endpoint: str) -> int:
        """Size of the inline create-request body these lines would send.

        Args:
            lines: The requests to measure.
            endpoint: Unused; Anthropic's lane is chat-only.

        Returns:
            Size in bytes.
        """
        del endpoint
        return len(json.dumps({"requests": self.build_requests(lines)}).encode())

    def submit(
        self,
        lines: Sequence[BatchLine],
        *,
        endpoint: str,
        window: str | None,
        api_key: str,
        key: str | None = None,
    ) -> BatchHandle:
        """Create the batch. No upload step -- requests go inline.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.
            window: Must be None; Anthropic accepts no window parameter.
            api_key: Anthropic API key.
            key: Accepted for interface parity and deliberately unused. The
                create body takes only ``requests`` -- there is no label,
                metadata or identifier field to stamp it on -- so recovery
                goes through :meth:`find_submitted` instead.

        Returns:
            A handle for the created job.
        """
        del key  # nowhere to put it; see find_submitted
        self.check(lines, endpoint=endpoint, window=window)
        job = request(
            "POST",
            BASE_URL,
            headers=self._headers(api_key),
            json_body={"requests": self.build_requests(lines)},
        ).json()
        return BatchHandle(
            provider="anthropic",
            job_id=job["id"],
            endpoint=endpoint,
            lane="batch_inline",
            created_at=utcnow(),
            # Model is per-line here, so the job carries none.
            model=None,
            extra={},
        )

    def find_submitted(
        self,
        key: str,
        *,
        api_key: str,
        expected_rows: int,
        since: datetime,
    ) -> BatchHandle | None:
        """Find a possibly-submitted batch, without a key to match on.

        Anthropic's create body accepts only ``requests``, so there is no
        stamp to look for. The available signals are creation time and row
        count, which together are suggestive but not unique. When more than
        one batch fits, this raises rather than choosing: adopting the wrong
        job and paying twice are both worse than an error naming the
        ambiguity.

        Args:
            key: The submission key, unused here beyond the error message.
            api_key: Anthropic API key.
            expected_rows: How many requests the chunk held.
            since: When the submission was attempted.

        Returns:
            The single matching handle, or None if nothing matches.

        Raises:
            BatchlaneError: If several batches match and none can be ruled out.
        """
        candidates = []
        page = request(
            "GET", BASE_URL, headers=self._headers(api_key), params={"limit": 100}
        ).json()
        for job in page.get("data") or []:
            counts = job.get("request_counts") or {}
            total = sum(
                counts.get(k) or 0
                for k in ("processing", "succeeded", "errored", "canceled", "expired")
            )
            created = job.get("created_at")
            if total != expected_rows or not created:
                continue
            if datetime.fromisoformat(created) >= since:
                candidates.append(job["id"])

        if not candidates:
            return None
        if len(candidates) > 1:
            raise BatchlaneError(
                f"Anthropic gives a batch no field to label, so a submission "
                f"interrupted before it was recorded cannot be identified "
                f"exactly. {len(candidates)} batches created after "
                f"{since:%Y-%m-%d %H:%M:%S} hold {expected_rows} requests: "
                f"{', '.join(candidates)}. "
                f"Collect the right one with batchlane.status() on its id, or "
                f"cancel the strays, then rerun. Refusing to guess (key={key})."
            )
        return BatchHandle(
            provider="anthropic",
            job_id=candidates[0],
            endpoint="chat.completions",
            lane="batch_inline",
            created_at=utcnow(),
            model=None,
            extra={},
        )

    def status(self, handle: BatchHandle, *, api_key: str) -> JobStatus:
        """Poll the job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Anthropic API key.

        Returns:
            The job's normalized status.
        """
        return self.parse_status(
            request(
                "GET", f"{BASE_URL}/{handle.job_id}", headers=self._headers(api_key)
            ).json()
        )

    def parse_status(self, job: dict[str, Any]) -> JobStatus:
        """Normalize an Anthropic batch object.

        ``processing_status`` alone is not enough: ``ended`` covers a job that
        succeeded, one that was cancelled and one that expired. The counts
        disambiguate, so they are what this reads.

        Args:
            job: The raw batch payload.

        Returns:
            The normalized status, retaining the provider's own string.
        """
        raw = str(job.get("processing_status", ""))
        counts = job.get("request_counts") or {}
        succeeded = counts.get("succeeded")
        errored = counts.get("errored") or 0
        canceled = counts.get("canceled") or 0
        expired = counts.get("expired") or 0
        total = sum(
            counts.get(k) or 0
            for k in ("processing", "succeeded", "errored", "canceled", "expired")
        )

        state: State
        if raw != "ended":
            state = "running"
        elif expired and expired == total:
            state = "expired"
        elif canceled and not succeeded:
            state = "cancelled"
        elif succeeded:
            # Per-line failures do not fail the job, matching OpenAI's semantics.
            state = "succeeded"
        else:
            state = "failed"

        return JobStatus(
            state=state,
            raw_state=raw,
            total=total or None,
            succeeded=succeeded,
            failed=errored + canceled + expired or None,
        )

    def results(self, handle: BatchHandle, *, api_key: str) -> Iterator[RequestResult]:
        """Stream the ended job's results.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Anthropic API key.

        Yields:
            RequestResult: One result per line, joined on ``custom_id``.

        Raises:
            RuntimeError: If the job has not ended, so no results exist yet.
        """
        job = request(
            "GET", f"{BASE_URL}/{handle.job_id}", headers=self._headers(api_key)
        ).json()
        results_url = job.get("results_url")
        if not results_url:
            raise RuntimeError(
                f"Anthropic batch {handle.job_id} has no results yet "
                f"(processing_status={job.get('processing_status')!r}); "
                f"results_url stays null until the job ends."
            )
        text = request("GET", results_url, headers=self._headers(api_key)).text
        for raw_line in text.splitlines():
            if raw_line.strip():
                yield _parse_result_line(json.loads(raw_line))

    def list_jobs(self, *, limit: int = 20, api_key: str) -> Iterator[BatchHandle]:
        """List recent batches.

        Args:
            limit: Maximum jobs to return.
            api_key: Anthropic API key.

        Yields:
            BatchHandle: A handle per batch the provider reports.
        """
        page = request(
            "GET", BASE_URL, headers=self._headers(api_key), params={"limit": limit}
        ).json()
        for job in page.get("data") or []:
            yield BatchHandle(
                provider="anthropic",
                job_id=job["id"],
                endpoint="chat.completions",
                lane="batch_inline",
                created_at=utcnow(),
                model=None,
                extra={},
            )

    def cancel(self, handle: BatchHandle, *, api_key: str) -> None:
        """Cancel the running job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Anthropic API key.
        """
        request(
            "POST",
            f"{BASE_URL}/{handle.job_id}/cancel",
            headers=self._headers(api_key),
        )


def _parse_result_line(payload: dict[str, Any]) -> RequestResult:
    """Turn one results-JSONL line into a result.

    Args:
        payload: The decoded line, ``{custom_id, result: {type, ...}}``.

    Returns:
        The result, with a succeeded message decoded to OpenAI shape.
    """
    result = payload.get("result") or {}
    custom_id = payload.get("custom_id", "")
    if result.get("type") != "succeeded":
        return RequestResult(custom_id=custom_id, error=result)
    message = result.get("message") or {}
    decoded = decode_response("anthropic", message.get("model", ""), message)
    return RequestResult(
        custom_id=custom_id, response=decoded.model_dump(), status_code=200
    )
