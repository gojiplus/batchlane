"""Gemini AI Studio's batch lane.

The gap this package exists to close: litellm can batch Gemini models through
Vertex AI but not through AI Studio. ``batches/main.py`` never mentions
``gemini``, and there is no ``llms/gemini/batches/`` -- only a files API with
no batch API to use it. So the same model behind the same 50% discount is
reachable with GCP credentials and unreachable with a ``GEMINI_API_KEY``.

Shape notes, all of which differ from the OpenAI-compatible providers:

* Auth is ``x-goog-api-key``, not a bearer token.
* The model lives in the **URL path** (``models/{model}:batchGenerateContent``),
  so a batch is scoped to one model. ``check()`` rejects a mixed-model batch
  via ``model_scope="url"``.
* State is nested at ``metadata.state`` with ``JOB_STATE_*`` values, resolving
  a documentation conflict with a ``BATCH_STATE_*`` enum listed elsewhere.
* Cancel uses an RPC-style ``:cancel`` suffix, not a sub-path.
* There is no caller-settable completion window.

v1 submits **inline** requests only (a documented 20MB ceiling). File input
needs the resumable File API upload and moves the join key to a different
place in the payload; both are deferred.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .._http import request
from ..capabilities import CAPABILITIES
from ..handle import BatchHandle, JobStatus, RequestResult, State, utcnow
from ..translate import decode_response, encode_body
from .base import BatchAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..handle import BatchLine

__all__ = ["GeminiAdapter"]

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_STATE_MAP: dict[str, State] = {
    "JOB_STATE_PENDING": "pending",
    "JOB_STATE_QUEUED": "pending",
    "JOB_STATE_RUNNING": "running",
    "JOB_STATE_SUCCEEDED": "succeeded",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLING": "running",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


class GeminiAdapter(BatchAdapter):
    """Batch lane for Gemini via Google AI Studio."""

    def __init__(self) -> None:
        """Bind the adapter to Gemini's capability descriptor."""
        self.capabilities = CAPABILITIES["gemini"]

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers.

        Args:
            api_key: Google AI Studio API key.

        Returns:
            Headers to send.
        """
        return {"x-goog-api-key": api_key, "content-type": "application/json"}

    def build_batch(
        self, lines: Sequence[BatchLine], *, display_name: str
    ) -> dict[str, Any]:
        """Render lines as Gemini's inline batch payload.

        Split out from :meth:`submit` so it can be tested without a network.

        Args:
            lines: The requests to run.
            display_name: A label Gemini requires on every batch.

        Returns:
            The full create-request body.
        """
        requests = [
            {
                "request": encode_body(
                    "gemini", line.model, line.messages, dict(line.params)
                ),
                "metadata": {"key": line.custom_id},
            }
            for line in lines
        ]
        return {
            "batch": {
                "display_name": display_name,
                "input_config": {"requests": {"requests": requests}},
            }
        }

    def submit(
        self,
        lines: Sequence[BatchLine],
        *,
        endpoint: str,
        window: str | None,
        api_key: str,
    ) -> BatchHandle:
        """Create the batch against the model named in the URL.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.
            window: Must be None; Gemini accepts no window parameter.
            api_key: Google AI Studio API key.

        Returns:
            A handle for the created job.
        """
        self.check(lines, endpoint=endpoint, window=window)
        model = lines[0].model
        job = request(
            "POST",
            f"{BASE_URL}/models/{model}:batchGenerateContent",
            headers=self._headers(api_key),
            json_body=self.build_batch(lines, display_name="batchlane"),
        ).json()
        return BatchHandle(
            provider="gemini",
            job_id=job["name"],
            endpoint=endpoint,
            lane="batch_inline",
            created_at=utcnow(),
            model=model,
            # Gemini's docs say inline results map to requests by array index,
            # not by the key we supply. Carry the submitted order on the handle
            # so results can still be labelled after a cold start, and so an
            # index join is at least checkable against a known length.
            extra={"keys": json.dumps([line.custom_id for line in lines])},
        )

    def status(self, handle: BatchHandle, *, api_key: str) -> JobStatus:
        """Poll the job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Google AI Studio API key.

        Returns:
            The job's normalized status.
        """
        return self.parse_status(self._fetch(handle, api_key))

    def _fetch(self, handle: BatchHandle, api_key: str) -> dict[str, Any]:
        """Fetch the raw job object.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Google AI Studio API key.

        Returns:
            The decoded operation payload.
        """
        return request(
            "GET", f"{BASE_URL}/{handle.job_id}", headers=self._headers(api_key)
        ).json()

    def parse_status(self, job: dict[str, Any]) -> JobStatus:
        """Normalize a Gemini operation payload.

        Args:
            job: The raw operation.

        Returns:
            The normalized status, retaining the provider's own string.
        """
        meta = job.get("metadata") or {}
        raw = str(meta.get("state", ""))
        stats = meta.get("batchStats") or {}

        def _int(key: str) -> int | None:
            value = stats.get(key)
            return int(value) if value is not None else None

        return JobStatus(
            state=_STATE_MAP.get(raw, "running"),
            raw_state=raw,
            total=_int("requestCount"),
            succeeded=_int("successfulRequestCount"),
            failed=_int("failedRequestCount"),
            error=(job.get("error") or {}).get("message"),
        )

    def results(self, handle: BatchHandle, *, api_key: str) -> Iterator[RequestResult]:
        """Stream the finished job's inline results.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Google AI Studio API key.

        Yields:
            One result per line, labelled with the submitted ``custom_id``.

        Raises:
            RuntimeError: If the job has no results yet, or if the number
                returned does not match the number submitted -- which would
                make an index-based join silently mislabel every row.
        """
        job = self._fetch(handle, api_key)
        inlined = ((job.get("response") or {}).get("inlinedResponses")) or []
        if not inlined:
            raise RuntimeError(
                f"Gemini batch {handle.job_id} has no inline results yet "
                f"(state={(job.get('metadata') or {}).get('state')!r})."
            )
        submitted = json.loads(handle.extra.get("keys") or "[]")
        yield from _join(submitted, inlined, model=handle.model or "")

    def cancel(self, handle: BatchHandle, *, api_key: str) -> None:
        """Cancel the running job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Google AI Studio API key.
        """
        request(
            "POST",
            f"{BASE_URL}/{handle.job_id}:cancel",
            headers=self._headers(api_key),
        )

    def list_jobs(self, *, limit: int = 20, api_key: str) -> Iterator[BatchHandle]:
        """List recent batches.

        Args:
            limit: Maximum jobs to return.
            api_key: Google AI Studio API key.

        Yields:
            A handle per batch the provider reports.
        """
        page = request(
            "GET",
            f"{BASE_URL}/batches",
            headers=self._headers(api_key),
            params={"pageSize": limit},
        ).json()
        for job in page.get("operations") or page.get("batches") or []:
            yield BatchHandle(
                provider="gemini",
                job_id=job["name"],
                endpoint="chat.completions",
                lane="batch_inline",
                created_at=utcnow(),
                model=None,
                extra={},
            )


def _echoed_key(item: dict[str, Any]) -> str | None:
    """Find a request key echoed back on an inline result, if there is one.

    Gemini's docs describe index-based matching, but the payload may still
    carry the key we supplied. Prefer it wherever it appears: a key join is
    correct regardless of ordering, an index join is not.

    Args:
        item: One entry of ``response.inlinedResponses``.

    Returns:
        The echoed key, or None if the payload carries none.
    """
    for candidate in (item.get("metadata") or {}).get("key"), item.get("key"):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _join(
    submitted: list[str], inlined: list[dict[str, Any]], *, model: str
) -> Iterator[RequestResult]:
    """Label inline results with the custom_id they belong to.

    Args:
        submitted: The custom_ids in submission order.
        inlined: The provider's ``inlinedResponses`` array.
        model: The model the batch ran against, for decoding.

    Yields:
        One labelled result per entry.

    Raises:
        RuntimeError: If no key is echoed and the counts disagree, so an index
            join cannot be trusted.
    """
    keys = [_echoed_key(item) for item in inlined]
    if all(keys):
        pairs = zip(keys, inlined, strict=True)
    else:
        # Falling back to position. Only safe if the provider returned exactly
        # what we sent; otherwise every row would be labelled with someone
        # else's answer and nothing would look wrong.
        if len(submitted) != len(inlined):
            raise RuntimeError(
                f"Gemini returned {len(inlined)} results for {len(submitted)} "
                f"requests and echoed no key, so results cannot be matched to "
                f"rows. Refusing to guess."
            )
        pairs = zip(submitted, inlined, strict=True)

    for custom_id, item in pairs:
        error = item.get("error")
        if error or "response" not in item:
            yield RequestResult(custom_id=str(custom_id), error=error or item)
            continue
        decoded = decode_response("gemini", model, item["response"])
        yield RequestResult(
            custom_id=str(custom_id), response=decoded.model_dump(), status_code=200
        )
