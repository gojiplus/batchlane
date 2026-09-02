"""Mistral's batch lane.

Close to the OpenAI-shaped providers in spirit and different in three ways
that matter:

* The path is ``/v1/batch/jobs``, not ``/v1/batches``.
* The **model lives on the job**, not on each line, so a batch cannot mix
  models. ``check()`` already refuses that via ``model_scope="job"``.
* Turnaround is ``timeout_hours``, a bare integer, rather than a window enum.

It does accept a ``metadata`` map, so a submission key can be stamped and a
job orphaned by a crash found again.

Shape verified against docs.mistral.ai/api/endpoint/batch. Not yet exercised
against a live API.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .._http import request
from ..capabilities import CAPABILITIES
from ..handle import BatchHandle, JobStatus, RequestResult, State, utcnow
from ..translate import encode_body
from .base import KEY_FIELD, BatchAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..handle import BatchLine

__all__ = ["MistralAdapter"]

BASE_URL = "https://api.mistral.ai/v1"
JOBS_PATH = "/batch/jobs"

_STATE_MAP: dict[str, State] = {
    "QUEUED": "pending",
    "RUNNING": "running",
    "CANCELLATION_REQUESTED": "running",
    "SUCCESS": "succeeded",
    "FAILED": "failed",
    "TIMEOUT_EXCEEDED": "expired",
    "CANCELLED": "cancelled",
}

DEFAULT_TIMEOUT_HOURS = 24


def _timeout_hours(window: str | None) -> int:
    """Turn a window string into Mistral's integer hours.

    Args:
        window: A window such as ``"24h"``, or None for the default.

    Returns:
        Hours as an integer.
    """
    if not window:
        return DEFAULT_TIMEOUT_HOURS
    return int(window.rstrip("hH") or DEFAULT_TIMEOUT_HOURS)


class MistralAdapter(BatchAdapter):
    """Batch lane for Mistral's La Plateforme."""

    def __init__(self) -> None:
        """Bind the adapter to Mistral's capability descriptor."""
        self.capabilities = CAPABILITIES["mistral"]

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers.

        Args:
            api_key: Mistral API key.

        Returns:
            Headers to send.
        """
        return {"Authorization": f"Bearer {api_key}"}

    def build_jsonl(self, lines: Sequence[BatchLine]) -> bytes:
        """Render lines as Mistral's input JSONL.

        Each line is ``{custom_id, body}``. The model stays in the body even
        though the job also carries it: Mistral treats the per-line model as
        optional rather than forbidden, and leaving it makes the file readable
        on its own.

        Args:
            lines: The requests to run.

        Returns:
            The JSONL file contents.
        """
        rendered = [
            json.dumps(
                {
                    "custom_id": line.custom_id,
                    "body": encode_body(
                        "mistral", line.model, line.messages, dict(line.params)
                    ),
                }
            )
            for line in lines
        ]
        return ("\n".join(rendered) + "\n").encode()

    def payload_bytes(self, lines: Sequence[BatchLine], *, endpoint: str) -> int:
        """Size of the JSONL these lines would upload.

        Args:
            lines: The requests to measure.
            endpoint: Unused; the lane is chat-only here.

        Returns:
            Size in bytes.
        """
        del endpoint
        return len(self.build_jsonl(lines))

    def submit(
        self,
        lines: Sequence[BatchLine],
        *,
        endpoint: str,
        window: str | None,
        api_key: str,
        key: str | None = None,
    ) -> BatchHandle:
        """Upload the JSONL, then create a job scoped to one model.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.
            window: Turnaround, converted to ``timeout_hours``.
            api_key: Mistral API key.
            key: Submission key, stamped into the job's metadata.

        Returns:
            A handle for the created job.
        """
        self.check(lines, endpoint=endpoint, window=window)
        model = lines[0].model

        upload = request(
            "POST",
            f"{BASE_URL}/files",
            headers=self._headers(api_key),
            files={
                "file": (
                    "batchlane.jsonl",
                    self.build_jsonl(lines),
                    "application/jsonl",
                )
            },
            data={"purpose": "batch"},
        ).json()

        body: dict[str, Any] = {
            "endpoint": "/v1/chat/completions",
            "model": model,
            "input_files": [upload["id"]],
            "timeout_hours": _timeout_hours(window),
        }
        if key is not None:
            body["metadata"] = {KEY_FIELD: key}

        job = request(
            "POST",
            f"{BASE_URL}{JOBS_PATH}",
            headers=self._headers(api_key),
            json_body=body,
        ).json()

        return BatchHandle(
            provider="mistral",
            job_id=job["id"],
            endpoint=endpoint,
            lane="batch_file",
            created_at=utcnow(),
            model=model,
            extra=(
                {"input_file_id": upload["id"], KEY_FIELD: key}
                if key is not None
                else {"input_file_id": upload["id"]}
            ),
        )

    def status(self, handle: BatchHandle, *, api_key: str) -> JobStatus:
        """Poll the job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Mistral API key.

        Returns:
            The job's normalized status.
        """
        return self.parse_status(self._fetch(handle, api_key))

    def _fetch(self, handle: BatchHandle, api_key: str) -> dict[str, Any]:
        """Fetch the raw job object.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Mistral API key.

        Returns:
            The decoded job payload.
        """
        return request(
            "GET",
            f"{BASE_URL}{JOBS_PATH}/{handle.job_id}",
            headers=self._headers(api_key),
        ).json()

    def parse_status(self, job: dict[str, Any]) -> JobStatus:
        """Normalize a Mistral job object.

        Args:
            job: The raw job payload.

        Returns:
            The normalized status, retaining the provider's own string.
        """
        raw = str(job.get("status", ""))
        return JobStatus(
            state=_STATE_MAP.get(raw, "running"),
            raw_state=raw,
            total=job.get("total_requests"),
            succeeded=job.get("succeeded_requests"),
            failed=job.get("failed_requests"),
        )

    def results(self, handle: BatchHandle, *, api_key: str) -> Iterator[RequestResult]:
        """Stream the finished job's output and error files.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Mistral API key.

        Yields:
            RequestResult: One result per line, joined on ``custom_id``.
        """
        job = self._fetch(handle, api_key)
        for field in ("output_file", "error_file"):
            file_id = job.get(field)
            if not file_id:
                continue
            text = request(
                "GET",
                f"{BASE_URL}/files/{file_id}/content",
                headers=self._headers(api_key),
            ).text
            for raw_line in text.splitlines():
                if raw_line.strip():
                    yield _parse_result_line(json.loads(raw_line))

    def cancel(self, handle: BatchHandle, *, api_key: str) -> None:
        """Cancel the running job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Mistral API key.
        """
        request(
            "POST",
            f"{BASE_URL}{JOBS_PATH}/{handle.job_id}/cancel",
            headers=self._headers(api_key),
        )

    def list_jobs(self, *, limit: int = 20, api_key: str) -> Iterator[BatchHandle]:
        """List recent jobs.

        Args:
            limit: Maximum jobs to return.
            api_key: Mistral API key.

        Yields:
            BatchHandle: A handle per job the provider reports.
        """
        page = request(
            "GET",
            f"{BASE_URL}{JOBS_PATH}",
            headers=self._headers(api_key),
            params={"page_size": limit},
        ).json()
        for job in page.get("data") or []:
            yield BatchHandle(
                provider="mistral",
                job_id=job["id"],
                endpoint="chat.completions",
                lane="batch_file",
                created_at=utcnow(),
                model=job.get("model"),
                extra={KEY_FIELD: (job.get("metadata") or {}).get(KEY_FIELD, "")},
            )


def _parse_result_line(payload: dict[str, Any]) -> RequestResult:
    """Turn one output line into a result.

    Args:
        payload: The decoded JSONL line.

    Returns:
        The result, joined on ``custom_id``.
    """
    response = payload.get("response") or {}
    return RequestResult(
        custom_id=payload.get("custom_id", ""),
        response=response.get("body"),
        error=payload.get("error") or response.get("error"),
        status_code=response.get("status_code"),
    )
