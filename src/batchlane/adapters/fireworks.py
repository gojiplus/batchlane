"""Fireworks' batch lane.

Structurally the furthest from OpenAI's shape of any lane here:

* Everything is account-scoped, so ``FIREWORKS_ACCOUNT_ID`` must be set. The
  docs do not say how to find it; it is the account slug in the console URL.
* Input is a **dataset**, not a file. Register it, then upload to it, then
  point a job at it. Output lands in a second dataset you name up front.
* Results are fetched through a signed-URL indirection rather than a content
  endpoint.
* **There is no cancel endpoint.** ``supports_cancel`` is False and the ABC's
  refusing default stands, so cancel raises rather than pretending.

One thing it does better than every other provider: ``batchInferenceJobId`` is
a client-settable query parameter, so the submission key becomes the job's own
id rather than a label attached to it. Resubmitting the same work collides on
the server, which is the closest thing to true idempotency in this package.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from .._http import request
from ..capabilities import CAPABILITIES
from ..errors import BatchlaneError
from ..handle import BatchHandle, JobStatus, RequestResult, State, utcnow
from ..translate import encode_body
from .base import KEY_FIELD, BatchAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import datetime

    from ..handle import BatchLine

__all__ = ["FireworksAdapter"]

BASE_URL = "https://api.fireworks.ai/v1"
ACCOUNT_ENV = "FIREWORKS_ACCOUNT_ID"

_STATE_MAP: dict[str, State] = {
    "JOB_STATE_CREATING": "pending",
    "JOB_STATE_VALIDATING": "pending",
    "JOB_STATE_PENDING": "pending",
    "JOB_STATE_RUNNING": "running",
    "JOB_STATE_WRITING_RESULTS": "running",
    "JOB_STATE_COMPLETED": "succeeded",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
    # The docs also use bare forms without the JOB_STATE_ prefix.
    "VALIDATING": "pending",
    "PENDING": "pending",
    "RUNNING": "running",
    "COMPLETED": "succeeded",
    "FAILED": "failed",
    "EXPIRED": "expired",
}

MIN_DURATION_HOURS = 12
MAX_DURATION_HOURS = 72


class FireworksAdapter(BatchAdapter):
    """Batch lane for Fireworks AI."""

    def __init__(self) -> None:
        """Bind the adapter to Fireworks' capability descriptor."""
        self.capabilities = CAPABILITIES["fireworks_ai"]

    def _account(self) -> str:
        """Read the account slug every Fireworks URL needs.

        Returns:
            The account id.

        Raises:
            BatchlaneError: If the environment variable is not set.
        """
        account = os.environ.get(ACCOUNT_ENV)
        if not account:
            raise BatchlaneError(
                f"Fireworks scopes every batch URL to an account, so {ACCOUNT_ENV} "
                f"must be set. It is the account slug shown in your Fireworks "
                f"console URL."
            )
        return account

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers.

        Args:
            api_key: Fireworks API key.

        Returns:
            Headers to send.
        """
        return {"Authorization": f"Bearer {api_key}"}

    def build_jsonl(self, lines: Sequence[BatchLine]) -> bytes:
        """Render lines as the dataset's JSONL.

        Fireworks takes ``{custom_id, body}`` with no model on the line: the
        job is bound to one deployed model.

        Args:
            lines: The requests to run.

        Returns:
            The JSONL contents.
        """
        rendered = []
        for line in lines:
            body = encode_body(
                "fireworks_ai", line.model, line.messages, dict(line.params)
            )
            body.pop("model", None)
            rendered.append(json.dumps({"custom_id": line.custom_id, "body": body}))
        return ("\n".join(rendered) + "\n").encode()

    def payload_bytes(self, lines: Sequence[BatchLine], *, endpoint: str) -> int:
        """Size of the dataset these lines would upload.

        Args:
            lines: The requests to measure.
            endpoint: Unused; the lane is chat-only.

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
        """Register a dataset, upload to it, then start a job against it.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.
            window: Turnaround, clamped to the documented [12h, 72h] range.
            api_key: Fireworks API key.
            key: Submission key, used as the job's own id.

        Returns:
            A handle for the created job.
        """
        self.check(lines, endpoint=endpoint, window=window)
        account = self._account()
        root = f"{BASE_URL}/accounts/{account}"
        headers = self._headers(api_key)

        job_id = key or f"batchlane-{utcnow():%Y%m%d%H%M%S}"
        in_id, out_id = f"{job_id}-in", f"{job_id}-out"

        for dataset_id in (in_id, out_id):
            request(
                "POST",
                f"{root}/datasets",
                headers=headers,
                json_body={"datasetId": dataset_id, "dataset": {"userUploaded": {}}},
            )
        request(
            "POST",
            f"{root}/datasets/{in_id}:upload",
            headers=headers,
            files={
                "file": (
                    "batchlane.jsonl",
                    self.build_jsonl(lines),
                    "application/jsonl",
                )
            },
        )

        body: dict[str, Any] = {
            "model": lines[0].model,
            "inputDatasetId": f"accounts/{account}/datasets/{in_id}",
            "outputDatasetId": f"accounts/{account}/datasets/{out_id}",
        }
        if window:
            body["maxJobDuration"] = f"{_duration_hours(window)}h"

        job = request(
            "POST",
            f"{root}/batchInferenceJobs",
            headers=headers,
            json_body=body,
            params={"batchInferenceJobId": job_id},
        ).json()

        return BatchHandle(
            provider="fireworks_ai",
            job_id=job.get("name", "").rsplit("/", 1)[-1] or job_id,
            endpoint=endpoint,
            lane="batch_file",
            created_at=utcnow(),
            model=lines[0].model,
            extra={"output_dataset": out_id, KEY_FIELD: job_id},
        )

    def status(self, handle: BatchHandle, *, api_key: str) -> JobStatus:
        """Poll the job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Fireworks API key.

        Returns:
            The job's normalized status.
        """
        job = request(
            "GET",
            f"{BASE_URL}/accounts/{self._account()}/batchInferenceJobs/{handle.job_id}",
            headers=self._headers(api_key),
        ).json()
        return self.parse_status(job)

    def parse_status(self, job: dict[str, Any]) -> JobStatus:
        """Normalize a Fireworks job object.

        Args:
            job: The raw job payload.

        Returns:
            The normalized status, retaining the provider's own string.
        """
        raw = str(job.get("state", ""))
        progress = job.get("jobProgress") or {}
        return JobStatus(
            state=_STATE_MAP.get(raw, "running"),
            raw_state=raw,
            total=progress.get("totalInputRequests"),
            succeeded=progress.get("successfullyProcessedRequests"),
            failed=progress.get("failedRequests"),
        )

    def results(self, handle: BatchHandle, *, api_key: str) -> Iterator[RequestResult]:
        """Stream the output dataset through its signed URLs.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Fireworks API key.

        Yields:
            RequestResult: One result per line, joined on ``custom_id``.

        Raises:
            RuntimeError: If the output dataset carries no files yet.
        """
        out_id = handle.extra.get("output_dataset")
        endpoint = request(
            "GET",
            f"{BASE_URL}/accounts/{self._account()}/datasets/{out_id}:getDownloadEndpoint",
            headers=self._headers(api_key),
        ).json()
        urls = endpoint.get("filenameToSignedUrls") or {}
        if not urls:
            raise RuntimeError(
                f"Fireworks output dataset {out_id!r} has no files yet; the job "
                f"has not finished writing results."
            )
        for url in urls.values():
            for raw_line in request("GET", url, headers={}).text.splitlines():
                if raw_line.strip():
                    yield _parse_result_line(json.loads(raw_line))

    def list_jobs(self, *, limit: int = 20, api_key: str) -> Iterator[BatchHandle]:
        """List recent jobs.

        Args:
            limit: Maximum jobs to return.
            api_key: Fireworks API key.

        Yields:
            BatchHandle: A handle per job the provider reports.
        """
        page = request(
            "GET",
            f"{BASE_URL}/accounts/{self._account()}/batchInferenceJobs",
            headers=self._headers(api_key),
            params={"pageSize": limit},
        ).json()
        for job in page.get("batchInferenceJobs") or page.get("jobs") or []:
            name = str(job.get("name", "")).rsplit("/", 1)[-1]
            yield BatchHandle(
                provider="fireworks_ai",
                job_id=name,
                endpoint="chat.completions",
                lane="batch_file",
                created_at=utcnow(),
                model=job.get("model"),
                # The job id is the submission key, so recovery matches on it
                # directly rather than on a separate label.
                extra={"output_dataset": f"{name}-out", KEY_FIELD: name},
            )

    def find_submitted(
        self,
        key: str,
        *,
        api_key: str,
        expected_rows: int,
        since: datetime,
    ) -> BatchHandle | None:
        """Find a job whose id is the submission key.

        Fireworks lets the client set the job id, so this is an exact lookup
        rather than a scan.

        Args:
            key: The submission key, which is also the job id.
            api_key: Fireworks API key.
            expected_rows: Unused; the id match is exact.
            since: Unused; the id match is exact.

        Returns:
            The handle if the job exists, otherwise None.

        Raises:
            HttpError: If the lookup fails for any reason other than the job
                not existing. A 404 is the answer "never submitted"; anything
                else is a real failure and must not be read as one.
        """
        del expected_rows, since
        from .._http import HttpError

        try:
            job = request(
                "GET",
                f"{BASE_URL}/accounts/{self._account()}/batchInferenceJobs/{key}",
                headers=self._headers(api_key),
            ).json()
        except HttpError as exc:
            if exc.status == 404:
                return None
            raise
        return BatchHandle(
            provider="fireworks_ai",
            job_id=str(job.get("name", key)).rsplit("/", 1)[-1],
            endpoint="chat.completions",
            lane="batch_file",
            created_at=utcnow(),
            model=job.get("model"),
            extra={"output_dataset": f"{key}-out", KEY_FIELD: key},
        )


def _duration_hours(window: str) -> int:
    """Clamp a window to the documented job-duration range.

    Args:
        window: A window such as ``"24h"``.

    Returns:
        Hours within [12, 72].
    """
    hours = int(window.rstrip("hH") or 24)
    return max(MIN_DURATION_HOURS, min(hours, MAX_DURATION_HOURS))


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
        response=response.get("body") or (response or None),
        error=payload.get("error"),
        status_code=response.get("status_code"),
    )
