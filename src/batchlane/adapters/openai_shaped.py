"""One client for the providers that copied OpenAI's files-plus-batches shape.

Groq, Together and DeepInfra all upload a JSONL file with a purpose, create a
job referencing the file id, poll for a status, and hand back an output file id
whose lines join on ``custom_id``. Their differences are small enough to be
data, so each is a :class:`ProviderRow` rather than a subclass.

Everything else -- Gemini, Mistral, Fireworks -- differs structurally and gets
its own module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._http import request
from ..capabilities import CAPABILITIES
from ..handle import BatchHandle, BatchLine, JobStatus, RequestResult, State, utcnow
from ..translate import encode_body
from .base import BatchAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["ROWS", "OpenAIShapedAdapter", "ProviderRow"]

#: OpenAI's batch status vocabulary, normalized. Together uppercases its own
#: values, so lookups are done case-insensitively.
_STATE_MAP: dict[str, State] = {
    "validating": "pending",
    "queued": "pending",
    "in_progress": "running",
    "finalizing": "running",
    "cancelling": "running",
    "completed": "succeeded",
    "failed": "failed",
    "expired": "expired",
    "cancelled": "cancelled",
}


@dataclass(frozen=True, slots=True)
class ProviderRow:
    """Everything that differs between the OpenAI-shaped providers."""

    provider: str
    base_url: str
    api_key_env: str
    upload_purpose: str = "batch"
    files_path: str = "/files"
    batches_path: str = "/batches"
    #: Together reports a single progress float instead of request_counts.
    counts_key: str | None = "request_counts"


ROWS: dict[str, ProviderRow] = {
    "openai": ProviderRow(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "groq": ProviderRow(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "together_ai": ProviderRow(
        provider="together_ai",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        # Not the OpenAI-standard "batch"; Together rejects that value.
        upload_purpose="batch-api",
        files_path="/files/upload",
        counts_key=None,
    ),
    "deepinfra": ProviderRow(
        provider="deepinfra",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_TOKEN",
    ),
}

_ENDPOINT_PATHS = {
    "chat.completions": "/v1/chat/completions",
    "embeddings": "/v1/embeddings",
}


class OpenAIShapedAdapter(BatchAdapter):
    """Batch lane for a provider that follows OpenAI's files-plus-batches shape."""

    def __init__(self, row: ProviderRow) -> None:
        """Bind the adapter to one provider row.

        Args:
            row: The per-provider data describing paths and quirks.
        """
        self.row = row
        self.capabilities = CAPABILITIES[row.provider]

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers.

        Args:
            api_key: Credential for this provider.

        Returns:
            Headers to send.
        """
        return {"Authorization": f"Bearer {api_key}"}

    def build_jsonl(self, lines: Sequence[BatchLine], *, endpoint: str) -> bytes:
        """Render batch lines as an OpenAI-format JSONL payload.

        Split out from :meth:`submit` so it can be tested without any network.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.

        Returns:
            The JSONL file contents.
        """
        url = _ENDPOINT_PATHS[endpoint]
        rendered = []
        for line in lines:
            body = encode_body(
                self.row.provider, line.model, line.messages, dict(line.params)
            )
            rendered.append(
                json.dumps(
                    {
                        "custom_id": line.custom_id,
                        "method": "POST",
                        "url": url,
                        "body": body,
                    }
                )
            )
        return ("\n".join(rendered) + "\n").encode()

    def payload_bytes(self, lines: Sequence[BatchLine], *, endpoint: str) -> int:
        """Size of the JSONL file these lines would upload.

        Args:
            lines: The requests to measure.
            endpoint: Which endpoint the lines target.

        Returns:
            Size in bytes.
        """
        return len(self.build_jsonl(lines, endpoint=endpoint))

    def submit(
        self,
        lines: Sequence[BatchLine],
        *,
        endpoint: str,
        window: str | None,
        api_key: str,
    ) -> BatchHandle:
        """Upload the JSONL and create the batch job.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.
            window: Requested turnaround, or None for the provider default.
            api_key: Credential for this provider.

        Returns:
            A handle for the created job.
        """
        self.check(lines, endpoint=endpoint, window=window)
        payload = self.build_jsonl(lines, endpoint=endpoint)

        upload = request(
            "POST",
            f"{self.row.base_url}{self.row.files_path}",
            headers=self._headers(api_key),
            files={"file": ("batchlane.jsonl", payload, "application/jsonl")},
            data={"purpose": self.row.upload_purpose},
        ).json()

        caps = self.capabilities
        body: dict[str, Any] = {
            "input_file_id": upload["id"],
            "endpoint": _ENDPOINT_PATHS[endpoint],
            "completion_window": window
            or (caps.window.default if caps.window else "24h"),
        }
        job = request(
            "POST",
            f"{self.row.base_url}{self.row.batches_path}",
            headers=self._headers(api_key),
            json_body=body,
        ).json()

        return BatchHandle(
            provider=self.row.provider,
            job_id=job["id"],
            endpoint=endpoint,
            lane="batch_file",
            created_at=utcnow(),
            model=None,
            extra={"input_file_id": upload["id"]},
        )

    def status(self, handle: BatchHandle, *, api_key: str) -> JobStatus:
        """Poll the job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Credential for this provider.

        Returns:
            The job's normalized status.
        """
        job = request(
            "GET",
            f"{self.row.base_url}{self.row.batches_path}/{handle.job_id}",
            headers=self._headers(api_key),
        ).json()
        return self.parse_status(job)

    def parse_status(self, job: dict[str, Any]) -> JobStatus:
        """Normalize a provider job object.

        Args:
            job: The raw job payload.

        Returns:
            The normalized status, retaining the provider's own string.
        """
        raw = str(job.get("status", ""))
        counts = job.get(self.row.counts_key) or {} if self.row.counts_key else {}
        return JobStatus(
            state=_STATE_MAP.get(raw.lower(), "running"),
            raw_state=raw,
            total=counts.get("total"),
            succeeded=counts.get("completed"),
            failed=counts.get("failed"),
            error=_first_error(job),
        )

    def results(self, handle: BatchHandle, *, api_key: str) -> Iterator[RequestResult]:
        """Stream the completed job's output file.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Credential for this provider.

        Yields:
            RequestResult: One result per line, joined on ``custom_id``.
        """
        job = request(
            "GET",
            f"{self.row.base_url}{self.row.batches_path}/{handle.job_id}",
            headers=self._headers(api_key),
        ).json()
        for key in ("output_file_id", "error_file_id"):
            file_id = job.get(key)
            if not file_id:
                continue
            text = request(
                "GET",
                f"{self.row.base_url}/files/{file_id}/content",
                headers=self._headers(api_key),
            ).text
            for raw_line in text.splitlines():
                if raw_line.strip():
                    yield _parse_result_line(json.loads(raw_line))

    def list_jobs(self, *, limit: int = 20, api_key: str) -> Iterator[BatchHandle]:
        """List recent jobs.

        Args:
            limit: Maximum jobs to return.
            api_key: Credential for this provider.

        Yields:
            BatchHandle: A handle per job the provider reports.
        """
        if not self.capabilities.supports_list:
            yield from BatchAdapter.list_jobs(self, limit=limit, api_key=api_key)
            return
        page = request(
            "GET",
            f"{self.row.base_url}{self.row.batches_path}",
            headers=self._headers(api_key),
            params={"limit": limit},
        ).json()
        for job in page.get("data") or []:
            yield BatchHandle(
                provider=self.row.provider,
                job_id=job["id"],
                endpoint=job.get("endpoint") or "chat.completions",
                lane="batch_file",
                created_at=utcnow(),
                model=None,
                extra={"input_file_id": job.get("input_file_id") or ""},
            )

    def cancel(self, handle: BatchHandle, *, api_key: str) -> None:
        """Cancel the running job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Credential for this provider.
        """
        request(
            "POST",
            f"{self.row.base_url}{self.row.batches_path}/{handle.job_id}/cancel",
            headers=self._headers(api_key),
        )


def _first_error(job: dict[str, Any]) -> str | None:
    """Extract a human-readable error from a job payload.

    Args:
        job: The raw job payload.

    Returns:
        The first error message, or None.
    """
    errors = (job.get("errors") or {}).get("data") or []
    if errors:
        return str(errors[0].get("message"))
    return None


def _parse_result_line(payload: dict[str, Any]) -> RequestResult:
    """Turn one output-file line into a result.

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
