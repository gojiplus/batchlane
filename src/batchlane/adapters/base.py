"""The adapter contract every provider lane implements.

Deliberately not a subclass of LiteLLM's ``BaseBatchesConfig``. That ABC is
file-shaped by construction -- it transforms around an ``input_file_id`` and a
``completion_window: Literal["24h"]`` -- and it declares no cancel or list
method at all. Four of our six providers would have to fabricate fields to fit
it, and LiteLLM's dispatcher never routes to it for cancel or list regardless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..errors import CapabilityNotSupportedError, MixedModelBatchError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..capabilities import LaneCapabilities
    from ..handle import BatchHandle, BatchLine, JobStatus, RequestResult

__all__ = ["BatchAdapter"]


class BatchAdapter(ABC):
    """One provider's asynchronous batch lane."""

    #: Set by each adapter; instance-level because one class can serve
    #: several providers via a data row.
    capabilities: LaneCapabilities

    @abstractmethod
    def submit(
        self,
        lines: Sequence[BatchLine],
        *,
        endpoint: str,
        window: str | None,
        api_key: str,
    ) -> BatchHandle:
        """Submit a batch and return a receipt.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.
            window: Requested turnaround, or None for the provider default.
            api_key: Credential for this provider.

        Returns:
            A handle that can poll and collect the job.
        """

    @abstractmethod
    def status(self, handle: BatchHandle, *, api_key: str) -> JobStatus:
        """Poll a submitted job.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Credential for this provider.

        Returns:
            The job's normalized status.
        """

    @abstractmethod
    def results(self, handle: BatchHandle, *, api_key: str) -> Iterator[RequestResult]:
        """Stream a completed job's results.

        Yields rather than returns so a 50,000-line output file does not have
        to be held in memory.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Credential for this provider.

        Yields:
            One result per submitted line, joined on ``custom_id``.
        """

    def cancel(self, handle: BatchHandle, *, api_key: str) -> None:
        """Cancel a running job.

        The default refuses. A provider with no cancel endpoint (Fireworks)
        therefore needs no special case and cannot silently no-op.

        Args:
            handle: The receipt from :meth:`submit`.
            api_key: Credential for this provider.

        Raises:
            CapabilityNotSupportedError: Always, unless a subclass overrides this.
        """
        del handle, api_key  # the default refuses; overriding subclasses use them
        raise CapabilityNotSupportedError(
            self.capabilities.provider,
            "cancel",
            "no cancel endpoint is documented for this provider",
        )

    def list_jobs(self, *, limit: int = 20, api_key: str) -> Iterator[BatchHandle]:
        """List recent jobs on this provider.

        The default refuses, so a provider with no documented list endpoint
        (DeepInfra) needs no special case and cannot return a silent empty
        list, which would read as "you have no jobs".

        Args:
            limit: Maximum jobs to return.
            api_key: Credential for this provider.

        Yields:
            One handle per job the provider reports.

        Raises:
            CapabilityNotSupportedError: Always, unless a subclass overrides.
        """
        del limit, api_key  # the default refuses; overriding subclasses use them
        raise CapabilityNotSupportedError(
            self.capabilities.provider,
            "list",
            "no list-batches endpoint is documented for this provider",
        )
        yield  # pragma: no cover - unreachable; marks this a generator

    def check(
        self, lines: Sequence[BatchLine], *, endpoint: str, window: str | None
    ) -> None:
        """Validate a submission against the capability descriptor.

        Called by every ``submit`` before any network I/O, so a request the
        provider would reject costs nothing and fails with a precise reason.

        Args:
            lines: The requests to run.
            endpoint: Which endpoint the lines target.
            window: Requested turnaround, or None for the provider default.

        Raises:
            CapabilityNotSupportedError: If the endpoint, window or size is unsupported.
            MixedModelBatchError: If a job-scoped provider got more than one model.
        """
        caps = self.capabilities
        if endpoint not in caps.endpoints:
            raise CapabilityNotSupportedError(
                caps.provider,
                endpoint,
                f"lane supports {sorted(caps.endpoints)}",
            )
        if window is not None and caps.window is None:
            raise CapabilityNotSupportedError(
                caps.provider,
                "completion_window",
                "the provider sets turnaround itself and accepts no window",
            )
        if (
            window is not None
            and caps.window is not None
            and caps.window.allowed
            and window not in caps.window.allowed
        ):
            raise CapabilityNotSupportedError(
                caps.provider,
                "completion_window",
                f"allowed values are {caps.window.allowed}, got {window!r}",
            )
        if caps.model_scope in ("job", "url"):
            models = tuple(sorted({line.model for line in lines}))
            if len(models) > 1:
                raise MixedModelBatchError(caps.provider, models)
        if caps.max_requests is not None and len(lines) > caps.max_requests:
            raise CapabilityNotSupportedError(
                caps.provider,
                "max_requests",
                f"lane caps a batch at {caps.max_requests} lines, got {len(lines)}",
            )
