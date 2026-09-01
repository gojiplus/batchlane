"""Errors raised by batchlane.

Every error here is raised *before* any network call, so a misuse costs
nothing. The refusal errors deliberately explain *why* a lane is missing
rather than only reporting that it is.
"""

from __future__ import annotations

__all__ = [
    "AdapterNotShippedError",
    "BatchlaneError",
    "CapabilityNotSupportedError",
    "MixedModelBatchError",
    "NoBatchLaneError",
]


class BatchlaneError(Exception):
    """Base class for every error batchlane raises."""


class NoBatchLaneError(BatchlaneError, LookupError):
    """The provider runs no asynchronous batch lane we can use.

    This is usually structural rather than temporary. A batch discount exists
    because the provider backfills otherwise-idle GPUs, which requires owning
    the fleet. Resellers that buy capacity at retail cannot offer one however
    long you wait.
    """

    def __init__(
        self, provider: str, *, reason: str, alternatives: tuple[str, ...] = ()
    ) -> None:
        """Build the refusal.

        Args:
            provider: The resolved provider name that has no lane.
            reason: Why it has none, in one sentence.
            alternatives: Providers that do run a lane, if any are known.
        """
        self.provider = provider
        self.reason = reason
        self.alternatives = alternatives
        message = f"{provider!r} has no asynchronous batch lane: {reason.rstrip('.')}."
        if alternatives:
            message += f" Providers with a lane: {', '.join(alternatives)}."
        super().__init__(message)


class CapabilityNotSupportedError(BatchlaneError):
    """The provider has a lane, but not this particular capability."""

    def __init__(self, provider: str, capability: str, detail: str) -> None:
        """Build the error.

        Args:
            provider: The provider that lacks the capability.
            capability: Short name of what is missing, e.g. ``"cancel"``.
            detail: What the provider does instead, or why it cannot.
        """
        self.provider = provider
        self.capability = capability
        super().__init__(f"[{provider}] does not support {capability!r}: {detail}")


class MixedModelBatchError(BatchlaneError):
    """A multi-model batch was sent to a provider that scopes model per job."""

    def __init__(self, provider: str, models: tuple[str, ...]) -> None:
        """Build the error.

        Args:
            provider: The job-scoped provider.
            models: The distinct models found across the submitted lines.
        """
        self.provider = provider
        self.models = models
        super().__init__(
            f"[{provider}] scopes a batch to one model, but {len(models)} were given: "
            f"{', '.join(models)}. Split into one batch per model; batchlane does "
            f"not fan out automatically."
        )


class AdapterNotShippedError(BatchlaneError):
    """The provider runs a lane, but batchlane has not implemented it.

    Kept distinct from :class:`NoBatchLaneError` so a refusal never claims a lane is
    absent when it is only unimplemented -- the two call for different actions
    from the reader.
    """

    def __init__(
        self, provider: str, *, reason: str, alternatives: tuple[str, ...] = ()
    ) -> None:
        """Build the error.

        Args:
            provider: The provider whose adapter is missing.
            reason: Why it has not been shipped.
            alternatives: Providers that are supported today.
        """
        self.provider = provider
        self.reason = reason
        message = (
            f"{provider!r} runs a batch lane, but batchlane has no adapter "
            f"for it: {reason.rstrip('.')}."
        )
        if alternatives:
            message += f" Supported today: {', '.join(alternatives)}."
        super().__init__(message)
