"""The handle-shaped job abstraction.

A batch outlives the process that submitted it, so :class:`BatchHandle` is
JSON-serializable and carries everything needed to resume against the provider
from a cold start.

The public shape is deliberately *not* file-shaped. ``input_file_id`` plus
``purpose=batch`` is one provider's 2024 implementation detail; Gemini has no
file id, xAI has no completion window, Fireworks has a dataset. The ``lane``
discriminator is what lets OpenAI Responses ``background=True`` and Bedrock
async-invoke be added later without changing this dataclass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["BatchHandle", "BatchLine", "JobStatus", "Lane", "RequestResult", "State"]

Lane = Literal["batch_file", "batch_inline", "background", "async_invoke"]
State = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]


@dataclass(frozen=True, slots=True)
class BatchLine:
    """One request within a batch.

    ``model`` is always required here even for providers that scope the model
    to the job: the adapter checks that every line agrees before dropping it,
    which is what makes :class:`~batchlane.errors.MixedModelBatchError` detectable.
    """

    custom_id: str
    model: str
    messages: list[dict[str, Any]]
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BatchHandle:
    """A receipt for a submitted job."""

    provider: str
    job_id: str
    endpoint: str
    lane: Lane
    created_at: datetime
    model: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize so the job can be polled from another process.

        Returns:
            A JSON string accepted by :meth:`from_json`.
        """
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["extra"] = dict(self.extra)
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> BatchHandle:
        """Rebuild a handle produced by :meth:`to_json`.

        Args:
            raw: The JSON string.

        Returns:
            The reconstructed handle.
        """
        payload = json.loads(raw)
        payload["created_at"] = datetime.fromisoformat(payload["created_at"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Normalized job state.

    Seven providers use seven status vocabularies. ``state`` is the normalized
    one; ``raw_state`` keeps the provider's own string so nothing is lost.
    """

    state: State
    raw_state: str
    total: int | None = None
    succeeded: int | None = None
    failed: int | None = None
    error: str | None = None
    expires_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether the job has stopped and will not change again.

        Returns:
            True when no further polling is useful.
        """
        return self.state in ("succeeded", "failed", "cancelled", "expired")


@dataclass(frozen=True, slots=True)
class RequestResult:
    """One line's outcome, joined back to its ``custom_id``."""

    custom_id: str
    response: Any | None = None
    error: Mapping[str, Any] | None = None
    status_code: int | None = None

    @property
    def ok(self) -> bool:
        """Whether this line produced a usable response.

        Returns:
            True when a response is present and no error was recorded.
        """
        return self.error is None and self.response is not None


def utcnow() -> datetime:
    """Timezone-aware now, for handle creation timestamps.

    Returns:
        The current UTC time.
    """
    return datetime.now(UTC)
