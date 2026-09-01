"""Resolve a model to the adapter that can batch it, or refuse with a reason."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .adapters.anthropic import AnthropicAdapter
from .adapters.fireworks import FireworksAdapter
from .adapters.gemini import GeminiAdapter
from .adapters.mistral import MistralAdapter
from .adapters.openai_shaped import ROWS, OpenAIShapedAdapter
from .capabilities import CAPABILITIES, LOCAL_RUNTIME, NO_LANE, NOT_SHIPPED
from .errors import AdapterNotShippedError, BatchlaneError, NoBatchLaneError
from .translate import resolve

if TYPE_CHECKING:
    from .adapters.base import BatchAdapter

__all__ = ["get_adapter", "resolve_api_key", "supported_providers"]

#: Providers whose lane differs structurally enough to need its own adapter
#: rather than a row in the OpenAI-shaped table.
_BESPOKE: dict[str, type[BatchAdapter]] = {
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "mistral": MistralAdapter,
    "fireworks_ai": FireworksAdapter,
}

_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "together_ai": ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
    "deepinfra": ("DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "fireworks_ai": ("FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY"),
}


def supported_providers() -> tuple[str, ...]:
    """Providers batchlane can currently submit batches to.

    Returns:
        Provider keys, sorted.
    """
    return tuple(sorted({*ROWS, *_BESPOKE}))


def get_adapter(provider: str) -> BatchAdapter:
    """Return the adapter for a provider.

    Args:
        provider: A LiteLLM provider key.

    Returns:
        The adapter that can run batches for it.

    Raises:
        NoBatchLaneError: If the provider runs no usable lane, with the reason.
        AdapterNotShippedError: If it runs a lane batchlane has not implemented.
    """
    if provider in ROWS:
        return OpenAIShapedAdapter(ROWS[provider])
    if provider in _BESPOKE:
        return _BESPOKE[provider]()
    if provider in LOCAL_RUNTIME:
        raise NoBatchLaneError(
            provider, reason=LOCAL_RUNTIME[provider], alternatives=supported_providers()
        )
    if provider in NO_LANE:
        raise NoBatchLaneError(
            provider, reason=NO_LANE[provider], alternatives=supported_providers()
        )
    if provider in NOT_SHIPPED:
        raise AdapterNotShippedError(
            provider, reason=NOT_SHIPPED[provider], alternatives=supported_providers()
        )
    if provider in CAPABILITIES:
        raise AdapterNotShippedError(
            provider,
            reason="its adapter is planned but not written yet",
            alternatives=supported_providers(),
        )
    raise NoBatchLaneError(
        provider,
        reason=(
            "batchlane has no record of one. If a lane has shipped since, please "
            "open an issue"
        ),
        alternatives=supported_providers(),
    )


def adapter_for_model(model: str) -> tuple[BatchAdapter, str, str]:
    """Resolve a model string to its adapter.

    Args:
        model: A LiteLLM-style model string.

    Returns:
        A ``(adapter, provider, bare_model)`` triple.
    """
    provider, bare, _base = resolve(model)
    return get_adapter(provider), provider, bare


def resolve_api_key(provider: str, explicit: str | None = None) -> str:
    """Find the credential for a provider.

    Args:
        provider: A LiteLLM provider key.
        explicit: A key passed by the caller, which always wins.

    Returns:
        The API key.

    Raises:
        BatchlaneError: If no key was given and none is in the environment.
    """
    if explicit:
        return explicit
    for name in _ENV_VARS.get(provider, ()):
        value = os.environ.get(name)
        if value:
            return value
    names = " or ".join(_ENV_VARS.get(provider, ("<unknown>",)))
    raise BatchlaneError(
        f"No API key for {provider!r}. Set {names}, or pass api_key= explicitly."
    )
