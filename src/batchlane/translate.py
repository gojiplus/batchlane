"""The single seam between batchlane and LiteLLM.

Nothing else in this package imports ``litellm``. Everything called here is a
pure, synchronous transform -- no network, no auth, no callbacks -- which is
what makes it safe to use for building batch payloads offline.

These are LiteLLM *internals*, not published API. ``tests/test_translate.py``
pins their output against committed fixtures so a version bump that reshapes
them fails in CI rather than silently emitting wrong JSONL into a 50,000-row
job.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from .errors import BatchlaneError

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.llms.openai import AllMessageValues

__all__ = ["decode_response", "encode_body", "resolve"]


def resolve(model: str) -> tuple[str, str, str | None]:
    """Split a model string into provider, bare model name, and default base URL.

    Args:
        model: A LiteLLM-style model string, e.g. ``"groq/llama-3.3-70b"``.

    Returns:
        A ``(provider, bare_model, api_base)`` triple. ``api_base`` is the
        provider's synchronous default and may be None.
    """
    from litellm import get_llm_provider

    bare, provider, _key, api_base = get_llm_provider(model=model)
    return provider, bare, api_base


def encode_body(
    provider: str,
    bare_model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build the provider-shaped request body for one batch line.

    For OpenAI-shaped providers this is the same JSON LiteLLM would POST for an
    ordinary chat call, which is exactly what a JSONL line needs.

    Args:
        provider: LiteLLM provider key.
        bare_model: Model name with the provider prefix already stripped.
        messages: OpenAI-format chat messages.
        params: OpenAI-format request parameters (temperature, tools, ...).

    Returns:
        The request body as a plain dict.

    Raises:
        BatchlaneError: If LiteLLM has no chat config registered for the provider.
    """
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager, get_optional_params

    # litellm's transforms mutate the messages list in place -- AnthropicConfig
    # pops the system message out of it. Callers routinely reuse one list across
    # lines (a shared system prompt, a template in a loop), and without this copy
    # every line after the first silently loses its system prompt. Verified
    # against litellm 1.80.10; see tests/test_translate.py.
    messages = deepcopy(messages)

    if provider == "gemini":
        return _encode_gemini(bare_model, messages, params)

    config = ProviderConfigManager.get_provider_chat_config(
        bare_model, LlmProviders(provider)
    )
    if config is None:
        raise BatchlaneError(
            f"litellm has no chat config for provider {provider!r}; cannot build "
            f"a batch line body for it."
        )
    optional = get_optional_params(
        model=bare_model, custom_llm_provider=provider, **params
    )
    body = config.transform_request(
        model=bare_model,
        messages=cast("list[AllMessageValues]", messages),
        optional_params=optional,
        litellm_params={},
        headers={},
    )
    # litellm adds two fields that belong to a live call rather than a batch
    # line. Both are the default value, so dropping them changes nothing about
    # the request, while sending them can only ever be rejected: an empty
    # extra_body is an unknown key to a strict validator, and stream has no
    # meaning in a batch, which is not a streaming context. Mistral's config
    # already omits stream, so removing it makes the lanes agree.
    for noise, default in (("extra_body", {}), ("stream", False)):
        if body.get(noise) == default:
            body.pop(noise, None)
    return dict(body)


def _encode_gemini(
    bare_model: str, messages: list[dict[str, Any]], params: dict[str, Any]
) -> dict[str, Any]:
    """Build a Gemini GenerateContentRequest body.

    Gemini takes a different entry point: ``VertexGeminiConfig.transform_request``
    raises NotImplementedError, and the module-level builder requires every
    ``vertex_*`` argument even when the provider is AI Studio.

    Args:
        bare_model: Gemini model name.
        messages: OpenAI-format chat messages.
        params: OpenAI-format request parameters.

    Returns:
        The GenerateContentRequest body.
    """
    from litellm.llms.vertex_ai.gemini.transformation import sync_transform_request_body
    from litellm.utils import get_optional_params

    optional = get_optional_params(
        model=bare_model, custom_llm_provider="gemini", **params
    )
    body = sync_transform_request_body(
        gemini_api_key=None,
        messages=cast("list[AllMessageValues]", messages),
        api_base=None,
        model=bare_model,
        client=None,
        timeout=None,
        extra_headers=None,
        optional_params=optional,
        # Typed as required, but the sync builder never touches it and
        # tolerates None; verified against litellm 1.80.10.
        logging_obj=cast("Logging", None),
        custom_llm_provider="gemini",
        litellm_params={},
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )
    return dict(body)


def decode_response(provider: str, bare_model: str, payload: dict[str, Any]) -> Any:
    """Turn a provider's raw result body into an OpenAI-shaped response.

    Providers whose batch lane returns native (non-OpenAI) response bodies --
    Gemini and Anthropic -- need this. The OpenAI-shaped providers return chat
    completions already and skip it entirely.

    Args:
        provider: LiteLLM provider key.
        bare_model: The model the batch ran against.
        payload: One raw response body from the results file.

    Returns:
        A LiteLLM ``ModelResponse`` in OpenAI chat-completion shape.

    Raises:
        BatchlaneError: If the provider needs no decoding and none is defined.
    """
    import httpx
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import ModelResponse

    if provider == "gemini":
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        config: Any = VertexGeminiConfig()
    elif provider == "anthropic":
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        config = AnthropicConfig()
    else:
        raise BatchlaneError(
            f"{provider!r} returns OpenAI-shaped responses; no decode is needed."
        )

    logging_obj = Logging(
        model=bare_model,
        messages=[],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="batchlane",
        function_id="batchlane",
    )
    # The constructor does not set this, and transform_response reads it.
    # Without it litellm raises VertexAIError rather than a clear AttributeError.
    logging_obj.optional_params = {}
    return config.transform_response(
        model=bare_model,
        raw_response=httpx.Response(
            200, json=payload, request=httpx.Request("POST", "https://batchlane.local")
        ),
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
