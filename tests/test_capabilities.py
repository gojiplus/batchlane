"""The capability table must describe what the code actually does.

A descriptor that drifts from behavior is worse than none: it makes a caller
confident about a claim nothing enforces.
"""

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.base import BatchAdapter
from batchlane.adapters.openai_shaped import ROWS, OpenAIShapedAdapter
from batchlane.capabilities import (
    CAPABILITIES,
    IMPLEMENTED_ENDPOINTS,
    NO_LANE,
    NOT_SHIPPED,
)
from batchlane.errors import CapabilityNotSupportedError, MixedModelBatchError

# Derived from the registry, not hand-listed, so a new adapter is covered by
# the whole contract suite the moment it is wired up.
IDS = list(bl.supported_providers())
ADAPTERS = [bl.get_adapter(p) for p in IDS]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_every_adapter_has_a_descriptor_naming_itself(adapter):
    assert adapter.capabilities.provider in bl.supported_providers()
    assert adapter.capabilities.endpoints


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
@respx.mock
def test_guard_fires_before_any_network_call(adapter):
    # respx with no routes registered: any HTTP attempt raises. So if check()
    # did not fire first, this fails with a connection error instead.
    with pytest.raises(CapabilityNotSupportedError):
        adapter.submit(
            [bl.BatchLine("a", "m", [{"role": "user", "content": "x"}])],
            endpoint="video.generation",
            window=None,
            api_key="k",
        )
    assert not respx.calls


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_unsupported_cancel_refuses_rather_than_silently_no_opping(adapter):
    if adapter.capabilities.supports_cancel:
        pytest.skip("provider documents a cancel endpoint")
    handle = bl.BatchHandle(
        provider=adapter.capabilities.provider,
        job_id="j",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )
    with pytest.raises(CapabilityNotSupportedError):
        adapter.cancel(handle, api_key="k")


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
@respx.mock
def test_list_matches_what_the_descriptor_claims(adapter):
    # Behavioral, not structural. An earlier version of this test asserted the
    # class overrode list_jobs, which passed even with a false supports_list
    # claim because one class serves several providers. Exercise the call.
    # A union of every list-response shape our providers use, so one fixture
    # serves them all: OpenAI-shaped providers read "data"/"id", Gemini reads
    # "operations"/"name".
    respx.route(method="GET").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"id": "job_1"}],
                "operations": [{"name": "batches/job_1"}],
            },
        )
    )
    if adapter.capabilities.supports_list:
        handles = list(adapter.list_jobs(api_key="k"))
        assert handles, "claims supports_list but returned nothing"
        assert handles[0].provider == adapter.capabilities.provider
        assert respx.calls, "claims supports_list but made no request"
    else:
        with pytest.raises(CapabilityNotSupportedError, match="list"):
            list(adapter.list_jobs(api_key="k"))
        assert not respx.calls, "refused, but still hit the network"


def test_window_outside_the_documented_set_is_refused():
    adapter = OpenAIShapedAdapter(ROWS["groq"])
    with pytest.raises(CapabilityNotSupportedError, match="completion_window"):
        adapter.check(
            [bl.BatchLine("a", "m", [{"role": "user", "content": "x"}])],
            endpoint="chat.completions",
            window="90d",
        )


def test_groq_accepts_the_seven_day_window_litellm_cannot_express():
    # litellm hardcodes completion_window: Literal["24h"]. Groq documents 7d.
    adapter = OpenAIShapedAdapter(ROWS["groq"])
    adapter.check(
        [bl.BatchLine("a", "m", [{"role": "user", "content": "x"}])],
        endpoint="chat.completions",
        window="7d",
    )


def test_per_line_providers_allow_a_mixed_model_batch():
    adapter = OpenAIShapedAdapter(ROWS["groq"])
    assert adapter.capabilities.model_scope == "line"
    adapter.check(
        [
            bl.BatchLine("a", "model-one", [{"role": "user", "content": "x"}]),
            bl.BatchLine("b", "model-two", [{"role": "user", "content": "y"}]),
        ],
        endpoint="chat.completions",
        window=None,
    )


def test_job_scoped_providers_reject_a_mixed_model_batch():
    # Mistral and Fireworks put the model on the job, so a mixed batch is not
    # expressible. Refuse rather than silently running only one model.
    adapter = OpenAIShapedAdapter(ROWS["groq"])
    object.__setattr__(adapter, "capabilities", CAPABILITIES["mistral"])
    with pytest.raises(MixedModelBatchError, match="one model"):
        adapter.check(
            [
                bl.BatchLine("a", "model-one", [{"role": "user", "content": "x"}]),
                bl.BatchLine("b", "model-two", [{"role": "user", "content": "y"}]),
            ],
            endpoint="chat.completions",
            window=None,
        )


def test_batch_larger_than_the_documented_cap_is_refused():
    adapter = OpenAIShapedAdapter(ROWS["groq"])
    cap = adapter.capabilities.max_requests
    lines = [
        bl.BatchLine(str(i), "m", [{"role": "user", "content": "x"}])
        for i in range(cap + 1)
    ]
    with pytest.raises(CapabilityNotSupportedError, match="max_requests"):
        adapter.check(lines, endpoint="chat.completions", window=None)


def test_no_lane_and_not_shipped_are_disjoint():
    # A provider claimed to have no lane must not also be listed as one whose
    # adapter is merely unwritten -- the two say opposite things to a reader.
    assert not set(NO_LANE) & set(NOT_SHIPPED)


def test_shipped_adapters_are_not_listed_as_refusals():
    assert not set(ROWS) & (set(NO_LANE) | set(NOT_SHIPPED))


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_no_descriptor_claims_an_endpoint_we_cannot_build(adapter):
    # The bug this guards: three descriptors advertised "embeddings" while
    # translate only builds chat-completion bodies, so check() accepted an
    # embeddings batch and then uploaded {"messages": ...} to /v1/embeddings.
    # A descriptor must never promise more than the code can produce.
    unbuildable = adapter.capabilities.endpoints - IMPLEMENTED_ENDPOINTS
    assert not unbuildable, (
        f"{adapter.capabilities.provider} claims {sorted(unbuildable)} but "
        f"translate can only build {sorted(IMPLEMENTED_ENDPOINTS)}"
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_an_unbuildable_endpoint_is_refused_before_any_upload(adapter):
    with pytest.raises(CapabilityNotSupportedError):
        adapter.check(
            [bl.BatchLine("e1", "m", [{"role": "user", "content": "x"}])],
            endpoint="embeddings",
            window=None,
        )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
@respx.mock
def test_an_adapter_that_claims_to_stamp_a_key_actually_sends_it(adapter):
    # Reconciliation after a crash depends entirely on the key reaching the
    # provider. An adapter that quietly dropped it would still pass every
    # other test and would resubmit paid-for work on the next resume.
    if not adapter.stamps_key:
        pytest.skip(f"{adapter.capabilities.provider} accepts no batch-level label")

    respx.route().mock(return_value=httpx.Response(200, json={"id": "j", "name": "j"}))
    adapter.submit(
        [bl.BatchLine("r0", "m", [{"role": "user", "content": "x"}])],
        endpoint="chat.completions",
        window=None,
        api_key="k",
        key="bl-sentinel-0",
    )
    sent = b"".join(call.request.content for call in respx.calls)
    assert b"bl-sentinel-0" in sent, (
        f"{adapter.capabilities.provider} declares stamps_key but the key never "
        f"reached the wire"
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_an_adapter_that_cannot_stamp_overrides_recovery(adapter):
    # The default find_submitted matches on the stamped key, so an adapter
    # that cannot stamp must supply its own recovery or it silently has none.
    if adapter.stamps_key:
        pytest.skip("uses the default key match")
    assert type(adapter).find_submitted is not BatchAdapter.find_submitted
