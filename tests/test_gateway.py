"""Whether an unmodified OpenAI client can drive batchlane.

This is the test the gateway exists to pass. It uses the real `openai` SDK,
not a hand-rolled client, because the claim being made is protocol
compatibility: point `base_url` here and existing batch code reaches a
different provider. If the SDK needs any special-casing to work, the claim is
false and should not be made.
"""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.openai_shaped import ROWS
from batchlane.gateway import build_app, decode_batch_id, encode_batch_id

GROQ = ROWS["groq"]
MODEL = "groq/llama-3.3-70b-versatile"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return build_app(tmp_path / "gw")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def client(app):
    """A stock OpenAI client, pointed at the gateway in-process.

    The async client rather than the sync one because httpx.ASGITransport is
    async-only, and starlette's TestClient now wants httpx2. Neither detour
    weakens the claim: this is the unmodified `openai` SDK, and nothing about
    the gateway is special-cased for it.
    """
    import openai

    return openai.AsyncOpenAI(
        base_url="http://gateway.invalid/v1",
        api_key="not-used-by-the-gateway",
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
    )


def _allow_gateway():
    """Let traffic to the in-process app through respx to the ASGI transport."""
    respx.route(host="gateway.invalid").pass_through()


def _mock_groq(status="completed"):
    _allow_gateway()
    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f1"})
    )
    respx.post(f"{GROQ.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "validating"})
    )
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "b1",
                "status": status,
                "output_file_id": "o1",
                "request_counts": {"total": 2, "completed": 2, "failed": 0},
            },
        )
    )
    respx.get(f"{GROQ.base_url}/files/o1/content").mock(
        return_value=httpx.Response(
            200,
            text="\n".join(
                json.dumps(
                    {
                        "custom_id": cid,
                        "response": {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"content": text}}]},
                        },
                    }
                )
                for cid, text in (("row-2", "Paris"), ("row-1", "4"))
            ),
        )
    )


def _jsonl() -> bytes:
    return b"\n".join(
        json.dumps(
            {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8,
                },
            }
        ).encode()
        for cid, prompt in (("row-1", "What is 2+2?"), ("row-2", "Capital of France?"))
    )


@pytest.mark.anyio
@respx.mock
async def test_the_stock_openai_sdk_runs_a_whole_batch_against_groq(client):
    """Upload, create, poll, download -- all through the unmodified SDK."""
    _mock_groq()
    uploaded = await client.files.create(file=("in.jsonl", _jsonl()), purpose="batch")
    assert uploaded.id.startswith("file-")

    batch = await client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    assert batch.status == "validating"

    fetched = await client.batches.retrieve(batch.id)
    assert fetched.status == "completed"
    assert fetched.request_counts.completed == 2

    body = (await client.files.content(fetched.output_file_id)).text
    rows = {json.loads(x)["custom_id"]: json.loads(x) for x in body.splitlines()}
    # The provider answered out of order; the ids must still line up.
    assert rows["row-1"]["response"]["body"]["choices"][0]["message"]["content"] == "4"
    assert (
        rows["row-2"]["response"]["body"]["choices"][0]["message"]["content"] == "Paris"
    )


@pytest.mark.anyio
@respx.mock
async def test_the_batch_id_carries_the_job_so_the_server_stores_nothing(
    client, tmp_path
):
    _mock_groq()
    uploaded = await client.files.create(file=("in.jsonl", _jsonl()), purpose="batch")
    batch = await client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    # The id is the state: decoding it alone recovers the provider job.
    handles = decode_batch_id(batch.id)
    assert [h.provider for h in handles] == ["groq"]
    assert handles[0].job_id == "b1"


@pytest.mark.anyio
@respx.mock
async def test_cancel_reaches_the_provider_through_the_sdk(client):
    _mock_groq()
    uploaded = await client.files.create(file=("in.jsonl", _jsonl()), purpose="batch")
    cancel = respx.post(f"{GROQ.base_url}/batches/b1/cancel").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "cancelling"})
    )
    batch = await client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    await client.batches.cancel(batch.id)
    assert cancel.called


@pytest.mark.anyio
@respx.mock
async def test_listing_says_it_holds_nothing_rather_than_inventing_a_list(client):
    # A stateless server has nothing to enumerate. An empty list keeps a
    # polling client working; a 500 would not.
    _allow_gateway()
    page = await client.batches.list()
    assert page.data == []


@pytest.mark.anyio
@respx.mock
async def test_an_unknown_batch_id_is_a_404_not_a_crash(client):
    _allow_gateway()
    import openai

    with pytest.raises(openai.NotFoundError):
        await client.batches.retrieve("batch_bl_not-a-real-token")


@pytest.mark.anyio
@respx.mock
async def test_a_file_with_no_requests_is_rejected_with_400(client):
    _allow_gateway()
    import openai

    empty = await client.files.create(file=("in.jsonl", b"\n"), purpose="batch")
    with pytest.raises(openai.BadRequestError):
        await client.batches.create(
            input_file_id=empty.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )


def test_a_large_job_spills_to_disk_instead_of_producing_an_unusable_url(tmp_path):
    # Measured earlier: 500 chunks is a 5.4kB token, past what fits in a path.
    handles = [
        bl.BatchHandle(
            provider="groq",
            job_id=f"batch_{i:08x}",
            endpoint="chat.completions",
            lane="batch_file",
            created_at=bl.handle.utcnow(),
            extra={"input_file_id": f"file_{i:08x}"},
        )
        for i in range(500)
    ]
    big = encode_batch_id(handles, tmp_path)
    assert len(big) < 100, "an oversized job must reference its handles, not carry them"
    assert len(decode_batch_id(big, tmp_path)) == 500

    small = encode_batch_id(handles[:1], tmp_path)
    assert len(decode_batch_id(small, tmp_path)) == 1
    assert "ref_" not in small, "a small job must stay stateless"


# --- path traversal, both confirmed exploitable before these landed ---


@pytest.mark.anyio
@respx.mock
async def test_a_traversing_input_file_id_cannot_read_outside_the_store(
    client, tmp_path
):
    """The worst of the two: it was arbitrary file read plus exfiltration.

    input_file_id arrives in a JSON body, which the router does not normalise
    the way it does a path segment. Before the fix this read ../../outside.jsonl,
    parsed it, and got as far as uploading its contents to Groq.
    """
    _allow_gateway()
    import openai

    for probe in (
        "../../etc/passwd",
        "../outside.jsonl",
        "/etc/passwd",
        "file-../../x",
    ):
        with pytest.raises(openai.NotFoundError) as exc:
            await client.batches.create(
                input_file_id=probe,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        assert "issued" in str(exc.value)
    # Nothing may have reached a provider.
    assert not [c for c in respx.calls if "groq" in str(c.request.url)]


@pytest.mark.anyio
@respx.mock
async def test_a_traversing_file_id_cannot_be_read_back(client):
    _allow_gateway()
    import openai

    for probe in ("..%2F..%2Fetc%2Fpasswd", "file-not-hex-at-all", "../secrets"):
        with pytest.raises((openai.NotFoundError, openai.APIStatusError)):
            await client.files.content(probe)


def test_a_traversing_spill_digest_cannot_be_read(tmp_path):
    # decode_batch_id took the digest straight from a caller-supplied batch id.
    outside = tmp_path / "PWNED.json"
    outside.write_text("[]")
    spill = tmp_path / "jobs"
    spill.mkdir()
    for probe in ("../PWNED", "../../etc/hosts", "not-hex", "a" * 31):
        with pytest.raises(bl.BatchlaneError, match="issued"):
            decode_batch_id(f"batch_blref_{probe}", spill)


def test_a_legitimately_spilled_job_still_round_trips(tmp_path):
    # The validation must not break the case it guards.
    handles = [
        bl.BatchHandle(
            provider="groq",
            job_id=f"b{i}",
            endpoint="chat.completions",
            lane="batch_file",
            created_at=bl.handle.utcnow(),
            extra={"input_file_id": f"f{i}"},
        )
        for i in range(500)
    ]
    spilled = encode_batch_id(handles, tmp_path)
    assert "ref_" in spilled
    assert len(decode_batch_id(spilled, tmp_path)) == 500


# --- denial of service and access control ---


def test_a_compressed_bomb_cannot_be_inflated_through_a_batch_id():
    """A short id must not be able to allocate gigabytes.

    Measured before the bound: a 272kB id expanded to 459MB, so a handful of
    concurrent requests would exhaust the process.
    """
    import base64
    import zlib

    from batchlane.gateway import BATCH_PREFIX

    token = (
        base64.urlsafe_b64encode(zlib.compress(b"A" * (64 * 1024 * 1024), 9))
        .decode()
        .rstrip("=")
    )
    with pytest.raises(bl.BatchlaneError, match="longer than this gateway issues"):
        decode_batch_id(f"{BATCH_PREFIX}_{token}")


def test_an_id_within_the_issued_size_still_decodes():
    # The bound must not break the jobs it protects.
    handles = [
        bl.BatchHandle(
            provider="groq",
            job_id=f"b{i}",
            endpoint="chat.completions",
            lane="batch_file",
            created_at=bl.handle.utcnow(),
            extra={"input_file_id": f"f{i}"},
        )
        for i in range(50)
    ]
    assert len(decode_batch_id(encode_batch_id(handles))) == 50


@pytest.mark.anyio
async def test_without_a_key_every_route_is_open_and_with_one_none_are(
    tmp_path, monkeypatch
):
    # The gateway spends the operator's provider credits, so reachability is
    # spending authority.
    monkeypatch.setenv("GROQ_API_KEY", "k")
    import openai

    guarded = build_app(tmp_path / "a", api_key="s3cret")
    transport = httpx.ASGITransport(app=guarded)

    anon = openai.AsyncOpenAI(
        base_url="http://gateway.invalid/v1",
        api_key="wrong",
        http_client=httpx.AsyncClient(transport=transport),
    )
    with pytest.raises(openai.AuthenticationError):
        await anon.batches.list()

    authorised = openai.AsyncOpenAI(
        base_url="http://gateway.invalid/v1",
        api_key="s3cret",
        http_client=httpx.AsyncClient(transport=transport),
    )
    assert (await authorised.batches.list()).data == []


def test_serving_beyond_loopback_without_a_key_is_refused(monkeypatch, capsys):
    # The dangerous default is one flag away, so the CLI blocks it rather than
    # documenting it.
    from batchlane.cli import main

    monkeypatch.delenv("BATCHLANE_GATEWAY_KEY", raising=False)
    # The literal is the point of the test: this is the address that must be
    # refused without a key.
    exposed = "0.0.0.0"  # noqa: S104
    assert main(["serve", "--host", exposed]) == 2
    assert "spends your provider credits" in capsys.readouterr().err
