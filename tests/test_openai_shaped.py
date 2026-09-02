"""Full submit -> poll -> collect lifecycle against a mocked provider."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.openai_shaped import ROWS, OpenAIShapedAdapter

GROQ = ROWS["groq"]
LINES = [
    bl.BatchLine(
        "q1", "llama-3.3-70b-versatile", [{"role": "user", "content": "2+2?"}]
    ),
    bl.BatchLine(
        "q2", "llama-3.3-70b-versatile", [{"role": "user", "content": "capital?"}]
    ),
]


def _output_jsonl():
    return "\n".join(
        json.dumps(
            {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": text}}]},
                },
            }
        )
        for cid, text in (("q1", "4"), ("q2", "Paris"))
    )


@respx.mock
def test_submit_uploads_jsonl_then_creates_the_job():
    upload = respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "file_1"})
    )
    create = respx.post(f"{GROQ.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    )

    handle = OpenAIShapedAdapter(GROQ).submit(
        LINES, endpoint="chat.completions", window="7d", api_key="k"
    )

    assert upload.called
    assert create.called
    assert handle.job_id == "batch_1"
    assert handle.provider == "groq"
    assert handle.extra["input_file_id"] == "file_1"

    body = json.loads(create.calls[0].request.content)
    assert body["input_file_id"] == "file_1"
    assert body["completion_window"] == "7d"
    assert body["endpoint"] == "/v1/chat/completions"

    # The uploaded file must be one JSONL line per request, model on the line.
    uploaded = upload.calls[0].request.content.decode(errors="ignore")
    assert uploaded.count('"custom_id"') == 2


@respx.mock
def test_status_normalizes_seven_vocabularies_but_keeps_the_original():
    respx.get(f"{GROQ.base_url}/batches/batch_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "batch_1",
                "status": "in_progress",
                "request_counts": {"total": 2, "completed": 1, "failed": 0},
            },
        )
    )
    handle = bl.BatchHandle(
        provider="groq",
        job_id="batch_1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )
    status = OpenAIShapedAdapter(GROQ).status(handle, api_key="k")
    assert status.state == "running"
    assert status.raw_state == "in_progress"
    assert (status.total, status.succeeded) == (2, 1)
    assert not status.is_terminal


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("validating", "pending"),
        ("finalizing", "running"),
        ("completed", "succeeded"),
        ("COMPLETED", "succeeded"),  # Together uppercases its enum
        ("failed", "failed"),
        ("expired", "expired"),
        ("cancelled", "cancelled"),
    ],
)
def test_status_map(raw, expected):
    got = OpenAIShapedAdapter(GROQ).parse_status({"status": raw})
    assert got.state == expected
    assert got.raw_state == raw


@respx.mock
def test_results_stream_and_join_on_custom_id():
    respx.get(f"{GROQ.base_url}/batches/batch_1").mock(
        return_value=httpx.Response(
            200,
            json={"id": "batch_1", "status": "completed", "output_file_id": "out_1"},
        )
    )
    respx.get(f"{GROQ.base_url}/files/out_1/content").mock(
        return_value=httpx.Response(200, text=_output_jsonl())
    )
    handle = bl.BatchHandle(
        provider="groq",
        job_id="batch_1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )
    results = list(OpenAIShapedAdapter(GROQ).results(handle, api_key="k"))
    assert [r.custom_id for r in results] == ["q1", "q2"]
    assert all(r.ok for r in results)
    assert results[0].response["choices"][0]["message"]["content"] == "4"


@respx.mock
def test_together_uses_its_nonstandard_purpose_and_upload_path():
    row = ROWS["together_ai"]
    upload = respx.post(f"{row.base_url}/files/upload").mock(
        return_value=httpx.Response(200, json={"id": "f"})
    )
    respx.post(f"{row.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b", "status": "VALIDATING"})
    )
    OpenAIShapedAdapter(row).submit(
        [
            bl.BatchLine(
                "q1",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                [{"role": "user", "content": "x"}],
            )
        ],
        endpoint="chat.completions",
        window=None,
        api_key="k",
    )
    assert b"batch-api" in upload.calls[0].request.content


@respx.mock
def test_openai_is_shipped_and_uses_the_canonical_endpoints():
    # batchlane covers OpenAI even though litellm does, so a caller with mixed
    # providers needs one code path rather than two.
    row = ROWS["openai"]
    assert row.base_url == "https://api.openai.com/v1"
    upload = respx.post(f"{row.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "file-1"})
    )
    create = respx.post(f"{row.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    )
    handle = OpenAIShapedAdapter(row).submit(
        [bl.BatchLine("q1", "gpt-4o-mini", [{"role": "user", "content": "hi"}])],
        endpoint="chat.completions",
        window=None,
        api_key="sk-test",
    )
    assert handle.provider == "openai"
    assert b"batch" in upload.calls[0].request.content
    assert json.loads(create.calls[0].request.content)["completion_window"] == "24h"


# Each value below was read from the provider's own API reference. Nothing in
# the suite pinned these before: every other test reads the row dynamically, so
# changing a host or a purpose string broke nothing and would have shipped.
DOCUMENTED_ROWS = {
    "openai": ("https://api.openai.com/v1", "/files", "batch"),
    "groq": ("https://api.groq.com/openai/v1", "/files", "batch"),
    # docs.together.ai publishes api.together.ai, and a non-standard purpose.
    "together_ai": ("https://api.together.ai/v1", "/files/upload", "batch-api"),
    # DeepInfra's base path embeds /openai/.
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "/files", "batch"),
}


@pytest.mark.parametrize(("provider", "documented"), DOCUMENTED_ROWS.items())
def test_the_row_matches_the_providers_published_api(provider, documented):
    row = ROWS[provider]
    assert (row.base_url, row.files_path, row.upload_purpose) == documented


def test_every_shipped_openai_shaped_lane_has_its_endpoints_pinned():
    # A lane added without an entry here would go unverified against its docs.
    assert set(ROWS) == set(DOCUMENTED_ROWS)


def test_the_recovery_key_fits_openais_documented_metadata_limits():
    # OpenAI caps metadata at 16 pairs, 64-character keys and 512-character
    # values. The key is stamped there for crash recovery, so it has to fit.
    from batchlane.adapters.base import KEY_FIELD
    from batchlane.runner import _chunk_key

    key = _chunk_key([bl.BatchLine("r0", "m", [{"role": "user", "content": "x"}])], 0)
    assert len(KEY_FIELD) <= 64
    assert len(key) <= 512


def test_a_lane_with_no_counts_still_reports_progress():
    """Together publishes a percentage and no counts at all.

    Discarding it left a running job indistinguishable from a stalled one, so
    the percentage is carried through and normalised to the same 0..1 scale
    the counted lanes produce.
    """
    together = OpenAIShapedAdapter(ROWS["together_ai"])
    status = together.parse_status({"status": "IN_PROGRESS", "progress": 42.0})
    assert status.total is None, "Together genuinely reports no counts"
    assert status.progress == pytest.approx(0.42)
    assert status.fraction_done == pytest.approx(0.42)


def test_a_counted_lane_derives_the_same_scale():
    groq = OpenAIShapedAdapter(ROWS["groq"])
    status = groq.parse_status(
        {
            "status": "in_progress",
            "request_counts": {"total": 10, "completed": 3, "failed": 1},
        }
    )
    # Both lanes answer the same question in the same units, however the
    # provider happens to express it.
    assert status.fraction_done == pytest.approx(0.4)


def test_progress_is_clamped_to_a_sane_range():
    together = OpenAIShapedAdapter(ROWS["together_ai"])
    assert together.parse_status({"status": "x", "progress": 150}).progress == 1.0
    assert together.parse_status({"status": "x", "progress": -5}).progress == 0.0
    assert together.parse_status({"status": "x"}).fraction_done is None
