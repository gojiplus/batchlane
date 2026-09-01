"""The one place a crash costs money: submitting twice.

`run()` calls the provider and then records the receipt. Those are two steps,
so a crash between them leaves a job that was paid for and not remembered. No
batch provider offers an idempotency key, so the client has to close the gap
itself: write the intent first, stamp a key on the provider's job, and look
for that key before spending again.
"""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane import runner
from batchlane.adapters.base import KEY_FIELD
from batchlane.adapters.openai_shaped import ROWS

GROQ = ROWS["groq"]
MODEL = "groq/llama-3.3-70b-versatile"


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")


def _mock_provider():
    """A provider that remembers what it was given, the way a real one does."""
    submitted: list[dict] = []

    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f"})
    )

    def on_create(request):
        body = json.loads(request.content)
        submitted.append(body)
        return httpx.Response(200, json={"id": "b1", "status": "validating"})

    submit = respx.post(f"{GROQ.base_url}/batches").mock(side_effect=on_create)

    # The crux: a job that was accepted shows up when you list, carrying the
    # key we stamped. That is what makes recovery possible at all.
    def on_list(request):
        data = [
            {
                "id": "b1",
                "status": "in_progress",
                "metadata": body.get("metadata") or {},
            }
            for body in submitted
        ]
        return httpx.Response(200, json={"data": data})

    respx.get(f"{GROQ.base_url}/batches").mock(side_effect=on_list)
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        return_value=httpx.Response(
            200, json={"id": "b1", "status": "completed", "output_file_id": "o1"}
        )
    )
    respx.get(f"{GROQ.base_url}/files/o1/content").mock(
        return_value=httpx.Response(
            200,
            text=json.dumps(
                {
                    "custom_id": "r0",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "x"}}]},
                    },
                }
            ),
        )
    )
    return submit, submitted


@respx.mock
def test_a_crash_between_submit_and_record_does_not_resubmit(
    tmp_path, monkeypatch, groq_key
):
    submit, _ = _mock_provider()
    lines = [bl.BatchLine("r0", MODEL, [{"role": "user", "content": "a"}])]
    ckpt = tmp_path / "job.jsonl"

    # Run 1: the provider accepts the job, then the process dies before the
    # receipt is written. That inference is already paid for.
    real = runner._append_checkpoint
    monkeypatch.setattr(
        runner,
        "_append_checkpoint",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("died after submit")),
    )
    with pytest.raises(RuntimeError):
        list(bl.run(lines, checkpoint=ckpt, poll_interval=0))
    assert submit.call_count == 1

    # Run 2: resume must find the orphaned job rather than buy it again.
    monkeypatch.setattr(runner, "_append_checkpoint", real)
    out = list(bl.run(lines, checkpoint=ckpt, poll_interval=0))

    assert submit.call_count == 1, (
        f"resume paid for the same chunk twice: {submit.call_count} submits"
    )
    assert [line.custom_id for line, _r in out] == ["r0"]


@respx.mock
def test_the_intent_is_on_disk_before_the_provider_is_called(
    tmp_path, monkeypatch, groq_key
):
    # Ordering is the whole mechanism: if the intent were written after the
    # call, the crash window would still be open.
    ckpt = tmp_path / "job.jsonl"
    seen: list[bool] = []

    real_submit = ROWS["groq"]

    def spy(request):
        seen.append(ckpt.exists() and "key" in ckpt.read_text())
        return httpx.Response(200, json={"id": "b1", "status": "validating"})

    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f"})
    )
    respx.post(f"{GROQ.base_url}/batches").mock(side_effect=spy)
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "failed"})
    )
    del real_submit

    list(
        bl.run(
            [bl.BatchLine("r0", MODEL, [{"role": "user", "content": "a"}])],
            checkpoint=ckpt,
            poll_interval=0,
        )
    )
    assert seen == [True], "the intent must be durable before the provider is called"


@respx.mock
def test_the_submission_key_reaches_the_provider(tmp_path, groq_key):
    _, submitted = _mock_provider()
    list(
        bl.run(
            [bl.BatchLine("r0", MODEL, [{"role": "user", "content": "a"}])],
            checkpoint=tmp_path / "job.jsonl",
            poll_interval=0,
        )
    )
    # Without the stamp there is nothing to reconcile against.
    assert submitted[0]["metadata"][KEY_FIELD].startswith("bl-")


def test_the_key_is_deterministic_for_the_same_work():
    lines = [bl.BatchLine("r0", MODEL, [{"role": "user", "content": "a"}])]
    assert runner._chunk_key(lines, 0) == runner._chunk_key(lines, 0)
    assert runner._chunk_key(lines, 0) != runner._chunk_key(lines, 1)
