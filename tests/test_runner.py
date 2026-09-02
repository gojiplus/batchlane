"""The runner: chunking, waiting, stitching and resume."""

import dataclasses
import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

import batchlane as bl
from batchlane import runner
from batchlane.adapters.openai_shaped import ROWS
from batchlane.capabilities import CAPABILITIES

GROQ = ROWS["groq"]
MODEL = "groq/llama-3.3-70b-versatile"


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


@pytest.fixture
def tiny_caps(monkeypatch):
    """Shrink Groq's caps so chunking is exercised without megabytes of data."""

    def _apply(**kwargs):
        monkeypatch.setitem(
            CAPABILITIES, "groq", dataclasses.replace(CAPABILITIES["groq"], **kwargs)
        )

    return _apply


def _lines(n, text="hi"):
    return [
        bl.BatchLine(f"r{i}", MODEL, [{"role": "user", "content": f"{text} {i}"}])
        for i in range(n)
    ]


def _output(ids):
    return "\n".join(
        json.dumps(
            {
                "custom_id": cid,
                "response": {"status_code": 200, "body": {"answer": cid}},
            }
        )
        for cid in ids
    )


# --- plan() ---------------------------------------------------------------


def test_a_small_job_is_one_chunk():
    p = bl.plan(_lines(3))
    assert p.fits_in_one
    assert p.n_chunks == 1
    assert p.provider == "groq"
    assert p.total_bytes > 0


def test_a_job_over_the_request_cap_splits(tiny_caps):
    tiny_caps(max_requests=2)
    p = bl.plan(_lines(5))
    assert p.n_chunks == 3
    assert [len(c) for c in p.chunks] == [2, 2, 1]
    # Every row survives the split exactly once.
    assert sorted(ln.custom_id for c in p.chunks for ln in c) == [
        f"r{i}" for i in range(5)
    ]


def test_a_job_over_the_byte_cap_splits(tiny_caps):
    # Gemini's inline lane caps at 20MB, an order of magnitude under the
    # file-based providers, so byte-based splitting is not optional.
    one = bl.plan(_lines(1)).total_bytes
    tiny_caps(max_requests=None, max_input_bytes=int(one * 2 / 0.9) + 1)
    p = bl.plan(_lines(6))
    assert p.n_chunks > 1
    assert all(len(c) <= 2 for c in p.chunks)


def test_every_chunk_actually_fits_the_cap_it_was_split_for(tiny_caps):
    one = bl.plan(_lines(1)).total_bytes
    cap = int(one * 3 / 0.9)
    tiny_caps(max_requests=None, max_input_bytes=cap)
    adapter = bl.get_adapter("groq")
    for chunk in bl.plan(_lines(9)).chunks:
        assert adapter.payload_bytes(list(chunk), endpoint="chat.completions") <= cap


def test_a_single_row_too_large_to_send_raises_rather_than_making_a_bad_chunk(
    tiny_caps,
):
    tiny_caps(max_input_bytes=50)
    with pytest.raises(bl.BatchlaneError, match="on its own"):
        bl.plan(_lines(1, text="x" * 500))


def test_an_empty_job_is_refused():
    with pytest.raises(bl.BatchlaneError, match="empty"):
        bl.plan([])


# --- wait() ---------------------------------------------------------------


@respx.mock
def test_wait_polls_until_terminal(groq_key):
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        side_effect=[
            httpx.Response(200, json={"id": "b1", "status": "in_progress"}),
            httpx.Response(200, json={"id": "b1", "status": "completed"}),
        ]
    )
    handle = bl.BatchHandle(
        provider="groq",
        job_id="b1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )
    assert bl.wait(handle, poll_interval=0).state == "succeeded"


@respx.mock
def test_wait_times_out_without_losing_the_job(groq_key):
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "in_progress"})
    )
    handle = bl.BatchHandle(
        provider="groq",
        job_id="b1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )
    with pytest.raises(TimeoutError, match="not lost"):
        bl.wait(handle, poll_interval=0, timeout=0)


# --- run() ----------------------------------------------------------------


def _mock_one_job(ids, job_id="b1"):
    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": f"f-{job_id}"})
    )
    submit = respx.post(f"{GROQ.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": job_id, "status": "validating"})
    )
    respx.get(f"{GROQ.base_url}/batches/{job_id}").mock(
        return_value=httpx.Response(
            200,
            json={"id": job_id, "status": "completed", "output_file_id": f"o-{job_id}"},
        )
    )
    respx.get(f"{GROQ.base_url}/files/o-{job_id}/content").mock(
        return_value=httpx.Response(200, text=_output(ids))
    )
    return submit


@respx.mock
def test_run_hands_back_each_input_row_beside_its_answer(groq_key):
    _mock_one_job(["r0", "r1", "r2"])
    lines = _lines(3)
    pairs = list(bl.run(lines, poll_interval=0))
    assert len(pairs) == 3
    for line, result in pairs:
        # The caller's own object comes back, so nothing needs joining.
        assert isinstance(line, bl.BatchLine)
        assert result.response["answer"] == line.custom_id
        assert line.model.startswith("groq/"), "caller's model string must survive"


@respx.mock
def test_results_returned_out_of_order_still_pair_with_the_right_row(groq_key):
    # Anthropic and Gemini both do this in real life.
    _mock_one_job(["r2", "r0", "r1"])
    pairs = list(bl.run(_lines(3), poll_interval=0))
    assert all(result.response["answer"] == line.custom_id for line, result in pairs)


@respx.mock
def test_a_row_the_provider_never_answered_is_reported_not_dropped(groq_key):
    # Silence is not success. Every submitted row must be accounted for.
    _mock_one_job(["r0", "r2"])
    pairs = {
        line.custom_id: result for line, result in bl.run(_lines(3), poll_interval=0)
    }
    assert set(pairs) == {"r0", "r1", "r2"}
    assert pairs["r1"].error["reason"] == "no result returned"
    assert not pairs["r1"].ok


@respx.mock
def test_a_failed_job_yields_an_error_per_row_rather_than_raising(groq_key):
    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f1"})
    )
    respx.post(f"{GROQ.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "validating"})
    )
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "failed"})
    )
    pairs = list(bl.run(_lines(2), poll_interval=0))
    assert len(pairs) == 2
    assert all(r.error["job_state"] == "failed" for _l, r in pairs)


@respx.mock
def test_a_multi_chunk_job_submits_once_per_chunk_and_returns_every_row(
    groq_key, tiny_caps
):
    tiny_caps(max_requests=2)
    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f"})
    )
    submit = respx.post(f"{GROQ.base_url}/batches").mock(
        side_effect=[
            httpx.Response(200, json={"id": "b1", "status": "validating"}),
            httpx.Response(200, json={"id": "b2", "status": "validating"}),
        ]
    )
    for job, ids in (("b1", ["r0", "r1"]), ("b2", ["r2"])):
        respx.get(f"{GROQ.base_url}/batches/{job}").mock(
            return_value=httpx.Response(
                200,
                json={"id": job, "status": "completed", "output_file_id": f"o-{job}"},
            )
        )
        respx.get(f"{GROQ.base_url}/files/o-{job}/content").mock(
            return_value=httpx.Response(200, text=_output(ids))
        )
    pairs = list(bl.run(_lines(3), poll_interval=0))
    assert submit.call_count == 2
    assert sorted(line.custom_id for line, _r in pairs) == ["r0", "r1", "r2"]


# --- resume: the check written to be able to fail -------------------------


@respx.mock
def test_resume_re_reads_finished_jobs_instead_of_re_paying_for_them(
    groq_key, tiny_caps, tmp_path
):
    """A resume that silently re-submits would pass "did it finish" and fail here.

    Two chunks. The first run dies after chunk 0 is submitted. The second run
    resumes from the checkpoint. Across BOTH runs the submit endpoint must be
    hit exactly twice -- once per chunk -- not four times. Inference is the
    expensive part; re-reading results is free, which is why the checkpoint
    stores handles rather than results.
    """
    tiny_caps(max_requests=2)
    checkpoint = tmp_path / "job.jsonl"

    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f"})
    )
    submit = respx.post(f"{GROQ.base_url}/batches").mock(
        side_effect=[
            httpx.Response(200, json={"id": "b1", "status": "validating"}),
            httpx.Response(200, json={"id": "b2", "status": "validating"}),
        ]
    )
    for job, ids in (("b1", ["r0", "r1"]), ("b2", ["r2"])):
        respx.get(f"{GROQ.base_url}/batches/{job}").mock(
            return_value=httpx.Response(
                200,
                json={"id": job, "status": "completed", "output_file_id": f"o-{job}"},
            )
        )
        respx.get(f"{GROQ.base_url}/files/o-{job}/content").mock(
            return_value=httpx.Response(200, text=_output(ids))
        )

    # First run: take only the first chunk's rows, then abandon the generator.
    gen = bl.run(_lines(3), checkpoint=checkpoint, poll_interval=0)
    first = [next(gen), next(gen)]
    gen.close()

    assert submit.call_count == 1
    assert checkpoint.exists(), "the handle must be persisted before anything can fail"

    # Second run: same inputs, same checkpoint.
    second = list(bl.run(_lines(3), checkpoint=checkpoint, poll_interval=0))

    assert submit.call_count == 2, (
        f"resume re-submitted an already-paid-for chunk "
        f"({submit.call_count} submits for 2 chunks)"
    )
    assert sorted(line.custom_id for line, _r in second) == ["r0", "r1", "r2"]
    assert [line.custom_id for line, _r in first] == ["r0", "r1"]


@respx.mock
def test_the_checkpoint_records_what_was_submitted_not_the_results(groq_key, tmp_path):
    # Keeps the file small, and means a resume re-attaches to the provider's
    # own retained output rather than trusting a local copy.
    _mock_one_job(["r0", "r1"])
    checkpoint = tmp_path / "job.jsonl"
    list(bl.run(_lines(2), checkpoint=checkpoint, poll_interval=0))
    lines = [json.loads(x) for x in checkpoint.read_text().splitlines()]
    # Two records per chunk, in this order: the intent, written before the
    # provider is called, then the receipt once it answers. The order is the
    # mechanism -- reversed, a crash mid-submit would leave no trace.
    intent, receipt = lines[0], lines[1]
    assert set(intent) == {"chunk", "key", "at"}
    assert intent["key"].startswith("bl-")
    assert set(receipt) == {"chunk", "custom_ids", "handle"}
    assert receipt["handle"]["job_id"] == "b1"
    assert "response" not in checkpoint.read_text()


@respx.mock
def test_polling_backs_off_instead_of_hammering_a_job_that_runs_for_hours(
    groq_key, monkeypatch
):
    # A six-hour job at a fixed 30s is 720 polls per chunk that learn nothing.
    slept: list[float] = []
    monkeypatch.setattr(runner.time, "sleep", slept.append)
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        side_effect=[httpx.Response(200, json={"id": "b1", "status": "in_progress"})]
        * 6
        + [httpx.Response(200, json={"id": "b1", "status": "completed"})]
    )
    handle = bl.BatchHandle(
        provider="groq",
        job_id="b1",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=bl.handle.utcnow(),
    )
    bl.wait(handle, poll_interval=30)

    assert slept == sorted(slept), "intervals must not shrink"
    assert slept[-1] > slept[0], "the interval must actually grow"
    assert max(slept) <= runner._BACKOFF_CAP_SECONDS, "and must stay capped"


@respx.mock
def test_anthropic_refuses_to_guess_when_recovery_is_ambiguous(monkeypatch):
    # Anthropic gives a batch no label, so recovery matches on creation time
    # and row count. When two batches fit, adopting one risks collecting
    # someone else's answers; resubmitting risks paying twice. Neither is
    # acceptable silently.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    adapter = bl.get_adapter("anthropic")
    respx.get("https://api.anthropic.com/v1/messages/batches").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "msgbatch_a",
                        "created_at": "2026-09-01T12:00:00+00:00",
                        "request_counts": {"processing": 2},
                    },
                    {
                        "id": "msgbatch_b",
                        "created_at": "2026-09-01T12:00:01+00:00",
                        "request_counts": {"processing": 2},
                    },
                ]
            },
        )
    )
    with pytest.raises(bl.BatchlaneError, match="Refusing to guess"):
        adapter.find_submitted(
            "bl-x-0",
            api_key="k",
            expected_rows=2,
            since=datetime(2026, 9, 1, 11, 0, tzinfo=UTC),
        )


def test_a_plan_warns_about_the_lane_before_the_job_runs(monkeypatch):
    # The capability table always held these; a caller had to go looking.
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    rows = [
        bl.BatchLine(
            "r0", "gemini/gemini-2.5-flash", [{"role": "user", "content": "x"}]
        )
    ]
    caveats = " ".join(bl.plan(rows).caveats).lower()
    assert "array index" in caveats, "the join hazard must be stated up front"
    assert "48h" in caveats


def test_caveats_carry_user_consequences_not_maintainer_provenance():
    # `notes` records where a value came from and which doc contradicts which.
    # That belongs to whoever maintains the lane, not to a caller deciding
    # whether to run a job.
    from batchlane.capabilities import CAPABILITIES

    for provider, caps in CAPABILITIES.items():
        joined = " ".join(caps.caveats).lower()
        for leak in ("upload purpose", "metadata.state", "api reference", "docs.deep"):
            assert leak not in joined, (
                f"{provider} leaks implementation detail to users"
            )


def test_a_split_job_says_so_because_it_changes_what_to_expect(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    caps = CAPABILITIES["groq"]
    monkeypatch.setitem(CAPABILITIES, "groq", dataclasses.replace(caps, max_requests=2))
    rows = [
        bl.BatchLine(f"r{i}", MODEL, [{"role": "user", "content": "x"}])
        for i in range(5)
    ]
    assert any("3 separate provider jobs" in c for c in bl.plan(rows).caveats)
