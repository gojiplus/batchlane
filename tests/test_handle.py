from datetime import UTC, datetime

from batchlane.handle import BatchHandle, JobStatus, RequestResult


def test_handle_survives_a_round_trip_through_json():
    # The entire point of a handle: poll a job from a process that did not
    # submit it.
    original = BatchHandle(
        provider="groq",
        job_id="batch_abc",
        endpoint="chat.completions",
        lane="batch_file",
        created_at=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
        model=None,
        extra={"input_file_id": "file_xyz"},
    )
    assert BatchHandle.from_json(original.to_json()) == original


def test_terminal_states():
    assert JobStatus(state="succeeded", raw_state="completed").is_terminal
    assert JobStatus(state="expired", raw_state="expired").is_terminal
    assert not JobStatus(state="running", raw_state="in_progress").is_terminal


def test_result_ok_requires_a_response_and_no_error():
    assert RequestResult("a", response={"x": 1}).ok
    assert not RequestResult("a", error={"m": "boom"}).ok
    assert not RequestResult("a").ok
