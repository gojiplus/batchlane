"""The CLI: a file in, a file out, the caller's own columns preserved."""

import json

import httpx
import pytest
import respx

import batchlane as bl
from batchlane.adapters.openai_shaped import ROWS
from batchlane.cli import main

GROQ = ROWS["groq"]
MODEL = "groq/llama-3.3-70b-versatile"


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def _mock(answers):
    respx.post(f"{GROQ.base_url}/files").mock(
        return_value=httpx.Response(200, json={"id": "f1"})
    )
    respx.post(f"{GROQ.base_url}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "validating"})
    )
    respx.get(f"{GROQ.base_url}/batches/b1").mock(
        return_value=httpx.Response(
            200, json={"id": "b1", "status": "completed", "output_file_id": "o1"}
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
                            "body": {"choices": [{"message": {"content": t}}]},
                        },
                    }
                )
                for cid, t in answers.items()
            ),
        )
    )


def _write(tmp_path, lines, name="in.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


@respx.mock
def test_a_real_file_goes_through_to_a_real_file(tmp_path, groq_key):
    src = _write(tmp_path, [{"prompt": "a"}, {"prompt": "b"}])
    out = tmp_path / "out.jsonl"
    _mock({"row-0": "first", "row-1": "second"})

    assert (
        main(
            ["run", str(src), "--model", MODEL, "-o", str(out), "--poll-interval", "0"]
        )
        == 0
    )

    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert [g["answer"] for g in got] == ["first", "second"]


@respx.mock
def test_the_callers_own_columns_survive(tmp_path, groq_key):
    # The whole point of file-to-file: your data stays attached to its answer.
    src = _write(
        tmp_path,
        [
            {"id": "x1", "speaker": "Ana", "prompt": "a"},
            {"id": "x2", "speaker": "Bo", "prompt": "b"},
        ],
    )
    out = tmp_path / "out.jsonl"
    _mock({"row-0": "first", "row-1": "second"})
    main(["run", str(src), "--model", MODEL, "-o", str(out), "--poll-interval", "0"])

    got = {g["id"]: g for g in (json.loads(x) for x in out.read_text().splitlines())}
    assert got["x1"]["speaker"] == "Ana"
    assert got["x1"]["answer"] == "first"
    assert got["x2"]["speaker"] == "Bo"


@respx.mock
def test_bare_string_lines_work_without_a_prompt_field(tmp_path, groq_key):
    src = _write(tmp_path, ["a", "b"])
    out = tmp_path / "out.jsonl"
    _mock({"row-0": "first", "row-1": "second"})
    assert (
        main(
            ["run", str(src), "--model", MODEL, "-o", str(out), "--poll-interval", "0"]
        )
        == 0
    )
    assert len(out.read_text().splitlines()) == 2


@respx.mock
def test_id_field_is_reused_as_the_row_id(tmp_path, groq_key):
    src = _write(tmp_path, [{"sku": "A-1", "prompt": "a"}])
    out = tmp_path / "out.jsonl"
    _mock({"A-1": "answer for A-1"})
    main(
        [
            "run",
            str(src),
            "--model",
            MODEL,
            "-o",
            str(out),
            "--id-field",
            "sku",
            "--poll-interval",
            "0",
        ]
    )
    assert json.loads(out.read_text())["answer"] == "answer for A-1"


@respx.mock
def test_a_failed_row_is_written_with_its_error_not_dropped(tmp_path, groq_key):
    src = _write(tmp_path, [{"prompt": "a"}, {"prompt": "b"}])
    out = tmp_path / "out.jsonl"
    _mock({"row-0": "first"})  # row-1 never answered
    main(["run", str(src), "--model", MODEL, "-o", str(out), "--poll-interval", "0"])

    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(got) == 2, "a row without an answer must still appear in the output"
    missing = next(g for g in got if g["answer"] is None)
    assert "error" in missing


@respx.mock
def test_dry_run_reports_chunking_and_submits_nothing(tmp_path, groq_key, capsys):
    src = _write(tmp_path, [{"prompt": "a"}])
    out = tmp_path / "out.jsonl"
    assert (
        main(
            [
                "run",
                str(src),
                "--model",
                MODEL,
                "-o",
                str(out),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not respx.calls, "dry run must not touch the provider"
    assert not out.exists()
    assert "1 rows -> 1 batch" in capsys.readouterr().err


def test_a_missing_prompt_field_names_the_flag_that_fixes_it(tmp_path, capsys):
    src = _write(tmp_path, [{"text": "a"}])
    code = main(["run", str(src), "--model", MODEL, "-o", str(tmp_path / "o.jsonl")])
    assert code == 2
    assert "--prompt-field" in capsys.readouterr().err


def test_an_empty_input_file_is_refused(tmp_path, capsys):
    src = tmp_path / "empty.jsonl"
    src.write_text("\n")
    assert (
        main(["run", str(src), "--model", MODEL, "-o", str(tmp_path / "o.jsonl")]) == 2
    )
    assert "no rows" in capsys.readouterr().err


def test_providers_lists_every_shipped_lane(capsys):
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    for provider in bl.supported_providers():
        assert provider in out
