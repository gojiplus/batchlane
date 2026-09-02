"""Command line entry point, for the file-to-file case.

Bulk work usually arrives as a file rather than as a Python list, so this
reads JSONL, submits it, and writes JSONL back. Every field on an input line
is copied to its output line beside a new ``answer``, which keeps a caller's
own columns attached to their results. That mirrors what :func:`batchlane.run`
does by handing back the caller's own row.

Uses ``argparse`` rather than a CLI framework: batchlane declares three
runtime dependencies and a library is a poor place to add more for one
command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .errors import BatchlaneError
from .handle import BatchLine
from .registry import supported_providers
from .runner import answer_text, plan, run

__all__ = ["main"]

DEFAULT_PROMPT_FIELD = "prompt"


def _read_rows(path: Path, prompt_field: str) -> list[tuple[dict[str, Any], str]]:
    """Read JSONL input into (row, prompt) pairs.

    Args:
        path: The input file.
        prompt_field: Field holding the prompt when a line is an object.

    Returns:
        One pair per non-blank line, preserving file order.

    Raises:
        BatchlaneError: If a line is neither a string nor an object carrying
            ``prompt_field``.
    """
    rows: list[tuple[dict[str, Any], str]] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            rows.append(({prompt_field: parsed}, parsed))
        elif isinstance(parsed, dict) and prompt_field in parsed:
            rows.append((parsed, str(parsed[prompt_field])))
        else:
            raise BatchlaneError(
                f"{path}:{number} is neither a JSON string nor an object with a "
                f"{prompt_field!r} field. Use --prompt-field to name the right one."
            )
    if not rows:
        raise BatchlaneError(f"{path} has no rows.")
    return rows


def _build(
    rows: list[tuple[dict[str, Any], str]],
    model: str,
    system: str | None,
    id_field: str | None,
    params: dict[str, Any],
) -> list[BatchLine]:
    """Turn input rows into batch lines.

    Args:
        rows: (row, prompt) pairs from :func:`_read_rows`.
        model: Provider-prefixed model string.
        system: Optional system prompt for every row.
        id_field: Field to reuse as the row id, if present.
        params: Extra model parameters.

    Returns:
        One :class:`~batchlane.BatchLine` per row.
    """
    lines = []
    for index, (row, prompt) in enumerate(rows):
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        custom_id = (
            str(row.get(id_field, f"row-{index}")) if id_field else f"row-{index}"
        )
        lines.append(BatchLine(custom_id, model, messages, dict(params)))
    return lines


def _cmd_run(args: argparse.Namespace) -> int:
    """Submit a file and write answers beside their inputs.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit status.
    """
    rows = _read_rows(Path(args.input), args.prompt_field)
    params: dict[str, Any] = {}
    if args.max_tokens is not None:
        params["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        params["temperature"] = args.temperature
    lines = _build(rows, args.model, args.system, args.id_field, params)

    chunking = plan(lines)
    print(  # noqa: T201 - this is a CLI
        f"{len(lines)} rows -> {chunking.n_chunks} "
        f"{'batch' if chunking.n_chunks == 1 else 'batches'} "
        f"on {chunking.provider} ({chunking.total_bytes} bytes)",
        file=sys.stderr,
    )
    if args.dry_run:
        cost = chunking.cost
        print(f"  {cost}", file=sys.stderr)  # noqa: T201 - a CLI
        if cost.caveat:
            print(f"  note: {cost.caveat}", file=sys.stderr)  # noqa: T201 - a CLI
        return 0

    by_id = {line.custom_id: row for line, (row, _p) in zip(lines, rows, strict=True)}
    written = failed = 0
    with Path(args.output).open("w") as out:
        for line, result in run(
            lines, checkpoint=args.checkpoint, poll_interval=args.poll_interval
        ):
            answer = answer_text(result)
            failed += answer is None
            record = dict(by_id[line.custom_id])
            record["answer"] = answer
            if answer is None and result.error:
                record["error"] = result.error
            out.write(json.dumps(record) + "\n")
            out.flush()
            written += 1
    print(  # noqa: T201 - this is a CLI
        f"wrote {written} rows to {args.output}"
        + (f" ({failed} failed)" if failed else ""),
        file=sys.stderr,
    )
    return 1 if failed == written else 0


def _cmd_providers(args: argparse.Namespace) -> int:
    """List the lanes batchlane can reach.

    Args:
        args: Parsed arguments, unused.

    Returns:
        Process exit status.
    """
    del args
    from .capabilities import capabilities_for

    for name in supported_providers():
        caps = capabilities_for(name)
        if not caps or caps.window is None:
            window = "provider-set"
        elif caps.window.allowed:
            window = "/".join(caps.window.allowed)
        else:
            # An empty allowed tuple means unconstrained, not forbidden: these
            # lanes take an integer hour count rather than a fixed enum.
            window = f"any (default {caps.window.default})"
        # Notes carry real caveats (allowlists, excluded models) and vary a
        # lot in length, so they take the last column rather than a padded
        # middle one that only lines up for the short ones.
        note = (caps.discount_note or "") if caps else ""
        print(f"  {name:14s} window={window:12s} {note}")  # noqa: T201 - a CLI
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the OpenAI-compatible gateway.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit status.

    Raises:
        BatchlaneError: If the optional server extras are not installed.
    """
    # The safety check comes first on purpose. Whether uvicorn is installed
    # has nothing to do with whether this address is safe to serve on, and
    # reporting a missing dependency instead would bury the real problem.
    key = args.api_key or os.environ.get("BATCHLANE_GATEWAY_KEY")
    if key is None and args.host not in ("127.0.0.1", "localhost", "::1"):
        raise BatchlaneError(
            f"Refusing to serve on {args.host} without a key. This gateway "
            f"spends your provider credits, so anyone who can reach it can "
            f"spend money. Pass --api-key or set BATCHLANE_GATEWAY_KEY."
        )

    try:
        import uvicorn
    except ImportError as exc:
        raise BatchlaneError(
            "The gateway needs the optional server extras. Install with "
            "'pip install batchlane[serve]'."
        ) from exc

    from .gateway import build_app

    print(  # noqa: T201 - a CLI
        f"batchlane gateway on http://{args.host}:{args.port}/v1 -- point an "
        f"OpenAI client's base_url here",
        file=sys.stderr,
    )
    uvicorn.run(
        build_app(Path(args.storage), api_key=key), host=args.host, port=args.port
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The parser, exposed separately so tests can exercise it directly.
    """
    parser = argparse.ArgumentParser(
        prog="batchlane",
        description="Run batch inference at a provider's discount rate.",
    )
    parser.add_argument(
        "--version", action="version", version=f"batchlane {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="submit a JSONL file and write answers")
    run_cmd.add_argument("input", help="input .jsonl, one row per line")
    run_cmd.add_argument(
        "--model", required=True, help="e.g. groq/llama-3.3-70b-versatile"
    )
    run_cmd.add_argument("-o", "--output", required=True, help="output .jsonl")
    run_cmd.add_argument("--prompt-field", default=DEFAULT_PROMPT_FIELD)
    run_cmd.add_argument(
        "--id-field", default=None, help="reuse this field as the row id"
    )
    run_cmd.add_argument("--system", default=None, help="system prompt for every row")
    run_cmd.add_argument("--max-tokens", type=int, default=None)
    run_cmd.add_argument("--temperature", type=float, default=None)
    run_cmd.add_argument(
        "--checkpoint", default=None, help="record jobs here to allow resume"
    )
    run_cmd.add_argument("--poll-interval", type=float, default=30.0)
    run_cmd.add_argument(
        "--dry-run", action="store_true", help="report the chunking and stop"
    )
    run_cmd.set_defaults(func=_cmd_run)

    providers = sub.add_parser("providers", help="list reachable batch lanes")
    providers.set_defaults(func=_cmd_providers)

    serve = sub.add_parser(
        "serve", help="run an OpenAI-compatible batch endpoint over every lane"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--storage", default="batchlane-gateway")
    serve.add_argument(
        "--api-key",
        default=None,
        help="require this bearer token; mandatory off loopback",
    )
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments, or None to read from the command line.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BatchlaneError as exc:
        print(f"batchlane: {exc}", file=sys.stderr)  # noqa: T201 - this is a CLI
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
