"""An OpenAI-compatible batch endpoint in front of every lane batchlane reaches.

The point is not the server, it is the protocol. Point the stock OpenAI SDK at
this and ``client.batches.create(...)`` submits to Groq, Gemini or Mistral
without the client knowing they differ. An R, JavaScript or curl client works
the same way, with no batchlane dependency at all.

**It holds no job state.** OpenAI's batch protocol is already state-passing at
the client boundary: the client receives an opaque ``batch_id`` and hands it
back on every later call. Since :class:`~batchlane.BatchHandle` is
JSON-serializable by design, the handle *is* the state, so the id carries it
rather than pointing at a row. One process, no database, no migrations,
nothing lost on restart.

Two consequences are accepted rather than hidden. Uploaded files must live
somewhere until a batch references them, so file blobs go on local disk. And
``GET /v1/batches`` cannot be answered by a server holding nothing, so it
returns an empty list and says why in its own documentation rather than
pretending to enumerate.
"""

# No `from __future__ import annotations` here, deliberately: it turns every
# annotation into a string, and FastAPI cannot resolve the forward references
# that produces for handlers defined inside build_app.

import base64
import hashlib
import json
import re
import secrets
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from .errors import BatchlaneError
from .handle import BatchHandle
from .registry import adapter_for_model, get_adapter, resolve_api_key
from .runner import plan

__all__ = ["build_app", "decode_batch_id", "encode_batch_id"]

BATCH_PREFIX = "batch_bl"

#: Ids this gateway issues, and therefore the only ones it will read back. Both
#: are checked against the pattern *and* for containment after resolution:
#: identifiers arriving in a JSON body are not normalised by the router the way
#: a path segment is, so a body field alone could otherwise walk out of the
#: store and have its contents shipped to a provider.
_FILE_ID = re.compile(r"file-[0-9a-f]{24}\Z")
_SPILL_DIGEST = re.compile(r"[0-9a-f]{32}\Z")
OUTPUT_PREFIX = "file-out-"

#: Past this, a base64 id stops fitting comfortably in a URL path, so the
#: handle list spills to disk and the id references it instead. Measured:
#: about 200 chunks fit under this, and 500 chunks would be 5.4kB.
_MAX_INLINE_TOKEN = 1500

#: An id longer than this was never issued by us, so it is rejected before
#: anything tries to decompress it.
_MAX_ACCEPTED_TOKEN = 4096

#: zlib will happily turn a small token into gigabytes. Measured: a 272kB id
#: expanded to 459MB. Decoding is bounded to a little over the largest handle
#: list this gateway can produce.
_MAX_DECOMPRESSED = 8 * 1024 * 1024


def _in_store(
    base: Path, name: str, pattern: re.Pattern[str], suffix: str = ""
) -> Path:
    """Resolve a caller-supplied identifier to a path inside ``base``.

    Args:
        base: The directory the result must sit inside.
        name: The caller-supplied identifier.
        pattern: What a legitimate identifier looks like.
        suffix: Extension to append, if any.

    Returns:
        The resolved path.

    Raises:
        BatchlaneError: If the name is not one we issued, or resolves outside
            ``base``.
    """
    if not pattern.match(name):
        raise BatchlaneError(f"Not an identifier this gateway issued: {name!r}")
    resolved = (base / f"{name}{suffix}").resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise BatchlaneError(f"Not an identifier this gateway issued: {name!r}")
    return resolved


def encode_batch_id(
    handles: Sequence[BatchHandle], spill_dir: Path | None = None
) -> str:
    """Pack a job's handles into an opaque batch id.

    Args:
        handles: One handle per chunk of the job.
        spill_dir: Where to write the handle list if it is too large to carry
            in the id itself.

    Returns:
        An id the client passes back on every later call.
    """
    blob = json.dumps([json.loads(h.to_json()) for h in handles])
    token = (
        base64.urlsafe_b64encode(zlib.compress(blob.encode(), 9)).decode().rstrip("=")
    )
    if len(token) <= _MAX_INLINE_TOKEN or spill_dir is None:
        return f"{BATCH_PREFIX}_{token}"
    # Too long for a URL path. Fall back to a reference, which is the only
    # place this gateway keeps job state at all.
    digest = hashlib.sha256(blob.encode()).hexdigest()[:32]
    (spill_dir / f"{digest}.json").write_text(blob)
    return f"{BATCH_PREFIX}ref_{digest}"


def decode_batch_id(batch_id: str, spill_dir: Path | None = None) -> list[BatchHandle]:
    """Recover a job's handles from its id.

    Args:
        batch_id: An id previously produced by :func:`encode_batch_id`.
        spill_dir: Where spilled handle lists were written.

    Returns:
        The handles, one per chunk.

    Raises:
        BatchlaneError: If the id is not one of ours or cannot be read.
    """
    raw = batch_id.removeprefix(OUTPUT_PREFIX)
    if raw.startswith(f"{BATCH_PREFIX}ref_"):
        digest = raw.removeprefix(f"{BATCH_PREFIX}ref_")
        path = _in_store(spill_dir or Path(), digest, _SPILL_DIGEST, ".json")
        if not path.exists():
            raise BatchlaneError(f"Unknown batch id {batch_id!r}.")
        blob = path.read_text()
    elif raw.startswith(f"{BATCH_PREFIX}_"):
        token = raw.removeprefix(f"{BATCH_PREFIX}_")
        if len(token) > _MAX_ACCEPTED_TOKEN:
            raise BatchlaneError("Batch id is longer than this gateway issues.")
        padded = token + "=" * (-len(token) % 4)
        try:
            packed = base64.urlsafe_b64decode(padded)
            # Bounded rather than zlib.decompress: an unbounded inflate of a
            # caller-supplied token is a denial of service, not a parse. A
            # 272kB id was measured expanding to 459MB.
            engine = zlib.decompressobj()
            blob = engine.decompress(packed, _MAX_DECOMPRESSED).decode()
            if not engine.eof:
                raise BatchlaneError("Batch id expands beyond any job we issue.")
        except BatchlaneError:
            raise
        except Exception as exc:
            raise BatchlaneError(f"Malformed batch id {batch_id!r}.") from exc
    else:
        raise BatchlaneError(f"Unknown batch id {batch_id!r}.")
    return [BatchHandle.from_json(json.dumps(item)) for item in json.loads(blob)]


def _aggregate(statuses: list[Any]) -> tuple[str, dict[str, int]]:
    """Reduce per-chunk statuses to one OpenAI-shaped status and counts.

    Args:
        statuses: One :class:`~batchlane.JobStatus` per chunk.

    Returns:
        An ``(status, request_counts)`` pair in OpenAI's vocabulary.
    """
    counts = {
        "total": sum(s.total or 0 for s in statuses),
        "completed": sum(s.succeeded or 0 for s in statuses),
        "failed": sum(s.failed or 0 for s in statuses),
    }
    states = {s.state for s in statuses}
    if states == {"succeeded"}:
        return "completed", counts
    if "failed" in states:
        return "failed", counts
    if "cancelled" in states:
        return "cancelled", counts
    if "expired" in states:
        return "expired", counts
    if states <= {"pending"}:
        return "validating", counts
    return "in_progress", counts


def build_app(storage: Path | None = None, api_key: str | None = None) -> Any:
    """Construct the ASGI application.

    Args:
        storage: Directory for uploaded files and spilled handle lists.
            Defaults to ``./batchlane-gateway``.
        api_key: If set, every request must carry it as a bearer token. The
            gateway spends the operator's provider credits, so anyone who can
            reach it can spend money; on anything but loopback, set this.

    Returns:
        A FastAPI application.

    Raises:
        BatchlaneError: If the optional server dependencies are not installed.
    """
    try:
        from fastapi import (
            Depends,
            FastAPI,
            File,
            Form,
            Header,
            HTTPException,
            UploadFile,
        )
        from fastapi.responses import PlainTextResponse
    except ImportError as exc:  # pragma: no cover - exercised by install shape
        raise BatchlaneError(
            "The gateway needs the optional server extras. Install with "
            "'pip install batchlane[serve]'."
        ) from exc

    root = Path(storage or "batchlane-gateway")
    files = root / "files"
    spill = root / "jobs"
    for directory in (files, spill):
        directory.mkdir(parents=True, exist_ok=True)

    # Handlers are sync on purpose: each one makes blocking calls to a
    # provider, and FastAPI runs sync endpoints in a threadpool rather than on
    # the event loop.
    def guard(authorization: Annotated[str | None, Header()] = None) -> None:
        """Reject a request that does not carry the configured bearer token.

        Args:
            authorization: The Authorization header, if the client sent one.

        Raises:
            HTTPException: If a key is configured and the request lacks it.
        """
        if api_key is None:
            return
        if not secrets.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(status_code=401, detail="Invalid API key.")

    app = FastAPI(
        title="batchlane",
        description="OpenAI-compatible batch gateway",
        dependencies=[Depends(guard)],
    )

    @app.post("/v1/files")
    def upload(
        file: Annotated[UploadFile, File()],
        purpose: Annotated[str, Form()] = "batch",
    ) -> dict[str, Any]:
        content = file.file.read()
        file_id = f"file-{hashlib.sha256(content).hexdigest()[:24]}"
        (files / file_id).write_bytes(content)
        return {
            "id": file_id,
            "object": "file",
            "bytes": len(content),
            "filename": file.filename or "input.jsonl",
            "purpose": purpose,
        }

    @app.get("/v1/files/{file_id}/content", response_class=PlainTextResponse)
    def content(file_id: str) -> str:
        # An output id carries the job, so results are fetched from the
        # providers rather than read back off disk.
        if file_id.startswith(OUTPUT_PREFIX):
            return _collect(file_id)
        return _stored(file_id).read_text()

    @app.post("/v1/batches")
    def create(payload: dict[str, Any]) -> dict[str, Any]:
        file_id = str(payload.get("input_file_id", ""))
        path = _stored(file_id)
        try:
            handles = _submit(path, payload.get("completion_window"))
        except BatchlaneError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        batch_id = encode_batch_id(handles, spill)
        return _batch_object(
            batch_id, "validating", {"total": 0, "completed": 0, "failed": 0}
        )

    @app.get("/v1/batches/{batch_id}")
    def retrieve(batch_id: str) -> dict[str, Any]:
        handles = _handles(batch_id)
        statuses = [
            get_adapter(h.provider).status(h, api_key=resolve_api_key(h.provider))
            for h in handles
        ]
        status, counts = _aggregate(statuses)
        return _batch_object(batch_id, status, counts)

    @app.post("/v1/batches/{batch_id}/cancel")
    def cancel(batch_id: str) -> dict[str, Any]:
        for handle in _handles(batch_id):
            get_adapter(handle.provider).cancel(
                handle, api_key=resolve_api_key(handle.provider)
            )
        return _batch_object(
            batch_id, "cancelling", {"total": 0, "completed": 0, "failed": 0}
        )

    @app.get("/v1/batches")
    def index() -> dict[str, Any]:
        # A server that holds no job state has nothing to enumerate. Saying so
        # beats inventing a list or failing a client that merely polls it.
        return {"object": "list", "data": [], "has_more": False}

    def _stored(file_id: str) -> Path:
        try:
            path = _in_store(files, file_id, _FILE_ID)
        except BatchlaneError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No such file: {file_id}")
        return path

    def _handles(batch_id: str) -> list[BatchHandle]:
        try:
            return decode_batch_id(batch_id, spill)
        except BatchlaneError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _collect(file_id: str) -> str:
        out = [
            json.dumps(
                {
                    "id": f"batch_req_{result.custom_id}",
                    "custom_id": result.custom_id,
                    "response": {
                        "status_code": result.status_code or 200,
                        "body": result.response,
                    },
                    "error": result.error,
                }
            )
            for handle in _handles(file_id)
            for result in get_adapter(handle.provider).results(
                handle, api_key=resolve_api_key(handle.provider)
            )
        ]
        return "\n".join(out) + "\n" if out else ""

    return app


def _submit(path: Path, window: str | None) -> list[BatchHandle]:
    """Parse an uploaded JSONL and submit it to the right provider.

    Args:
        path: The stored input file.
        window: Requested completion window, if the client set one.

    Returns:
        One handle per chunk.

    Raises:
        BatchlaneError: If the file carries no usable requests.
    """
    from .handle import BatchLine

    lines = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        body = record.get("body") or {}
        lines.append(
            BatchLine(
                custom_id=record.get("custom_id", f"row-{len(lines)}"),
                model=body.get("model", ""),
                messages=body.get("messages") or [],
                params={
                    k: v for k, v in body.items() if k not in ("model", "messages")
                },
            )
        )
    if not lines:
        raise BatchlaneError("The uploaded file contains no requests.")

    adapter, provider, _bare = adapter_for_model(lines[0].model)
    api_key = resolve_api_key(provider)
    chunking = plan(lines)
    return [
        adapter.submit(
            list(chunk), endpoint="chat.completions", window=window, api_key=api_key
        )
        for chunk in chunking.chunks
    ]


def _batch_object(batch_id: str, status: str, counts: dict[str, int]) -> dict[str, Any]:
    """Render a batch in OpenAI's shape.

    Args:
        batch_id: The opaque id carrying the job.
        status: OpenAI-vocabulary status.
        counts: Per-request counts.

    Returns:
        The batch object.
    """
    body: dict[str, Any] = {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "",
        "completion_window": "24h",
        "status": status,
        "request_counts": counts,
    }
    if status == "completed":
        body["output_file_id"] = f"{OUTPUT_PREFIX}{batch_id}"
    return body
