"""`POST /api/{version}/temporal/ingest` (temporal plan §5.7 / §5.9 / P21).

New source material for RAPTOR's temporal pipeline, in exactly one of two forms:

  - **inline content** (`content` + `content_type`): plain text or Markdown prose. It becomes a RAPTOR
    asset like any upload (Module 04, archive / destroy in INLINE_RETENTION_DAYS) and is processed by the
    document worker's text path, so evidence, status, retry and deletion work exactly as for an upload.
    Its expiry is registered with Module 25, which archives the source -- out of every query -- the moment
    it passes; Module 04 deletes it physically later.
  - **an asset reference** (`asset: {asset_path, version_id}`): an asset the caller already uploaded, sent
    for analysis if it has not been (Module 04's register guard decides).

The client never supplies temporal facts, and never converts anything itself. A content type outside
the delivered phase (HTML, email, chat exports, CSV / JSON / logs come after v1.0) is refused with
UNSUPPORTED_CONTENT_TYPE -- never processed as plain text. `idempotency_key` makes a retried request
return the first one's result.

`operation` (plan P21's dispatch): `ingest` (the default) as above; `query` -- a question over the caller's
existing sources, optionally scoped by `source_version_ids`, never stored as a source; `ingest_and_query`
-- ingest, wait up to `wait_seconds` for the temporal job, then answer the question from that source only
(Module 21, mode direct, scoped). A job still running at the deadline returns `answer_status: pending` --
never an answer made without it; a failed or cancelled job is `not_answered`.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import httpx
from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from starlette.datastructures import Headers

_logger = logging.getLogger(__name__)

INLINE_RETENTION_DAYS = 7
MAX_INLINE_BYTES = 1_000_000
INLINE_CONTENT_TYPES = ("text/plain", "text/markdown")      # before v1.0 (§5.7); both go through the text path
# Asset types the temporal pipeline serves before v1.0: documents (PDF, DOCX, PPTX, TXT), images, audio, video.
ASSET_EXTENSIONS = ("pdf", "docx", "pptx", "txt", "jpg", "jpeg", "png", "mp3", "wav", "flac", "aac", "ogg",
                    "wma", "m4a", "mp4", "avi", "mov", "wmv", "flv", "webm", "mkv", "3gp")
IDEMPOTENCY_TTL_SECONDS = INLINE_RETENTION_DAYS * 86400


class AssetRef(BaseModel):
    asset_path: str = Field(..., min_length=1)
    version_id: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    operation: Literal["ingest", "ingest_and_query", "query"] = Field(
        "ingest", description="`ingest`; `query` (no source material); `ingest_and_query` (answer from it only)")
    content: str | None = Field(None, description="Inline source text (UTF-8). Exactly one of content / asset")
    content_type: str | None = Field(None, description="With content: `text/plain` or `text/markdown`")
    filename: str | None = Field(None, max_length=100, description="With content: a name for the source")
    asset: AssetRef | None = Field(None, description="An asset the caller already uploaded")
    idempotency_key: str | None = Field(None, min_length=1, max_length=200,
                                        description="Required to ingest: a retried request with the same key "
                                                    "returns the first ingest result")
    question: str | None = Field(None, min_length=1, max_length=4000,
                                 description="With query / ingest_and_query: the question")
    source_version_ids: list[str] | None = Field(None, max_length=100,
                                                 description="With query: answer from these sources only")
    wait_seconds: int = Field(60, ge=0, le=300, description="With ingest_and_query: how long to wait for processing")
    top_k: int = Field(5, ge=1, le=50, description="With query / ingest_and_query: passages used to answer")


def ingest_error(code: int, error: str, **detail: Any) -> HTTPException:
    return HTTPException(status_code=code, detail={"error": error, **detail})


def check(req: IngestRequest) -> None:
    if req.operation == "query":
        if req.content is not None or req.asset is not None:
            raise ingest_error(400, "INVALID_INGEST", reason="`query` takes no source material")
        if not (req.question or "").strip():
            raise ingest_error(400, "INVALID_INGEST", reason="`query` needs a `question`")
        return
    if req.idempotency_key is None:
        raise ingest_error(400, "INVALID_INGEST", reason="`idempotency_key` is required to ingest")
    if req.source_version_ids is not None:
        raise ingest_error(400, "INVALID_INGEST", reason="`source_version_ids` goes with `query`")
    if req.operation == "ingest_and_query" and not (req.question or "").strip():
        raise ingest_error(400, "INVALID_INGEST", reason="`ingest_and_query` needs a `question`")
    if (req.content is None) == (req.asset is None):
        raise ingest_error(400, "INVALID_INGEST", reason="give exactly one of `content` and `asset`")
    if req.content is not None:
        ctype = (req.content_type or "").split(";")[0].strip().lower()
        if ctype not in INLINE_CONTENT_TYPES:
            raise ingest_error(415, "UNSUPPORTED_CONTENT_TYPE", content_type=req.content_type,
                               supported=list(INLINE_CONTENT_TYPES))
        size = len(req.content.encode("utf-8"))
        if not req.content.strip():
            raise ingest_error(400, "INVALID_INGEST", reason="`content` is empty")
        if size > MAX_INLINE_BYTES:
            raise ingest_error(413, "CONTENT_TOO_LARGE", bytes=size, limit=MAX_INLINE_BYTES)


def extension(filename: str | None) -> str:
    return os.path.splitext(filename or "")[1].lstrip(".").lower()


def inline_file(req: IngestRequest) -> UploadFile:
    """The inline text as an upload: a .txt file (the document worker's text path reads Markdown as text)."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(req.filename or "")[0]).strip("._") or "inline"
    return UploadFile(file=io.BytesIO(req.content.encode("utf-8")), filename=f"{stem[:80]}.txt",
                      headers=Headers({"content-type": "text/plain; charset=utf-8"}))


def fingerprint(req: IngestRequest) -> str:
    body = req.model_dump(exclude={"idempotency_key", "wait_seconds"}, mode="json")
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def idempotency_key(user_id: str, key: str) -> str:
    return f"temporal_ingest:{user_id}:{hashlib.sha256(key.encode()).hexdigest()[:32]}"


async def replay(redis, user_id: str, req: IngestRequest) -> dict[str, Any] | None:
    """The first result for this key, or None. The same key with a different body is a 409."""
    if redis is None:
        return None
    try:
        raw = await redis.get(idempotency_key(user_id, req.idempotency_key))
    except Exception as exc:  # noqa: BLE001 -- Redis down: proceed; Module 04's content dedup still holds
        _logger.warning("idempotency lookup failed: %s", exc)
        return None
    if not raw:
        return None
    stored = json.loads(raw)
    if stored.get("fingerprint") != fingerprint(req):
        raise ingest_error(409, "IDEMPOTENCY_KEY_REUSED", reason="the key was used for a different request")
    return stored["response"]


async def remember(redis, user_id: str, req: IngestRequest, response: dict[str, Any]) -> None:
    if redis is None:
        return
    try:
        await redis.setex(idempotency_key(user_id, req.idempotency_key), IDEMPOTENCY_TTL_SECONDS,
                          json.dumps({"fingerprint": fingerprint(req), "response": response}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 -- best effort, as above
        _logger.warning("idempotency store failed: %s", exc)


def expiry() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=INLINE_RETENTION_DAYS)


async def job_status(pdb, user_id: str, version_id: str) -> str:
    """The source's temporal job status, or `pending` before it has one."""
    try:
        return (await pdb.temporal_status(user_id, version_id)).get("status") or "pending"
    except HTTPException as exc:
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            return "pending"
        raise


async def expires_of(pdb, user_id: str, version_id: str, refuse_expired: bool = False) -> str | None:
    """The source's `expires_at` (ISO, UTC), None for an upload. With *refuse_expired*, a source already past
    it is a 409: the same text sent again cannot bring an expired source back."""
    try:
        found = await pdb.temporal_expiry(user_id, version_id)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            return None                        # an upload: Module 04's own lifecycle
        raise
    ms = found.get("expires_at")
    when = datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z") if ms else None
    if refuse_expired and found.get("expired"):
        raise ingest_error(409, "SOURCE_EXPIRED", source_version_id=version_id, expires_at=when,
                           reason="the same content was ingested before and has expired; it is deleted within a "
                                  "day, after which it can be sent again")
    return when


FINISHED = ("complete", "partial", "failed", "cancelled")
POLL_SECONDS = 3.0


async def wait_for_job(pdb, user_id: str, version_id: str, seconds: int) -> str:
    """The job's status once it finishes, or its last status at the deadline."""
    deadline = asyncio.get_event_loop().time() + seconds
    while True:
        current = await job_status(pdb, user_id, version_id)
        if current in FINISHED or asyncio.get_event_loop().time() >= deadline:
            return current
        await asyncio.sleep(POLL_SECONDS)


async def ask(agent_url: str, question: str, user_id: str, authorization: str | None,
              scope: list[str] | None, top_k: int) -> dict[str, Any]:
    """Module 21's answer (mode direct), scoped to *scope* when given."""
    headers = {"X-User-ID": user_id, "X-Branch-ID": user_id, **({"Authorization": authorization} if authorization else {})}
    body = {"question": question, "mode": "direct", "top_k": top_k,
            **({"source_version_ids": scope} if scope else {})}
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{agent_url.rstrip('/')}/query", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        _logger.error("ask failed: %s", exc)
        raise ingest_error(502, "ANSWER_UNAVAILABLE", reason=str(exc)) from exc


def answered(result: dict[str, Any]) -> dict[str, Any]:
    """What a temporal/ingest response carries of Module 21's answer."""
    return {"answer": result.get("answer"), "sources": result.get("sources") or [],
            "graph_context": result.get("graph_context") or "", "temporal_claims": result.get("temporal_claims") or []}
