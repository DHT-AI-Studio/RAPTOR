"""Temporal operations — `/api/{version}/temporal/*` (temporal plan §5.9).

P12: `status` -- where the temporal processing of one uploaded source stands. Proxies Module 25's
`/personal/temporal/status/{version_id}`, scoped to the caller's own database.
P21: `ingest` -- new source material, inline text or an uploaded asset (app/services/temporal_ingest.py);
`expiry` -- when an inline source expires.
P17: `verify` -- an answer's temporal claims checked against the caller's facts.
TMP-7: `parse` -- the time a question asks about, read by Module 25's normaliser.
"""
from __future__ import annotations

from typing import Any

from aiokafka import AIOKafkaProducer  # type: ignore[import]
from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_storage_service
from app.core.config import Settings, get_settings
from app.routers.asset import _save_to_nfs, get_kafka_producer
from app.routers.search import get_personal_db_service
from app.services import temporal_ingest as ingest
from app.services.kafka_service import KafkaService
from app.services.personal_db_service import PersonalDBService
from app.services.storage_service import StorageService, User

router = APIRouter()


@router.get(
    "/capabilities",
    tags=["Temporal"],
    status_code=status.HTTP_200_OK,
    summary="Temporal features this deployment serves",
    description="""
Contract versions of `/search/tkg` (2: ranked and paged with `sort` / `cursor`, strictly checked), its match
modes, Allen relations and sorts, the temporal operations available, and the extraction pipeline's media
types, model and versions. Fields the plan defines but this deployment does not serve are listed under
`tkg.unsupported`; a version-2 request using one is refused with 400.
""",
)
async def temporal_capabilities(
    svc: PersonalDBService = Depends(get_personal_db_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    caps = await svc.temporal_capabilities(current_user["sub"])
    caps.setdefault("operations", {}).update(
        {"status": "/api/{version}/temporal/status/{version_id}", "capabilities": "/api/{version}/temporal/capabilities",
         "reason": "/api/{version}/temporal/reason/{version_id}", "ingest": "/api/{version}/temporal/ingest",
         "verify": "/api/{version}/temporal/verify", "expiry": "/api/{version}/temporal/expiry/{version_id}",
         "parse": "/api/{version}/temporal/parse"})
    caps["ingest"] = {"inline_content_types": list(ingest.INLINE_CONTENT_TYPES),      # P21: what is live
                      "asset_extensions": list(ingest.ASSET_EXTENSIONS),
                      "inline_retention_days": ingest.INLINE_RETENTION_DAYS,
                      "max_inline_bytes": ingest.MAX_INLINE_BYTES}
    if isinstance(caps.get("tkg"), dict):
        caps["tkg"]["route"] = "/api/{version}/search/tkg"
    return caps


@router.get(
    "/reason/{version_id}",
    tags=["Temporal"],
    status_code=status.HTTP_200_OK,
    summary="Reasoning result of one source",
    description="""
What RAPTOR inferred from the source's dates and the before / after / during links its text states:
`status` (consistent, inconsistent with a `witness` of the conflicting constraints, or no_links), the
`derived_facts` with their sound bounds and supporting facts, and the links used or left unresolved.
`stale` while the source's temporal job is still running.
""",
)
async def temporal_reason(
    version_id: str = Path(..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$"),
    svc: PersonalDBService = Depends(get_personal_db_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await svc.temporal_reason(current_user["sub"], version_id)


@router.post(
    "/reason/{version_id}/recompute",
    tags=["Temporal"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a recompute of one source's temporal reasoning",
)
async def temporal_reason_recompute(
    version_id: str = Path(..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$"),
    svc: PersonalDBService = Depends(get_personal_db_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await svc.temporal_reason(current_user["sub"], version_id, recompute=True)


class VerifyRequest(BaseModel):
    claims: list[dict[str, Any]] = Field(..., max_length=50)          # Module 25 claims.MAX_CLAIMS
    evidence: list[str] = Field(..., max_length=500)                # claims.MAX_EVIDENCE
    complete: bool = False
    timezone: str | None = None


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    timezone: str | None = None
    anchor: str | None = None


@router.post(
    "/parse",
    tags=["Temporal"],
    status_code=status.HTTP_200_OK,
    summary="The time a question asks about",
    description="""
Reads a question's time condition with the same grammar RAPTOR extracts dates with:

- `RESOLVED`: a `window` (`start` / `end` instants with their offset; a null side is unbounded) and its
  `match`, and/or a `reference` event with its `reference_relations`. "after 2011" →
  [2012-01-01T00:00:00.000, unbounded); "before 2013" → (unbounded, 2012-12-31T23:59:59.999]; `since` /
  `until` include the period; 「2011年以後」, 「2013年之前」, 以來, 為止 alike;
- `NO_TEMPORAL`: nothing in the text constrains time;
- `UNRESOLVED`: a date phrase the grammar cannot read, with its `reason` -- do not run the question
  without its time condition.

`timezone`: the zone calendar dates are read in (default: the deployment's). `anchor` (ISO 8601 with an
offset): the question's own time, which "last year" / 去年 are read from; without it they are `UNRESOLVED`.
Also returns `keywords`, `phrase`, `operator`, `precision` and `grammar_version`.
""",
)
async def temporal_parse(
    req: ParseRequest,
    svc: PersonalDBService = Depends(get_personal_db_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await svc.temporal_parse(current_user["sub"], req.model_dump(exclude_none=True))


@router.post(
    "/verify",
    tags=["Temporal"],
    status_code=status.HTTP_200_OK,
    summary="Verify an answer's temporal claims against your facts",
    description="""
Each claim cites fact IDs from the answer's `evidence` (the facts it was built from) and is checked
deterministically:

- `date` {`fact`, `date`, `endpoint`?}: the fact's dates entail the claimed date at the claimed precision
  ("2005" holds for a fact dated 2005-07; "July 12, 2005" does not);
- `ordering` {`subject`, `object`, `relations`, `wording`: `certain` | `possible`}: the claimed Allen
  relations cover every relation the two facts' dates allow (`certain`) or at least one (`possible`);
- `absence`: holds only when that retrieval was `complete`.

A cited fact must be in `evidence` and still active. Returns `verdicts` (`verified`, `reason`),
`all_verified` and the cited `facts`. `timezone`: the zone claimed calendar dates are read in.
""",
)
async def temporal_verify(
    req: VerifyRequest,
    svc: PersonalDBService = Depends(get_personal_db_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await svc.temporal_verify(current_user["sub"], req.model_dump(exclude_none=True))


@router.get(
    "/expiry/{version_id}",
    tags=["Temporal"],
    status_code=status.HTTP_200_OK,
    summary="When an inline source expires",
    description="""
Text sent inline to `POST /temporal/ingest` is kept for 7 days. `expires_at`, and `state`: `pending`,
`expired` (archived: left out of every query, never reactivated) or `deleted`. 404 for an uploaded
asset, which keeps its own lifecycle, or a source that is not yours.
""",
)
async def temporal_expiry(
    version_id: str = Path(..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$"),
    svc: PersonalDBService = Depends(get_personal_db_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await svc.temporal_expiry(current_user["sub"], version_id)


@router.get(
    "/status/{version_id}",
    tags=["Temporal"],
    status_code=status.HTTP_200_OK,
    summary="Temporal processing status of one source",
    description="""
Where RAPTOR's temporal fact extraction stands for one uploaded source version (the `version_id`
of the upload). Every medium -- document, video, audio, image -- gets one durable job.

**`status`:** `pending`, `running`, `complete`, `partial` (some segments failed or are in an
unsupported language), `failed` (nothing could be extracted) or `cancelled` (the source was
archived or deleted while it ran).

Also returned: segment counts, the number of facts written, the extractor / normaliser versions and
model, and the segments that failed or could not be read. 404 when the source has no temporal job
(not processed yet, processed before temporal jobs existed, deleted, or not the caller's).
""",
)
async def temporal_status(
    version_id: str = Path(..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$"),
    svc: PersonalDBService = Depends(get_personal_db_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return await svc.temporal_status(current_user["sub"], version_id)
    except HTTPException:
        raise                       # Module 25's status (404 = no job) passes through unchanged
    except Exception as e:  # noqa: BLE001 -- malformed-JSON ValueError etc.; converts to 500
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ingest",
    tags=["Temporal"],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest new source material for temporal processing",
    description="""
New source material for RAPTOR's temporal pipeline, in **exactly one** form:

- **Inline content** -- `content` with `content_type` `text/plain` or `text/markdown` (free-form notes and
  prose included, up to 1 MB). It becomes an asset of yours like an upload and is processed the same way.
  It **expires after 7 days**: from `expires_at` it is left out of every search and answer, and it is
  deleted afterwards.
- **An asset you uploaded** -- `asset: {asset_path, version_id}` (PDF, DOCX, PPTX, TXT, images, audio,
  video). Sent for analysis if it has not been; uploads keep their own expiry settings.

No temporal facts or conversions are expected from the client. `idempotency_key` is required: a retried
request with the same key returns the first result (a different request with the same key is `409`).

**Returns** `source_version_id` (also the `job_id`: poll `GET /api/{version}/temporal/status/{version_id}`),
`status` (`pending` until processing starts), `expires_at` (inline content) and `duplicate` (the same content
was already ingested; its source is returned).

**`operation`:**
- `ingest` (default) -- as above.
- `query` -- `question` over your existing sources, optionally only `source_version_ids`; nothing is stored.
  Returns `answer`, `sources`, `graph_context`, `temporal_claims` (200).
- `ingest_and_query` -- ingest, wait up to `wait_seconds` (default 60, max 300) for processing, then answer
  `question` from **this source only** (200, `answer_status: answered`). Still processing at the deadline:
  `answer_status: pending` and no answer -- poll the status, then ask with `operation: query` and
  `source_version_ids: [source_version_id]`. A failed job: `not_answered`.

**Errors:** `400 INVALID_INGEST` (both or neither form, empty content), `413 CONTENT_TOO_LARGE`,
`415 UNSUPPORTED_CONTENT_TYPE` (HTML, email, chat exports, CSV / JSON / logs are not served yet -- never
processed as plain text), `409 IDEMPOTENCY_KEY_REUSED`, `409 SOURCE_EXPIRED` (the same text expired and is
not deleted yet), `404` (an asset that is not yours), `502 ANSWER_UNAVAILABLE` (the answering service failed).
""",
)
async def temporal_ingest(
    req: ingest.IngestRequest,
    request: Request,
    response: Response,
    svc: StorageService = Depends(get_storage_service),
    pdb: PersonalDBService = Depends(get_personal_db_service),
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    settings: Settings = Depends(get_settings),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = current_user["sub"]
    ingest.check(req)
    authorization = request.headers.get("authorization")
    if req.operation == "query":                        # a question, never stored as a source
        response.status_code = status.HTTP_200_OK
        result = await ingest.ask(settings.agent_protocol_url, req.question, user_id, authorization,
                                  req.source_version_ids, req.top_k)
        return {"operation": "query", "source_version_ids": req.source_version_ids, **ingest.answered(result)}

    redis = getattr(request.app.state, "redis_client", None)
    out = await ingest.replay(redis, user_id, req)
    if out is None:
        out = await _ingest(req, user_id, svc, pdb, KafkaService(producer, settings, redis), settings)
        await ingest.remember(redis, user_id, req, out)
    if req.operation != "ingest_and_query":
        return out

    version_id = out["source_version_id"]
    final = await ingest.wait_for_job(pdb, user_id, version_id, req.wait_seconds)
    out = {**out, "operation": "ingest_and_query", "status": final}
    if final in ("complete", "partial"):
        response.status_code = status.HTTP_200_OK
        result = await ingest.ask(settings.agent_protocol_url, req.question, user_id, authorization,
                                  [version_id], req.top_k)
        return {**out, "answer_status": "answered", **ingest.answered(result)}
    # still running at the deadline: poll status, then ask with operation=query and this source --
    # an answer made without its facts would be a guess
    return {**out, "answer_status": "not_answered" if final in ingest.FINISHED else "pending", "answer": None}


async def _ingest(req: ingest.IngestRequest, user_id: str, svc: StorageService, pdb: PersonalDBService,
                  kafka: KafkaService, settings: Settings) -> dict[str, Any]:
    """Store / register the source material and send it for processing; the ingest result."""
    user = User(user_id=user_id, branch_id=user_id)
    duplicate = False
    expires_at: str | None = None
    await pdb.ensure_database(user_id)

    if req.content is not None:
        upload = ingest.inline_file(req)
        when = ingest.expiry()
        result = await svc.upload_file(primary_file=upload, archive_date_or_ttl=str(ingest.INLINE_RETENTION_DAYS),
                                       destroy_date_or_ttl=str(ingest.INLINE_RETENTION_DAYS), user=user)
        version_id, asset_path = result["version_id"], result["asset_path"]
        duplicate = bool((result.get("existence_info") or {}).get("exists"))
        if duplicate:                                  # the same text was ingested before: that source
            try:
                await svc.register_asset(asset_path, version_id, user)
            except HTTPException:
                pass                                   # already active
            expires_at = await ingest.expires_of(pdb, user_id, version_id, refuse_expired=True)
        else:
            temp = await _save_to_nfs(upload, result)
            await kafka.send_processing_request(upload_result=result, user_id=user_id, branch_id=user_id,
                                                processing_mode=None, temp_file_path=temp)
            expires_at = when.isoformat().replace("+00:00", "Z")
            await pdb.set_temporal_expiry(user_id, version_id, expires_at)
    else:
        version_id, asset_path = req.asset.version_id, req.asset.asset_path
        try:
            registered = await svc.register_asset(asset_path, version_id, user)
        except HTTPException as exc:
            if exc.status_code != status.HTTP_409_CONFLICT:
                raise                                  # 404: not the caller's asset
            registered = {"outcome": "already_processing_or_active"}
        if registered.get("outcome") == "needs_processing":
            meta = registered.get("metadata") or {}
            if ingest.extension(meta.get("primary_filename")) not in ingest.ASSET_EXTENSIONS:
                raise ingest.ingest_error(415, "UNSUPPORTED_CONTENT_TYPE", filename=meta.get("primary_filename"),
                                          supported=list(ingest.ASSET_EXTENSIONS))
            await kafka.send_processing_request(upload_result=meta, user_id=user_id, branch_id=user_id,
                                                processing_mode=None)
        expires_at = await ingest.expires_of(pdb, user_id, version_id)

    return {
        "operation": "ingest", "source_version_id": version_id, "job_id": version_id, "asset_path": asset_path,
        "status": await ingest.job_status(pdb, user_id, version_id), "expires_at": expires_at,
        "duplicate": duplicate, "status_url": f"/api/{settings.api_version}/temporal/status/{version_id}",
    }
