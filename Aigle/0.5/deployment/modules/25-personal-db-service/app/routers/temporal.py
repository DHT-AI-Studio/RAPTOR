"""Temporal operations (temporal plan §5.9). P12: `status` -- the durable temporal job of one source.
P09: `capabilities` -- what this deployment serves, so a client checks before it relies on a feature.
TMP-7: `parse` -- the time a question asks about, read by the same normaliser as extraction."""
from __future__ import annotations

import logging
import zoneinfo
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.graph_search import V2_UNSUPPORTED
from app.services import searcher, source_expiry, temporal_jobs
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import DatabaseNotInitializedError
from app.services.temporal_normalize import NORMALIZER_VERSION
from app.temporal import allen, anchor, contract, reference
from app.temporal import claims as claims_mod
from app.temporal import query as query_mod
from app.temporal.extract import EXTRACTOR_VERSION
from app.temporal.normalize import NORMALIZER_VERSION as GRAMMAR_VERSION

logger = logging.getLogger("personal_db.routers.temporal")

router = APIRouter(prefix="/personal/temporal", tags=["temporal"])
client = ArcadeDBClient()


def _require_branch(x_branch_id: str | None) -> str:
    if not x_branch_id:
        raise HTTPException(status_code=400, detail="Missing X-Branch-ID header")
    return x_branch_id


def capabilities() -> dict[str, Any]:
    """The contract versions, TKG features and temporal pipeline this deployment actually serves (§5.9)."""
    return {
        "contract_versions": list(contract.CONTRACT_VERSIONS),
        # PD_TEMPORAL_V2: Module 18 reads questions through `parse` only once this is on (TMP-7)
        "temporal_v2": settings.temporal_v2,
        "tkg": {
            "route": "/personal/search/tkg",
            "match": ["inside", "overlaps", "contains", "any"],
            "relations": list(allen.RELATIONS),
            "certainty": ["possible", "certain"],
            "sort": list(contract.SORTS),              # v2
            "pagination": "cursor",                    # v2: keyset, bound to its query
            "timelines": list(contract.TIMELINES),
            "timezone": {"default": settings.temporal_default_timezone, "per_request": True},   # v2
            "retrieval": ["entity", "keywords", "temporal"],
            "fact_fields": ["when", "precision", "allen", "date_text", "source_sentence", "extent", "anchor",
                            "derived", "support_ids"],                 # P15
            "reference": {"forms": ["fact_id", "text"], "statuses": list(reference.STATUSES),   # P16 (v2)
                          "min_coverage": reference.MIN_COVERAGE},
            "unsupported": list(V2_UNSUPPORTED),       # refused in a v2 request, never ignored
        },
        "operations": {"status": "/personal/temporal/status/{version_id}", "capabilities": "/personal/temporal/capabilities",
                       "reason": "/personal/temporal/reason/{version_id}",     # P15
                       "verify": "/personal/temporal/verify",                  # P17
                       "expiry": "/personal/temporal/expiry/{version_id}",     # P21
                       "parse": "/personal/temporal/parse",                   # TMP-7
                       "ingest": None},
        "extraction": {
            "jobs": settings.temporal_jobs_enabled,
            "media_types": ["documents", "videos", "audios", "images"],
            "model": settings.chat_model_name if settings.temporal_llm_enabled else None,
            "extractor_version": EXTRACTOR_VERSION,
            "normalizer_version": NORMALIZER_VERSION,
            "grammar_version": GRAMMAR_VERSION,
            "languages": ["en", "zh"],
            "deictic_anchors": list(anchor.SOURCES),        # P13: file dates "two years ago" is read from
        },
    }


@router.get("/capabilities", summary="What this deployment's temporal features are (P09)")
async def temporal_capabilities() -> dict[str, Any]:
    """Contract versions, TKG match modes / Allen relations / sorts / pagination, operations, and the
    extraction pipeline's media types, model and versions. Fields the plan defines but this deployment does
    not serve are listed under `tkg.unsupported` (a v2 request using them is refused) or as null operations."""
    return capabilities()


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    timezone: str | None = None             # the zone calendar dates are read in (default: the deployment's)
    anchor: str | None = None               # ISO 8601: the query's own time, for "last year" / 去年


@router.post("/parse", summary="The time a question asks about (TMP-7)")
async def temporal_parse(req: ParseRequest) -> dict[str, Any]:
    """`status`: `RESOLVED` -- a `window` ({start, end} as instants with their offset, plus `start_ms` /
    `end_ms`; a null side is unbounded) and its `match`, and/or a `reference` event with its
    `reference_relations`; `NO_TEMPORAL` -- nothing in the text constrains time; `UNRESOLVED` -- a date
    phrase the grammar cannot read, with its `reason` (e.g. `invalid_calendar`, or `missing_anchor` for
    "yesterday" without an `anchor`). A caller must not run the question without its time condition.

    after 2011 -> [2012-01-01T00:00:00.000, unbounded); before 2013 -> (unbounded, 2012-12-31T23:59:59.999];
    since / until include the period; 「2011年以後」 / 「2013年之前」 / 以來 / 為止 alike. Also `phrase`,
    `operator`, `precision`, `keywords` (the terms a TKG search filters the evidence by), `timezone`,
    `grammar_version`. No database is read; no tenant is needed."""
    tz = req.timezone or settings.temporal_default_timezone
    try:
        zoneinfo.ZoneInfo(tz)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "INVALID_TIMEZONE", "timezone": tz}) from exc
    if req.anchor:
        try:
            if datetime.fromisoformat(req.anchor.replace("Z", "+00:00")).tzinfo is None:
                raise ValueError("no offset")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": "INVALID_ANCHOR", "anchor": req.anchor,
                                                         "expected": "ISO 8601 with an offset"}) from exc
    return query_mod.parse_query(req.text, tz, req.anchor)


class VerifyRequest(BaseModel):
    claims: list[dict[str, Any]] = Field(..., max_length=claims_mod.MAX_CLAIMS)
    evidence: list[str] = Field(..., max_length=claims_mod.MAX_EVIDENCE)   # the fact IDs the answer was built from
    complete: bool = False                  # whether that retrieval was complete (TKG `complete`)
    timezone: str | None = None             # the zone claimed calendar dates are read in


@router.post("/verify", summary="Verify an answer's temporal claims against the facts (P17)")
async def temporal_verify(req: VerifyRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")) -> dict[str, Any]:
    """Each claim -- `date` {fact, date, endpoint}, `ordering` {subject, object, relations, wording} or
    `absence` -- is checked deterministically (app.temporal.claims): its cited facts are in `evidence` and
    active; a date is entailed by the fact's dates at the claimed precision; an ordering's claimed relations
    cover every relation the dates allow (`certain`) or at least one (`possible`); absence needs a complete
    retrieval. Returns `verdicts` (verified, reason, what was found), `all_verified`, and the cited `facts`."""
    branch_id = _require_branch(x_branch_id)
    if req.timezone:
        try:
            zoneinfo.ZoneInfo(req.timezone)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"error": "INVALID_TIMEZONE", "timezone": req.timezone}) from exc
    try:
        return await searcher.verify_claims(client, branch_id, req.claims, req.evidence, req.complete, req.timezone)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class ExpiryRequest(BaseModel):
    expires_at: str                         # ISO 8601 with a zone, e.g. 2026-10-28T03:00:00Z


@router.put("/expiry/{version_id}", summary="Register when an inline source expires (P21)")
async def temporal_expiry_set(version_id: str, req: ExpiryRequest,
                              x_branch_id: str = Header(None, alias="X-Branch-ID")) -> dict[str, Any]:
    """Called by the gateway's temporal/ingest for inline content, before the source is indexed. At
    `expires_at` the source is archived -- left out of every query, its jobs cancelled -- and it is never
    reactivated; Module 04 deletes it physically later. An expiry already reached is not lifted."""
    branch_id = _require_branch(x_branch_id)
    try:
        when = datetime.fromisoformat(req.expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": "INVALID_EXPIRES_AT", "expires_at": req.expires_at}) from exc
    if when.tzinfo is None:
        raise HTTPException(status_code=400, detail={"error": "INVALID_EXPIRES_AT", "reason": "a zone is required"})
    return await source_expiry.register(client, branch_id, version_id, int(when.timestamp() * 1000))


@router.get("/expiry/{version_id}", summary="When an inline source expires, and its state (P21)")
async def temporal_expiry_get(version_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")) -> dict[str, Any]:
    """`state`: pending | expired | deleted; `expired` is true once `expires_at` has passed. 404 for a source
    with no registered expiry (an upload keeps Module 04's own lifecycle)."""
    found = await source_expiry.get(client, _require_branch(x_branch_id), version_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"No expiry registered for version {version_id}")
    return found


@router.get("/reason/{version_id}", summary="Stored reasoning result of one source version (P15)")
async def temporal_reason(version_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")) -> dict[str, Any]:
    """The result of ingestion-time reasoning over the source's facts and the qualitative links between them:
    `status` consistent | inconsistent (with a `witness`: the conflicting constraints and what states each) |
    no_links; the `derived_facts` it inferred (sound bounds, with the facts they rest on); the links it used
    and those it could not resolve. `stale` while the source's temporal job has not finished."""
    branch_id = _require_branch(x_branch_id)
    try:
        result = await temporal_jobs.reasoning_of(client, branch_id, version_id)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No temporal job for version {version_id}")
    return result


@router.post("/reason/{version_id}/recompute", status_code=202,
             summary="Queue a recompute of one source's temporal facts and reasoning (P15)")
async def temporal_reason_recompute(version_id: str,
                                    x_branch_id: str = Header(None, alias="X-Branch-ID")) -> dict[str, Any]:
    """Reopens the source's temporal job and hands it to the worker; returns at once. Its segments stay done,
    so the model is not called again: the facts, candidates and reasoning are rebuilt from stored replies."""
    branch_id = _require_branch(x_branch_id)
    try:
        queued = await temporal_jobs.recompute(client, branch_id, version_id)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not queued:
        raise HTTPException(status_code=404, detail=f"No temporal job for version {version_id}")
    return {"source_version_id": version_id, "queued": True}


@router.get("/status/{version_id}", summary="Temporal job state of one source version (P12)")
async def temporal_status(version_id: str,
                          x_branch_id: str = Header(None, alias="X-Branch-ID")) -> dict[str, Any]:
    """pending | running | complete | partial | failed | cancelled, with segment counts, the facts written,
    the extractor / normaliser versions and model, and the segments that failed or could not be read
    (`unsupported_language`). 404 when the source has no job under the current extractor and normaliser
    versions -- not indexed yet, indexed before P12, or deleted."""
    branch_id = _require_branch(x_branch_id)
    try:
        result = await temporal_jobs.status(client, branch_id, version_id)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- wraps ArcadeDB calls; converts any failure to 500, logged
        logger.error(f"temporal_status failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No temporal job for version {version_id}")
    return result
