"""Video search router — `/api/{version}/search/video_search`.

4-way RRF (BM25 + Vector + GraphRAG + TKG) + rerank, scoped to module 25
(personal-db) as of the 17/20 retirement. Endpoint path and request/response
shapes unchanged; only the backend swapped. This is where personal_db.py's
own `personal_video_search` copied its logic from (verbatim, imported not
re-copied), so this is the tested implementation, not new code.

People (app/routers/search.py's own POST /search/people) runs alongside the
4 content retrievers but is NOT folded into the RRF pool. A name/alias match
is a different kind of result from a semantic-relevance segment -- no
start_sec/end_sec granularity in common with Chunk-boundary moments (see
_seg_key), no score the cross-encoder or RRF's own fusion math means
anything for, and a payload shape (identity_id, trace_id, keyframe) that
doesn't fit Segment.text/score at all. An earlier version of this file did
force People through that pool (text="person:{name}", a score that always
bypassed score_threshold) -- kept People *findable* here, but every one of
those compromises was a symptom of forcing a fit that was never really
there. Attaching `people: list[PersonHit]` per VideoResult instead keeps the
identity data structured and leaves the 4-way RRF exactly as it was before
People existed.

Old module 17/20-backed implementation kept commented out below, not
deleted.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.services.consent_check import ConsentCheck, get_consent_check
from app.services.face_urls import add_image_urls
from app.services.personal_db_service import PersonalDBService
from app.services.search_service import HybridSearchService
from app.services.storage_service import StorageService

_logger = logging.getLogger(__name__)

router = APIRouter()

_RRF_K = 60          # standard RRF constant
_RERANK_CANDIDATES = 50  # max segments sent to cross-encoder


def get_personal_db_service(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> PersonalDBService:
    return PersonalDBService(client=client, settings=settings)


def _get_search_service(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage_svc: StorageService = Depends(get_storage_service),
) -> HybridSearchService:
    """Kept for `resolve_version_urls` -- a pure URL-resolution helper with
    no dependency on module 17 being alive."""
    return HybridSearchService(
        hybrid_search_url=settings.hybrid_search_url,
        http_client=client,
        storage_service=storage_svc,
    )


def _user_dict(user_id: str) -> dict[str, str]:
    return {"user_id": user_id, "branch_id": user_id}


# ── Request / Response models ─────────────────────────────────────────────────

class VideoSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of videos to return")
    candidate_multiplier: int = Field(5, ge=1, le=20,
        description="Each retriever fetches top_k × candidate_multiplier segments as candidates before RRF")
    score_threshold: float = Field(0.52, ge=0.0, le=1.0,
        description="Minimum rerank score (sigmoid-normalised) for a segment to appear in results; sigmoid(0)=0.5 so 0.52 filters near-zero cross-encoder scores")


class Segment(BaseModel):
    start_time: float | None = None
    end_time: float | None = None
    score: float
    text: str | None = None
    sources: list[str] = Field(default_factory=list,
                               description="Which backends matched this segment: bm25/vector/graphrag/tkg")


class PersonHit(BaseModel):
    """One name/alias match's appearance in this video -- module 25's
    PeopleSegment, trimmed to what the frontend needs to show and play it.
    Not an RRF-fused Segment: see this file's module docstring for why."""
    identity_id: str
    identity_name: str
    trace_id: str
    start_time: float
    end_time: float
    confidence: float | None = None
    keyframe_url: str | None = None


class VideoResult(BaseModel):
    video_id: str
    version_id: str = Field(description="Same value as video_id -- module 25's own identifier for this "
                                         "asset version is what video_id already carries; exposed again "
                                         "under its storage-layer name since that's the param module 04's "
                                         "/thumbnail endpoint expects, alongside asset_path below")
    asset_path: str | None = None
    filename: str | None = None
    score: float = Field(description="Highest segment score for this video")
    asset_url: str | None = None
    upload_time: str | None = None
    segments: list[Segment] = Field(description="Matching segments sorted by score desc")
    people: list[PersonHit] = Field(default_factory=list,
        description="Name/alias matches appearing in this video; independent of segments/score_threshold")


class VideoSearchResponse(BaseModel):
    query: str
    total: int
    results: list[VideoResult]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _format_graph_text(m: dict) -> str:
    parts = []
    if m.get("contextual_text"):
        parts.append(f"contextual:{{{m['contextual_text']}}}")
    if m.get("ocr_text"):
        parts.append(f"ocr:{{{m['ocr_text']}}}")
    if m.get("asr_text"):
        parts.append(f"asr:{{{m['asr_text']}}}")
    if m.get("lvlm_description"):
        parts.append(f"lvlm:{{{m['lvlm_description']}}}")
    return " / ".join(parts)


def _seg_key(source_id: str, start_sec: float | None) -> str:
    """Dedup key: same video + same second = same moment."""
    ts = round(start_sec or 0.0)
    return f"{source_id}|{ts}"


def _rrf_add(
    pool: dict[str, dict],
    video_meta: dict[str, dict],
    video_id: str,
    start_sec,
    end_sec,
    rank: int,
    text: str,
    source_tag: str,
    asset_path=None,
    filename=None,
    upload_time=None,
) -> None:
    """Add one moment to the RRF pool, accumulating 1/(rank+k) per source."""
    key = _seg_key(video_id, start_sec)
    contrib = 1.0 / (rank + _RRF_K)
    if key not in pool:
        pool[key] = {
            "video_id": video_id,
            "start_time": start_sec,
            "end_time": end_sec,
            "score": contrib,
            "text": text or "",
            "sources": [source_tag],
        }
    else:
        pool[key]["score"] += contrib
        if text and not pool[key]["text"]:
            pool[key]["text"] = text
        if end_sec and not pool[key]["end_time"]:
            pool[key]["end_time"] = end_sec
        if source_tag not in pool[key]["sources"]:
            pool[key]["sources"].append(source_tag)

    if video_id not in video_meta:
        video_meta[video_id] = {
            "filename": filename,
            "asset_path": asset_path,
            "upload_time": upload_time,
        }
    else:
        meta = video_meta[video_id]
        if asset_path and not meta.get("asset_path"):
            meta["asset_path"] = asset_path
        if filename and not meta.get("filename"):
            meta["filename"] = filename
        if upload_time and not meta.get("upload_time"):
            meta["upload_time"] = upload_time


async def _people_coro(svc: PersonalDBService, consent: ConsentCheck, user_id: str,
                       query: str, limit: int) -> dict[str, Any]:
    """Module 25's /personal/search/people, run alongside the 4 RRF
    retrievers (not folded into their pool -- see module docstring).

    Consent (face_detection) gates this the same way it gates POST
    /search/people itself -- but here the caller never asked specifically
    for people results, so a user who hasn't granted it should see
    video_search behave exactly as it did before people was added (4-way
    RRF, no error), not get a 403 for a plain content search. Only the
    expected "not granted" 403 is swallowed; anything else (e.g. 503
    CONSENT_CHECK_UNAVAILABLE, a network error) is left to propagate into
    asyncio.gather's return_exceptions=True, logged and skipped the same
    way a failed bm25/vector/graphrag/tkg call already is.
    """
    try:
        await consent.require(user_id, "face_detection")
    except HTTPException as exc:
        if exc.status_code == 403:
            return {"results": []}
        raise
    return await svc.people_search(user_id, {"query": query, "limit": limit})


def _people_hits_by_video(people_res: dict[str, Any]) -> tuple[dict[str, list[PersonHit]], dict[str, dict]]:
    """Group People's raw segments by video_id into PersonHit lists, plus
    the (asset_path, filename, asset_url) each video_id needs if it turns
    out to have no content-retriever hit at all (see people_only below).
    image_url on each segment's keyframe must already be resolved by the
    caller (add_image_urls) -- this function only reshapes, no I/O."""
    hits: dict[str, list[PersonHit]] = {}
    meta: dict[str, dict] = {}
    for seg in people_res.get("results", []):
        vid_id = seg.get("version_id", "")
        if not vid_id:
            continue
        hits.setdefault(vid_id, []).append(PersonHit(
            identity_id=seg["identity_id"], identity_name=seg.get("identity_name", ""),
            trace_id=seg["trace_id"], start_time=seg["first_sec"], end_time=seg["last_sec"],
            confidence=seg.get("confidence"), keyframe_url=(seg.get("keyframe") or {}).get("image_url"),
        ))
        if vid_id not in meta:
            meta[vid_id] = {"asset_path": seg.get("asset_path"), "filename": seg.get("filename"),
                            "asset_url": seg.get("asset_url")}
    return hits, meta


async def _attach_scoped_people(
    svc: Any, storage_svc: Any, user_id: str, people_res: dict[str, Any] | BaseException,
    people_by_video: dict[str, list[PersonHit]], video_ids: set[str],
) -> dict[str, list[PersonHit]]:
    """search_people()'s own global, limit-truncated segment list can crowd a
    video's correct people out just because the matched identity also appears
    in many *other* videos -- confirmed live (a 72-segment identity across 7
    videos, under a 50-segment video_search-internal limit, silently dropped
    2 of those 7 videos' people entirely). Re-query via the scoped
    people_in_videos endpoint, restricted to `video_ids` -- bounded by
    (identities) x (these videos), never by an identity's total appearance
    count system-wide -- and merge its (complete) results over `people_by_video`.
    Falls back to the original, unscoped dict on any failure (never breaks
    content search over a people-only concern), or when there's nothing to
    scope (no identity matched, or no candidate videos at all)."""
    identity_ids = [m["identity_id"] for m in people_res.get("identities", [])] \
        if not isinstance(people_res, BaseException) else []
    if not identity_ids or not video_ids:
        return people_by_video
    try:
        scoped_res = await svc.people_in_videos(
            user_id, {"identity_ids": identity_ids, "version_ids": list(video_ids)})
        await add_image_urls(scoped_res.get("results", []), storage_svc, user_id)
        scoped_by_video, _ = _people_hits_by_video({"results": scoped_res.get("results", [])})
        people_by_video.update(scoped_by_video)  # prefer the scoped, complete data where we have it
    except Exception as exc:  # noqa: BLE001 -- svc.people_in_videos()/add_image_urls() can raise a range of errors; all fall back the same way
        _logger.warning(f"video_search scoped people lookup failed, keeping unscoped results: {exc}")
    return people_by_video


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "",
    tags=["Video Search"],
    status_code=status.HTTP_200_OK,
    response_model=VideoSearchResponse,
    summary="Video Search (4-way RRF + Rerank + People)",
    description="""
Runs **BM25, Vector, GraphRAG, and TKG** retrievers in parallel, merges their ranked lists with Reciprocal Rank Fusion (RRF), then re-ranks the top content candidates (up to 50) with a cross-encoder reranker.

A query that matches a person's name or alias (same matching as `POST /search/people`) also surfaces their appearances, under each matching video's `people` field -- independent of RRF/rerank/`score_threshold`, since a name match isn't a semantic-relevance score those were built for. A video found *only* by name (no content-retriever hit at all) is appended after the content-ranked videos, filling whatever of `top_k` they didn't use, sorted by face-detection confidence. Requires the caller's `face_detection` consent for this contribution only: without it, video_search still runs normally (BM25/Vector/GraphRAG/TKG), just with every `people` list empty, no error.

**Results are grouped by video:**
- Each video entry includes matching segments with `start_time` / `end_time`, segment text, and source tags
- `people` — name/alias matches appearing in this video, each with its own `start_time`/`end_time`/`confidence`/`keyframe_url`
- `asset_url` — pre-signed URL for direct video access
- `asset_path` / `version_id` — module 04's own identifiers for this asset version, e.g. to call its `/thumbnail/{asset_path}/{version_id}` endpoint directly

**Response headers:**
`X-Process-Time-*` fields report per-stage latency in milliseconds.
""",
)
async def video_search(
    req: VideoSearchRequest,
    response: Response,
    svc: PersonalDBService = Depends(get_personal_db_service),
    search_svc: HybridSearchService = Depends(_get_search_service),
    storage_svc: StorageService = Depends(get_storage_service),
    consent: ConsentCheck = Depends(get_consent_check),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> VideoSearchResponse:
    """Multi-source video search: 4-way RRF → cross-encoder rerank, scoped to
    the caller's own personal database (module 25). People runs alongside as
    an independent lookup, attached per-video afterwards -- never part of
    the RRF pool or rerank (see module docstring).

    BM25, Vector, GraphRAG, and TKG run in parallel as independent
    retrievers against module 25. Reciprocal Rank Fusion merges their ranked
    lists without cross-source score normalisation. Module 25's own
    cross-encoder reranker then precision-ranks the top *content* candidates.
    """
    import time as _time
    _t0 = _time.perf_counter()

    user_id = current_user["sub"]
    await svc.ensure_database(user_id)
    user = _user_dict(user_id)
    candidate_k = req.top_k * req.candidate_multiplier
    graph_limit = min(candidate_k, 200)  # module 25's TKGRequest/GraphRAGRequest limit cap
    people_limit = min(candidate_k, 500)  # module 25's PeopleSearchRequest limit cap

    # ── Parallel fan-out to all 5 retrievers, all scoped to the caller's own database ──
    bm25_coro = svc.bm25_search(
        user_id, {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"})
    vector_coro = svc.vector_search(
        user_id, {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"})
    graphrag_coro = svc.graphrag_search(
        user_id, {"query": req.query, "max_depth": 2, "limit": graph_limit, "score_threshold": 0.3})
    tkg_coro = svc.temporal_search(
        user_id, {"query": req.query, "max_depth": 2, "limit": graph_limit, "score_threshold": 0.3})
    people_coro = _people_coro(svc, consent, user_id, req.query, people_limit)

    bm25_res, vector_res, graphrag_res, tkg_res, people_res = await asyncio.gather(
        bm25_coro, vector_coro, graphrag_coro, tkg_coro, people_coro, return_exceptions=True)
    _t_fanout = _time.perf_counter()
    pool: dict[str, dict] = {}
    video_meta: dict[str, dict] = {}

    # ── 1. BM25 ──────────────────────────────────────────────────────────────
    if not isinstance(bm25_res, Exception):
        for rank, hit in enumerate(bm25_res.get("results", [])):
            p = hit.get("payload", {})
            vid_id = p.get("version_id") or hit.get("id", "")
            if not vid_id:
                continue
            _rrf_add(
                pool, video_meta, video_id=vid_id,
                start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
                end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
                rank=rank, text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
                source_tag="bm25", asset_path=p.get("asset_path"),
                filename=p.get("filename"), upload_time=p.get("upload_time"),
            )
    else:
        _logger.warning(f"video_search BM25 search failed: {bm25_res}")

    # ── 2. Vector ────────────────────────────────────────────────────────────
    if not isinstance(vector_res, Exception):
        for rank, hit in enumerate(vector_res.get("results", [])):
            p = hit.get("payload", {})
            vid_id = p.get("version_id") or hit.get("id", "")
            if not vid_id:
                continue
            _rrf_add(
                pool, video_meta, video_id=vid_id,
                start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
                end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
                rank=rank, text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
                source_tag="vector", asset_path=p.get("asset_path"),
                filename=p.get("filename"), upload_time=p.get("upload_time"),
            )
    else:
        _logger.warning(f"video_search vector search failed: {vector_res}")

    # ── 3. GraphRAG ──────────────────────────────────────────────────────────
    if not isinstance(graphrag_res, Exception):
        gr = graphrag_res
        matched = sorted(gr.get("matched_moments", []), key=lambda m: m.get("score", 0.0), reverse=True)
        for rank, m in enumerate(matched):
            src = m.get("version_id", "")
            if not src:
                continue
            _rrf_add(
                pool, video_meta, video_id=src, start_sec=m.get("start_sec"), end_sec=m.get("end_sec"),
                rank=rank, text=m.get("asr_text") or m.get("contextual_text") or "",
                source_tag="graphrag", asset_path=m.get("asset_path"),
                filename=m.get("filename"), upload_time=m.get("upload_time"),
            )

        subgraph_rank = len(matched)
        matched_ids = {m.get("id") or m.get("moment_id") for m in matched}
        for m in gr.get("moment_ids", []):
            src = m.get("version_id", "")
            mid = m.get("moment_id", "")
            if not src or mid in matched_ids:
                continue
            _rrf_add(
                pool, video_meta, video_id=src, start_sec=m.get("start_sec"), end_sec=m.get("end_sec"),
                rank=subgraph_rank, text=_format_graph_text(m), source_tag="graphrag",
                asset_path=m.get("asset_path"), filename=m.get("filename"), upload_time=m.get("upload_time"),
            )
    else:
        _logger.warning(f"video_search GraphRAG search failed: {graphrag_res}")

    # ── 4. TKG ───────────────────────────────────────────────────────────────
    if not isinstance(tkg_res, Exception):
        for m in tkg_res.get("moment_ids", []):
            src = m.get("version_id", "")
            if not src:
                continue
            _rrf_add(
                pool, video_meta, video_id=src, start_sec=m.get("start_sec"), end_sec=m.get("end_sec"),
                rank=0, text=_format_graph_text(m), source_tag="tkg",
                asset_path=m.get("asset_path"), filename=m.get("filename"), upload_time=m.get("upload_time"),
            )
    else:
        _logger.warning(f"video_search TKG search failed: {tkg_res}")

    # ── People (independent lookup; not RRF-fused, see module docstring) ────
    people_by_video: dict[str, list[PersonHit]] = {}
    people_video_meta: dict[str, dict] = {}
    if not isinstance(people_res, Exception):
        try:
            await add_image_urls(people_res.get("results", []), storage_svc, user_id)
        except Exception as exc:  # noqa: BLE001 -- a presign failure must not drop people results, just their thumbnails
            _logger.warning(f"video_search people image URLs failed: {exc}")
        people_by_video, people_video_meta = _people_hits_by_video(people_res)
    else:
        _logger.warning(f"video_search people search failed: {people_res}")

    # ── Rerank top candidates with module 25's own cross-encoder ─────────────
    candidates = sorted(pool.values(), key=lambda s: s["score"], reverse=True)[:_RERANK_CANDIDATES]
    if candidates:
        try:
            rerank_docs = [
                {"id": _seg_key(s["video_id"], s["start_time"]), "text": s["text"]}
                for s in candidates if s.get("text")
            ]
            for s in candidates:
                if not s.get("text"):
                    pool[_seg_key(s["video_id"], s["start_time"])]["score"] = 0.0

            if rerank_docs:
                rerank_data = await svc.rerank(user_id, {"query": req.query, "documents": rerank_docs})
                for item in rerank_data.get("results", []):
                    if item["id"] in pool:
                        pool[item["id"]]["score"] = item["score"]
        except Exception as exc:  # noqa: BLE001 -- svc.rerank() can raise HTTPException/ValueError, response shape can raise KeyError; fails open to RRF scores, logged
            _logger.warning(f"video_search rerank failed, keeping RRF scores: {exc}")
    _t_rerank = _time.perf_counter()

    # ── URL resolution ─────────────────────────────────────────────────────
    version_map = {
        vid_id: (meta["asset_path"], meta.get("filename"))
        for vid_id, meta in video_meta.items() if meta.get("asset_path")
    }
    url_map = await search_svc.resolve_version_urls(version_map, user)
    _t_url = _time.perf_counter()

    # ── Group segments by video ───────────────────────────────────────────────
    videos: dict[str, dict] = {}
    for seg in candidates:
        if seg["score"] < req.score_threshold:
            continue
        vid_id = seg["video_id"]
        if vid_id not in videos:
            meta = video_meta.get(vid_id, {})
            videos[vid_id] = {
                "video_id": vid_id, "filename": meta.get("filename"),
                "asset_path": meta.get("asset_path"),
                "asset_url": url_map.get(vid_id), "upload_time": meta.get("upload_time"),
                "score": 0.0, "segments": [], "people": [],
            }
        videos[vid_id]["segments"].append(seg)
        videos[vid_id]["score"] = max(videos[vid_id]["score"], seg["score"])

    # Scoped people lookup -- see _attach_scoped_people's own
    # docstring. Covers every video this request could possibly show: both
    # the RRF content candidates and the videos people search alone found.
    people_by_video = await _attach_scoped_people(
        svc, storage_svc, user_id, people_res, people_by_video, set(videos) | set(people_by_video))

    for vid_id, hits in people_by_video.items():
        if vid_id in videos:
            videos[vid_id]["people"] = hits

    for v in videos.values():
        v["segments"].sort(key=lambda s: s["score"], reverse=True)

    ranked = sorted(videos.values(), key=lambda v: v["score"], reverse=True)[: req.top_k]

    # People-only videos: name/alias matched, but no content retriever hit
    # them at all. Appended after the content-ranked list, filling whatever
    # of top_k it didn't use, sorted by face-detection confidence -- never
    # mixed into the content-relevance ordering above (see module docstring).
    extra_slots = max(0, req.top_k - len(ranked))
    if extra_slots:
        people_only = sorted(
            (
                {
                    "video_id": vid_id, "filename": meta.get("filename"),
                    "asset_path": meta.get("asset_path"), "asset_url": meta.get("asset_url"),
                    "upload_time": None, "score": max((h.confidence or 0.0) for h in hits),
                    "segments": [], "people": hits,
                }
                for vid_id, hits in people_by_video.items() if vid_id not in videos
                for meta in (people_video_meta[vid_id],)
            ),
            key=lambda v: v["score"], reverse=True,
        )
        ranked = ranked + people_only[:extra_slots]

    results = [
        VideoResult(
            video_id=v["video_id"], version_id=v["video_id"], asset_path=v["asset_path"],
            filename=v["filename"], score=v["score"],
            asset_url=v["asset_url"], upload_time=v["upload_time"],
            segments=[
                Segment(start_time=s["start_time"], end_time=s["end_time"], score=s["score"],
                       text=s["text"] or None, sources=s["sources"])
                for s in v["segments"]
            ],
            people=v["people"],
        )
        for v in ranked
    ]

    _t_end = _time.perf_counter()
    ms = lambda a, b: str(round((b - a) * 1000, 1))
    response.headers["X-Process-Time-Total"] = ms(_t0, _t_end)
    response.headers["X-Process-Time-Fanout"] = ms(_t0, _t_fanout)
    response.headers["X-Process-Time-Rerank"] = ms(_t_fanout, _t_rerank)
    response.headers["X-Process-Time-Url"] = ms(_t_rerank, _t_url)

    return VideoSearchResponse(query=req.query, total=len(results), results=results)


# async def video_search(
#     req: VideoSearchRequest,
#     response: Response,
#     svc: HybridSearchService = Depends(_get_search_service),
#     client: httpx.AsyncClient = Depends(get_http_client),
#     settings: Settings = Depends(get_settings),
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ) -> VideoSearchResponse:
#     """Multi-source video search: 4-way RRF → cross-encoder rerank.
#
#     BM25, Vector, GraphRAG, and TKG run in parallel as independent retrievers.
#     Reciprocal Rank Fusion merges their ranked lists without cross-source score
#     normalisation. A cross-encoder reranker then precision-ranks the top candidates.
#     """
#     import time as _time
#     _t0 = _time.perf_counter()
#
#     user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
#     branch_id = current_user["sub"]
#     candidate_k = req.top_k * req.candidate_multiplier
#     graph_limit = min(candidate_k, 200)  # module 20 schema hard cap
#
#     # ── Parallel fan-out to all 4 retrievers ─────────────────────────────────
#     bm25_coro = svc.bm25_search(
#         {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"},
#         user=user, resolve_urls=False,
#     )
#     vector_coro = svc.vector_search(
#         {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"},
#         user=user, resolve_urls=False,
#     )
#     graphrag_coro = client.post(
#         f"{settings.graph_service_url}/query/graph_rag",
#         json={"query": req.query, "max_depth": 2, "limit": graph_limit,
#               "score_threshold": 0.3, "branch_id": branch_id},
#         timeout=30.0,
#     )
#     tkg_coro = client.post(
#         f"{settings.graph_service_url}/tkg/query",
#         json={"query": req.query, "max_depth": 2, "limit": graph_limit,
#               "score_threshold": 0.3, "branch_id": branch_id},
#         timeout=30.0,
#     )
#
#     bm25_res, vector_res, graphrag_res, tkg_res = await asyncio.gather(
#         bm25_coro, vector_coro, graphrag_coro, tkg_coro,
#         return_exceptions=True,
#     )
#     _t_fanout = _time.perf_counter()
#     pool: Dict[str, dict] = {}
#     video_meta: Dict[str, dict] = {}
#
#     # ── 1. BM25 ──────────────────────────────────────────────────────────────
#     if not isinstance(bm25_res, Exception):
#         for rank, hit in enumerate(bm25_res.get("results", [])):
#             p = hit.get("payload", {})
#             vid_id = p.get("version_id") or p.get("video_id") or hit.get("id", "")
#             if not vid_id:
#                 continue
#             _rrf_add(
#                 pool, video_meta,
#                 video_id=vid_id,
#                 start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
#                 end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
#                 rank=rank,
#                 text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
#                 source_tag="bm25",
#                 asset_path=p.get("asset_path"),
#                 filename=p.get("filename"),
#                 upload_time=p.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"BM25 search failed: {bm25_res}")
#
#     # ── 2. Vector ────────────────────────────────────────────────────────────
#     if not isinstance(vector_res, Exception):
#         for rank, hit in enumerate(vector_res.get("results", [])):
#             p = hit.get("payload", {})
#             vid_id = p.get("version_id") or p.get("video_id") or hit.get("id", "")
#             if not vid_id:
#                 continue
#             _rrf_add(
#                 pool, video_meta,
#                 video_id=vid_id,
#                 start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
#                 end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
#                 rank=rank,
#                 text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
#                 source_tag="vector",
#                 asset_path=p.get("asset_path"),
#                 filename=p.get("filename"),
#                 upload_time=p.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"Vector search failed: {vector_res}")
#
#     # ── 3. GraphRAG ──────────────────────────────────────────────────────────
#     if not isinstance(graphrag_res, Exception):
#         try:
#             graphrag_res.raise_for_status()
#             gr = graphrag_res.json()
#         except Exception as exc:
#             _logger.warning(f"GraphRAG response error: {exc}")
#             gr = {}
#
#         # matched_moments: text-searched with Lucene score → sort → rank by position
#         matched = sorted(gr.get("matched_moments", []),
#                          key=lambda m: m.get("score", 0.0), reverse=True)
#         for rank, m in enumerate(matched):
#             src = m.get("version_id", "")
#             if not src:
#                 continue
#             _rrf_add(
#                 pool, video_meta,
#                 video_id=src,
#                 start_sec=m.get("start_sec"),
#                 end_sec=m.get("end_sec"),
#                 rank=rank,
#                 text=m.get("asr_text") or m.get("contextual_text") or "",
#                 source_tag="graphrag",
#                 asset_path=m.get("asset_path"),
#                 filename=m.get("filename"),
#                 upload_time=m.get("upload_time"),
#             )
#
#         # moment_ids (subgraph traversal): no real score, rank just after matched_moments.
#         # Skip moments already processed via matched_moments to avoid double-counting.
#         subgraph_rank = len(matched)
#         matched_ids = {m.get("id") or m.get("moment_id") for m in matched}
#         for m in gr.get("moment_ids", []):
#             src = m.get("version_id", "")
#             mid = m.get("moment_id", "")
#             if not src or mid in matched_ids:
#                 continue
#             _rrf_add(
#                 pool, video_meta,
#                 video_id=src,
#                 start_sec=m.get("start_sec"),
#                 end_sec=m.get("end_sec"),
#                 rank=subgraph_rank,
#                 text=_format_graph_text(m),
#                 source_tag="graphrag",
#                 asset_path=m.get("asset_path"),
#                 filename=m.get("filename"),
#                 upload_time=m.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"GraphRAG search failed: {graphrag_res}")
#
#     # ── 4. TKG ───────────────────────────────────────────────────────────────
#     # TKG is a temporal retriever: moments are found via graph traversal, not
#     # scored. All moments get rank=0 (equal contribution of 1/60 each).
#     if not isinstance(tkg_res, Exception):
#         try:
#             tkg_res.raise_for_status()
#             tk = tkg_res.json()
#         except Exception as exc:
#             _logger.warning(f"TKG response error: {exc}")
#             tk = {}
#
#         for m in tk.get("moment_ids", []):
#             src = m.get("version_id", "")
#             if not src:
#                 continue
#             _rrf_add(
#                 pool, video_meta,
#                 video_id=src,
#                 start_sec=m.get("start_sec"),
#                 end_sec=m.get("end_sec"),
#                 rank=0,
#                 text=_format_graph_text(m),
#                 source_tag="tkg",
#                 asset_path=m.get("asset_path"),
#                 filename=m.get("filename"),
#                 upload_time=m.get("upload_time"),
#             )
#     else:
#         _logger.warning(f"TKG search failed: {tkg_res}")
#
#     # ── Rerank top candidates with cross-encoder ──────────────────────────────
#     candidates = sorted(pool.values(), key=lambda s: s["score"], reverse=True)[:_RERANK_CANDIDATES]
#     if candidates:
#         try:
#             # Video-level vector hits have no text; sending text="" to the cross-encoder
#             # yields sigmoid(0)≈0.596, which inflates scores above score_threshold.
#             # Zero them out before reranking so they cannot pass the threshold.
#             rerank_docs = [
#                 {"id": _seg_key(s["video_id"], s["start_time"]), "text": s["text"]}
#                 for s in candidates if s.get("text")
#             ]
#             for s in candidates:
#                 if not s.get("text"):
#                     pool[_seg_key(s["video_id"], s["start_time"])]["score"] = 0.0
#
#             if rerank_docs:
#                 rerank_resp = await client.post(
#                     f"{settings.hybrid_search_url}/api/v1/search/rerank",
#                     json={"query": req.query, "documents": rerank_docs},
#                     timeout=30.0,
#                 )
#                 rerank_resp.raise_for_status()
#                 for item in rerank_resp.json().get("results", []):
#                     if item["id"] in pool:
#                         pool[item["id"]]["score"] = item["score"]
#         except Exception as exc:
#             _logger.warning(f"Rerank failed, keeping RRF scores: {exc}")
#     _t_rerank = _time.perf_counter()
#
#     # ── Unified URL resolution ────────────────────────────────────────────────
#     # Pass (asset_path, filename) tuple when filename is known so resolve_version_urls
#     # can use the fast /presign-url endpoint (1 LakeFS call) instead of /filedownload
#     # (which fetches all associated files).
#     version_map = {
#         vid_id: (meta["asset_path"], meta.get("filename"))
#         for vid_id, meta in video_meta.items()
#         if meta.get("asset_path")
#     }
#     url_map = await svc.resolve_version_urls(version_map, user)
#     _t_url = _time.perf_counter()
#
#     # ── Group segments by video ───────────────────────────────────────────────
#     videos: Dict[str, dict] = {}
#     for seg in candidates:
#         if seg["score"] < req.score_threshold:
#             continue
#         vid_id = seg["video_id"]
#         if vid_id not in videos:
#             meta = video_meta.get(vid_id, {})
#             videos[vid_id] = {
#                 "video_id": vid_id,
#                 "filename": meta.get("filename"),
#                 "asset_url": url_map.get(vid_id),
#                 "upload_time": meta.get("upload_time"),
#                 "score": 0.0,
#                 "segments": [],
#             }
#         videos[vid_id]["segments"].append(seg)
#         if seg["score"] > videos[vid_id]["score"]:
#             videos[vid_id]["score"] = seg["score"]
#
#     for v in videos.values():
#         v["segments"].sort(key=lambda s: s["score"], reverse=True)
#
#     ranked = sorted(videos.values(), key=lambda v: v["score"], reverse=True)[: req.top_k]
#
#     results = [
#         VideoResult(
#             video_id=v["video_id"],
#             filename=v["filename"],
#             score=v["score"],
#             asset_url=v["asset_url"],
#             upload_time=v["upload_time"],
#             segments=[
#                 Segment(
#                     start_time=s["start_time"],
#                     end_time=s["end_time"],
#                     score=s["score"],
#                     text=s["text"] or None,
#                     sources=s["sources"],
#                 )
#                 for s in v["segments"]
#             ],
#         )
#         for v in ranked
#     ]
#
#     _t_end = _time.perf_counter()
#     ms = lambda a, b: str(round((b - a) * 1000, 1))
#     response.headers["X-Process-Time-Total"] = ms(_t0, _t_end)
#     response.headers["X-Process-Time-Fanout"] = ms(_t0, _t_fanout)
#     response.headers["X-Process-Time-Rerank"] = ms(_t_fanout, _t_rerank)
#     response.headers["X-Process-Time-Url"] = ms(_t_rerank, _t_url)
#
#     return VideoSearchResponse(query=req.query, total=len(results), results=results)
