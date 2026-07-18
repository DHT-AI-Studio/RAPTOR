"""Video search router — 4-way RRF (BM25 + Vector + GraphRAG + TKG) + rerank."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.services.search_service import HybridSearchService
from app.services.storage_service import StorageService

_logger = logging.getLogger(__name__)

router = APIRouter()

_RRF_K = 60          # standard RRF constant
_RERANK_CANDIDATES = 50  # max segments sent to cross-encoder


def _get_search_service(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage_svc: StorageService = Depends(get_storage_service),
) -> HybridSearchService:
    return HybridSearchService(
        hybrid_search_url=settings.hybrid_search_url,
        http_client=client,
        storage_service=storage_svc,
    )


# ── Request / Response models ─────────────────────────────────────────────────

class VideoSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of videos to return")
    candidate_multiplier: int = Field(5, ge=1, le=20,
        description="Each retriever fetches top_k × candidate_multiplier segments as candidates before RRF")
    score_threshold: float = Field(0.52, ge=0.0, le=1.0,
        description="Minimum rerank score (sigmoid-normalised) for a segment to appear in results; sigmoid(0)=0.5 so 0.52 filters near-zero cross-encoder scores")


class Segment(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    score: float
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list,
                               description="Which backends matched this segment: bm25/vector/graphrag/tkg")


class VideoResult(BaseModel):
    video_id: str
    filename: Optional[str] = None
    score: float = Field(description="Highest segment score for this video")
    asset_url: Optional[str] = None
    upload_time: Optional[str] = None
    segments: List[Segment] = Field(description="Matching segments sorted by score desc")


class VideoSearchResponse(BaseModel):
    query: str
    total: int
    results: List[VideoResult]


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


def _seg_key(source_id: str, start_sec: Optional[float]) -> str:
    """Dedup key: same video + same second = same moment."""
    ts = round(start_sec or 0.0)
    return f"{source_id}|{ts}"


def _rrf_add(
    pool: Dict[str, dict],
    video_meta: Dict[str, dict],
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


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "",
    tags=["Video Search"],
    status_code=status.HTTP_200_OK,
    response_model=VideoSearchResponse,
    summary="Video Search (4-way RRF + Rerank)",
    description="""
Runs **BM25, Vector, GraphRAG, and TKG** retrievers in parallel, merges their ranked lists with Reciprocal Rank Fusion (RRF), then re-ranks the top candidates (up to 50) with a cross-encoder reranker.

**Results are grouped by video:**
- Each video entry includes matching segments with `start_time` / `end_time`, segment text, and source tags
- `asset_url` — pre-signed URL for direct video access

**Response headers:**
`X-Process-Time-*` fields report per-stage latency in milliseconds.
""",
)
async def video_search(
    req: VideoSearchRequest,
    response: Response,
    svc: HybridSearchService = Depends(_get_search_service),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> VideoSearchResponse:
    """Multi-source video search: 4-way RRF → cross-encoder rerank.

    BM25, Vector, GraphRAG, and TKG run in parallel as independent retrievers.
    Reciprocal Rank Fusion merges their ranked lists without cross-source score
    normalisation. A cross-encoder reranker then precision-ranks the top candidates.
    """
    import time as _time
    _t0 = _time.perf_counter()

    user = {"user_id": current_user["sub"], "branch_id": current_user["sub"]}
    branch_id = current_user["sub"]
    candidate_k = req.top_k * req.candidate_multiplier
    graph_limit = min(candidate_k, 200)  # module 20 schema hard cap

    # ── Parallel fan-out to all 4 retrievers ─────────────────────────────────
    bm25_coro = svc.bm25_search(
        {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"},
        user=user, resolve_urls=False,
    )
    vector_coro = svc.vector_search(
        {"query": req.query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"},
        user=user, resolve_urls=False,
    )
    graphrag_coro = client.post(
        f"{settings.graph_service_url}/query/graph_rag",
        json={"query": req.query, "max_depth": 2, "limit": graph_limit,
              "score_threshold": 0.3, "branch_id": branch_id},
        timeout=30.0,
    )
    tkg_coro = client.post(
        f"{settings.graph_service_url}/tkg/query",
        json={"query": req.query, "max_depth": 2, "limit": graph_limit,
              "score_threshold": 0.3, "branch_id": branch_id},
        timeout=30.0,
    )

    bm25_res, vector_res, graphrag_res, tkg_res = await asyncio.gather(
        bm25_coro, vector_coro, graphrag_coro, tkg_coro,
        return_exceptions=True,
    )
    _t_fanout = _time.perf_counter()
    pool: Dict[str, dict] = {}
    video_meta: Dict[str, dict] = {}

    # ── 1. BM25 ──────────────────────────────────────────────────────────────
    if not isinstance(bm25_res, Exception):
        for rank, hit in enumerate(bm25_res.get("results", [])):
            p = hit.get("payload", {})
            vid_id = p.get("version_id") or p.get("video_id") or hit.get("id", "")
            if not vid_id:
                continue
            _rrf_add(
                pool, video_meta,
                video_id=vid_id,
                start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
                end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
                rank=rank,
                text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
                source_tag="bm25",
                asset_path=p.get("asset_path"),
                filename=p.get("filename"),
                upload_time=p.get("upload_time"),
            )
    else:
        _logger.warning(f"BM25 search failed: {bm25_res}")

    # ── 2. Vector ────────────────────────────────────────────────────────────
    if not isinstance(vector_res, Exception):
        for rank, hit in enumerate(vector_res.get("results", [])):
            p = hit.get("payload", {})
            vid_id = p.get("version_id") or p.get("video_id") or hit.get("id", "")
            if not vid_id:
                continue
            _rrf_add(
                pool, video_meta,
                video_id=vid_id,
                start_sec=float(p["start_time"]) if p.get("start_time") is not None else None,
                end_sec=float(p["end_time"]) if p.get("end_time") is not None else None,
                rank=rank,
                text=p.get("asr_text") or p.get("contextual_text") or p.get("text") or "",
                source_tag="vector",
                asset_path=p.get("asset_path"),
                filename=p.get("filename"),
                upload_time=p.get("upload_time"),
            )
    else:
        _logger.warning(f"Vector search failed: {vector_res}")

    # ── 3. GraphRAG ──────────────────────────────────────────────────────────
    if not isinstance(graphrag_res, Exception):
        try:
            graphrag_res.raise_for_status()
            gr = graphrag_res.json()
        except Exception as exc:
            _logger.warning(f"GraphRAG response error: {exc}")
            gr = {}

        # matched_moments: text-searched with Lucene score → sort → rank by position
        matched = sorted(gr.get("matched_moments", []),
                         key=lambda m: m.get("score", 0.0), reverse=True)
        for rank, m in enumerate(matched):
            src = m.get("version_id", "")
            if not src:
                continue
            _rrf_add(
                pool, video_meta,
                video_id=src,
                start_sec=m.get("start_sec"),
                end_sec=m.get("end_sec"),
                rank=rank,
                text=m.get("asr_text") or m.get("contextual_text") or "",
                source_tag="graphrag",
                asset_path=m.get("asset_path"),
                filename=m.get("filename"),
                upload_time=m.get("upload_time"),
            )

        # moment_ids (subgraph traversal): no real score, rank just after matched_moments.
        # Skip moments already processed via matched_moments to avoid double-counting.
        subgraph_rank = len(matched)
        matched_ids = {m.get("id") or m.get("moment_id") for m in matched}
        for m in gr.get("moment_ids", []):
            src = m.get("version_id", "")
            mid = m.get("moment_id", "")
            if not src or mid in matched_ids:
                continue
            _rrf_add(
                pool, video_meta,
                video_id=src,
                start_sec=m.get("start_sec"),
                end_sec=m.get("end_sec"),
                rank=subgraph_rank,
                text=_format_graph_text(m),
                source_tag="graphrag",
                asset_path=m.get("asset_path"),
                filename=m.get("filename"),
                upload_time=m.get("upload_time"),
            )
    else:
        _logger.warning(f"GraphRAG search failed: {graphrag_res}")

    # ── 4. TKG ───────────────────────────────────────────────────────────────
    # TKG is a temporal retriever: moments are found via graph traversal, not
    # scored. All moments get rank=0 (equal contribution of 1/60 each).
    if not isinstance(tkg_res, Exception):
        try:
            tkg_res.raise_for_status()
            tk = tkg_res.json()
        except Exception as exc:
            _logger.warning(f"TKG response error: {exc}")
            tk = {}

        for m in tk.get("moment_ids", []):
            src = m.get("version_id", "")
            if not src:
                continue
            _rrf_add(
                pool, video_meta,
                video_id=src,
                start_sec=m.get("start_sec"),
                end_sec=m.get("end_sec"),
                rank=0,
                text=_format_graph_text(m),
                source_tag="tkg",
                asset_path=m.get("asset_path"),
                filename=m.get("filename"),
                upload_time=m.get("upload_time"),
            )
    else:
        _logger.warning(f"TKG search failed: {tkg_res}")

    # ── Rerank top candidates with cross-encoder ──────────────────────────────
    candidates = sorted(pool.values(), key=lambda s: s["score"], reverse=True)[:_RERANK_CANDIDATES]
    if candidates:
        try:
            # Video-level vector hits have no text; sending text="" to the cross-encoder
            # yields sigmoid(0)≈0.596, which inflates scores above score_threshold.
            # Zero them out before reranking so they cannot pass the threshold.
            rerank_docs = [
                {"id": _seg_key(s["video_id"], s["start_time"]), "text": s["text"]}
                for s in candidates if s.get("text")
            ]
            for s in candidates:
                if not s.get("text"):
                    pool[_seg_key(s["video_id"], s["start_time"])]["score"] = 0.0

            if rerank_docs:
                rerank_resp = await client.post(
                    f"{settings.hybrid_search_url}/api/v1/search/rerank",
                    json={"query": req.query, "documents": rerank_docs},
                    timeout=30.0,
                )
                rerank_resp.raise_for_status()
                for item in rerank_resp.json().get("results", []):
                    if item["id"] in pool:
                        pool[item["id"]]["score"] = item["score"]
        except Exception as exc:
            _logger.warning(f"Rerank failed, keeping RRF scores: {exc}")
    _t_rerank = _time.perf_counter()

    # ── Unified URL resolution ────────────────────────────────────────────────
    # Pass (asset_path, filename) tuple when filename is known so resolve_version_urls
    # can use the fast /presign-url endpoint (1 LakeFS call) instead of /filedownload
    # (which fetches all associated files).
    version_map = {
        vid_id: (meta["asset_path"], meta.get("filename"))
        for vid_id, meta in video_meta.items()
        if meta.get("asset_path")
    }
    url_map = await svc.resolve_version_urls(version_map, user)
    _t_url = _time.perf_counter()

    # ── Group segments by video ───────────────────────────────────────────────
    videos: Dict[str, dict] = {}
    for seg in candidates:
        if seg["score"] < req.score_threshold:
            continue
        vid_id = seg["video_id"]
        if vid_id not in videos:
            meta = video_meta.get(vid_id, {})
            videos[vid_id] = {
                "video_id": vid_id,
                "filename": meta.get("filename"),
                "asset_url": url_map.get(vid_id),
                "upload_time": meta.get("upload_time"),
                "score": 0.0,
                "segments": [],
            }
        videos[vid_id]["segments"].append(seg)
        if seg["score"] > videos[vid_id]["score"]:
            videos[vid_id]["score"] = seg["score"]

    for v in videos.values():
        v["segments"].sort(key=lambda s: s["score"], reverse=True)

    ranked = sorted(videos.values(), key=lambda v: v["score"], reverse=True)[: req.top_k]

    results = [
        VideoResult(
            video_id=v["video_id"],
            filename=v["filename"],
            score=v["score"],
            asset_url=v["asset_url"],
            upload_time=v["upload_time"],
            segments=[
                Segment(
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    score=s["score"],
                    text=s["text"] or None,
                    sources=s["sources"],
                )
                for s in v["segments"]
            ],
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
