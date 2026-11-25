"""Shared video-search service — 4-way RRF (BM25 + Vector + GraphRAG + TKG) + rerank.

Extracted from app/routers/video_search.py so both the REST endpoint and the
grpc-gateway SearchByPerson/StreamSearchByPerson handlers share the exact
same retrieval/ranking logic instead of it living inline in a FastAPI route
function. No FastAPI types in the public surface — this module (plus
search_service.py, storage_service.py and personal_db_service.py) is copied
into grpc-gateway's own container image, which never imports FastAPI's
request-handling machinery.
"""
from __future__ import annotations

import asyncio
import logging

from app.services.personal_db_service import PersonalDBService
from app.services.search_service import HybridSearchService
from fastapi import HTTPException
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)

_RRF_K = 60              # standard RRF constant
_RERANK_CANDIDATES = 50  # max segments sent to cross-encoder


# ── Result types (plain pydantic — no FastAPI) ────────────────────────────────

class SegmentHit(BaseModel):
    start_time: float | None = None
    end_time: float | None = None
    score: float
    text: str | None = None
    sources: list[str] = Field(default_factory=list)


class VideoHit(BaseModel):
    video_id: str
    filename: str | None = None
    score: float
    asset_url: str | None = None
    upload_time: str | None = None
    segments: list[SegmentHit]


class VideoSearchResult(BaseModel):
    query: str
    videos: list[VideoHit]
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ClipHitData(BaseModel):
    """Flat, one-per-segment shape — used by the gRPC SearchByPerson /
    StreamSearchByPerson handlers to build ClipHit proto messages."""

    clip_id: str
    video_id: str
    filename: str | None = None
    score: float
    asset_url: str | None = None
    upload_time: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    text: str | None = None
    sources: list[str] = Field(default_factory=list)


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


class VideoSearchService:
    """4-way RRF + cross-encoder rerank, shared by REST /search/video_search
    and grpc-gateway's SearchByPerson/StreamSearchByPerson. Retrieval is
    scoped to the caller's own personal database (module 25)."""

    def __init__(
        self,
        personal_db: PersonalDBService,
        url_resolver: HybridSearchService,
    ):
        self._personal_db = personal_db
        self._url_resolver = url_resolver

    async def search(
        self,
        query: str,
        user: dict[str, str],
        top_k: int = 10,
        candidate_multiplier: int = 5,
        score_threshold: float = 0.52,
    ) -> VideoSearchResult:
        """Multi-source video search: 4-way RRF -> cross-encoder rerank.

        BM25, Vector, GraphRAG, and TKG run in parallel as independent retrievers
        against module 25. Reciprocal Rank Fusion merges their ranked lists
        without cross-source score normalisation. Module 25's own cross-encoder
        reranker then precision-ranks the top candidates.
        """
        import time as _time
        _t0 = _time.perf_counter()

        user_id = user["user_id"]
        await self._personal_db.ensure_database(user_id)
        candidate_k = top_k * candidate_multiplier
        graph_limit = min(candidate_k, 200)  # module 25's TKGRequest/GraphRAGRequest limit cap

        # ── Parallel fan-out to all 4 retrievers, all scoped to the caller's own database ──
        bm25_coro = self._personal_db.bm25_search(
            user_id, {"query": query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"})
        vector_coro = self._personal_db.vector_search(
            user_id, {"query": query, "top_k": candidate_k, "type": "videos", "embedding_type": "text"})
        graphrag_coro = self._personal_db.graphrag_search(
            user_id, {"query": query, "max_depth": 2, "limit": graph_limit, "score_threshold": score_threshold})
        tkg_coro = self._personal_db.temporal_search(
            user_id, {"query": query, "max_depth": 2, "limit": graph_limit, "score_threshold": score_threshold})

        bm25_res, vector_res, graphrag_res, tkg_res = await asyncio.gather(
            bm25_coro, vector_coro, graphrag_coro, tkg_coro, return_exceptions=True)
        _t_fanout = _time.perf_counter()
        # One retriever failing is tolerated (the others still rank results). All
        # four failing means module 25 itself is down or rejecting us -- returning
        # an empty result would read as "nobody matched", so surface it instead.
        # (Local change to the 0.5 copy; the gateway maps it to UNAVAILABLE.)
        fanout = (bm25_res, vector_res, graphrag_res, tkg_res)
        if all(isinstance(r, Exception) for r in fanout):
            raise bm25_res
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
                    rerank_data = await self._personal_db.rerank(user_id, {"query": query, "documents": rerank_docs})
                    for item in rerank_data.get("results", []):
                        if item["id"] in pool:
                            pool[item["id"]]["score"] = item["score"]
            except (HTTPException, ValueError, KeyError) as exc:
                _logger.warning(f"video_search rerank failed, keeping RRF scores: {exc}")
        _t_rerank = _time.perf_counter()

        # ── URL resolution ─────────────────────────────────────────────────────
        version_map = {
            vid_id: (meta["asset_path"], meta.get("filename"))
            for vid_id, meta in video_meta.items() if meta.get("asset_path")
        }
        url_map = await self._url_resolver.resolve_version_urls(version_map, user)
        _t_url = _time.perf_counter()

        # ── Group segments by video ───────────────────────────────────────────────
        videos: dict[str, dict] = {}
        for seg in candidates:
            if seg["score"] < score_threshold:
                continue
            vid_id = seg["video_id"]
            if vid_id not in videos:
                meta = video_meta.get(vid_id, {})
                videos[vid_id] = {
                    "video_id": vid_id, "filename": meta.get("filename"),
                    "asset_url": url_map.get(vid_id), "upload_time": meta.get("upload_time"),
                    "score": 0.0, "segments": [],
                }
            videos[vid_id]["segments"].append(seg)
            videos[vid_id]["score"] = max(videos[vid_id]["score"], seg["score"])

        for v in videos.values():
            v["segments"].sort(key=lambda s: s["score"], reverse=True)

        ranked = sorted(videos.values(), key=lambda v: v["score"], reverse=True)[:top_k]

        result_videos = [
            VideoHit(
                video_id=v["video_id"], filename=v["filename"], score=v["score"],
                asset_url=v["asset_url"], upload_time=v["upload_time"],
                segments=[
                    SegmentHit(start_time=s["start_time"], end_time=s["end_time"], score=s["score"],
                               text=s["text"] or None, sources=s["sources"])
                    for s in v["segments"]
                ],
            )
            for v in ranked
        ]

        _t_end = _time.perf_counter()
        ms = lambda a, b: round((b - a) * 1000, 1)
        return VideoSearchResult(
            query=query,
            videos=result_videos,
            timings_ms={
                "total": ms(_t0, _t_end),
                "fanout": ms(_t0, _t_fanout),
                "rerank": ms(_t_fanout, _t_rerank),
                "url": ms(_t_rerank, _t_url),
            },
        )

    async def search_by_person(
        self,
        person_name: str,
        user: dict[str, str],
        top_k: int = 10,
        candidate_multiplier: int = 5,
        score_threshold: float = 0.52,
    ) -> VideoSearchResult:
        """Video search filtered by person.

        No person-entity index exists in Raptor today (graphrag/tkg take a
        free-text query, not an entity filter) — person_name is passed
        straight through as the query string to the same 4-way RRF pipeline,
        so this surfaces ASR/OCR/contextual-text mentions of the name, not
        true face/entity recognition.
        """
        return await self.search(
            query=person_name,
            user=user,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            score_threshold=score_threshold,
        )

    @staticmethod
    def flatten_to_hits(result: VideoSearchResult) -> list[ClipHitData]:
        """One ClipHitData per segment, not per video — carries the parent
        video's fields alongside each segment's own fields."""
        hits: list[ClipHitData] = []
        for v in result.videos:
            for s in v.segments:
                hits.append(ClipHitData(
                    clip_id=_seg_key(v.video_id, s.start_time),
                    video_id=v.video_id,
                    filename=v.filename,
                    score=s.score,
                    asset_url=v.asset_url,
                    upload_time=v.upload_time,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    text=s.text,
                    sources=s.sources,
                ))
        return hits
