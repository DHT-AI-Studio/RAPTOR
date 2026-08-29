from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from core.config import settings
from core.dependencies import get_current_user
from services import meta_index
from services.long_term_memory import LongTermMemoryService, SearchRequest as LTSearchRequest
from services.memvid_store import (
    normalize_scores,
    sync_search,
    sync_search_counted,
    _get_embedder,
    get_search_pool,
)
from services.multimedia_memory import MultimediaMemoryService, MediaSearchRequest
from services.session_shards import parse_session_shard_filename

router = APIRouter(prefix="/memory", tags=["Global Search"])

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_iso(s: str) -> float:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _unix_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── Request / Response models ─────────────────────────────────────────────────

class GlobalSearchRequest(BaseModel):
    query: str = Field(..., description="查詢文字，做 BM25 + bge-m3 語意向量混合搜尋")
    top_k: int = Field(5, ge=1, le=50, description="每個 scope 各自的回傳筆數上限")
    scope: list[Literal["sessions", "longterm", "multimedia"]] = Field(
        default=["sessions", "longterm", "multimedia"],
        description="要搜尋的記憶範圍，可縮小至任意子集；預設全選",
    )


class SessionScopeHit(BaseModel):
    text: str = Field(..., description="命中內容片段")
    score: float = Field(..., description="正規化後的相關性分數（0~1），與其他 scope 的分數用同一把尺，可跨 scope 比較排序")
    timestamp: str = Field(..., description="該輪的 ISO 8601 時間戳")
    turn_index: int = Field(..., description="該輪在 session 中的序號")
    user_message: str = Field(..., description="使用者訊息原文")
    assistant_response: str = Field(..., description="助手回覆原文")
    session_id: str = Field(..., description="命中的來源 session ID")


class LongTermScopeHit(BaseModel):
    text: str = Field(..., description="命中的事實/偏好/實體文字")
    score: float = Field(..., description="正規化後的相關性分數（0~1），與其他 scope 的分數用同一把尺，可跨 scope 比較排序")
    timestamp: str = Field(..., description="寫入時間的 ISO 8601 時間戳")
    frame_type: str = Field(..., description="frame 類型：conversation | preference | entity | fact")
    session_id: str = Field(..., description="來源 session ID（若寫入時有提供）")


class MultimediaScopeHit(BaseModel):
    text: str = Field(..., description="命中的文字片段（逐字稿 / OCR / 描述）")
    score: float = Field(..., description="正規化後的相關性分數（0~1），與其他 scope 的分數用同一把尺，可跨 scope 比較排序")
    timestamp: Optional[str] = Field(None, description="索引寫入時間的 ISO 8601 時間戳")
    asset_path: str = Field(..., description="命中資產的儲存路徑")
    version_id: str = Field(..., description="命中資產的版本 ID")
    media_type: str = Field(..., description="媒體類型：video | audio | image")
    start_sec: Optional[float] = Field(None, description="片段起始秒數（video/audio 才有）")
    end_sec: Optional[float] = Field(None, description="片段結束秒數（video/audio 才有）")
    session_id: str = Field(..., description="索引當時記錄的 session ID")


class GlobalSearchResponse(BaseModel):
    sessions: list[SessionScopeHit] = Field(default=[], description="session 記憶命中結果（scope 含 sessions 時才有）")
    longterm: list[LongTermScopeHit] = Field(default=[], description="long-term 記憶命中結果（scope 含 longterm 時才有）")
    multimedia: list[MultimediaScopeHit] = Field(default=[], description="多媒體記憶命中結果（scope 含 multimedia 時才有）")
    total_frames_searched: int = Field(..., description="本次搜尋掃描過的 frame 總數（跨所有選定 scope 加總）")


# ── Service ───────────────────────────────────────────────────────────────────

class GlobalSearchService:
    def __init__(self) -> None:
        self._root = Path(settings.storage_root)
        self._lt_svc = LongTermMemoryService()
        self._mm_svc = MultimediaMemoryService()

    def _session_files(self, user_id: str) -> list[tuple[str, str]]:
        """Return [(path, session_id), ...] for all session .mv2 shard files —
        a session can span more than one shard (session_shards.py), so the
        session_id is parsed back out rather than taken from the raw stem."""
        d = self._root / f"user_{user_id}"
        if not d.exists():
            return []
        result: list[tuple[str, str]] = []
        for p in sorted(d.glob("session_*.mv2")):
            parsed = parse_session_shard_filename(p.stem)
            if parsed is not None:
                result.append((str(p), parsed[0]))
        return result

    async def _search_sessions(
        self, user_id: str, query: str, top_k: int, *, _raw_scores: bool = False
    ) -> tuple[list[SessionScopeHit], int]:
        files = self._session_files(user_id)
        if not files:
            return [], 0

        # Pre-compute the query embedding once in the current process so workers
        # receive a plain list[float] argument (no model in the IPC payload).
        query_vec = await asyncio.to_thread(_get_embedder().embed_query, query)

        # Use a ProcessPoolExecutor so each worker runs find() in its own OS process,
        # bypassing the GIL that serialises concurrent asyncio.to_thread calls.
        loop = asyncio.get_running_loop()
        pool = get_search_pool()
        counted_tasks = [
            loop.run_in_executor(pool, sync_search_counted, path, query, top_k * 4, query_vec)
            for path, _ in files
        ]
        counted_results = await asyncio.gather(*counted_tasks)

        total = sum(cnt for _, cnt in counted_results)
        search_results = [hits for hits, _ in counted_results]

        candidates: list[dict] = []
        for hits, (_, session_id) in zip(search_results, files):
            for h in hits:
                candidates.append({**h, "_session_id": session_id})

        candidates.sort(key=lambda h: h.get("score", float("-inf")), reverse=True)
        candidates = candidates[:top_k]

        raw_scores = [h.get("score", 0.0) for h in candidates]
        scores = raw_scores if _raw_scores else normalize_scores(raw_scores)
        return [
            SessionScopeHit(
                text=h.get("text") or h.get("snippet", ""),
                score=score,
                timestamp=_unix_to_iso(float(h.get("timestamp", 0))),
                turn_index=int(h.get("turn_index", 0)),
                user_message=h.get("user_message", ""),
                assistant_response=h.get("assistant_response", ""),
                session_id=h["_session_id"],
            )
            for h, score in zip(candidates, scores)
        ], total

    async def _search_longterm(
        self, user_id: str, query: str, top_k: int, *, _raw_scores: bool = False
    ) -> tuple[list[LongTermScopeHit], int]:
        shards = self._lt_svc._shard_paths(user_id)
        if not shards:
            return [], 0

        counts = await asyncio.gather(*[asyncio.to_thread(meta_index.load, p) for p in shards])
        total = sum(len(idx) for idx in counts)

        req = LTSearchRequest(query=query, top_k=top_k)
        hits = await self._lt_svc.search(user_id, req, _raw_scores=_raw_scores)

        return [
            LongTermScopeHit(
                text=h.text,
                score=h.score,
                timestamp=_unix_to_iso(h.timestamp),
                frame_type=h.frame_type,
                session_id=h.session_id,
            )
            for h in hits
        ], total

    async def _search_multimedia(
        self, user_id: str, query: str, top_k: int, *, _raw_scores: bool = False
    ) -> tuple[list[MultimediaScopeHit], int]:
        media_path = self._root / f"user_{user_id}" / "media"
        paths = list(media_path.glob("*.mv2")) if media_path.exists() else []

        if not paths:
            return [], 0

        counts = await asyncio.gather(*[asyncio.to_thread(meta_index.load, str(p)) for p in paths])
        total = sum(len(idx) for idx in counts)

        req = MediaSearchRequest(query=query, top_k=top_k)
        hits = await self._mm_svc.search(user_id, req, _raw_scores=_raw_scores)

        return [
            MultimediaScopeHit(
                text=h.text_snippet,
                score=h.score,
                timestamp=_unix_to_iso(h.timestamp) if h.timestamp else None,
                asset_path=h.asset_path,
                version_id=h.version_id,
                media_type=h.media_type,
                start_sec=h.start_sec,
                end_sec=h.end_sec,
                session_id=h.session_id,
            )
            for h in hits
        ], total

    async def search(self, user_id: str, req: GlobalSearchRequest) -> GlobalSearchResponse:
        scope = set(req.scope) if req.scope else {"sessions", "longterm", "multimedia"}

        # Fetch raw (unnormalized) scores per scope, then normalize once across
        # all of them together below — so a score is comparable across scopes,
        # not just within the scope it came from.
        scope_list: list[str] = []
        coros = []
        if "sessions" in scope:
            scope_list.append("sessions")
            coros.append(self._search_sessions(user_id, req.query, req.top_k, _raw_scores=True))
        if "longterm" in scope:
            scope_list.append("longterm")
            coros.append(self._search_longterm(user_id, req.query, req.top_k, _raw_scores=True))
        if "multimedia" in scope:
            scope_list.append("multimedia")
            coros.append(self._search_multimedia(user_id, req.query, req.top_k, _raw_scores=True))

        gathered = await asyncio.gather(*coros)

        results: dict[str, tuple] = dict(zip(scope_list, gathered))
        total_frames = sum(r[1] for r in results.values())

        all_hits = [
            hit
            for hits, _ in results.values()
            for hit in hits
        ]
        normalized = normalize_scores([hit.score for hit in all_hits])
        for hit, score in zip(all_hits, normalized):
            hit.score = score

        return GlobalSearchResponse(
            sessions=results.get("sessions", ([], 0))[0],
            longterm=results.get("longterm", ([], 0))[0],
            multimedia=results.get("multimedia", ([], 0))[0],
            total_frames_searched=total_frames,
        )


# ── Router ────────────────────────────────────────────────────────────────────

def get_search_service() -> GlobalSearchService:
    return GlobalSearchService()


@router.post(
    "/search",
    response_model=GlobalSearchResponse,
    summary="全域搜尋（sessions + long-term + multimedia）",
    description="跨所有 session、long-term 記憶與多媒體記憶做混合搜尋，各 scope 平行查詢並各自回傳 top_k 筆。",
)
async def global_search(
    body: GlobalSearchRequest,
    user_id: str = Depends(get_current_user),
    svc: GlobalSearchService = Depends(get_search_service),
) -> GlobalSearchResponse:
    return await svc.search(user_id, body)
