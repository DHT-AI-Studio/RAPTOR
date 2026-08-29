from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from core.config import settings
from core.paths import user_dir as _ensure_user_dir
from services import meta_index
from services.compact_memory import NON_TURN_FRAME_TYPES
from services.memvid_store import (
    normalize_scores,
    sync_append,
    sync_search,
    sync_timeline,
    sync_get_remaining_capacity,
)
from services.session_shards import (
    get_active_shard_path,
    shard0_path,
    shard_paths,
)

_COMPACT_TRIGGER_THRESHOLD = 5 * 1024 * 1024  # trigger compaction when < 5 MB remaining

# Fire-and-forget background tasks (extraction, compaction) below MUST be kept
# in a strong reference somewhere, or asyncio is free to garbage-collect them
# mid-flight -- a bare asyncio.create_task() with no reference held anywhere
# else is a well-known asyncio pitfall (see the asyncio docs' own warning on
# this), and it's exactly what silently dropped extraction calls here: no log
# line, no error, no stored fact, confirmed live by comparing the fire-and-
# forget path (silently produced nothing) against calling extract_and_store()
# directly/synchronously via the /memory/longterm/extract endpoint (worked
# fine, {"stored": 1}) with the identical turn. Module 25's kafka_consumer.py
# already avoids this exact bug with its own _graph_extraction_tasks set --
# this mirrors that pattern.
_background_tasks: set[asyncio.Task] = set()


def _track_background_task(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_iso(s: str) -> float:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _unix_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TurnAppendRequest(BaseModel):
    user_message: str = Field(..., description="使用者這一輪輸入的訊息原文")
    assistant_response: str = Field(..., description="助手對這一輪訊息的回覆原文")
    search_results: list[dict] = Field(
        default_factory=list,
        description="這一輪回覆中引用的檢索結果（RAG hits），原樣存入 .mv2，供之後 timeline/export 回放",
    )
    tool_calls: list[dict] = Field(
        default_factory=list,
        description="這一輪回覆中觸發的工具呼叫紀錄（名稱、參數、結果等），原樣存入 .mv2",
    )
    timestamp: Optional[float] = Field(
        None, description="Unix timestamp（秒）；省略則使用伺服器目前時間"
    )
    provider_message_id: Optional[str] = Field(
        None,
        description="上游 LLM provider 的訊息 ID；串流回覆若被拆成多筆 turn 寫入，"
        "共用同一個 id 的 turns 在壓縮時會被視為同一組，tail 邊界不會切在組內",
    )
    mode: str = Field(
        "chat",
        description="這一輪的來源型態：chat（Module 15 對話）或 agent（Module 21 A2A 呼叫），"
        "供 timeline/export 區分來源",
    )


class TurnAppendResponse(BaseModel):
    frame_id: str = Field(..., description="這一輪對話在 .mv2 中的 frame ID")
    turn_index: int = Field(..., description="這一輪在該 session 中的序號（從 0 起算，遞增）")
    session_id: str = Field(..., description="所屬的 session ID（同路徑參數）")


class SessionInfo(BaseModel):
    session_id: str = Field(..., description="Session ID")
    turn_count: int = Field(..., description="該 session 目前累積的對話輪數")
    created_at: float = Field(..., description="Session 建立時間（Unix timestamp）")
    last_active_at: float = Field(..., description="最後一次寫入對話的時間（Unix timestamp）")


class TimelineEntry(BaseModel):
    turn_index: int = Field(..., description="該輪在 session 中的序號")
    timestamp: str = Field(..., description="ISO 8601 時間戳")
    frame_type: str = Field("session_turn", description="Frame 類型；一般為 session_turn，壓縮後的摘要為 summary")
    user_message: str = Field(..., description="使用者訊息原文")
    assistant_response: str = Field(..., description="助手回覆原文")
    media_refs: list[dict] = Field(default_factory=list, description="這一輪引用的多媒體參照")
    tool_calls: list[dict] = Field(default_factory=list, description="這一輪的工具呼叫紀錄")
    provider_message_id: Optional[str] = Field(
        None, description="上游 LLM provider 的訊息 ID（若有）"
    )
    session_id: Optional[str] = Field(
        None, description="所屬 session ID；只有跨 session 的使用者 timeline 才會帶這個欄位"
    )


class TimelineResponse(BaseModel):
    entries: list[TimelineEntry] = Field(..., description="本頁的對話輪次（時間升序）")
    total: int = Field(..., description="符合條件（含 time-travel 過濾）的總筆數")
    page: int = Field(..., description="目前頁碼（1-based）")
    page_size: int = Field(..., description="每頁筆數")
    has_next: bool = Field(..., description="是否還有下一頁")


class SessionSearchRequest(BaseModel):
    query: str = Field(..., description="查詢文字，會同時做 BM25 關鍵字比對與 bge-m3 語意向量比對")
    top_k: int = Field(5, ge=1, le=50, description="回傳筆數上限")
    from_date: Optional[str] = Field(
        None, description="ISO 8601 字串；只回傳此時間（含）之後的對話輪，可省略"
    )
    to_date: Optional[str] = Field(
        None, description="ISO 8601 字串；只回傳此時間（含）之前的對話輪，可省略"
    )


class SessionSearchHit(BaseModel):
    text: str = Field(..., description="命中內容片段（通常為 user_message + assistant_response 組合）")
    score: float = Field(..., description="正規化後的混合相關性分數（0~1，分數越高越相關）")
    timestamp: str = Field(..., description="該輪的 ISO 8601 時間戳")
    turn_index: int = Field(..., description="該輪在 session 中的序號")
    user_message: str = Field(..., description="使用者訊息原文")
    assistant_response: str = Field(..., description="助手回覆原文")


class SessionSearchResponse(BaseModel):
    hits: list[SessionSearchHit] = Field(..., description="依相關性分數降序排列的命中結果")
    total_frames_searched: int = Field(..., description="本次搜尋掃描過的 frame 總數（用於驗證涵蓋率）")


# ── Service ───────────────────────────────────────────────────────────────────

class SessionMemoryService:
    def __init__(self, redis: Redis, storage_root: str | None = None) -> None:
        self._redis = redis
        self._root = Path(storage_root or settings.storage_root)

    def _session_path(self, user_id: str, session_id: str) -> str:
        """Shard-0 path. Kept for unit-test direct seeding/inspection
        compatibility (mirrors LongTermMemoryService._lt_path) — internal
        reads/writes below use _shard_paths / _active_shard_path instead,
        since a session can span more than one shard."""
        user_dir = _ensure_user_dir(self._root, user_id)
        (user_dir / "media").mkdir(parents=True, exist_ok=True)
        (user_dir / "media").chmod(0o700)
        return shard0_path(self._root, user_id, session_id)

    def _shard_paths(self, user_id: str, session_id: str) -> list[str]:
        return shard_paths(self._root, user_id, session_id)

    async def _active_shard_path(self, user_id: str, session_id: str) -> str:
        _ensure_user_dir(self._root, user_id)
        return await get_active_shard_path(self._root, user_id, session_id)

    async def _next_turn_index(self, user_id: str, session_id: str) -> int:
        key = f"memory:session:{user_id}:{session_id}:turn_count"
        val = await self._redis.get(key)
        if val is None:
            # Redis was restarted — re-sync from .mv2 meta_index across all shards.
            shards = self._shard_paths(user_id, session_id)
            if shards:
                indices = await asyncio.gather(*(
                    asyncio.to_thread(meta_index.load, p) for p in shards
                ))
                total = sum(len(idx) for idx in indices)
                await self._redis.set(key, str(total), nx=True)
        return await self._redis.incr(key)

    async def append_turn(
        self, user_id: str, session_id: str, data: TurnAppendRequest
    ) -> TurnAppendResponse:
        ts = data.timestamp or time.time()
        turn_index = await self._next_turn_index(user_id, session_id)

        frame = {
            "title": f"Turn {turn_index}",
            "label": "session_turn",
            # Combined text field is what gets embedded by the configured embedder
            "text": f"User: {data.user_message}\nAssistant: {data.assistant_response}",
            "user_message": data.user_message,
            "assistant_response": data.assistant_response,
            "search_results": data.search_results,
            "tool_calls": data.tool_calls,
            "timestamp": ts,
            "session_id": session_id,
            "turn_index": turn_index,
            "provider_message_id": data.provider_message_id,
            "mode": data.mode,
        }

        path = await self._active_shard_path(user_id, session_id)
        frame_id = await asyncio.to_thread(sync_append, path, frame)

        # Fire-and-forget compaction when store is nearly full. This alone
        # doesn't prevent eventually running out of room — MV-12 compaction
        # archives rather than deletes turns, so it never shrinks the file —
        # _active_shard_path's rollover is what actually protects against
        # hitting the hard per-file capacity limit.
        remaining = await asyncio.to_thread(sync_get_remaining_capacity, path)
        if remaining < _COMPACT_TRIGGER_THRESHOLD:
            _track_background_task(self._trigger_compact(user_id, session_id))

        # Update Redis session index (sorted set keyed by last_active_at)
        sessions_key = f"memory:sessions:{user_id}"
        await self._redis.zadd(sessions_key, {session_id: ts})

        # Record created_at on first write (nx=True → only if key doesn't exist)
        created_key = f"memory:session:{user_id}:{session_id}:created_at"
        await self._redis.set(created_key, str(ts), nx=True)

        # Fire-and-forget: extract preferences/entities/facts from this turn
        _track_background_task(self._trigger_extraction(user_id, session_id, frame))

        return TurnAppendResponse(frame_id=frame_id, turn_index=turn_index, session_id=session_id)

    async def _trigger_extraction(self, user_id: str, session_id: str, turn: dict) -> None:
        try:
            from services.extractor import extract_and_store
            from services.long_term_memory import LongTermMemoryService
            lt_svc = LongTermMemoryService(storage_root=str(self._root))
            await extract_and_store(user_id, session_id, turn, lt_svc)
        except Exception:
            logger.warning("Extraction failed for %s/%s", user_id, session_id, exc_info=True)

    async def _trigger_compact(self, user_id: str, session_id: str) -> None:
        try:
            from services.compact_memory import CompactMemoryService
            compact = CompactMemoryService(storage_root=str(self._root))
            result = await compact.compact_session(user_id, session_id, context_window=128000)
            if result.compacted:
                logger.info(
                    "Auto-compact triggered for %s/%s: %d turns → %d turns",
                    user_id, session_id, result.turns_compacted, result.turns_kept,
                )
        except Exception:
            logger.warning("Auto-compact failed for %s/%s", user_id, session_id, exc_info=True)

    async def _all_session_entries(self, user_id: str, session_id: str) -> list[dict]:
        """Frames from every shard this session has, merged and sorted by
        timestamp — a session's history can span more than one shard."""
        shards = self._shard_paths(user_id, session_id)
        if not shards:
            return []
        per_shard = await asyncio.gather(*(
            asyncio.to_thread(sync_timeline, p) for p in shards
        ))
        entries = [e for shard_entries in per_shard for e in shard_entries]
        entries.sort(key=lambda e: float(e.get("timestamp", 0)))
        return entries

    async def get_recent(
        self, user_id: str, session_id: str, n: int
    ) -> list[dict]:
        entries = await self._all_session_entries(user_id, session_id)
        entries = [
            e for e in entries
            if e.get("frame_type") not in NON_TURN_FRAME_TYPES
            and e.get("label") not in NON_TURN_FRAME_TYPES
        ]
        return entries[-n:] if len(entries) > n else entries

    async def list_sessions(self, user_id: str) -> list[SessionInfo]:
        sessions_key = f"memory:sessions:{user_id}"
        raw = await self._redis.zrange(sessions_key, 0, -1, withscores=True)
        if not raw:
            return []

        async with self._redis.pipeline(transaction=False) as pipe:
            for session_id, _ in raw:
                pipe.get(f"memory:session:{user_id}:{session_id}:turn_count")
                pipe.get(f"memory:session:{user_id}:{session_id}:created_at")
            responses = await pipe.execute()

        result: list[SessionInfo] = []
        for i, (session_id, last_active_at) in enumerate(raw):
            turn_count = int(responses[i * 2] or 0)
            created_raw = responses[i * 2 + 1]
            created_at = float(created_raw) if created_raw else last_active_at
            result.append(SessionInfo(
                session_id=session_id,
                turn_count=turn_count,
                created_at=created_at,
                last_active_at=last_active_at,
            ))
        return result

    @staticmethod
    def _to_timeline_entry(entry: dict, session_id: Optional[str] = None) -> "TimelineEntry":
        frame_type = entry.get("frame_type") or entry.get("label") or "session_turn"
        if frame_type == "summary":
            return TimelineEntry(
                turn_index=int(entry.get("turn_index", 0)),
                timestamp=_unix_to_iso(float(entry.get("timestamp", 0))),
                frame_type="summary",
                user_message="",
                assistant_response=entry.get("text", ""),
                media_refs=[],
                tool_calls=[],
                session_id=session_id,
            )
        if frame_type == "compact_boundary":
            segment = entry.get("preserved_segment", {}) or {}
            tail_turn = entry.get("turn_index")
            return TimelineEntry(
                turn_index=int(tail_turn) if tail_turn is not None else 0,
                timestamp=_unix_to_iso(float(entry.get("timestamp", 0))),
                frame_type="compact_boundary",
                user_message="",
                assistant_response=(
                    f"[compact boundary: {entry.get('trigger', '')}/{entry.get('source', '')}"
                    f" — tail from frame {segment.get('tail_frame_id')}]"
                ),
                media_refs=[],
                tool_calls=[],
                session_id=session_id,
            )
        return TimelineEntry(
            turn_index=int(entry.get("turn_index", 0)),
            timestamp=_unix_to_iso(float(entry.get("timestamp", 0))),
            frame_type="session_turn",
            user_message=entry.get("user_message", ""),
            assistant_response=entry.get("assistant_response", ""),
            media_refs=entry.get("search_results") or [],
            tool_calls=entry.get("tool_calls") or [],
            provider_message_id=entry.get("provider_message_id"),
            session_id=session_id,
        )

    async def get_timeline(
        self,
        user_id: str,
        session_id: str,
        page: int,
        page_size: int,
        at: Optional[str] = None,
    ) -> TimelineResponse:
        all_entries = await self._all_session_entries(user_id, session_id)
        if not all_entries:
            return TimelineResponse(entries=[], total=0, page=page, page_size=page_size, has_next=False)

        if at is not None:
            at_ts = _parse_iso(at)
            all_entries = [e for e in all_entries if float(e.get("timestamp", 0)) <= at_ts]

        total = len(all_entries)
        start = (page - 1) * page_size
        page_entries = all_entries[start : start + page_size]

        return TimelineResponse(
            entries=[self._to_timeline_entry(e) for e in page_entries],
            total=total,
            page=page,
            page_size=page_size,
            has_next=(start + page_size) < total,
        )

    async def get_user_timeline(
        self, user_id: str, page: int, page_size: int,
    ) -> TimelineResponse:
        """Chronological timeline across every session this user has —
        unlike get_timeline (one session), this is the whole-user view."""
        sessions = await self.list_sessions(user_id)
        per_session = await asyncio.gather(*(
            self._all_session_entries(user_id, s.session_id)
            for s in sessions
        ))

        all_entries: list[tuple[dict, str]] = [
            (entry, s.session_id)
            for s, entries in zip(sessions, per_session)
            for entry in entries
        ]
        all_entries.sort(key=lambda pair: float(pair[0].get("timestamp", 0)))

        total = len(all_entries)
        start = (page - 1) * page_size
        page_entries = all_entries[start : start + page_size]

        return TimelineResponse(
            entries=[self._to_timeline_entry(e, session_id=sid) for e, sid in page_entries],
            total=total,
            page=page,
            page_size=page_size,
            has_next=(start + page_size) < total,
        )

    async def search(
        self, user_id: str, session_id: str, req: SessionSearchRequest
    ) -> SessionSearchResponse:
        shards = self._shard_paths(user_id, session_id)
        if not shards:
            return SessionSearchResponse(hits=[], total_frames_searched=0)

        from services.memvid_store import _get_embedder, get_search_pool
        query_vec = await asyncio.to_thread(_get_embedder().embed_query, req.query)

        indices = await asyncio.gather(*(
            asyncio.to_thread(meta_index.load, p) for p in shards
        ))
        total = sum(len(idx) for idx in indices)

        pool = get_search_pool()
        loop = asyncio.get_running_loop()
        shard_hits = await asyncio.gather(*(
            loop.run_in_executor(pool, sync_search, p, req.query, req.top_k * 3, query_vec)
            for p in shards
        ))
        raw_hits = [h for hits in shard_hits for h in hits]

        from_ts = _parse_iso(req.from_date) if req.from_date else None
        to_ts = _parse_iso(req.to_date) if req.to_date else None

        filtered: list[dict] = []
        for h in raw_hits:
            ts = float(h.get("timestamp", 0))
            if from_ts is not None and ts < from_ts:
                continue
            if to_ts is not None and ts > to_ts:
                continue
            filtered.append(h)

        filtered.sort(key=lambda h: h.get("score", float("-inf")), reverse=True)
        filtered = filtered[: req.top_k]

        scores = normalize_scores([h.get("score", 0.0) for h in filtered])
        hits = [
            SessionSearchHit(
                text=h.get("text") or h.get("snippet", ""),
                score=score,
                timestamp=_unix_to_iso(float(h.get("timestamp", 0))),
                turn_index=int(h.get("turn_index", 0)),
                user_message=h.get("user_message", ""),
                assistant_response=h.get("assistant_response", ""),
            )
            for h, score in zip(filtered, scores)
        ]
        return SessionSearchResponse(hits=hits, total_frames_searched=total)

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        shards = self._shard_paths(user_id, session_id)
        if not shards:
            return False

        for shard in shards:
            Path(shard).unlink()
            meta_index.delete(shard)

        await self._redis.zrem(f"memory:sessions:{user_id}", session_id)
        await self._redis.delete(
            f"memory:session:{user_id}:{session_id}:turn_count",
            f"memory:session:{user_id}:{session_id}:created_at",
        )
        return True
