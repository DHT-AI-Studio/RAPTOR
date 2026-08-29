from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.config import settings
from core.paths import user_dir as _ensure_user_dir
from services import meta_index
from services.memvid_store import (
    get_search_pool,
    normalize_scores,
    sync_append,
    sync_get_remaining_capacity,
    sync_remove,
    sync_search,
    sync_timeline,
)

FrameType = Literal["conversation", "preference", "entity", "fact"]

# Roll over to a new shard when remaining capacity drops below 1 MB.
# Each frame costs ~25 KB in lex-index overhead; 1 MB gives ~40 frames of
# headroom before hitting the 50 MB free-tier cap.
_SHARD_CAPACITY_THRESHOLD = 1 * 1024 * 1024


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class FactAddRequest(BaseModel):
    text: str = Field(..., description="要記住的事實 / 偏好 / 實體描述文字")
    frame_type: FrameType = Field(
        "fact",
        description="conversation（重要對話摘要）| preference（使用者偏好）| entity（實體，如人物/專案）| fact（一般事實）",
    )
    session_id: Optional[str] = Field(
        None, description="來源 session ID（可省略，僅作追溯用途，不影響跨 session 檢索）"
    )


class FactAddResponse(BaseModel):
    frame_id: str = Field(..., description="寫入/命中的 frame ID，可用於之後刪除")
    frame_type: str = Field(..., description="實際寫入的 frame 類型")
    timestamp: float = Field(..., description="寫入時間（Unix timestamp）；duplicate 時為既有 frame 的原始時間")
    status: Literal["created", "updated", "duplicate"] = Field(
        "created", description="created：新寫入 | updated：語意衝突，覆寫舊 frame | duplicate：語意重複，未寫入"
    )
    duplicate_of: Optional[str] = Field(
        None, description="status 為 duplicate 或 updated 時，對應到的既有 frame_id"
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="查詢文字，做 BM25 + bge-m3 語意向量混合搜尋")
    top_k: int = Field(5, ge=1, le=50, description="回傳筆數上限")
    from_date: Optional[float] = Field(
        None, description="Unix timestamp；只回傳此時間（含）之後寫入的記憶，可省略"
    )
    to_date: Optional[float] = Field(
        None, description="Unix timestamp；只回傳此時間（含）之前寫入的記憶，可省略"
    )


class SearchHit(BaseModel):
    text: str = Field(..., description="命中的事實/偏好/實體文字")
    score: float = Field(..., description="正規化後的混合相關性分數（0~1）")
    timestamp: float = Field(..., description="寫入時間（Unix timestamp）")
    session_id: str = Field(..., description="來源 session ID（若寫入時有提供）")
    frame_type: str = Field(..., description="frame 類型：conversation | preference | entity | fact")


# ── Service ───────────────────────────────────────────────────────────────────

class LongTermMemoryService:
    def __init__(self, storage_root: str | None = None) -> None:
        self._root = Path(storage_root or settings.storage_root)

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _user_dir(self, user_id: str) -> Path:
        """Return user dir, creating it and the media subdir on first write."""
        user_dir = _ensure_user_dir(self._root, user_id)
        (user_dir / "media").mkdir(parents=True, exist_ok=True)
        (user_dir / "media").chmod(0o700)
        return user_dir

    def _user_dir_ro(self, user_id: str) -> Path:
        """Return user dir without creating it (for read-only lookups)."""
        return self._root / f"user_{user_id}"

    def _shard_paths(self, user_id: str) -> list[str]:
        """Return all existing shard paths sorted by shard index."""
        user_dir = self._user_dir_ro(user_id)
        if not user_dir.exists():
            return []
        shards = sorted(
            user_dir.glob("long_term_*.mv2"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        return [str(p) for p in shards]

    def _lt_path(self, user_id: str) -> str:
        """Return shard-0 path. Kept for unit-test direct seeding compatibility."""
        return str(self._user_dir(user_id) / "long_term_0.mv2")

    async def _get_active_shard_path(self, user_id: str) -> str:
        """Return the writable shard, rolling over to a new one when nearly full."""
        shards = self._shard_paths(user_id)
        if not shards:
            return self._lt_path(user_id)  # shard 0, created on first write

        last = shards[-1]
        remaining = await asyncio.to_thread(sync_get_remaining_capacity, last)
        if remaining < _SHARD_CAPACITY_THRESHOLD:
            idx = int(Path(last).stem.rsplit("_", 1)[-1]) + 1
            return str(self._user_dir(user_id) / f"long_term_{idx}.mv2")
        return last

    # ── Public API ────────────────────────────────────────────────────────────

    async def add_fact(
        self, user_id: str, data: FactAddRequest, *, skip_dedup: bool = False
    ) -> FactAddResponse:
        if not skip_dedup:
            from services.dedup import check_duplicate

            existing_full = await self.get_facts(user_id)
            verdict, match_id = await check_duplicate(data.text, existing_full)

            if verdict == "DUPLICATE" and match_id:
                matched = next(
                    (f for f in existing_full if str(f.get("frame_id")) == match_id), None
                )
                if matched:
                    return FactAddResponse(
                        frame_id=match_id,
                        frame_type=matched.get("frame_type", data.frame_type),
                        timestamp=float(matched.get("timestamp", 0)),
                        status="duplicate",
                        duplicate_of=match_id,
                    )

            if verdict == "UPDATE" and match_id:
                shard = await asyncio.to_thread(self._find_shard_for_frame, user_id, match_id)
                if shard:
                    await asyncio.to_thread(sync_remove, shard, match_id)
                resp = await self._write_fact(user_id, data)
                resp.status = "updated"
                resp.duplicate_of = match_id
                return resp

        return await self._write_fact(user_id, data)

    async def _write_fact(self, user_id: str, data: FactAddRequest) -> FactAddResponse:
        ts = time.time()
        frame = {
            "title": f"{data.frame_type}: {data.text[:60]}",
            "label": data.frame_type,
            "text": data.text,
            "timestamp": ts,
            "session_id": data.session_id or "",
            "frame_type": data.frame_type,
        }
        shard_path = await self._get_active_shard_path(user_id)
        frame_id = await asyncio.to_thread(sync_append, shard_path, frame)
        return FactAddResponse(frame_id=frame_id, frame_type=data.frame_type, timestamp=ts)

    def _find_shard_for_frame(self, user_id: str, frame_id: str) -> str | None:
        """Scan shards to find which one contains frame_id (via meta_index)."""
        for path in self._shard_paths(user_id):
            idx = meta_index.load(path)
            if frame_id in idx:
                return path
        return None

    async def search(
        self, user_id: str, req: SearchRequest, *, _raw_scores: bool = False
    ) -> list[SearchHit]:
        """`_raw_scores=True` skips the per-batch normalization and returns
        unscaled SDK scores — used by GlobalSearchService to normalize once
        across all scopes together instead of once per scope."""
        shards = self._shard_paths(user_id)
        if not shards:
            return []

        from services.memvid_store import _get_embedder
        query_vec = await asyncio.to_thread(_get_embedder().embed_query, req.query)

        pool = get_search_pool()
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(pool, sync_search, path, req.query, req.top_k * 4, query_vec)
            for path in shards
        ]
        shard_results = await asyncio.gather(*tasks)

        # Merge candidates and apply date filter
        candidates: list[dict] = []
        for hits in shard_results:
            candidates.extend(hits)

        filtered: list[dict] = []
        for hit in candidates:
            ts = float(hit.get("timestamp", 0))
            if req.from_date is not None and ts < req.from_date:
                continue
            if req.to_date is not None and ts > req.to_date:
                continue
            filtered.append(hit)

        # SDK scores: closer to 0 = more relevant (scale varies by query)
        filtered.sort(key=lambda h: h.get("score", float("-inf")), reverse=True)
        filtered = filtered[: req.top_k]

        # Normalize relative to the other candidates in this batch
        raw_scores = [h.get("score", 0.0) for h in filtered]
        scores = raw_scores if _raw_scores else normalize_scores(raw_scores)
        results: list[SearchHit] = []
        for hit, score in zip(filtered, scores):
            results.append(SearchHit(
                text=hit.get("text") or hit.get("snippet", ""),
                score=score,
                timestamp=float(hit.get("timestamp", 0)),
                session_id=hit.get("session_id", ""),
                frame_type=hit.get("frame_type") or hit.get("label", ""),
            ))
        return results

    async def get_facts(self, user_id: str) -> list[dict]:
        """Return all preference + fact frames across all shards, sorted by recency."""
        shards = self._shard_paths(user_id)
        if not shards:
            return []

        tasks = [asyncio.to_thread(sync_timeline, path) for path in shards]
        timelines = await asyncio.gather(*tasks)

        all_entries: list[dict] = []
        for timeline in timelines:
            all_entries.extend(timeline)

        kept = [
            e for e in all_entries
            if e.get("frame_type") in ("preference", "fact", "entity")
            or e.get("label") in ("preference", "fact", "entity")
        ]
        kept.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return kept

    async def delete(self, user_id: str) -> bool:
        """Delete all shards for a user's long-term memory. Returns False if none exist."""
        shards = self._shard_paths(user_id)
        if not shards:
            return False
        for path in shards:
            Path(path).unlink()
            meta_index.delete(path)
        return True

    async def delete_fact(self, user_id: str, frame_id: str) -> bool:
        """Soft-delete a single fact/preference/entity frame. Returns False if not found."""
        for path in self._shard_paths(user_id):
            idx = await asyncio.to_thread(meta_index.load, path)
            if frame_id in idx:
                await asyncio.to_thread(sync_remove, path, frame_id)
                return True
        return False
