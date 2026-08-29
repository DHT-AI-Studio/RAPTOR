from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from core.config import settings
from core.dependencies import get_current_user, get_redis
from services import meta_index
from services.compact_memory import NON_TURN_FRAME_TYPES
from services.memvid_store import sync_timeline
from services.session_memory import SessionMemoryService, TimelineResponse
from services.session_shards import parse_session_shard_filename

router = APIRouter(prefix="/memory", tags=["Management"])

EXPORT_SCHEMA_VERSION = "1.0"


# ── Response models ───────────────────────────────────────────────────────────

class MemoryStats(BaseModel):
    session_count: int = Field(..., description="使用者目前擁有的 session 數量")
    total_turns: int = Field(..., description="所有 session 累計的對話輪數（不含 summary / compact_boundary 記帳用 frame）")
    summary_frame_count: int = Field(..., description="所有 session 累計的壓縮摘要（summary frame）數量")
    total_media_items: int = Field(..., description="已索引的多媒體項目數（video + audio + image）")
    long_term_frame_count: int = Field(..., description="long-term 記憶的 frame 總數（含所有 shard）")
    storage_bytes_used: int = Field(..., description="使用者所有 .mv2 / .meta.json 檔案的總儲存位元組數")


# ── Service ───────────────────────────────────────────────────────────────────

class ManagementService:
    def __init__(self) -> None:
        self._root = Path(settings.storage_root)

    def _user_dir(self, user_id: str) -> Path:
        return self._root / f"user_{user_id}"

    async def get_stats(self, user_id: str) -> MemoryStats:
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return MemoryStats(
                session_count=0,
                total_turns=0,
                summary_frame_count=0,
                total_media_items=0,
                long_term_frame_count=0,
                storage_bytes_used=0,
            )

        session_files = list(user_dir.glob("session_*.mv2"))
        lt_files = list(user_dir.glob("long_term_*.mv2"))
        media_dir = user_dir / "media"
        media_files = list(media_dir.glob("*.mv2")) if media_dir.exists() else []

        # Frame counts from lightweight sidecar — no need to open .mv2
        count_tasks = (
            [asyncio.to_thread(meta_index.load, str(p)) for p in session_files]
            + [asyncio.to_thread(meta_index.load, str(p)) for p in lt_files]
            + [asyncio.to_thread(meta_index.load, str(p)) for p in media_files]
        )
        all_indices = await asyncio.gather(*count_tasks) if count_tasks else []

        n_sess = len(session_files)
        sess_counts = all_indices[:n_sess]
        lt_counts = all_indices[n_sess : n_sess + len(lt_files)]
        media_counts = all_indices[n_sess + len(lt_files) :]

        # A session can span multiple shard files (session_shards.py) — count
        # distinct session_ids, not files.
        session_ids = {
            parsed[0]
            for p in session_files
            if (parsed := parse_session_shard_filename(p.stem)) is not None
        }
        session_count = len(session_ids)

        # session_*.mv2 frames mix real conversation turns with compaction
        # bookkeeping (summary / compact_boundary frames written by MV-12) — count
        # them separately so total_turns reflects actual conversation volume.
        total_turns = 0
        summary_frame_count = 0
        for idx in sess_counts:
            for meta in idx.values():
                frame_type = meta.get("frame_type", "")
                if frame_type == "summary":
                    summary_frame_count += 1
                if frame_type not in NON_TURN_FRAME_TYPES:
                    total_turns += 1
        long_term_frame_count = sum(len(idx) for idx in lt_counts)
        total_media_items = sum(len(idx) for idx in media_counts)

        storage_bytes_used = sum(
            f.stat().st_size for f in user_dir.rglob("*") if f.is_file()
        )

        return MemoryStats(
            session_count=session_count,
            total_turns=total_turns,
            summary_frame_count=summary_frame_count,
            total_media_items=total_media_items,
            long_term_frame_count=long_term_frame_count,
            storage_bytes_used=storage_bytes_used,
        )

    async def export_generator(self, user_id: str) -> AsyncGenerator[bytes, None]:
        """Yield JSON bytes incrementally — sessions streamed one by one."""
        user_dir = self._user_dir(user_id)
        exported_at = datetime.now(timezone.utc).isoformat()

        yield b'{\n'
        yield f'  "export_schema_version": {json.dumps(EXPORT_SCHEMA_VERSION)},\n'.encode()
        yield f'  "user_id": {json.dumps(user_id)},\n'.encode()
        yield f'  "exported_at": {json.dumps(exported_at)},\n'.encode()

        # ── Sessions (streamed one per chunk) ──────────────────────────────────
        # A session can span multiple shard files (session_shards.py) — group
        # by session_id and merge+sort each session's turns across its shards.
        yield b'  "sessions": [\n'
        session_files = sorted(user_dir.glob("session_*.mv2")) if user_dir.exists() else []
        shards_by_session: dict[str, list[Path]] = {}
        for p in session_files:
            parsed = parse_session_shard_filename(p.stem)
            if parsed is None:
                continue
            shards_by_session.setdefault(parsed[0], []).append(p)

        first = True
        for session_id, shard_paths in sorted(shards_by_session.items()):
            turns: list[dict] = []
            for path in shard_paths:
                try:
                    turns.extend(await asyncio.to_thread(sync_timeline, str(path)))
                except Exception:
                    pass
            turns.sort(key=lambda t: float(t.get("timestamp", 0)))
            obj = {"session_id": session_id, "turns": turns}
            prefix = b"" if first else b",\n"
            yield prefix + f"    {json.dumps(obj, ensure_ascii=False)}".encode()
            first = False
        yield b"\n  ],\n"

        # ── Long-term memory ──────────────────────────────────────────────────
        lt_files = sorted(user_dir.glob("long_term_*.mv2")) if user_dir.exists() else []
        all_lt: list[dict] = []
        for path in lt_files:
            try:
                entries = await asyncio.to_thread(sync_timeline, str(path))
                all_lt.extend(entries)
            except Exception:
                pass
        yield f'  "longterm": {json.dumps(all_lt, ensure_ascii=False)},\n'.encode()

        # ── Multimedia index (metadata only, no raw media) ────────────────────
        media_dir = user_dir / "media"
        media_meta: list[dict] = []
        if media_dir.exists():
            for path in sorted(media_dir.glob("*.mv2")):
                media_meta.extend(await asyncio.to_thread(meta_index.all_entries, str(path)))
        yield f'  "multimedia": {json.dumps(media_meta, ensure_ascii=False)}\n'.encode()

        yield b'}\n'

    async def delete_all(self, user_id: str, redis: Redis) -> bool:
        """GDPR erasure: remove all .mv2/.meta.json files and Redis keys for the user."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return False

        # Collect session IDs before wiping disk (needed for Redis cleanup).
        # A session can span multiple shard files — dedupe down to session_id.
        session_ids = {
            parsed[0]
            for p in user_dir.glob("session_*.mv2")
            if (parsed := parse_session_shard_filename(p.stem)) is not None
        }

        # Clean Redis first — if disk removal fails, Redis is already clean
        keys: list[str] = [f"memory:sessions:{user_id}"]
        for sid in session_ids:
            keys.append(f"memory:session:{user_id}:{sid}:turn_count")
            keys.append(f"memory:session:{user_id}:{sid}:created_at")

        # Module 15's short-term chat cache (`chat_memory:{user_id}` and
        # `chat_memory:{user_id}:{session_id}`) lives on this same shared
        # Redis cluster — GDPR erasure isn't complete while it's still
        # sitting there un-expired (up to MEMORY_TTL, default 1h).
        async for key in redis.scan_iter(match=f"chat_memory:{user_id}*"):
            keys.append(key)

        if keys:
            await redis.delete(*keys)

        # Retry with backoff: a concurrent search worker can still have a file
        # open when rmtree unlinks it — on the NFS-backed storage volume this
        # surfaces as a silly-rename ".nfsXXXX" file ("Device or resource
        # busy") until that worker closes its handle. It can also recreate a
        # sidecar .lock file (see memvid_store._lock_for) between rmtree's
        # directory listing and its rmdir call (ENOTEMPTY). Both clear once
        # the concurrent request finishes, typically well under a second.
        attempts = 8
        for attempt in range(attempts):
            try:
                await asyncio.to_thread(shutil.rmtree, str(user_dir))
                break
            except OSError:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(min(0.3 * (attempt + 1), 2.0))
        return True


# ── Endpoints ─────────────────────────────────────────────────────────────────

def get_management_service() -> ManagementService:
    return ManagementService()


def get_session_service_for_timeline(redis: Redis = Depends(get_redis)) -> SessionMemoryService:
    return SessionMemoryService(redis=redis)


@router.get(
    "/timeline",
    response_model=TimelineResponse,
    summary="跨 session 的時間軸（分頁）",
    description="依時間升序分頁回傳使用者所有 session 的對話輪次，彼此穿插排列；"
    "只想看單一 session 請用 GET /memory/sessions/{session_id}/timeline。",
)
async def get_user_timeline(
    page: int = Query(1, ge=1, description="頁碼（1-based）"),
    page_size: int = Query(20, ge=1, le=100, description="每頁筆數（最大 100）"),
    user_id: str = Depends(get_current_user),
    svc: SessionMemoryService = Depends(get_session_service_for_timeline),
) -> TimelineResponse:
    return await svc.get_user_timeline(user_id, page, page_size)


@router.get(
    "/stats",
    response_model=MemoryStats,
    summary="記憶統計",
    description="回傳目前使用者的 session 數、對話輪數、多媒體項目數、long-term frame 數與儲存用量。",
)
async def get_stats(
    user_id: str = Depends(get_current_user),
    svc: ManagementService = Depends(get_management_service),
) -> MemoryStats:
    return await svc.get_stats(user_id)


@router.get(
    "/export",
    summary="匯出完整記憶（GDPR / 資料可攜）",
    description="以串流方式回傳 JSON，內容含全部 sessions、long-term 記憶與多媒體 metadata（不含原始媒體檔案）。"
    "回應帶 `Content-Disposition: attachment`，可直接下載。",
)
async def export_memory(
    user_id: str = Depends(get_current_user),
    svc: ManagementService = Depends(get_management_service),
) -> StreamingResponse:
    filename = f"memory_export_{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    return StreamingResponse(
        svc.export_generator(user_id),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete(
    "",
    status_code=204,
    response_class=Response,
    summary="GDPR 抹除：刪除當前使用者所有記憶",
    description="永久刪除該使用者所有 .mv2、.meta.json 檔案及 Redis 索引鍵；找不到任何記憶則回 404。此操作不可復原。",
)
async def delete_all_memory(
    user_id: str = Depends(get_current_user),
    redis: Redis = Depends(get_redis),
    svc: ManagementService = Depends(get_management_service),
) -> Response:
    found = await svc.delete_all(user_id, redis)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No memory found for user")
    return Response(status_code=204)
