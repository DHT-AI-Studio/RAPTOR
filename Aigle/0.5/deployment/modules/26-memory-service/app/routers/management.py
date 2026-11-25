from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from core.dependencies import get_current_user, get_redis
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from services import meta_index
from services.compact_memory import NON_TURN_FRAME_TYPES
from services.long_term_memory import FACT_FRAME_TYPES
from services.memvid_store import sync_timeline
from services.session_memory import SessionMemoryService, TimelineResponse
from services.session_shards import parse_session_shard_filename

logger = logging.getLogger(__name__)

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

    async def get_gdpr_summary(self, user_id: str) -> dict:
        """GD-4 / VIE01-249: counts only, for Module 33's DSAR assembler --
        never frame contents (that's export_generator's job, and GD-2's cost).

        long_term_facts counts only fact-type frames (FACT_FRAME_TYPES), the
        same set GET /memory/longterm/facts (get_facts) returns -- not every
        long-term frame like get_stats' long_term_frame_count, which also
        includes "conversation" frames. Soft-deleted facts are already gone
        from the sidecar (memvid_store.sync_remove), so they don't count.
        The sidecar's frame_type is written as frame_type-or-label
        (memvid_store), so it alone covers get_facts' label fallback.
        oldest_session is the earliest frame timestamp
        across every session shard's .meta.json sidecar -- a real per-turn
        timestamp, unlike expire_sessions' filesystem mtime (which only tells
        when a shard was last written, not when the session began). Falls
        back to the oldest shard's mtime only for a session whose sidecar
        carries no timestamps at all."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return {"user_id": user_id, "sessions": 0, "long_term_facts": 0, "oldest_session": None}

        session_files = list(user_dir.glob("session_*.mv2"))
        lt_files = list(user_dir.glob("long_term_*.mv2"))
        indices = await asyncio.gather(
            *[asyncio.to_thread(meta_index.load, str(p)) for p in session_files + lt_files]
        ) if session_files or lt_files else []
        sess_indices = indices[: len(session_files)]
        lt_indices = indices[len(session_files):]

        session_ids = {
            parsed[0]
            for p in session_files
            if (parsed := parse_session_shard_filename(p.stem)) is not None
        }

        oldest_ts: float | None = None
        for path, idx in zip(session_files, sess_indices):
            stamps = [float(m["timestamp"]) for m in idx.values() if m.get("timestamp")]
            candidate = min(stamps) if stamps else path.stat().st_mtime
            if oldest_ts is None or candidate < oldest_ts:
                oldest_ts = candidate

        return {
            "user_id": user_id,
            "sessions": len(session_ids),
            "long_term_facts": sum(
                1 for idx in lt_indices for meta in idx.values() if meta.get("frame_type") in FACT_FRAME_TYPES
            ),
            "oldest_session": (
                datetime.fromtimestamp(oldest_ts, tz=timezone.utc).isoformat() if oldest_ts is not None else None
            ),
        }

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
                except Exception as e:  # noqa: BLE001 -- memvid SDK + file I/O, exception surface not documented; already logged with the data-loss tradeoff stated explicitly
                    logger.warning(f"Export: shard {path} unreadable, skipping (data loss in export): {e}")
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
            except Exception as e:  # noqa: BLE001 -- memvid SDK + file I/O, exception surface not documented; already logged with the data-loss tradeoff stated explicitly
                logger.warning(f"Export: long-term file {path} unreadable, skipping (data loss in export): {e}")
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
        """GDPR erasure: remove all .mv2/.meta.json files and Redis keys for the user.

        Redis cleanup must not be gated on the on-disk user directory existing:
        a user can have only Module 15's short-term chat cache (no memvid
        directory ever created, e.g. they chatted but nothing was archived
        yet) and still have stale Redis records an erasure request must clear."""
        user_dir = self._user_dir(user_id)
        dir_exists = user_dir.exists()

        # Collect session IDs before wiping disk (needed for Redis cleanup).
        # A session can span multiple shard files — dedupe down to session_id.
        session_ids = {
            parsed[0]
            for p in user_dir.glob("session_*.mv2")
            if (parsed := parse_session_shard_filename(p.stem)) is not None
        } if dir_exists else set()

        # Clean Redis first — if disk removal fails, Redis is already clean
        keys: list[str] = [f"memory:sessions:{user_id}"]
        for sid in session_ids:
            keys.append(f"memory:session:{user_id}:{sid}:turn_count")
            keys.append(f"memory:session:{user_id}:{sid}:created_at")

        # Module 15's short-term chat cache (`chat_memory:{user_id}` and
        # `chat_memory:{user_id}:{session_id}`) lives on this same shared
        # Redis cluster — GDPR erasure isn't complete while it's still
        # sitting there un-expired (up to MEMORY_TTL, default 1h). This scan
        # runs even when the user has no memvid directory at all.
        found_redis_keys = False
        async for key in redis.scan_iter(match=f"chat_memory:{user_id}*"):
            keys.append(key)
            found_redis_keys = True

        if len(keys) > 1 or found_redis_keys:
            await redis.delete(*keys)

        if not dir_exists:
            return found_redis_keys

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

    async def expire_sessions(self, older_than_days: int, redis: Redis, dry_run: bool = False) -> int:
        """GD-3 / VIE01-248: retention sweep across every user's storage, not
        just one -- unlike delete_all's per-user erasure. A session's "age" is
        its most recent shard's mtime, not its first: a session with any
        activity in the last `older_than_days` is kept even if shard 0 is
        ancient. No durable per-session timestamp exists anywhere else (not
        in the filename, not durably in Redis -- the *:created_at keys below
        are TTL'd caches), so filesystem mtime is the only real signal.
        """
        if not self._root.exists():
            return 0
        cutoff = time.time() - older_than_days * 86400
        expired_count = 0
        for user_dir in self._root.iterdir():
            if not user_dir.is_dir() or not user_dir.name.startswith("user_"):
                continue
            shards_by_session: dict[str, list[Path]] = {}
            for p in user_dir.glob("session_*.mv2"):
                parsed = parse_session_shard_filename(p.stem)
                if parsed is None:
                    continue
                shards_by_session.setdefault(parsed[0], []).append(p)

            user_id = user_dir.name[len("user_"):]
            for session_id, shards in shards_by_session.items():
                newest_mtime = max(p.stat().st_mtime for p in shards)
                if newest_mtime >= cutoff:
                    continue
                expired_count += 1
                if dry_run:
                    continue
                for p in shards:
                    p.unlink(missing_ok=True)
                    p.with_suffix(".meta.json").unlink(missing_ok=True)
                await redis.delete(
                    f"memory:session:{user_id}:{session_id}:turn_count",
                    f"memory:session:{user_id}:{session_id}:created_at",
                )
        return expired_count


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
