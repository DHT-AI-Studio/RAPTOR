"""Shard management for per-session .mv2 storage.

Mirrors LongTermMemoryService's shard rollover (services/long_term_memory.py)
but keyed per (user_id, session_id) instead of per user — a single session's
.mv2 file has no cap on how long a conversation can run, and MV-12 compaction
never shrinks it (summarized turns are archived, not deleted), so without
rollover a long-running session eventually hits the memvid-sdk's free-tier
50MB-per-file capacity and every subsequent append_turn starts failing.

Shared by session_memory.py (writes/reads), compact_memory.py (reads all
shards, writes summary/boundary frames to the active one), and
routers/management.py (directory-scan discovery for stats/export/delete —
no Redis session list available there).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from core.paths import user_dir as _ensure_user_dir
from services.memvid_store import sync_get_remaining_capacity

# Roll over to a new shard when remaining capacity drops below 1 MB — same
# threshold and rationale as LongTermMemoryService's _SHARD_CAPACITY_THRESHOLD.
SHARD_CAPACITY_THRESHOLD = 1 * 1024 * 1024


def shard_paths(root: Path, user_id: str, session_id: str) -> list[str]:
    """Return all existing shard paths for this session, sorted by shard index."""
    user_dir = root / f"user_{user_id}"
    if not user_dir.exists():
        return []
    shards = sorted(
        user_dir.glob(f"session_{session_id}_*.mv2"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return [str(p) for p in shards]


def shard0_path(root: Path, user_id: str, session_id: str) -> str:
    """Return shard-0 path, creating the user/media dirs if needed."""
    user_dir = _ensure_user_dir(root, user_id)
    (user_dir / "media").mkdir(parents=True, exist_ok=True)
    (user_dir / "media").chmod(0o700)
    return str(user_dir / f"session_{session_id}_0.mv2")


async def get_active_shard_path(root: Path, user_id: str, session_id: str) -> str:
    """Return the writable shard for this session, rolling over to a new one
    when the current last shard is nearly full."""
    shards = shard_paths(root, user_id, session_id)
    if not shards:
        return shard0_path(root, user_id, session_id)

    last = shards[-1]
    remaining = await asyncio.to_thread(sync_get_remaining_capacity, last)
    if remaining < SHARD_CAPACITY_THRESHOLD:
        idx = int(Path(last).stem.rsplit("_", 1)[-1]) + 1
        user_dir = _ensure_user_dir(root, user_id)
        return str(user_dir / f"session_{session_id}_{idx}.mv2")
    return last


def parse_session_shard_filename(stem: str) -> tuple[str, int] | None:
    """Parse a `session_{session_id}_{shard_idx}` file stem back into
    (session_id, shard_idx). Returns None if it doesn't match that shape
    (e.g. legacy pre-sharding `session_{session_id}` files with no index).

    Known limitation: a session_id that itself ends in `_<digits>` is
    ambiguous with the shard suffix — session IDs in this system are
    caller-supplied path segments, not expected to look like that.
    """
    if not stem.startswith("session_"):
        return None
    rest = stem[len("session_"):]
    session_id, _, idx_str = rest.rpartition("_")
    if not session_id or not idx_str.isdigit():
        return None
    return session_id, int(idx_str)
