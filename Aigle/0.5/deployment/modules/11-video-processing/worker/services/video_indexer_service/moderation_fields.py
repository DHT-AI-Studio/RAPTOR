# services/video_indexer_service/moderation_fields.py
"""CM-3 (VIE01-406): put a video's keyframe moderation verdicts on what the indexer
publishes to Module 25 -- each moment's flags on its moment entry (-> Chunk), the
video's rollup on its summary entry (-> Source). Set upstream by
video_chunking_service/video_moderation.py; absent when moderation is off."""
from __future__ import annotations

from typing import Any


def with_moment_moderation(moments: list[dict[str, Any]],
                           chunking_moments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CM-3: each moment's keyframe moderation_flags (set by video_chunking_service
    on the chunking moments) copied, by moment_index, onto the moments published
    to Module 25 -- which may be the contextualized copies, built without them.
    Moments without flags (moderation off) are returned unchanged."""
    by_index = {m.get("moment_index"): m["moderation_flags"] for m in chunking_moments
                if isinstance(m.get("moderation_flags"), dict)}
    if not by_index:
        return moments
    return [{**m, "moderation_flags": by_index[m.get("moment_index")]}
            if m.get("moment_index") in by_index else m for m in moments]


def with_summary_moderation(entries: list[dict[str, Any]],
                            flags: dict[str, Any] | None) -> list[dict[str, Any]]:
    """CM-3: the video's moderation rollup on its summary entry, which Module 25
    writes to Source.moderation_flags (as CM-2 does for images)."""
    if not isinstance(flags, dict):
        return entries
    out = []
    for entry in entries:
        payload = entry.get("payload", entry)
        if payload.get("embedding_type") == "summary":
            payload = {**payload, "moderation_flags": flags}
            entry = {**entry, "payload": payload} if "payload" in entry else payload
        out.append(entry)
    return out
