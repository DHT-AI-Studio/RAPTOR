"""Shared CM-5 (VIE01-408) client: reports an ingestion-time moderation verdict into
Module 23's admin-review store (``POST /moderation/flags``), one row per flagged
category.

Called by modules 09/10/11/12 -- whichever one holds the content id -- right where
each computes its own CM-2/3/4 verdict, using that verdict's already-parsed native
values (a category list, not the module's own `moderation_flags` blob shape written to
Module 04/25 -- those three shapes differ from each other and none of them matter here,
since this is called before any of them are assembled).

Fire-and-forget: Module 23 is the authority on flag storage, not on whether ingestion
should proceed. A failure here must never affect the caller's own moderation_flags
write to Module 04/25 (already done independently) or block ingestion -- same
fail-open convention as pii_scan.py / moderation_scan.py in modules 09/12.
"""
from __future__ import annotations

import asyncio
import logging

import aiohttp

logger = logging.getLogger("raptor_common.moderation_flag_reporter")


async def report_moderation_flags(
    session: aiohttp.ClientSession,
    guardrail_url: str,
    *,
    branch_id: str,
    version_id: str,
    asset_path: str,
    categories: list[str],
    model: str,
    scores: dict[str, float] | None = None,
    policy_version: str | None = None,
    moment_id: str | None = None,
    chunk_id: str | None = None,
) -> None:
    """POST one ``/moderation/flags`` row per entry in `categories` (deduplicated,
    order preserved; a no-op when empty -- call this only for content that was
    actually flagged, never for a clean verdict). `branch_id`, `version_id` and
    `asset_path` are all required on every call, regardless of content type --
    together they're the real, queryable Module 04 asset an admin reviewing this
    flag needs to actually see; a bare moment_id (a one-way hash) or chunk_id (a
    UUID) means nothing on its own. `moment_id`/`chunk_id` are an optional,
    finer-grained sub-reference within that asset -- at most one of the two
    (never both), the same rule CM-5's own ``ModerationFlagCreate`` enforces
    server-side; neither set means the flag is about the whole asset.

    Never raises: logs and moves on to the next category on any failure.
    """
    seen: list[str] = []
    for category in categories:
        if category not in seen:
            seen.append(category)
    if not seen:
        return

    scores = scores or {}
    url = guardrail_url.rstrip("/") + "/moderation/flags"
    ref = {"branch_id": branch_id, "version_id": version_id,
           "asset_path": asset_path, "moment_id": moment_id, "chunk_id": chunk_id}
    content_id = moment_id or chunk_id or asset_path

    for category in seen:
        body = {**ref, "category": category, "score": scores.get(category),
                "model": model, "policy_version": policy_version}
        try:
            async with session.post(url, json=body) as resp:
                if resp.status != 201:
                    logger.warning("[moderation-flag] %s on %s: HTTP %s (fail-open, not retried)",
                                   category, content_id, resp.status)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("[moderation-flag] %s on %s not reported (fail-open): %s",
                           category, content_id, exc)
