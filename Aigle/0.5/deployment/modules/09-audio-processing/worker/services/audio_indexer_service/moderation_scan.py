"""CM-4 / VIE01-403: Content moderation scan of document chunks and ASR
transcript segments before personal indexing.

Reuses the GD-6 pii_scan.py hook path (Module 12 document_indexer_service ->
Module 23 guardrail service). Calls the same POST /guardrail/policy/decide
endpoint; writes the full Llama Guard 3 category verdict as `moderation_flags`
onto each chunk/segment's payload. Module 25's kafka_consumer._map_entry()
already reads `ep.get("moderation_flags")` (CM-3 path) and writes it to the
Chunk vertex — no changes needed in Module 25.

Design mirrors pii_scan.py exactly:
  - Fail-open: a scan error or timeout leaves the entry unflagged and logs a
    warning; ingestion is never blocked.
  - No raw content in logs: only category codes and counts are logged.
  - Concurrent: MAX_CONCURRENT_SCANS semaphore, same value as pii_scan.py.
  - Settings passed in, not imported: no import-time dependency on config.

`moderation_flags` shape written to each chunk payload:
    {
        "safe": bool,
        "categories": ["S1", "S3"],   # empty list when safe
        "checked_at": "2026-11-12T...",
    }
Note: /guardrail/policy/decide (Module 23 DecideResponse) does not expose
which guard model answered; `model` is intentionally omitted from the flags.
Only written when the scan succeeds. An entry whose scan failed or timed out
gets no `moderation_flags` key at all (fail-open, same as pii_scan.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import aiohttp
from moderation_flag_reporter import report_moderation_flags

logger = logging.getLogger(__name__)

SCAN_DIRECTION = "input"
MAX_CONCURRENT_SCANS = 8
# /guardrail/policy/decide doesn't expose which guard model answered (see module
# docstring); text/transcript modality is always screened by Llama Guard 3's S-codes,
# so this is fixed rather than threaded through from a response that doesn't have it.
_MODEL = "llama-guard-3"


def _payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Unwrap {"id", "payload": {...}} shape; flat dict is its own payload."""
    payload = entry.get("payload")
    return payload if isinstance(payload, dict) else entry


def _chunk_text(entry: dict[str, Any]) -> str:
    ep = _payload(entry)
    # Audio segments may carry text in `text` or `asr_text`
    return (ep.get("text") or ep.get("asr_text") or "").strip()


async def _scan_one(
    session: aiohttp.ClientSession,
    url: str,
    text: str,
    scan_module: str,
) -> dict[str, Any] | None:
    """Call Module 23 and return a moderation_flags dict, or None on failure.

    Returns None (not raises) so the semaphore-guarded caller can stay
    fail-open without a try/except at every call site."""
    body = {
        "content": text,
        "direction": SCAN_DIRECTION,
        "module": scan_module,
        "request_id": str(uuid.uuid4()),
    }
    try:
        async with session.post(url, json=body) as resp:
            resp.raise_for_status()
            result = await resp.json()
    except (aiohttp.ClientError, json.JSONDecodeError) as exc:
        logger.debug("[moderation-scan] HTTP/parse error (fail-open): %s", type(exc).__name__)
        return None

    categories: list[str] = result.get("categories") or []
    # DecideResponse has no "safe" field; derive from absence of flagged categories
    # (mirrors pii_scan.py pattern — endpoint returns decision/reason/categories only)
    safe: bool = result.get("safe", not categories)
    return {
        "safe": safe,
        "categories": categories,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


async def scan_entries(
    entries: list[dict[str, Any]],
    guardrail_url: str,
    timeout_ms: int,
    scan_module: str,
) -> int:
    """Scan each entry and stamp `moderation_flags` onto its payload.

    Returns the count of entries that came back unsafe.
    Entries without text (e.g. summary-only entries) are skipped.
    Never raises.
    """
    url = f"{guardrail_url.rstrip('/')}/guardrail/policy/decide"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCANS)
    failures: list[str] = []

    async def scan(session: aiohttp.ClientSession, entry: dict[str, Any]) -> None:
        text = _chunk_text(entry)
        if not text:
            return
        async with semaphore:
            flags = await _scan_one(session, url, text, scan_module)
        if flags is None:
            failures.append("scan_one returned None")
            return
        _payload(entry)["moderation_flags"] = flags
        categories = flags.get("categories")
        # entry["id"] (top-level, not inside payload) is this chunk's id as Module 25's
        # kafka_consumer._map_entry() will read it (entry.get("id") or a derived
        # fallback) -- assigned upstream in *_summary_service, already present here.
        chunk_id = entry.get("id")
        branch_id = _payload(entry).get("branch_id")
        version_id = _payload(entry).get("version_id")
        asset_path = _payload(entry).get("asset_path")
        if not flags.get("safe", True) and categories and chunk_id and branch_id and version_id and asset_path:
            # CM-5 / VIE01-408: report to Module 23's review store, alongside (not
            # instead of) the moderation_flags just stamped above. Never affects the
            # caller's own scan result either way.
            try:
                await report_moderation_flags(
                    session, guardrail_url, branch_id=branch_id, version_id=version_id, asset_path=asset_path,
                    categories=categories, model=_MODEL, chunk_id=chunk_id,
                )
            except Exception as exc:  # noqa: BLE001 -- must never affect this chunk's own scan result
                logger.warning("[moderation-flag] report failed for %s: %s", chunk_id, exc)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(*(scan(session, entry) for entry in entries))
    except Exception as exc:  # noqa: BLE001 -- "never raises" contract; per-entry errors already isolated in scan()
        logger.warning(
            "[moderation-scan] scan aborted, indexing without moderation flags: %s", exc
        )
        return 0

    if failures:
        logger.warning(
            "[moderation-scan] %d of %d entries not scanned (indexed without moderation_flags)",
            len(failures), len(entries),
        )

    unsafe_count = sum(
        1 for e in entries
        if isinstance(_payload(e).get("moderation_flags"), dict)
        and not _payload(e)["moderation_flags"].get("safe", True)
    )
    total_scanned = sum(
        1 for e in entries if _payload(e).get("moderation_flags") is not None
    )
    logger.info(
        "[moderation-scan] module=%s scanned=%d unsafe=%d skipped=%d",
        scan_module,
        total_scanned,
        unsafe_count,
        len(failures),
        # ↑ No chunk text in logs — AC requirement
    )
    return unsafe_count
