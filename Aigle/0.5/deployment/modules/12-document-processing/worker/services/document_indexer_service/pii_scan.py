"""GD-6 / VIE01-251: PII scan of document chunks before personal indexing.

Each chunk's text goes to Module 23's policy-aware decision API
(POST {GUARDRAIL_URL}/guardrail/policy/decide, direction="input",
module="12"). A result whose `categories` -- the codes the guard model
(Llama Guard) reported, whatever the active policy then decided -- include
"S7" (Privacy) marks the chunk `pii_detected=True`; Module 25 turns that
into `pii_flag` on the chunk's ArcadeDB vertex. The `decision` itself is
not used: this scan only flags, it never blocks or redacts a chunk. With an
active policy, the guard model is prompted with that policy's categories,
so a policy without S7 means S7 is never reported (only CONTACT_PATTERNS
below still flag).

Llama Guard doesn't always treat plain contact details as private (e.g.
"Contact Jane at jane@example.com or call 0912-345-678" comes back safe),
so a chunk is also flagged when CONTACT_PATTERNS find an email address,
phone number or Taiwanese ID number -- but only once the guard call itself
succeeded, so a failed/timed-out scan still leaves the chunk unflagged.

Fail-open by design: a chunk whose scan errors or times out is indexed
without a flag (`pii_detected` stays unset) and a warning is logged. The
scan never blocks or fails ingestion.

Settings are passed in rather than read from `config` here, so this file
has no import-time dependency on the indexer's config module.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

PII_CATEGORY = "S7"
SCAN_MODULE = "12"
# Chunks scanned at once -- a document can have hundreds, and a sequential
# scan at up to M12_PII_SCAN_TIMEOUT_MS each would stall the indexer.
MAX_CONCURRENT_SCANS = 8

# Deterministic backup for what Llama Guard misses: identifiers that are
# personal data on their own. Names and addresses are left to the model.
CONTACT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",                     # email
    r"(?<!\d)09\d{2}[-\s]?\d{3}[-\s]?\d{3}(?!\d)",               # TW mobile
    r"(?<!\d)\(?0\d{1,2}\)?[-\s]?\d{3,4}-\d{4}(?!\d)",            # TW landline
    r"\+\d{1,3}[-\s]?\(?\d{1,4}\)?(?:[-\s]?\d{2,4}){2,4}(?!\d)",   # international
    r"(?<!\d)\(?\d{3}\)?[-.\s]\d{3}[-.]\d{4}(?!\d)",              # North American
    r"(?<![a-z0-9])[a-z][1289]\d{8}(?![a-z0-9])",                  # TW national / resident ID
)]


def has_contact_details(text: str) -> bool:
    return any(p.search(text) for p in CONTACT_PATTERNS)


def _payload(entry: dict[str, Any]) -> dict[str, Any]:
    """The {"id", "payload": {...}} entry shape every worker publishes; a flat
    dict is its own payload (same fallback as Module 25's consumer)."""
    payload = entry.get("payload")
    return payload if isinstance(payload, dict) else entry


def chunk_text(entry: dict[str, Any]) -> str:
    return (_payload(entry).get("text") or "").strip()


async def _scan_one(session: aiohttp.ClientSession, url: str, text: str) -> bool:
    body = {"content": text, "direction": "input", "module": SCAN_MODULE, "request_id": str(uuid.uuid4())}
    async with session.post(url, json=body) as resp:
        resp.raise_for_status()
        result = await resp.json()
    return PII_CATEGORY in (result.get("categories") or [])


async def scan_chunks(chunks: list[dict[str, Any]], guardrail_url: str, timeout_ms: int) -> int:
    """Mark each chunk's payload with `pii_detected`, and stamp every chunk
    with the document's `pii_chunks_detected` total. Returns that total.

    A chunk without text (e.g. a whole-document summary entry) isn't
    scanned and gets no `pii_detected`. Never raises."""
    url = f"{guardrail_url.rstrip('/')}/guardrail/policy/decide"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCANS)
    failures: list[str] = []
    by_pattern_only: list[int] = []

    async def scan(session: aiohttp.ClientSession, entry: dict[str, Any]) -> None:
        text = chunk_text(entry)
        if not text:
            return
        async with semaphore:
            try:
                detected = await _scan_one(session, url, text)
            except (aiohttp.ClientError, json.JSONDecodeError) as exc:  # unreachable, timeout, bad response: fail open
                failures.append(f"{type(exc).__name__}: {exc}")
                return
        if not detected and has_contact_details(text):
            detected = True
            by_pattern_only.append(1)
        _payload(entry)["pii_detected"] = detected

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(*(scan(session, entry) for entry in chunks))
    except Exception as exc:  # noqa: BLE001 -- "never raises" contract; scan() already isolates per-chunk HTTP errors, this covers session setup/malformed-chunk failures too
        logger.warning("[pii-scan] scan aborted, indexing without PII flags: %s", exc)
        return 0

    if failures:
        logger.warning("[pii-scan] %d of %d chunk(s) not scanned (indexed without pii_flag); first error: %s",
                       len(failures), len(chunks), failures[0])

    detected_count = sum(1 for entry in chunks if _payload(entry).get("pii_detected"))
    for entry in chunks:
        _payload(entry)["pii_chunks_detected"] = detected_count
    logger.info("[pii-scan] %d of %d chunk(s) contain PII (%d by contact pattern only)",
                detected_count, len(chunks), len(by_pattern_only))
    return detected_count
