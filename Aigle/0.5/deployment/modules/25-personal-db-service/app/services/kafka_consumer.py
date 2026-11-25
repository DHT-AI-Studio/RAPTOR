"""
Kafka consumer for the personal-index-requests topic (PA-8).

Workers 09-12 publish the same payload they send to the global DBs onto this
topic after their global indexing step completes. The consumer here routes each
message to the correct per-user ArcadeDB database, keeping personal data in
sync with uploads automatically.

Deduplication happens at two levels (VIE01-190):

  * whole message — `event_id` is claimed in PostgreSQL `personal_index_events`
    before any work starts; a replayed event is acknowledged and dropped.
  * per chunk — a Redis key (personal:indexed:{branch_id}:{chunk_id}) with a
    7-day TTL catches a chunk that arrives twice inside *different* events. It is
    per tenant, and a hit is trusted only while the Chunk row still exists (the
    key outlives a delete-by-version); version_id and chunk_index are optional (whole-asset media
    like image/audio summaries have neither).

A message that fails PD_KAFKA_MAX_ATTEMPTS times is parked on the DLQ topic so
one poisoned payload cannot block the partition behind it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from collections import defaultdict
from typing import Any

import redis.asyncio as aioredis
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError

from app.core.config import settings
from app.models.graph_index import (
    EntityIndexRequest,
    RelationshipIndexRequest,
    TemporalFactIndexRequest,
)
from app.models.index import ChunkIndexRequest, SourceSummaryIndexRequest
from app.services import graph_extractor, graph_indexer, temporal_jobs
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.embedder import embed_texts
from app.services.index_events import claim_event, compute_event_id, mark_event_failed
from app.services.indexer import (
    DatabaseNotInitializedError,
    index_chunk,
    index_source_summary,
    version_exists,
)
from app.services.schema_init import initialize_schema
from app.temporal import anchor

logger = logging.getLogger("personal_db.consumer")

_DEDUP_PREFIX = "personal:indexed"

# ── video graph extraction: background tasks, not awaited inline ──────────────
# extract_and_index_video() makes one LLM call for the summary plus one per
# moment batch (90s timeout each) -- awaiting it inline would block this consumer's
# single sequential loop (personal-index-requests is one topic shared by every
# media type and every user) for as long as those calls take. Spawned as a
# fire-and-forget task instead: _handle_message() returns and the Kafka offset
# commits as soon as chunk indexing is done, so a slow or hung extraction only
# delays that one video's graph data, never anyone else's chunk indexing.
# Bounded by a semaphore so a burst of uploads can't fire unlimited concurrent
# LLM calls. Tasks are kept in a module-level set only so asyncio doesn't
# garbage-collect them mid-flight (a bare fire-and-forget create_task() holds
# no strong reference anywhere else) -- see asyncio docs on this exact pitfall.
_graph_extraction_semaphore = asyncio.Semaphore(settings.graph_extraction_max_concurrency)
_graph_extraction_tasks: set = set()


async def _run_graph_extraction(
    client: ArcadeDBClient, branch_id: str, ensured: set, version_id: str,
    summary_text: str, moment_texts: list[tuple],
) -> None:
    async with _graph_extraction_semaphore:
        try:
            await _ensure_db(client, branch_id, ensured)
            await graph_extractor.extract_and_index_video(
                client, branch_id, version_id, summary_text, moment_texts)
        except Exception as exc:  # noqa: BLE001 -- background best-effort task; ArcadeDB/LLM errors must not propagate, logged
            logger.error("[consumer] video graph extraction failed for branch %s: %s",
                        branch_id, exc)


def _spawn_graph_extraction(
    client: ArcadeDBClient, branch_id: str, ensured: set, version_id: str,
    summary_text: str, moment_texts: list[tuple],
) -> None:
    task = asyncio.create_task(_run_graph_extraction(
        client, branch_id, ensured, version_id, summary_text, moment_texts))
    _graph_extraction_tasks.add(task)
    task.add_done_callback(_graph_extraction_tasks.discard)


async def drain_graph_extraction_tasks() -> None:
    """Called from app/main.py's shutdown alongside the main consumer task.

    Cancels rather than waits out in-flight extractions -- each can take
    several 90s LLM calls, and blocking a redeploy on that would
    turn a routine restart into a multi-minute stall for a best-effort feature.
    A cancelled extraction is no worse than one lost to a crash, which this
    service already tolerates (see _run_graph_extraction's docstring above).
    """
    if not _graph_extraction_tasks:
        return
    for task in list(_graph_extraction_tasks):
        task.cancel()
    await asyncio.gather(*_graph_extraction_tasks, return_exceptions=True)


# Module 12's contextual chunking (document_analysis.py) wraps each chunk as
# "[FILENAME] name [/FILENAME]\n[CONTEXT] an LLM's sentence [/CONTEXT]\n[CHUNK] the text [/CHUNK]".
_DOC_CHUNK = re.compile(r"\[CHUNK\] (.*) \[/CHUNK\]\s*\Z", re.DOTALL)


def _evidence_text(ep: dict[str, Any], text: str) -> tuple[str, int]:
    """(the part of a Chunk's text that is evidence, where it starts). What a model or the pipeline wrote
    around the source's own words is not evidence, so temporal extraction never reads it:
    - a document chunk from Module 12 is wrapped in [FILENAME] / [CONTEXT] (an LLM's summary sentence) /
      [CHUNK] tags -- only the [CHUNK] text counts (a year in the file name or the context is not a date
      the document states);
    - a video moment's synthesized text opens with `contextual:{...} / `, Module 11's LLM-written context --
      only what follows (ASR / OCR / visual description) counts.
    Offsets into the returned text plus the start are offsets into Chunk.text."""
    if text.startswith("[FILENAME]") and (m := _DOC_CHUNK.search(text)):
        return m.group(1), m.start(1)
    prefix = ep.get("contextual_prefix") or ep.get("contextual_text")
    if ep.get("type") == "videos" and not ep.get("text") and prefix:
        head = f"contextual:{{{prefix}}} / "
        if text.startswith(head):
            return text[len(head):], len(head)
    return text, 0


def _temporal_segments(
    entries: list[dict[str, Any]], top_branch_id: str, top_version_id: str, indexed: set,
) -> dict[tuple[str, str], tuple[str | None, list[temporal_jobs.SourceSegment], temporal_jobs.Anchor]]:
    """The temporal job's segments per (branch, version): every non-summary entry that has a Chunk
    row, any medium, with the evidence part of the Chunk's text (_evidence_text) and where it starts,
    so evidence offsets map back into Chunk.text; in document order (chunk_index / moment_index,
    else arrival order) -- plus the source's capture / publish time (P13), validated, from the first entry that
    carries one."""
    found: dict[tuple[str, str], list[tuple[tuple, temporal_jobs.SourceSegment]]] = defaultdict(list)
    media: dict[tuple[str, str], str | None] = {}
    anchors: dict[tuple[str, str], temporal_jobs.Anchor] = {}
    for n, entry in enumerate(entries):
        ep = entry.get("payload", entry)
        if ep.get("embedding_type") == "summary":
            continue
        version_id = ep.get("version_id") or top_version_id
        branch_id = ep.get("branch_id") or top_branch_id
        chunk_id = entry.get("id") or _stable_chunk_id(ep, version_id)
        text, _, _ = _extract_content(ep)
        if chunk_id not in indexed or not branch_id or not version_id or not (text or "").strip():
            continue
        idx = ep.get("moment_index") if ep.get("moment_index") is not None else ep.get("chunk_index")
        start, end = _to_float(ep.get("start_time")), _to_float(ep.get("end_time"))
        pages = ep.get("page_numbers") or []
        evidence, offset = _evidence_text(ep, text)
        if not evidence.strip():
            continue
        seg = temporal_jobs.SourceSegment(
            chunk_id=chunk_id, text=evidence, text_offset=offset, seq=0, heading=ep.get("section_heading") or "",
            page=",".join(str(p) for p in pages) if isinstance(pages, list) else str(pages),
            media_start_ms=round(start * 1000) if start is not None else None,
            media_end_ms=round(end * 1000) if end is not None else None)
        order = (0, int(idx), n) if isinstance(idx, int) or (isinstance(idx, str) and idx.isdigit()) else (1, 0, n)
        found[(branch_id, version_id)].append((order, seg))
        media.setdefault((branch_id, version_id), ep.get("type"))
        if ep.get("capture_time") and (branch_id, version_id) not in anchors:
            when, why = anchor.validate(ep["capture_time"], ep.get("capture_time_source"),
                                        settings.temporal_default_timezone)
            anchors[(branch_id, version_id)] = temporal_jobs.Anchor(when, ep.get("capture_time_source"), why)
    out = {}
    for key, items in found.items():
        segments = [seg for _, seg in sorted(items, key=lambda it: it[0])]
        for i, seg in enumerate(segments):
            seg.seq = i
        out[key] = (media[key], segments, anchors.get(key, temporal_jobs.Anchor()))
    return out


def _photo_observations(
    entries: list[dict[str, Any]], top_branch_id: str, top_version_id: str,
) -> dict[tuple[str, str], tuple[str, str | None, str, temporal_jobs.Anchor]]:
    """Per (branch, version) of an image that carries an EXIF capture time (P13): (file name, caption -- the
    summary entry's description --, the raw value, the validated anchor). Read from the summary entry too:
    a photo with no OCR text has no chunk, but it was still taken at that time."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        ep = entry.get("payload", entry)
        if ep.get("type") != "images":
            continue
        key = (ep.get("branch_id") or top_branch_id, ep.get("version_id") or top_version_id)
        if not all(key):
            continue
        info = found.setdefault(key, {})
        info.setdefault("name", ep.get("filename"))
        if ep.get("embedding_type") == "summary" and ep.get("summary"):
            info["caption"] = ep["summary"]
        if ep.get("capture_time") and "raw" not in info:
            info["raw"], info["source"] = str(ep["capture_time"]), ep.get("capture_time_source")
    out = {}
    for key, info in found.items():
        if info.get("raw"):
            when, why = anchor.validate(info["raw"], info.get("source"), settings.temporal_default_timezone)
            out[key] = (info.get("name") or key[1], info.get("caption"), info["raw"],
                        temporal_jobs.Anchor(when, info.get("source"), why))
    return out


# ── auto-create user DB on first message (idempotent) ─────────────────────────

async def _ensure_db(client: ArcadeDBClient, branch_id: str, ensured: set) -> None:
    """Create + schema-init the user's database if it does not exist yet.

    This is the consumer-side equivalent of POST /internal/db/init — the pipeline
    never has to pre-create a user's DB. `ensured` caches branches already checked
    within one message so we hit the server at most once per branch per message.
    """
    if branch_id in ensured:
        return
    db = db_name_for(branch_id)
    if not await client.database_exists(db):
        await client.create_database(db)
        await initialize_schema(client, db)
        logger.info("[consumer] auto-created database for branch %s", branch_id)
    ensured.add(branch_id)


# ── Redis dedup helpers ───────────────────────────────────────────────────────

def _dedup_key(branch_id: str, chunk_id: str) -> str:
    """Per tenant: the same chunk_id in two users' uploads (content-derived ids, clones, replays) must not let
    the second user's chunk be skipped as "already indexed" when it only ever was in the first user's database."""
    return f"{_DEDUP_PREFIX}:{branch_id}:{chunk_id}"


async def _already_indexed(redis: aioredis.Redis, branch_id: str, chunk_id: str) -> bool:
    return bool(await redis.exists(_dedup_key(branch_id, chunk_id)))


async def _mark_indexed(redis: aioredis.Redis, branch_id: str, chunk_id: str) -> None:
    await redis.set(_dedup_key(branch_id, chunk_id), "1", ex=settings.redis_dedup_ttl)


async def _row_exists(client: ArcadeDBClient, branch_id: str, ensured: set, dedup_id: str,
                      is_summary: bool, version_id: str) -> bool:
    """Whether the row a dedup key vouches for is really there. The key outlives a delete-by-version (and a
    dropped database), so without this a re-upload within the TTL would be skipped and leave nothing indexed."""
    await _ensure_db(client, branch_id, ensured)
    db = db_name_for(branch_id)
    if is_summary:
        rows = await client.query(db, "SELECT count(*) AS c FROM Source WHERE version_id = :v AND summary IS NOT NULL",
                                  params={"v": version_id})
    else:
        rows = await client.query(db, "SELECT count(*) AS c FROM Chunk WHERE chunk_id = :c", params={"c": dedup_id})
    return bool(rows and int(rows[0].get("c") or 0))


def _to_float(val: Any) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _stable_chunk_id(ep: dict[str, Any], version_id: str) -> str:
    """Video moment entries carry no `id` at all (unlike every other entry
    shape), so `entry.get("id") or uuid.uuid4()` would mint a fresh random id
    on every reprocess — breaking both Redis dedup and ArcadeDB's UPSERT
    (both keyed on chunk_id), so a redelivered/retried message would create a
    duplicate Chunk instead of overwriting the same one. Derive a
    deterministic id from version_id + moment_index instead, same
    separator-safe join as compute_event_id. Falls back to a random id only
    when there's truly nothing stable to key on (best effort, not expected
    in practice — every moment carries a moment_index).
    """
    idx = ep.get("moment_index")
    if idx is None:
        idx = ep.get("chunk_index")
    if idx is None or not version_id:
        return str(uuid.uuid4())
    raw = "\x1f".join([version_id, "moment", str(idx)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tag_entry_type(entry: dict[str, Any], media_type: str) -> dict[str, Any]:
    """Stamp `type` onto an entry's payload if it doesn't already have one,
    handling both the wrapped {"id":..., "payload": {...}} and flat shapes."""
    ep = entry.get("payload", entry)
    if ep.get("type"):
        return entry
    tagged_ep = {**ep, "type": media_type}
    return {**entry, "payload": tagged_ep} if "payload" in entry else tagged_ep


def _synthesize_moment_text(ep: dict[str, Any]) -> str:
    """Video 'moment' entries (built by video_orchestrator_service's
    _assemble_moments, enriched by video_contextualize_service) never carry a
    combined `text` field — content lives in separate contextual_prefix/
    ocr_text/asr_text/lvlm_desc fields instead, unlike every other entry shape
    this consumer handles. Without this, every moment silently fails the "no
    embeddable content" check and is dropped.

    Format and ordering match module 11's own indexer exactly
    (video_indexer_service/kafka_handler.py, the one feeding Qdrant/
    OpenSearch: "contextual:{...} / ocr:{...} / asr:{...} / lvlm:{...}",
    braces literal, contextual prepended only when ocr/asr/lvlm content
    exists) — `text` is the only field vector search embeds from and BM25
    full-text-indexes (see searcher.py), so this is what actually makes
    contextual_text search-relevant here, not just a display field."""
    parts = []
    if ep.get("ocr_text"):
        parts.append(f"ocr:{{{ep['ocr_text']}}}")
    if ep.get("asr_text"):
        parts.append(f"asr:{{{ep['asr_text']}}}")
    if ep.get("lvlm_desc"):
        parts.append(f"lvlm:{{{ep['lvlm_desc']}}}")
    combined = " / ".join(parts)
    if not combined:
        return ""
    contextual_prefix = ep.get("contextual_prefix") or ep.get("contextual_text")
    if contextual_prefix:
        combined = f"contextual:{{{contextual_prefix}}} / {combined}"
    return combined


def _extract_content(ep: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """Return (text, summary, embedding_type) an entry actually carries.

    summary_val/emb_type are one rule for every media type: images/documents
    summary entries carry `summary` + `embedding_type="summary"` and no
    `text`; images/documents per-item entries carry `text` +
    `embedding_type="text"` and no `summary`; audio entries are the same,
    explicitly one or the other.

    Earlier versions of this function hardcoded a field/embedding_type per
    media_type (e.g. images always forced to embedding_type="summary",
    text=None) instead of reading what the entry actually carries. Since
    images and documents each send TWO entries per asset — one "summary"
    shaped, one "text" shaped — the hardcoding silently discarded every
    "text"-shaped entry's real content for images, and (once a separate
    upstream bug in module 12 was fixed) would have discarded documents'
    summary entries the same way. Reading the entry's own fields instead of
    assuming based on media_type fixes both.

    text_val keeps one media_type-specific branch, deliberately: video
    moments (type="videos", tagged by _tag_entry_type before this runs) carry
    a raw `asr_text` field alongside `ocr_text`/`lvlm_desc`/`contextual_prefix`
    that _synthesize_moment_text() combines into one string — but `asr_text`
    is also genuinely standalone content on audio entries. A single
    `ep.get("text") or ep.get("asr_text") or _synthesize_moment_text(ep)`
    fallback chain would let a video moment's own non-empty `asr_text` (true
    for nearly every real moment) short-circuit before
    _synthesize_moment_text() ever runs, silently dropping the ocr/lvlm/
    contextual content that fix was written to combine in the first place.
    """
    media_type = ep.get("type", "documents")
    if media_type == "videos":
        text_val = ep.get("text") or _synthesize_moment_text(ep) or None
    else:
        text_val = ep.get("text") or ep.get("asr_text") or None
    summary_val = ep.get("summary")
    emb_type = ep.get("embedding_type", "text")
    return text_val, summary_val, emb_type


def _map_entry(ep: dict[str, Any], chunk_id: str, embedding: list[float]) -> ChunkIndexRequest:
    """Build a ChunkIndexRequest from an unwrapped entry payload."""
    media_type: str = ep.get("type", "documents")
    text_val, summary_val, emb_type = _extract_content(ep)

    return ChunkIndexRequest(
        chunk_id=chunk_id,
        type=media_type,
        embedding_type=emb_type,
        embedding=embedding,
        text=text_val,
        summary=summary_val,
        filename=ep.get("filename"),
        source=ep.get("source"),
        asset_path=ep.get("asset_path"),
        version_id=ep.get("version_id") or None,
        upload_time=ep.get("upload_time"),
        # Always active on creation, matching index_source_summary() -- not
        # ep.get("status"), which can carry a stale pre-analysis value.
        status="active",
        start_sec=_to_float(ep.get("start_time")),
        end_sec=_to_float(ep.get("end_time")),
        speaker=ep.get("speaker"),
        # `or` would coerce moment_index=0 (every video's first moment) to
        # the chunk_index fallback (or None) since 0 is falsy in Python.
        # Module 25 unifies both upstream field names (video events carry
        # moment_index, document events carry chunk_index) into one property,
        # chunk_index -- there's no separate "Moment" node type here to
        # justify a video-specific name, unlike Module 20's Neo4j schema.
        chunk_index=(ep.get("moment_index") if ep.get("moment_index") is not None
                     else ep.get("chunk_index")),
        asr_text=ep.get("asr_text"),
        lvlm_desc=ep.get("lvlm_desc"),
        ocr_text=ep.get("ocr_text"),
        contextual_text=ep.get("contextual_prefix") or ep.get("contextual_text"),
        audio_labels=ep.get("audio_labels"),
        page_numbers=ep.get("page_numbers"),
        section_heading=ep.get("section_heading"),
        element_types=ep.get("element_types"),
        char_count=ep.get("char_count"),
        # GD-6 / VIE01-251: Module 12's ingestion scan marks chunks containing
        # PII with pii_detected=true. Only a positive result is stored --
        # unscanned (fail-open) and clean chunks alike carry no flag.
        pii_flag=True if ep.get("pii_detected") is True else None,
        # CM-3 / VIE01-406: Module 11 puts each video moment's keyframe moderation
        # verdict on the moment entry.
        moderation_flags=ep.get("moderation_flags") if isinstance(ep.get("moderation_flags"), dict) else None,
        capture_time=str(ep["capture_time"]) if ep.get("capture_time") else None,
        capture_time_source=ep.get("capture_time_source"),
    )


# ── graph-layer entry mappers (PA-5 indexers) ────────────────────────────────
# Worker payload field names for the graph layer are not yet frozen, so these
# mappers accept a couple of common aliases and skip an entry (logged) if a
# required field is missing rather than crash the whole message.

def _map_entity(ep: dict[str, Any]) -> EntityIndexRequest | None:
    eid = ep.get("entity_id") or ep.get("id")
    name = ep.get("name")
    if not eid or not name:
        return None
    return EntityIndexRequest(
        entity_id=eid, name=name,
        type=ep.get("type") or ep.get("entity_type") or "UNKNOWN",
        description=ep.get("description"),
        source_chunk_id=ep.get("source_chunk_id") or ep.get("chunk_id"),
        modality=ep.get("modality"),
    )


def _map_relationship(ep: dict[str, Any]) -> RelationshipIndexRequest | None:
    frm = ep.get("from_entity_id") or ep.get("source") or ep.get("from")
    to = ep.get("to_entity_id") or ep.get("target") or ep.get("to")
    rel = ep.get("relation") or ep.get("type")
    if not frm or not to or not rel:
        return None
    return RelationshipIndexRequest(
        from_entity_id=frm, to_entity_id=to, relation=rel,
        confidence=_to_float(ep.get("confidence")),
        source_version_id=ep.get("source_version_id") or ep.get("version_id"),
    )


def _map_temporal_fact(ep: dict[str, Any]) -> TemporalFactIndexRequest | None:
    fid = ep.get("fact_id") or ep.get("id")
    if not fid or not ep.get("relation") or ep.get("value") is None:
        return None
    return TemporalFactIndexRequest(
        fact_id=fid, entity=ep.get("entity"), relation=ep.get("relation"),
        value=ep.get("value"), entity_id=ep.get("entity_id"),
        time_start=ep.get("time_start"), time_end=ep.get("time_end"),
        confidence=_to_float(ep.get("confidence")),
        chunk_id=ep.get("chunk_id"),
        source_version_id=ep.get("source_version_id") or ep.get("version_id"),
    )


# ── message handler (pure function — testable without Kafka) ──────────────────

async def _handle_message(
    client: ArcadeDBClient,
    redis: aioredis.Redis,
    raw: dict[str, Any],
) -> dict[str, Any] | None:
    """Returns a completion payload once a real indexing pass actually ran (for
    `run_consumer()` to publish onto `kafka_topic_completed`), or None on an
    early-return path (gdpr-erased skip, or an already-claimed duplicate --
    the earlier successful pass already published completion for this event).
    """
    payload = raw.get("payload", {})
    top_branch_id: str = payload.get("branch_id") or payload.get("user_id", "")
    correlation_id: str = payload.get("correlation_id", "")
    params = payload.get("parameters", {})
    top_version_id: str = params.get("version_id", "")
    top_asset_path: str = params.get("asset_path", "")

    # GD-1 / VIE01-246: a user mid-erasure (or already erased -- the set
    # entry outlives the job by design, 30-day TTL) must not have new
    # content indexed for them. Checked before the event-dedup claim below
    # so an erased user's messages don't even burn a claim_event() write.
    if top_branch_id and await redis.sismember("gdpr:erased_users", top_branch_id):
        logger.info("[consumer] branch_id=%s is in gdpr:erased_users — skipping message", top_branch_id)
        return                              # ack and drop

    # Message-level dedup (VIE01-190). Publishers 09-12 stamp event_id; a message
    # without one is from an older publisher, so fall back to computing it from
    # the same three fields rather than skipping the check.
    event_id = raw.get("event_id") or compute_event_id(
        params.get("asset_path"), top_version_id, top_branch_id)
    if not await claim_event(
        event_id, top_branch_id,
        asset_path=params.get("asset_path"),
        version_id=top_version_id,
        source_module=raw.get("source_module"),
    ):
        return                              # already processed — ack and drop

    # All workers publish entries as {"id": <uuid>, "payload": {...}}.
    # Documents arrive under params["chunks"]; video moments under params["moments"].
    # Moments never carry a `type` field (video_orchestrator_service's
    # _assemble_moments doesn't set one), so _map_entry()'s `ep.get("type",
    # "documents")` default would silently mislabel every moment Chunk as
    # type="documents" — tag them "videos" here, where we still know which
    # list an entry came from (that distinction is lost once merged below).
    all_entries = params.get("chunks", []) + [
        _tag_entry_type(e, "videos") for e in params.get("moments", [])
    ]
    ensured: set = set()          # branches whose DB we've confirmed this message
    indexed = 0
    # chunk_ids confirmed to have a real Chunk row -- either indexed just now,
    # or already indexed by an earlier pass (Redis dedup hit). Graph extraction
    # below must only anchor MENTIONS/OBSERVED_IN edges to chunk_ids in this
    # set; an entry that was skipped here (no branch_id, no embeddable content,
    # embed failure, DB not initialised, index_chunk() error) has no Chunk row
    # for `CREATE EDGE ... FROM (SELECT FROM Chunk WHERE chunk_id=:c)` to find.
    indexed_chunk_ids: set = set()

    # version_ids whose whole-asset summary was written to Source this pass
    # (or already written by an earlier pass -- Redis dedup hit). Graph
    # extraction below reads this instead of a summary chunk_id: summary
    # entries no longer create a Chunk row at all, see index_source_summary().
    summary_indexed_versions: set = set()

    for entry in all_entries:
        # Unwrap nested payload; fall back to entry itself for flat dicts.
        ep: dict[str, Any] = entry.get("payload", entry)

        branch_id = ep.get("branch_id") or top_branch_id
        version_id = ep.get("version_id") or top_version_id
        # Video "moment" entries carry neither version_id nor asset_path per
        # item — _map_entry() reads both straight off `ep`, so without this,
        # req.version_id stays None and indexer.py's _link_source() (gated on
        # `if req.version_id`) never runs for moments, leaving them
        # un-linked to their Source even once they're indexed as Chunks.
        if not ep.get("version_id") or not ep.get("asset_path"):
            ep = {**ep, "version_id": ep.get("version_id") or version_id,
                  "asset_path": ep.get("asset_path") or top_asset_path}

        is_summary = ep.get("embedding_type") == "summary"
        # Summary entries have no per-item id worth keying Redis dedup on
        # (there's no Chunk row to point back at any more) -- one summary
        # per asset, so version_id alone is the natural dedup key.
        dedup_id = f"summary:{ep['version_id']}" if is_summary else (entry.get("id") or _stable_chunk_id(ep, version_id))

        if not branch_id:
            logger.warning("[consumer] entry %s missing branch_id — skipping", dedup_id)
            continue

        try:
            if await _already_indexed(redis, branch_id, dedup_id) and await _row_exists(
                    client, branch_id, ensured, dedup_id, is_summary, ep["version_id"]):
                logger.debug("[consumer] %s already indexed — skipping", dedup_id)
                if is_summary:
                    summary_indexed_versions.add(ep["version_id"])
                else:
                    indexed_chunk_ids.add(dedup_id)  # indexed earlier -- the Chunk row exists
                continue

            if is_summary:
                summary_text = (ep.get("summary") or "").strip()
                if not summary_text:
                    logger.debug("[consumer] summary entry %s has no summary text — skipping", dedup_id)
                    continue
                embedding = ep.get("embedding")
                if not embedding:
                    try:
                        embedding = (await embed_texts([summary_text]))[0]
                    except Exception as embed_exc:  # noqa: BLE001 -- embed_texts() can raise model/httpx errors; one entry's failure must not block the batch, logged
                        logger.warning("[consumer] failed to embed summary %s: %s — skipping",
                                       dedup_id, embed_exc)
                        continue
                await _ensure_db(client, branch_id, ensured)
                await index_source_summary(client, branch_id, SourceSummaryIndexRequest(
                    version_id=ep["version_id"], summary=summary_text, embedding=embedding,
                    filename=ep.get("filename"), asset_path=ep.get("asset_path"),
                    media_type=ep.get("type"),
                    capture_time=str(ep["capture_time"]) if ep.get("capture_time") else None,
                    capture_time_source=ep.get("capture_time_source"),
                    # CM-2: Module 10 puts the image's moderation verdict on its summary entry.
                    moderation_flags=ep.get("moderation_flags") if isinstance(ep.get("moderation_flags"), dict) else None,
                ))
                await _mark_indexed(redis, branch_id, dedup_id)
                indexed += 1
                summary_indexed_versions.add(ep["version_id"])
                continue

            chunk_id = dedup_id
            embedding = ep.get("embedding")
            if not embedding:
                # Embed whichever of text/summary this entry actually carries
                # — see _extract_content()'s docstring for why this is a
                # single rule, not a per-media_type branch.
                text_val, summary_val, _ = _extract_content(ep)
                raw_text = (text_val or summary_val or "").strip()

                if not raw_text:
                    logger.debug("[consumer] entry %s has no embeddable content — skipping", chunk_id)
                    continue
                try:
                    vectors = await embed_texts([raw_text])
                    embedding = vectors[0]
                except Exception as embed_exc:  # noqa: BLE001 -- embed_texts() can raise model/httpx errors; one entry's failure must not block the batch, logged
                    logger.warning("[consumer] failed to embed %s: %s — skipping", chunk_id, embed_exc)
                    continue

            await _ensure_db(client, branch_id, ensured)   # auto-create on first upload
            req = _map_entry(ep, chunk_id, embedding)
            await index_chunk(client, branch_id, req)
            await _mark_indexed(redis, branch_id, chunk_id)
            indexed += 1
            indexed_chunk_ids.add(chunk_id)

        except DatabaseNotInitializedError:
            logger.warning("[consumer] DB not initialised for branch %s — skipping entry", branch_id)
            continue
        except Exception as exc:  # noqa: BLE001 -- ArcadeDB write can raise httpx/validation errors; one entry's failure must not block the batch, logged
            logger.error("[consumer] failed to index entry %s for branch %s: %s", dedup_id, branch_id, exc)

    # ── temporal job (P12): every medium, straight from its chunks ───────────
    # Recorded durably before the offset commits; the extraction itself runs in
    # temporal_jobs' background worker. Independent of the embedding dedup above:
    # a redelivered message whose chunks were all indexed already still finds --
    # and finishes -- its job.
    for (branch_id, version_id), (media_type, segments, source_anchor) in _temporal_segments(
            all_entries, top_branch_id, top_version_id, indexed_chunk_ids).items():
        try:
            key = await temporal_jobs.enqueue(client, branch_id, version_id, media_type, segments, source_anchor)
            if key:
                temporal_jobs.submit(branch_id, key)
        except Exception as exc:  # noqa: BLE001 -- ArcadeDB write can raise httpx errors; chunk indexing above already succeeded, logged
            logger.error("[consumer] could not record the temporal job for branch %s version %s: %s",
                         branch_id, version_id, exc)

    # ── a photo's own capture time is an observation event (P13) ─────────────
    for (branch_id, version_id), (name, caption, exif_raw, photo_anchor) in _photo_observations(
            all_entries, top_branch_id, top_version_id).items():
        try:
            if photo_anchor.time and await version_exists(client, branch_id, version_id, status="active"):
                await temporal_jobs.write_observation(client, branch_id, version_id, name, caption, exif_raw, photo_anchor)
            elif photo_anchor.reason:
                logger.info("[consumer] photo %s: capture time %r not used (%s)", version_id, exif_raw, photo_anchor.reason)
        except Exception as exc:  # noqa: BLE001 -- ArcadeDB write can raise httpx errors; indexing above already succeeded, logged
            logger.error("[consumer] could not record the capture of photo %s: %s", version_id, exc)

    # ── video graph extraction (PA-7): entities/relationships
    # derived from this video's own summary + moments, video only for now.
    # `params["moments"]` is only ever populated by video -- other media types
    # publish everything under `params["chunks"]` -- so its presence is the
    # signal this message is a video, without needing a separate media_type
    # field on the envelope itself. The actual extraction runs in a spawned
    # background task (_spawn_graph_extraction) and has its own try/except
    # (_run_graph_extraction) -- this one only guards scanning all_entries for
    # the summary/moment texts to hand off.
    if settings.graph_extraction_enabled and params.get("moments"):
        try:
            summary_ep: dict[str, Any] | None = None
            moment_texts: list[tuple] = []
            for entry in all_entries:
                ep = entry.get("payload", entry)
                if ep.get("embedding_type") == "summary":
                    # No Chunk row for this any more (index_source_summary()
                    # writes Source directly) -- check summary_indexed_versions,
                    # not indexed_chunk_ids.
                    if (ep.get("version_id") or top_version_id) in summary_indexed_versions:
                        summary_ep = ep
                    continue
                cid = entry.get("id") or _stable_chunk_id(ep, ep.get("version_id") or top_version_id)
                if cid not in indexed_chunk_ids:
                    # No Chunk row for this entry (skipped above -- no content,
                    # embed failure, etc.) -- anchoring a MENTIONS/OBSERVED_IN
                    # edge to it would point at a Chunk that was never created.
                    continue
                text_val, _, _ = _extract_content(ep)
                if text_val:
                    moment_texts.append((cid, text_val))
            if summary_ep and summary_ep.get("summary"):
                _spawn_graph_extraction(
                    client, top_branch_id, ensured, top_version_id,
                    summary_ep["summary"], moment_texts)
        except Exception as exc:  # noqa: BLE001 -- only guards scanning all_entries for the extraction handoff; must not block chunk indexing above, logged
            logger.error("[consumer] failed to scan video entries for graph extraction, "
                        "branch %s: %s", top_branch_id, exc)

    # ── graph layer: entities / relationships / temporal facts ────────────────
    graphed = 0
    for key, mapper, index_fn in (
        ("entities", _map_entity, graph_indexer.index_entity),
        ("relationships", _map_relationship, graph_indexer.index_relationship),
        ("temporal_facts", _map_temporal_fact, graph_indexer.index_temporal_fact),
    ):
        for entry in params.get(key, []):
            ep = entry.get("payload", entry)
            branch_id = ep.get("branch_id") or top_branch_id
            if not branch_id:
                continue
            try:
                req = mapper(ep)
                if req is None:
                    logger.warning("[consumer] %s entry missing required fields — skipping", key)
                    continue
                await _ensure_db(client, branch_id, ensured)
                await index_fn(client, branch_id, req)
                graphed += 1
            except Exception as exc:  # noqa: BLE001 -- ArcadeDB write can raise httpx/validation errors; one entry's failure must not block the batch, logged
                logger.error("[consumer] failed to index %s for branch %s: %s", key, branch_id, exc)

    logger.info("[consumer] indexed branch=%s version=%s chunks=%d graph=%d",
                top_branch_id, top_version_id, indexed, graphed)

    # Shaped to match what 09-12's orchestrators already expect from every
    # other topic on their result_consumer (handle_service_result() does an
    # unconditional message["payload"] / message.get("correlation_id")).
    return {
        "correlation_id": correlation_id,
        "payload": {
            "branch_id": top_branch_id,
            "asset_path": top_asset_path,
            "version_id": top_version_id,
        },
    }


# ── consumer loop ─────────────────────────────────────────────────────────────

async def _send_to_dlq(
    producer: AIOKafkaProducer,
    raw: dict[str, Any],
    attempts: int,
    error: str,
) -> None:
    """Park a message that failed `attempts` times, with why it failed.

    The original envelope is nested rather than merged so a DLQ message can be
    replayed onto the main topic by lifting `original` out verbatim — no need to
    strip the diagnostic fields back off.
    """
    await producer.send_and_wait(settings.kafka_dlq_topic, {
        "original": raw,
        "attempts": attempts,
        "error": error,
        "consumer_group": settings.kafka_group_id,
    })
    # Flip the claim to 'failed' only after the DLQ write lands. Doing it first
    # would open a window where the event is re-claimable but not yet parked.
    event_id = raw.get("event_id")
    if event_id:
        await mark_event_failed(event_id, error)
    logger.error("[consumer] message sent to DLQ after %d attempts: %s", attempts, error)


async def run_consumer(client: ArcadeDBClient) -> None:
    """Subscribe to personal-index-requests and process messages indefinitely.

    Designed to run as a background asyncio.Task. Offset is committed only
    after _handle_message completes so no message is lost on crash/restart.
    Responds to asyncio.CancelledError for clean shutdown.

    A message that fails `PD_KAFKA_MAX_ATTEMPTS` times goes to the DLQ topic and
    the offset is committed (VIE01-190). Retrying forever would block every later
    message behind one poisoned payload, so the failure is parked where Module
    14's alert rule can see it and the partition keeps moving.
    """
    consumer = AIOKafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id=settings.kafka_group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode()),
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    await consumer.start()
    await producer.start()
    logger.info(
        "[consumer] started — topic=%s group=%s dlq=%s",
        settings.kafka_topic, settings.kafka_group_id, settings.kafka_dlq_topic,
    )
    try:
        async for msg in consumer:
            settled = False          # handled, or safely parked in the DLQ
            completion: dict[str, Any] | None = None
            for attempt in range(1, settings.kafka_max_attempts + 1):
                try:
                    completion = await _handle_message(client, redis, msg.value)
                    settled = True
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- _handle_message() spans ArcadeDB/embedding/graph calls; retry loop logs each attempt, eventually DLQs
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.error("[consumer] attempt %d/%d failed: %s",
                                 attempt, settings.kafka_max_attempts, last_error)
                    if attempt == settings.kafka_max_attempts:
                        try:
                            await _send_to_dlq(producer, msg.value, attempt, last_error)
                            settled = True
                        except KafkaError as dlq_exc:
                            logger.critical("[consumer] DLQ publish failed: %s", dlq_exc)

            if completion:
                try:
                    await producer.send_and_wait(settings.kafka_topic_completed, completion)
                except KafkaError as exc:
                    logger.error("[consumer] failed to publish completion for %s: %s",
                                 completion.get("correlation_id"), exc)

            if settled:
                await consumer.commit()
            else:
                # Neither indexed nor parked. Committing would drop the message
                # silently, so leave the offset where it is and let a restart
                # re-read it — a stalled partition is visible, a lost message is not.
                logger.critical("[consumer] offset NOT committed — message left on the topic")
    finally:
        await consumer.stop()
        await producer.stop()
        await redis.aclose()
        logger.info("[consumer] stopped")
