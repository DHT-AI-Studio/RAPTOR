"""Durable per-source temporal jobs (temporal plan §5.7 / P12, D7 + D15).

Every indexed source -- document, video, audio or image -- gets one job, enqueued by the Kafka
consumer as soon as its chunks are stored. Nothing depends on video moments or on a summary any
more: the job reads the chunks themselves.

    enqueue()  -> TemporalJob (pending) + one TemporalJobSegment per chunk, in document order
    run_job()  -> per segment: language check, one Module 07 call (throttled, retried), the reply
                  stored on the segment row; then app.temporal.extract over all segments with the
                  stored replies, Entities for the named subjects, facts written with their
                  evidence and linked to them, the undated candidates kept as
                  TemporalCandidate rows (evidence + links, for reasoning), stale facts of the source
                  removed, and a final status: complete | partial | failed | cancelled
    status()   -> what the status operation (§5.9) returns

**Durable.** The job key is (tenant, source version, extractor version, normaliser version), so it
is independent of the embedding dedup: a redelivered message whose chunks are already indexed
still finds -- and finishes -- its job. Each segment's model reply is stored as it arrives; a
restart (startup recovery re-queues pending/running jobs) only calls the model for segments that
are not done, then rebuilds the facts deterministically from the stored replies. Fact writes are
idempotent upserts keyed by the source-specific fact_id of §5.6.

**Generation checks.** Before the model calls and again before any fact is written, the source must
still be live (an active Chunk or Source for the version). Archive or delete in between cancels the
job; a late run never writes facts for a source that is gone.

**Language (F5).** A segment whose letters are mostly neither Latin nor Han is `unsupported`
(reason `unsupported_language`) and never sent to the extractor; a job with only such segments
is `failed`, one with some is `partial`. Checked per segment -- finer per-span coverage is P21.

Not here yet: ingestion-time reasoning (P15). `complete` currently means extraction is done.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings
from app.models.graph_index import EntityIndexRequest, TemporalFactIndexRequest
from app.services import graph_indexer
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.indexer import _ensure_ready, version_exists
from app.services.schema_init import initialize_job_schema
from app.services.temporal_normalize import NORMALIZER_VERSION, normalize_fact_dates
from app.temporal import reasoning
from app.temporal.anchor import anchor_day
from app.temporal.extract import (
    CANDIDATE_SCHEMA,
    EXTRACTOR_VERSION,
    Candidate,
    Segment,
    extract,
    segment_messages,
)

logger = logging.getLogger("personal_db.temporal_jobs")

TERMINAL = ("complete", "partial", "failed", "cancelled")


def _now_ms() -> int:
    return int(time.time() * 1000)


def job_key(branch_id: str, version_id: str) -> str:
    raw = f"{branch_id}\x1f{version_id}\x1f{EXTRACTOR_VERSION}\x1f{NORMALIZER_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- segments

@dataclass
class SourceSegment:
    """One chunk as the job sees it. `text` is the evidence part of the Chunk's text, starting at
    `text_offset` in it (0 except for a video moment's LLM-written context prefix)."""
    chunk_id: str
    text: str
    seq: int
    text_offset: int = 0
    heading: str = ""
    page: str = ""
    media_start_ms: int | None = None
    media_end_ms: int | None = None


@dataclass
class Anchor:
    """The source's capture / publish time (app.temporal.anchor): `time` when it validated, else `reason`."""
    time: str | None = None
    source: str | None = None
    reason: str | None = None


_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def language_supported(text: str) -> bool:
    """English / Chinese (§5.2). Letters that are neither Latin nor Han -- kana, Hangul, Cyrillic,
    Arabic, ... -- making up most of the segment mean the extractor cannot read it. A segment with no
    letters at all (a bare date, a number) is left to the extractor: dates are language-neutral."""
    latin = han = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if _HAN.match(ch):
            han += 1
        elif "LATIN" in unicodedata.name(ch, ""):
            latin += 1
        else:
            other += 1
    letters = latin + han + other
    return letters == 0 or other * 2 <= letters


# ---------------------------------------------------------------- Module 07

class ModelError(Exception):
    """A segment's model call failed for good (after its retries)."""


_RETRY_STATUS = {502, 503, 504}
_llm_slots: asyncio.Semaphore | None = None


def _slots() -> asyncio.Semaphore:
    global _llm_slots
    if _llm_slots is None:
        _llm_slots = asyncio.Semaphore(max(1, settings.temporal_llm_concurrency))
    return _llm_slots


def parse_reply(text: str) -> dict[str, Any]:
    """The model's JSON object. Tolerant (§5.10's bounded second line of defence): drops a <think> block and
    code fences, takes the outermost {...}, and allows raw control characters inside strings (strict=False).
    Anything else is a ValueError -- never a guess."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    t = re.sub(r"```(?:json)?", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(t[start:end + 1], strict=False)
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def _error_code(resp: httpx.Response) -> str | None:
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail.get("code") if isinstance(detail, dict) else None


async def call_model(messages: list[dict[str, str]]) -> dict[str, Any]:
    """One extraction call through Module 07 (/inference/infer, think off). Serialised by
    temporal_llm_concurrency; connection errors, timeouts, 502/503/504 and unreadable JSON are retried
    with backoff. Any other HTTP error (a guardrail block, a bad request) fails at once.

    P10: the reply is asked for as CANDIDATE_SCHEMA (`options.format`); Module 07 constrains the model with
    it and validates the output, returning it as `result.parsed`. An output that does not match
    (`structured_output_error`) is not retried -- the same prompt at temperature 0 gives the same reply.
    A model that cannot take a schema (`unsupported_capability`), or a Module 07 that predates P10 (no
    `parsed`), is read the old way, through parse_reply."""
    delay = settings.temporal_llm_retry_delay
    last: Exception | None = None
    options: dict[str, Any] = {"temperature": 0.0, "think": settings.inference_think, "format": CANDIDATE_SCHEMA}
    for attempt in range(settings.temporal_llm_retries + 1):
        if attempt:
            await asyncio.sleep(delay)
            delay *= 2
        try:
            async with _slots(), httpx.AsyncClient(timeout=settings.temporal_llm_timeout) as client:
                resp = await client.post(f"{settings.inference_url}/inference/infer", json={
                    "task": "text-generation", "engine": "ollama", "model_name": settings.chat_model_name,
                    "data": {"messages": messages}, "options": options,
                })
            code = _error_code(resp) if resp.status_code >= 400 else None
            if code == "unsupported_capability" and "format" in options:
                options = {k: v for k, v in options.items() if k != "format"}
                last = ModelError("Module 07 cannot constrain this model's output")
                continue
            if resp.status_code in _RETRY_STATUS and code != "structured_output_error":
                last = ModelError(f"Module 07 returned {resp.status_code}")
                continue
            if resp.status_code >= 400:
                raise ModelError(f"Module 07 returned {resp.status_code}: {resp.text[:200]}")
            result = resp.json()["result"]
            if isinstance(result.get("parsed"), dict):
                return result["parsed"]
            return parse_reply(result["response"])
        except (httpx.TransportError, ValueError, KeyError, TypeError) as exc:
            last = exc
    raise ModelError(f"{type(last).__name__}: {last}")


# ---------------------------------------------------------------- storage

_schema_ready: set[str] = set()       # databases this process has already given the job types


async def _ensure_job_schema(client: ArcadeDBClient, db: str) -> None:
    if db not in _schema_ready:
        await initialize_job_schema(client, db)
        _schema_ready.add(db)


async def _job(client: ArcadeDBClient, db: str, key: str) -> dict[str, Any] | None:
    rows = await client.query(db, "SELECT FROM TemporalJob WHERE job_key = :k", params={"k": key})
    return rows[0] if rows else None


async def _set_job(client: ArcadeDBClient, db: str, key: str, **fields: Any) -> None:
    fields["updated_at"] = _now_ms()
    sets = ", ".join(f"{f} = :{f}" for f in fields)
    await client.command(db, f"UPDATE TemporalJob SET {sets} WHERE job_key = :job_key",  # nosec B608 -- field names are this module's own keyword arguments
                         params={**fields, "job_key": key})


async def _segments(client: ArcadeDBClient, db: str, key: str) -> list[dict[str, Any]]:
    return await client.query(db, "SELECT FROM TemporalJobSegment WHERE job_key = :k ORDER BY seq",
                              params={"k": key})


async def _set_segment(client: ArcadeDBClient, db: str, segment_key: str, **fields: Any) -> None:
    sets = ", ".join(f"{f} = :{f}" for f in fields)
    await client.command(db, f"UPDATE TemporalJobSegment SET {sets} WHERE segment_key = :segment_key",  # nosec B608 -- field names are this module's own keyword arguments
                         params={**fields, "segment_key": segment_key})


async def _counts(client: ArcadeDBClient, db: str, key: str) -> dict[str, int]:
    rows = await client.query(db, "SELECT status, count(*) AS n FROM TemporalJobSegment WHERE job_key = :k "
                                  "GROUP BY status", params={"k": key})
    return {r["status"]: int(r["n"]) for r in rows}


async def enqueue(client: ArcadeDBClient, branch_id: str, version_id: str, media_type: str | None,
                  segments: list[SourceSegment], source_anchor: Anchor | None = None) -> str | None:
    """Record the job for one source version (idempotent) and return its key, or None when there is
    nothing to record. A job that already exists keeps its done segments; segments it has not seen
    are added, a finished job with failed segments left is reopened so they are retried, and a cancelled job
    is reopened (its source is back). *source_anchor*: the capture / publish time the source's deictic
    phrases are dated from; recorded on a new job, or on an existing one that has none yet."""
    if not settings.temporal_jobs_enabled or not segments:
        return None
    db = db_name_for(branch_id)
    await _ensure_job_schema(client, db)
    key = job_key(branch_id, version_id)
    existing = await _job(client, db, key)
    if existing is None:
        now = _now_ms()
        await client.command(db, "INSERT INTO TemporalJob SET job_key = :k, branch_id = :b, source_version_id = :v, "
                                 "media_type = :m, extractor_version = :ev, normalizer_version = :nv, status = 'pending', "
                                 "segments_total = 0, segments_done = 0, segments_failed = 0, segments_unsupported = 0, "
                                 "facts = 0, unresolved = 0, generation = :g, created_at = :t, updated_at = :t",
                             params={"k": key, "b": branch_id, "v": version_id, "m": media_type,
                                     "ev": EXTRACTOR_VERSION, "nv": NORMALIZER_VERSION, "t": now,
                                     "g": await _generation(client, db, version_id)})
    if source_anchor and (source_anchor.time or source_anchor.reason) and not (existing or {}).get("anchor_time"):
        await client.command(db, "UPDATE TemporalJob SET anchor_time = :at, anchor_source = :src, anchor_reason = :why "
                                 "WHERE job_key = :k",
                             params={"at": source_anchor.time, "src": source_anchor.source, "why": source_anchor.reason,
                                     "k": key})
    known = {s["chunk_id"] for s in await _segments(client, db, key)}
    for seg in segments:
        if seg.chunk_id in known:
            continue
        await client.command(db, "INSERT INTO TemporalJobSegment SET segment_key = :sk, job_key = :k, chunk_id = :c, "
                                 "seq = :seq, text = :t, text_offset = :o, heading = :h, page = :p, media_start_ms = :ms, "
                                 "media_end_ms = :me, status = 'pending', attempts = 0",
                             params={"sk": f"{key}:{seg.chunk_id}", "k": key, "c": seg.chunk_id, "seq": seg.seq,
                                     "t": seg.text, "o": seg.text_offset, "h": seg.heading, "p": seg.page,
                                     "ms": seg.media_start_ms, "me": seg.media_end_ms})
    total = len(known | {s.chunk_id for s in segments})
    if existing is None or existing.get("status") not in TERMINAL:
        await _set_job(client, db, key, segments_total=total)
    else:
        retry = await client.query(db, "SELECT count(*) AS n FROM TemporalJobSegment WHERE job_key = :k AND "
                                       "status = 'failed' AND attempts < :max",
                                   params={"k": key, "max": settings.temporal_segment_max_attempts})
        # reopened for new segments, for failed segments with attempts left, and for a job cancelled because its
        # source was archived or gone -- the source is being indexed again, so it is live again
        if total > len(known) or (retry and int(retry[0]["n"])) or existing.get("status") == "cancelled":
            await _set_job(client, db, key, segments_total=total, status="pending", reason=None,
                           generation=await _generation(client, db, version_id))
    return key


# ---------------------------------------------------------------- facts

def fact_id(version_id: str, c: Candidate, offsets: dict[str, Any], entity_id: str,
            dates: dict[str, Any]) -> str:
    """Source-specific assertion identity (§5.6): evidence, assertion and its normalised interval."""
    parts = [version_id, c.chunk_id, offsets, entity_id, c.relation, c.value, dates["axis"], dates["timeline_id"], c.extent,
             dates["s_state"], dates["s_lo"], dates["s_hi"], dates["e_state"], dates["e_lo"], dates["e_hi"],
             dates["precision"], dates["normalizer_version"]]
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"tf-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


def entity_id_for(name: str) -> str:
    """Same slug graph_extractor gives an Entity of that name, so a document's and a video's mentions of one
    name are one Entity."""
    slug = re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", name.lower().strip()))
    return slug[:64] or hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


_NOT_A_NAME = {"the", "a", "an", "this", "that", "these", "those", "it", "its", "he", "she", "they", "his", "her",
               "their", "we", "our", "i", "you", "one", "some", "each", "both", "such"}


def is_named(c: Candidate) -> bool:
    """Whether a fact's subject is a name worth an Entity: stated in the text (not a resolved coreference),
    capitalised or Chinese, and not a description ("the company", "the acquisition", "users")."""
    name = (c.entity or "").strip()
    if not name or c.coreference or name.split()[0].casefold() in _NOT_A_NAME:
        return False
    return name[0].isupper() or bool(_HAN.match(name[0]))


def entity_requests(accepted: list[Candidate]) -> list[EntityIndexRequest]:
    """One Entity per named subject, with a MENTIONS edge from each chunk that states a fact about it -- what
    the TKG search's entity retrieval needs ("What was Hugo Barra doing in 2010?")."""
    seen: dict[tuple[str, str], EntityIndexRequest] = {}
    for c in accepted:
        if is_named(c):
            eid = entity_id_for(c.entity)
            seen.setdefault((eid, c.chunk_id), EntityIndexRequest(
                entity_id=eid, name=c.entity, type=c.entity_type or "ENTITY", source_chunk_id=c.chunk_id,
                modality="text"))
    return list(seen.values())


def fact_requests(version_id: str, accepted: list[Candidate], media: dict[str, tuple[int | None, int | None]],
                  text_offsets: dict[str, int] | None = None) -> list[TemporalFactIndexRequest]:
    """*text_offsets*: where each chunk's segment text starts in Chunk.text, so the stored evidence offsets
    point into Chunk.text."""
    out = []
    for c in accepted:
        shift = (text_offsets or {}).get(c.chunk_id, 0)
        offsets = {k: [v[0] + shift, v[1] + shift] for k, v in (("date", c.date_offsets), ("entity", c.entity_offsets),
                                                                ("value", c.value_offsets)) if v}
        dates = normalize_fact_dates(c.time_start, c.time_end, settings.temporal_default_timezone)
        eid = entity_id_for(c.entity)
        start_ms, end_ms = media.get(c.chunk_id, (None, None))
        out.append(TemporalFactIndexRequest(
            fact_id=fact_id(version_id, c, offsets, eid, dates), entity=c.entity, relation=c.relation,
            value=c.value, entity_id=eid if is_named(c) else None, time_start=c.time_start, time_end=c.time_end, confidence=c.confidence,
            chunk_id=c.chunk_id, source_version_id=version_id, date_text=c.date_text, source_sentence=c.sentence,
            evidence_offsets=json.dumps(offsets, separators=(",", ":")), extent=c.extent, anchor=c.kind,
            extractor_version=EXTRACTOR_VERSION, media_start_ms=start_ms, media_end_ms=end_ms,
            links=json.dumps(c.links, ensure_ascii=False) if c.links else None))
    return out


def candidate_rows(version_id: str, unresolved: list[Candidate],
                   text_offsets: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """The unresolved candidates as TemporalCandidate rows, evidence offsets mapped into Chunk.text."""
    out = []
    for c in unresolved:
        shift = (text_offsets or {}).get(c.chunk_id, 0)
        offsets = {k: [v[0] + shift, v[1] + shift] for k, v in (("date", c.date_offsets), ("entity", c.entity_offsets),
                                                                ("value", c.value_offsets)) if v}
        ident = json.dumps([version_id, c.chunk_id, offsets, c.entity, c.relation, c.value, c.date_text, c.reason],
                           ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        out.append({
            "candidate_id": f"tc-{hashlib.sha256(ident.encode('utf-8')).hexdigest()[:32]}",
            "source_version_id": version_id, "chunk_id": c.chunk_id, "entity": c.entity, "relation": c.relation,
            "value": c.value, "date_text": c.date_text, "reason": c.reason,
            "links": json.dumps(c.links, ensure_ascii=False), "source_sentence": c.sentence,
            "evidence_offsets": json.dumps(offsets, separators=(",", ":")), "extent": c.extent, "origin": c.origin,
            "confidence": c.confidence, "extractor_version": EXTRACTOR_VERSION, "status": "active"})
    return out


async def _write_candidates(client: ArcadeDBClient, db: str, version_id: str, rows: list[dict[str, Any]]) -> int:
    """Replace the source's unresolved candidates with this job's (deduplicated by candidate_id)."""
    await client.command(db, "DELETE FROM TemporalCandidate WHERE source_version_id = :v", params={"v": version_id})
    seen: set[str] = set()
    for row in rows:
        if row["candidate_id"] in seen:
            continue
        seen.add(row["candidate_id"])
        sets = ", ".join(f"{k} = :{k}" for k in row)
        await client.command(db, f"INSERT INTO TemporalCandidate SET {sets}", params=row)  # nosec B608 -- keys are candidate_rows()' fixed field names
    return len(seen)


async def _write_facts(client: ArcadeDBClient, branch_id: str, db: str, version_id: str,
                       entities: list[EntityIndexRequest], requests: list[TemporalFactIndexRequest]) -> int:
    """Upsert the named subjects' Entities, then this job's facts (linked to them by HAS_TEMPORAL_FACT), then
    remove every other fact of the source -- the job owns the source's temporal facts, so an older extractor's
    (or the retired video path's) rows do not linger beside them."""
    for ent in entities:
        known = await client.query(db, "SELECT name, type FROM Entity WHERE entity_id = :e", params={"e": ent.entity_id})
        if known:            # an Entity another source made (a video's LLM extraction, say) keeps its own name and type
            ent = ent.model_copy(update={"name": known[0].get("name") or ent.name, "type": known[0].get("type") or ent.type})
        await graph_indexer.index_entity(client, branch_id, ent)
    written = 0
    for req in requests:
        await graph_indexer.index_temporal_fact(client, branch_id, req)
        written += 1
    keep = [r.fact_id for r in requests]
    await client.command(db, "DELETE VERTEX FROM TemporalFact WHERE source_version_id = :v AND fact_id NOT IN :keep "
                             "AND (anchor IS NULL OR anchor <> 'capture')",     # a photo's observation event is not the job's
                         params={"v": version_id, "keep": keep or ["-"]})
    return written


# ---------------------------------------------------------------- running

async def _generation(client: ArcadeDBClient, db: str, version_id: str) -> int:
    """The source's generation (P14): advanced by every archive / reactivation, so work started under one
    generation can tell it is stale. 0 for a source that was never archived (or has no Source row yet)."""
    rows = await client.query(db, "SELECT generation FROM Source WHERE version_id = :v", params={"v": version_id})
    return int((rows[0].get("generation") if rows else None) or 0)


async def _live(client: ArcadeDBClient, branch_id: str, version_id: str, generation: int | None = None) -> bool:
    """The source is active and, when the job recorded one, still at the generation the job was started for
    -- an archive and reactivation in between makes the job stale even though the source is active again."""
    if not await version_exists(client, branch_id, version_id, status="active"):
        return False
    return generation is None or await _generation(client, db_name_for(branch_id), version_id) == int(generation)


async def run_job(client: ArcadeDBClient, branch_id: str, key: str) -> str | None:
    """Run (or resume) one job to a final status, which it returns. Never raises for a segment's failure."""
    db = db_name_for(branch_id)
    await _ensure_job_schema(client, db)
    job = await _job(client, db, key)
    if job is None or job.get("status") in TERMINAL:
        return job.get("status") if job else None
    version_id = job["source_version_id"]
    generation = job.get("generation")
    if not await _live(client, branch_id, version_id, generation):
        await _set_job(client, db, key, status="cancelled", reason="source_not_active")
        return "cancelled"
    await _set_job(client, db, key, status="running",
                   model=settings.chat_model_name if settings.temporal_llm_enabled else None)

    rows = await _segments(client, db, key)
    segments = [Segment(r["chunk_id"], r.get("text") or "", r.get("heading") or "", r.get("page") or "") for r in rows]
    messages = segment_messages(segments)
    for row, msgs in zip(rows, messages, strict=True):
        if row["status"] in ("done", "unsupported"):
            continue
        if row["status"] == "failed" and (row.get("attempts") or 0) >= settings.temporal_segment_max_attempts:
            continue
        if not language_supported(row.get("text") or ""):
            await _set_segment(client, db, row["segment_key"], status="unsupported", reason="unsupported_language")
            continue
        if not settings.temporal_llm_enabled:
            await _set_segment(client, db, row["segment_key"], status="done", reply=None)
            continue
        attempts = (row.get("attempts") or 0) + 1
        try:
            reply = await call_model(msgs)
        except ModelError as exc:
            logger.warning("[temporal_jobs] %s segment %s failed (attempt %d): %s",
                           version_id, row["chunk_id"], attempts, exc)
            await _set_segment(client, db, row["segment_key"], status="failed", reason=str(exc)[:500],
                               attempts=attempts)
            continue
        await _set_segment(client, db, row["segment_key"], status="done", reason=None, attempts=attempts,
                           reply=json.dumps(reply, ensure_ascii=False))
        await _set_job(client, db, key, **_progress(await _counts(client, db, key)))

    rows = await _segments(client, db, key)
    counts = _progress(await _counts(client, db, key))
    usable = [r for r in rows if r["status"] != "unsupported"]
    replies = {r["chunk_id"]: json.loads(r["reply"]) for r in usable if r["status"] == "done" and r.get("reply")}
    job = await _job(client, db, key) or job
    published = anchor_day(job["anchor_time"], settings.temporal_default_timezone) if job.get("anchor_time") else None
    accepted, unresolved = extract(
        [Segment(r["chunk_id"], r.get("text") or "", r.get("heading") or "", r.get("page") or "") for r in usable],
        replies=replies, published=published)

    if not await _live(client, branch_id, version_id, generation):   # archived / deleted / reactivated meanwhile
        now = await _job(client, db, key)
        if now and now.get("generation") != generation:
            # reopened at a newer generation while this run was busy: that run decides, not this stale one
            return "superseded"
        await _set_job(client, db, key, status="cancelled", reason="source_not_active", **counts)
        return "cancelled"
    media = {r["chunk_id"]: (r.get("media_start_ms"), r.get("media_end_ms")) for r in rows}
    text_offsets = {r["chunk_id"]: r.get("text_offset") or 0 for r in rows}
    written = await _write_facts(client, branch_id, db, version_id, entity_requests(accepted),
                                 fact_requests(version_id, accepted, media, text_offsets))
    kept = await _write_candidates(client, db, version_id, candidate_rows(version_id, unresolved, text_offsets))

    # P15 (§5.5): reasoning is the last step of the job -- `complete` means extraction *and* reasoning
    await _set_job(client, db, key, status="reasoning", **counts)
    try:
        outcome = await reason_source(client, branch_id, db, version_id)
    except Exception as exc:  # noqa: BLE001 -- the facts above stand; a failed reasoning step is reported, not fatal
        logger.error("[temporal_jobs] reasoning failed for %s: %s", version_id, exc)
        outcome = "failed"

    total = len(rows)
    if counts["segments_done"] == 0:
        status = "failed"
        reason = "unsupported_language" if counts["segments_unsupported"] == total else "extraction_failed"
    elif counts["segments_done"] < total:
        status, reason = "partial", ("unsupported_language" if counts["segments_unsupported"] else "segments_failed")
    else:
        status, reason = "complete", None
    if outcome == "failed" and status == "complete":
        status, reason = "partial", "reasoning_failed"
    await _set_job(client, db, key, status=status, reason=reason, facts=written, unresolved=kept, **counts)
    logger.info("[temporal_jobs] branch=%s version=%s %s: %d facts, %d unresolved, segments %s",
                branch_id, version_id, status, written, len(unresolved), counts)
    return status


def _progress(counts: dict[str, int]) -> dict[str, int]:
    return {"segments_done": counts.get("done", 0), "segments_failed": counts.get("failed", 0),
            "segments_unsupported": counts.get("unsupported", 0)}


async def status(client: ArcadeDBClient, branch_id: str, version_id: str) -> dict[str, Any] | None:
    """The status operation (§5.9): the current-version job of one source, or None if it has none.
    Raises DatabaseNotInitializedError for a tenant with no database."""
    db = await _ensure_ready(client, branch_id)
    await _ensure_job_schema(client, db)
    job = await _job(client, db, job_key(branch_id, version_id))
    if job is None:
        return None
    failures = await client.query(db, "SELECT chunk_id, status, reason, attempts FROM TemporalJobSegment "
                                      "WHERE job_key = :k AND status IN ['failed', 'unsupported'] ORDER BY seq",
                                  params={"k": job["job_key"]})
    keys = ("source_version_id", "media_type", "status", "reason", "extractor_version", "normalizer_version",
            "model", "anchor_time", "anchor_source", "anchor_reason", "segments_total", "segments_done", "segments_failed", "segments_unsupported", "facts",
            "unresolved", "created_at", "updated_at")
    return {"job_id": job["job_key"], **{k: job.get(k) for k in keys},
            "segments_with_problems": [{k: f.get(k) for k in ("chunk_id", "status", "reason", "attempts")}
                                       for f in failures]}


async def write_observation(client: ArcadeDBClient, branch_id: str, version_id: str, name: str,
                            caption: str | None, raw: str | None, source_anchor: Anchor) -> str | None:
    """A photo is itself an event in world time (temporal plan §5.7): "this photo was taken at <EXIF time>".
    One fact per photo version, dated to the instant, `anchor = capture`, with the raw EXIF value as its
    date text and the image's caption (model-written, so not evidence of anything else) as its value. None
    when the photo has no validated capture time."""
    if not source_anchor.time:
        return None
    ident = hashlib.sha256(f"{version_id}\x1fcapture".encode()).hexdigest()[:32]
    req = TemporalFactIndexRequest(
        fact_id=f"tf-obs-{ident}", entity=name or version_id, relation="was taken",
        value=(caption or "").strip()[:140], time_start=source_anchor.time, time_end=source_anchor.time,
        confidence=0.9, source_version_id=version_id, date_text=raw, extent="point", anchor="capture",
        extractor_version=EXTRACTOR_VERSION)
    await graph_indexer.index_temporal_fact(client, branch_id, req)
    return req.fact_id


# ---------------------------------------------------------------- reasoning (P15)

_MAX = 2 ** 62                                   # an unbounded side is stored as unknown, never as a sentinel


def _finite(v: float) -> int | None:
    return int(v) if abs(v) < _MAX else None


def _day(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(
        ZoneInfo(settings.temporal_default_timezone)).date().isoformat()


def describe_derived(start: tuple[float, float], end: tuple[float, float]) -> str:
    """How a derived interval may be stated: its sound bounds by day, labelled as derived."""
    s_lo, e_hi = _finite(start[0]), _finite(end[1])
    if s_lo is not None and e_hi is not None:
        text = f"between {_day(s_lo)} and {_day(e_hi)}"
    elif s_lo is not None:
        text = f"after {_day(s_lo)}"
    elif e_hi is not None:
        text = f"before {_day(e_hi)}"
    else:
        text = "undated"
    return f"{text} (derived)"


async def reason_source(client: ArcadeDBClient, branch_id: str, db: str, version_id: str) -> str:
    """Reason over one source's facts and candidates (§5.5); store the result and its derived facts, replacing
    the previous ones. Returns the status."""
    facts = await client.query(db, "SELECT fact_id, entity, relation, value, source_sentence, s_state, s_lo, s_hi, "
                                   "e_state, e_lo, e_hi, extent, links FROM TemporalFact WHERE source_version_id = :v "
                                   "AND (derived IS NULL OR derived = false)", params={"v": version_id})
    cands = await client.query(db, "SELECT candidate_id, chunk_id, entity, relation, value, source_sentence, links, "
                                   "extent, date_text, confidence FROM TemporalCandidate WHERE source_version_id = :v "
                                   "AND (status IS NULL OR status = 'active')", params={"v": version_id})
    result = reasoning.reason(facts, cands)
    await client.command(db, "DELETE VERTEX FROM TemporalFact WHERE source_version_id = :v AND derived = true",
                         params={"v": version_id})
    by_id = {c["candidate_id"]: c for c in cands}
    for d in result.derived:
        c = by_id[d.candidate_id]
        s_lo, s_hi, e_lo, e_hi = (_finite(v) for v in (*d.start, *d.end))
        s_known, e_known = s_lo is not None and s_hi is not None, e_lo is not None and e_hi is not None
        fid = "tf-der-" + hashlib.sha256(f"{version_id}\x1f{d.candidate_id}".encode()).hexdigest()[:32]
        row = {
            "fact_id": fid, "entity": c.get("entity") or "", "relation": c.get("relation") or "",
            "value": c.get("value") or "", "status": "active", "source_version_id": version_id,
            "derived": True, "support_ids": d.support_ids, "network_version": result.network_version,
            "anchor": "derived", "extractor_version": EXTRACTOR_VERSION, "precision": "derived",
            "date_text": describe_derived(d.start, d.end), "source_sentence": c.get("source_sentence"),
            "extent": c.get("extent"), "confidence": round(float(c.get("confidence") or 0.5) * 0.8, 3),
            # a side bounded on one end only stays `unknown`: sound (a superset) and never a sentinel
            "s_state": "known" if s_known else "unknown", "s_lo": s_lo if s_known else None,
            "s_hi": s_hi if s_known else None,
            "e_state": "known" if e_known else "unknown", "e_lo": e_lo if e_known else None,
            "e_hi": e_hi if e_known else None,
            "time_start": _day(s_lo) if s_known else None, "time_end": _day(e_hi) if e_known else None,
            "timeline_id": "world", "axis": "world", "time_unit": "ms", "timezone": settings.temporal_default_timezone,
            "normalizer_version": NORMALIZER_VERSION, "migration_version": NORMALIZER_VERSION,
            "requires_reextraction": False,
        }
        sets = ", ".join(f"{k} = :{k}" for k in row)
        await client.command(db, f"UPDATE TemporalFact SET {sets} UPSERT WHERE fact_id = :fact_id", params=row)  # nosec B608 -- keys are this function's own literals
        if c.get("chunk_id"):
            await client.command(db, "CREATE EDGE OBSERVED_IN FROM (SELECT FROM TemporalFact WHERE fact_id = :f) "
                                     "TO (SELECT FROM Chunk WHERE chunk_id = :c)", params={"f": fid, "c": c["chunk_id"]})
    doc = {"source_version_id": version_id, "status": result.status, "network_version": result.network_version,
           "derived": len(result.derived), "witness": json.dumps(result.witness, ensure_ascii=False),
           "links_used": json.dumps(result.links_used, ensure_ascii=False),
           "links_unresolved": json.dumps(result.links_unresolved, ensure_ascii=False), "computed_at": _now_ms()}
    sets = ", ".join(f"{k} = :{k}" for k in doc)
    await client.command(db, f"UPDATE TemporalReasoning SET {sets} UPSERT WHERE source_version_id = :source_version_id",  # nosec B608 -- keys are literals
                         params=doc)
    return result.status


async def reasoning_of(client: ArcadeDBClient, branch_id: str, version_id: str) -> dict[str, Any] | None:
    """The `reason` operation (§5.9): a source's stored result; `stale` while its job has not finished."""
    db = await _ensure_ready(client, branch_id)
    await _ensure_job_schema(client, db)
    rows = await client.query(db, "SELECT FROM TemporalReasoning WHERE source_version_id = :v", params={"v": version_id})
    job = await _job(client, db, job_key(branch_id, version_id))
    if not rows and job is None:
        return None
    r = {k: v for k, v in (rows[0] if rows else {}).items() if not k.startswith("@")}
    derived = await client.query(db, "SELECT fact_id, entity, relation, value, date_text, support_ids FROM TemporalFact "
                                     "WHERE source_version_id = :v AND derived = true", params={"v": version_id})
    return {"source_version_id": version_id, "status": r.get("status"), "network_version": r.get("network_version"),
            "stale": job is not None and job.get("status") not in ("complete", "partial", "failed"),
            "job_status": job.get("status") if job else None, "computed_at": r.get("computed_at"),
            "derived_facts": [{k: v for k, v in f.items() if not k.startswith("@")} for f in derived],
            **{k: json.loads(r[k]) if r.get(k) else [] for k in ("witness", "links_used", "links_unresolved")}}


async def recompute(client: ArcadeDBClient, branch_id: str, version_id: str) -> bool:
    """Queue a recompute (does not wait): the job is reopened -- its segments stay done, so no model call -- and
    runs again: facts, candidates, reasoning. False when the source has no job."""
    db = await _ensure_ready(client, branch_id)
    await _ensure_job_schema(client, db)
    key = job_key(branch_id, version_id)
    job = await _job(client, db, key)
    if job is None:
        return False
    if job.get("status") in TERMINAL:
        await _set_job(client, db, key, status="pending", reason=None,
                       generation=await _generation(client, db, version_id))
    submit(branch_id, key)
    return True


async def on_status_change(client: ArcadeDBClient, db: str, branch_id: str, version_id: str, status: str) -> dict:
    """Archive / reactivate a source (P14), alongside its Chunks, Source and facts: its undated candidates follow
    the same status; archiving cancels its pending or running jobs; reactivating reopens the jobs an archive
    cancelled, at the source's new generation, and hands them to the worker."""
    await _ensure_job_schema(client, db)
    out = {"temporal_candidates": 0, "temporal_jobs_cancelled": 0, "temporal_jobs_reopened": 0}
    rows = await client.command(db, "UPDATE TemporalCandidate SET status = :s WHERE source_version_id = :v",
                                params={"s": status, "v": version_id})
    out["temporal_candidates"] = int((rows[0].get("count") if rows else 0) or 0)
    jobs = await client.query(db, "SELECT job_key, status, reason FROM TemporalJob WHERE source_version_id = :v",
                              params={"v": version_id})
    for j in jobs:
        if status != "active" and j.get("status") in ("pending", "running"):
            await _set_job(client, db, j["job_key"], status="cancelled", reason="source_archived")
            out["temporal_jobs_cancelled"] += 1
        elif status == "active" and j.get("status") == "cancelled" \
                and j.get("reason") in ("source_archived", "source_not_active"):
            await _set_job(client, db, j["job_key"], status="pending", reason=None,
                           generation=await _generation(client, db, version_id))
            submit(branch_id, j["job_key"])
            out["temporal_jobs_reopened"] += 1
    return out


async def clone_temporal(client: ArcadeDBClient, db: str, branch_id: str, source_version: str, target_version: str,
                         chunk_ids: dict[str, str], new_id) -> dict:
    """Clone a source's temporal job, its segments (model replies included) and its undated candidates onto the
    target version (P14), chunk ids remapped -- so the target's status answers at once and a later run of it
    never calls the model again. A job still pending or running is cloned as pending and handed to the worker.
    *new_id(target_version, old_id)*: the clone's id scheme (indexer._clone_id)."""
    await _ensure_job_schema(client, db)
    out = {"temporal_jobs": 0, "temporal_candidates": 0}
    src_key, dst_key = job_key(branch_id, source_version), job_key(branch_id, target_version)
    job = await _job(client, db, src_key)
    if job and not await _job(client, db, dst_key):
        fields = {k: v for k, v in job.items() if not k.startswith("@")}
        fields.update(job_key=dst_key, source_version_id=target_version, updated_at=_now_ms(),
                      generation=await _generation(client, db, target_version))
        if fields.get("status") in ("running", "cancelled"):
            fields["status"], fields["reason"] = "pending", None
        sets = ", ".join(f"{k} = :{k}" for k in fields)
        await client.command(db, f"INSERT INTO TemporalJob SET {sets}", params=fields)  # nosec B608 -- keys are an existing TemporalJob row's own columns
        for seg in await _segments(client, db, src_key):
            row = {k: v for k, v in seg.items() if not k.startswith("@")}
            cid = chunk_ids.get(row["chunk_id"], row["chunk_id"])
            row.update(job_key=dst_key, chunk_id=cid, segment_key=f"{dst_key}:{cid}")
            sets = ", ".join(f"{k} = :{k}" for k in row)
            await client.command(db, f"INSERT INTO TemporalJobSegment SET {sets}", params=row)  # nosec B608 -- keys are an existing segment row's own columns
        out["temporal_jobs"] = 1
        if fields["status"] == "pending":
            submit(branch_id, dst_key)
    for cand in await client.query(db, "SELECT FROM TemporalCandidate WHERE source_version_id = :v",
                                   params={"v": source_version}):
        row = {k: v for k, v in cand.items() if not k.startswith("@")}
        row.update(candidate_id=new_id(target_version, row["candidate_id"]), source_version_id=target_version,
                   chunk_id=chunk_ids.get(row.get("chunk_id"), row.get("chunk_id")), status="active")
        sets = ", ".join(f"{k} = :{k}" for k in row)
        await client.command(db, f"UPDATE TemporalCandidate SET {sets} UPSERT WHERE candidate_id = :candidate_id",  # nosec B608 -- keys are an existing candidate row's own columns
                             params=row)
        out["temporal_candidates"] += 1
    return out


async def delete_jobs(client: ArcadeDBClient, db: str, version_id: str) -> int:
    """Remove a source version's jobs, segments, unresolved candidates and reasoning (delete-by-version)."""
    await _ensure_job_schema(client, db)
    await client.command(db, "DELETE FROM TemporalCandidate WHERE source_version_id = :v", params={"v": version_id})
    await client.command(db, "DELETE FROM TemporalReasoning WHERE source_version_id = :v", params={"v": version_id})
    keys = [r["job_key"] for r in await client.query(
        db, "SELECT job_key FROM TemporalJob WHERE source_version_id = :v", params={"v": version_id})]
    for key in keys:
        await client.command(db, "DELETE FROM TemporalJobSegment WHERE job_key = :k", params={"k": key})
    await client.command(db, "DELETE FROM TemporalJob WHERE source_version_id = :v", params={"v": version_id})
    return len(keys)


# ---------------------------------------------------------------- runner

class TemporalJobRunner:
    """temporal_job_workers background workers draining a queue of (branch_id, job_key). Jobs run side by side
    while every model call still takes its turn (temporal_llm_concurrency), so their segments interleave and a
    short source is not stuck behind a long one. One job never runs twice at once: submitted again while it
    runs, it is run once more afterwards (it may have new segments). start() re-queues every job a previous
    process left pending or running, so nothing is lost to a restart."""

    def __init__(self, client: ArcadeDBClient) -> None:
        self.client = client
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.queued: set[tuple[str, str]] = set()
        self.running: set[tuple[str, str]] = set()
        self.again: set[tuple[str, str]] = set()
        self.tasks: list[asyncio.Task] = []

    def submit(self, branch_id: str, key: str) -> None:
        if (branch_id, key) not in self.queued:
            self.queued.add((branch_id, key))
            self.queue.put_nowait((branch_id, key))

    async def recover(self) -> int:
        n = 0
        for db in await self.client.list_databases():
            if not db.startswith("user_"):
                continue
            try:
                rows = await self.client.query(db, "SELECT branch_id, job_key FROM TemporalJob "
                                                   "WHERE status IN ['pending', 'running']")
            except Exception as exc:  # noqa: BLE001 -- a database from before P12 has no TemporalJob type; nothing to resume there
                logger.debug("[temporal_jobs] no jobs to resume in %s: %s", db, exc)
                continue
            for r in rows:
                self.submit(r["branch_id"], r["job_key"])
                n += 1
        return n

    async def _loop(self) -> None:
        while True:
            item = await self.queue.get()
            self.queued.discard(item)
            if item in self.running:
                self.again.add(item)
                continue
            self.running.add(item)
            branch_id, key = item
            try:
                await run_job(self.client, branch_id, key)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- ArcadeDB down etc.; the job stays pending/running and is resumed at the next start
                logger.error("[temporal_jobs] job %s for branch %s stopped: %s", key[:12], branch_id, exc)
            finally:
                self.running.discard(item)
            if item in self.again:
                self.again.discard(item)
                self.submit(branch_id, key)

    async def start(self) -> None:
        self.tasks = [asyncio.create_task(self._loop()) for _ in range(max(1, settings.temporal_job_workers))]
        try:
            n = await self.recover()
            if n:
                logger.info("[temporal_jobs] resumed %d unfinished job(s)", n)
        except Exception as exc:  # noqa: BLE001 -- recovery is best effort; new jobs still run
            logger.error("[temporal_jobs] recovery failed: %s", exc)

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


runner: TemporalJobRunner | None = None


def submit(branch_id: str, key: str) -> None:
    """Hand a recorded job to the running worker. Without one (tests, or the API without the consumer)
    the job stays pending and is picked up at the next start."""
    if runner is not None:
        runner.submit(branch_id, key)
