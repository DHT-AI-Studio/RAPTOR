"""LLM-based entity/relationship/temporal-fact extraction for the personal
graph (PA-7 data source), video only for now.

Ported from Module 20 (graph-service)'s app/graph_builder.py + app/tkg.py --
same prompt, same _has_temporal_signal heuristic -- but writing into this
service's own per-user ArcadeDB via graph_indexer instead of 20's shared
Neo4j. LLM calls (`_call_llm()` below) go through module 07's
/inference/infer, same as Module 20's now do (graph_builder.py's
_call_llm_via_07) -- switched from calling settings.llm_base_url's
OpenAI-compat endpoint directly after confirming live that qwen3.x-family
models "think" by default on that path with no way to turn it off, adding
10-20x latency for an identical final answer (4.8s vs 59s measured on the
same prompt). Module 07's Ollama adapter defaults think=false. This is NOT
Module 12 (document_graph_service)'s known-buggy module-07 usage (a
nonexistent TEMPORAL_MODEL_URL call) -- that bug is a separate call this
file has no equivalent of; the /inference/infer task name and shape used
here (`task=text-generation`, `data.messages`) were verified live before
this switch, not assumed.

Every public entry point here is fire-and-forget from the caller's
perspective: extraction failure must never affect the chunk-indexing result
that already succeeded for the same Kafka message.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from app.core.config import settings
from app.models.graph_index import (EntityIndexRequest, RelationshipIndexRequest,
                                     TemporalFactIndexRequest)
from app.services import graph_indexer
from app.services.arcadedb_client import ArcadeDBClient
from app.services.date_utils import fill_relation_dates

try:
    from rapidfuzz import fuzz
    _USE_FUZZY = True
except ImportError:  # pragma: no cover -- rapidfuzz is a hard requirement now
    # (requirements.txt), this mirrors Module 20's own defensive fallback in
    # match_entities_to_moments() rather than assuming the import can't fail.
    _USE_FUZZY = False

logger = logging.getLogger("personal_db.graph_extractor")

_SYSTEM_PROMPT = (
    "You are a precise JSON-only information extraction engine. "
    "Output only valid JSON, no explanation."
)

_EXTRACTION_PROMPT = """\
Extract all named entities and their relations from the text below.
The text may be in Chinese (Traditional/Simplified) or English — handle both.

Return ONLY valid JSON in this exact structure (no markdown, no explanation):
{{
  "entities": [
    {{"name": "Entity Name", "type": "PERSON|ORG|PLACE|EVENT|CONCEPT|OTHER", "description": "one sentence"}}
  ],
  "relations": [
    {{"subject": "Entity Name A", "predicate": "VERB_PHRASE", "object": "Entity Name B", "time_start": null, "time_end": null, "confidence": 0.9}}
  ]
}}

Rules:
- name: copy the entity's name EXACTLY as it appears in the text — do not translate,
  transliterate, or romanize it (e.g. keep "川普" as "川普", not "Trump" or "chuan_pu")
- CRITICAL: subject and object MUST be copied EXACTLY from a "name" in the entities list
- type:
    PERSON   — real-world individuals (politicians, public figures, celebrities, officials)
    ORG      — companies, governments, military units, institutions
    PLACE    — countries, cities, regions, geographic locations
    EVENT    — named events, operations, incidents, meetings
    CONCEPT  — abstract ideas, policies, technologies, doctrines
    OTHER    — anything else
- predicate: uppercase verb phrase (e.g. "LEADS", "ATTACKS", "ANNOUNCED", "SIGNED_WITH")
- time_start / time_end: ISO 8601 date string or null
- confidence: 0.0–1.0
- Return empty lists if nothing found

Text:
{text}
"""

_TEMPORAL_KEYWORDS = (
    "年", "月", "日", "世紀", "年代", "之前", "之後", "以前", "以後",
    "去年", "今年", "明年", "昨日", "昨天", "今天", "明天", "最近", "後來", "當時", "那時",
    "year", "month", "day", "century", "decade", "before", "after",
    "yesterday", "today", "tomorrow", "ago", "later", "recently",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    return slug[:64] or hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]


def _stable_fact_id(entity_id: str, relation: str, value: str, time_start: Optional[str]) -> str:
    raw = f"{entity_id or ''}|{relation}|{value}|{time_start or ''}"
    return f"tf-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _has_temporal_signal(text: str) -> bool:
    if not text:
        return False
    if any(kw in text for kw in _TEMPORAL_KEYWORDS):
        return True
    return bool(re.search(r"\b(?:19|20)\d{2}\b", text))


_BATCH_MOMENT_PROMPT = """\
For each numbered moment below, list the named entities (people, characters,
organizations, places, concepts) that appear in that moment.

Return ONLY valid JSON in this exact structure (no markdown):
{{
  "moments": [
    {{"moment_index": 0, "entity_names": ["Name 1", "Name 2"]}},
    {{"moment_index": 1, "entity_names": []}}
  ]
}}

Rules:
- Only return entity NAMES (the canonical name as it appears in the text)
- Do not invent entities — only ones explicitly mentioned
- Empty list if nothing named
- Cap each moment at 5 entities

Moments:
{moments_block}
"""


class _ContentPolicyBlockedError(Exception):
    """Raised by _call_llm_raw() when module 07 returns its guardrail-block
    shape (HTTP 400, {"error_type": "PolicyViolationError", ...}) -- a
    permanent result: retrying the identical prompt against the same policy
    can never succeed. Re-raised immediately in the retry loop below instead
    of going through the normal retries/retry_delay bookkeeping -- the one
    caller that passes retries>0 (the video-level summary call) would
    otherwise burn retries*retry_delay seconds waiting on something that
    will never pass, same as every other non-retryable failure this
    function's broad `except Exception` used to treat identically to a
    genuine transient one."""


async def _call_llm_raw(messages: list, timeout: float = 90.0, retries: int = 0,
                        retry_delay: float = 2.0) -> str:
    """Goes through module 07's /inference/infer, not settings.llm_base_url's
    OpenAI-compat endpoint directly -- module 07's Ollama adapter defaults
    think=false and talks to Ollama's native /api/chat (proper chat-template
    handling for the messages passed in here); the OpenAI-compat layer has no
    think control at all, and qwen3.x-family models think by default. Measured
    live on an identical extraction prompt: 4.8s with think=false vs 59s with
    think=true for the same final answer -- the likely cause of this call
    occasionally missing its own timeout on longer videos. See config.py's
    inference_url for why this doesn't repeat document_graph_service's
    (module 12) known TEMPORAL_MODEL_URL bug: no such call exists here.

    retries: extra attempts after the first, with retry_delay seconds between
    them. Default 0 (no retry) -- the moment-batch loop in
    extract_moment_entities_batched() and the per-moment temporal-fact loop
    in extract_and_index_video() already tolerate a single batch/moment
    failing (skip it, keep going), so retrying every one of those calls would
    just add load during exactly the kind of transient LLM-endpoint trouble
    this is meant to help with -- confirmed live once, as a shared Ollama
    instance's model runner crashing ("model runner has unexpectedly
    stopped... may be due to resource limitations or an internal error");
    root cause not confirmed (our own concurrent load vs. another team's much
    larger model on the same shared instance -- no log evidence either way).
    Only the video-level summary call passes retries>0 -- it is the one
    remaining single point of failure: if it fails, extract_and_index_video()
    returns immediately with zero entities/relations/temporal_facts for the
    whole video, no matter how long the video is or how much of the rest of
    the pipeline would otherwise have worked."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{settings.inference_url}/inference/infer",
                    json={
                        "task": "text-generation",
                        "engine": "ollama",
                        "model_name": settings.chat_model_name,
                        "data": {"messages": messages},
                        "options": {"temperature": 0.0, "think": settings.inference_think},
                    },
                )
                if resp.status_code == 400:
                    try:
                        err_body = resp.json()
                    except ValueError:
                        err_body = {}
                    # FastAPI's HTTPException(detail={...}) nests the payload
                    # under "detail" -- never at the top level of the body.
                    detail = err_body.get("detail")
                    if not isinstance(detail, dict):
                        detail = {}
                    if detail.get("error_type") == "PolicyViolationError":
                        raise _ContentPolicyBlockedError(
                            f"內容被guardrail政策擋下 (category={detail.get('category')}, "
                            f"direction={detail.get('direction')})"
                        )
                resp.raise_for_status()
                return resp.json()["result"]["response"].strip()
        except _ContentPolicyBlockedError:
            raise  # permanent -- skip the retries/retry_delay bookkeeping entirely
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning("[graph_extractor] LLM call failed (%s: %s), retrying "
                              "(%d/%d) in %.0fs", type(exc).__name__, exc,
                              attempt + 1, retries, retry_delay)
                await asyncio.sleep(retry_delay)
    raise last_exc


async def _call_llm(prompt: str, timeout: float = 90.0, retries: int = 0) -> Dict[str, Any]:
    try:
        content = await _call_llm_raw([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], timeout=timeout, retries=retries)
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception as exc:
        # type(exc).__name__ because httpx timeout-family exceptions often
        # stringify to "" -- "LLM extraction failed ()" told us nothing about
        # what actually happened when this fired live on the Charade video.
        logger.warning("[graph_extractor] LLM extraction failed (%s: %s), returning empty results",
                       type(exc).__name__, exc)
        return {"entities": [], "relations": []}

    # async with httpx.AsyncClient(timeout=timeout) as client:
    #     resp = await client.post(
    #         f"{settings.llm_base_url}/chat/completions",
    #         json={
    #             "model": settings.chat_model_name,
    #             "temperature": 0.0,
    #             "messages": [
    #                 {"role": "system", "content": _SYSTEM_PROMPT},
    #                 {"role": "user", "content": prompt},
    #             ],
    #         },
    #     )
    #     resp.raise_for_status()
    #     content = resp.json()["choices"][0]["message"]["content"].strip()
    #     if content.startswith("```"):
    #         content = re.sub(r"^```(?:json)?\s*", "", content)
    #         content = re.sub(r"\s*```$", "", content)
    #     return json.loads(content)


async def extract_moment_entities_batched(
    moments: List[Tuple[str, str]], batch_size: int = 8,
) -> Dict[str, List[str]]:
    """For each moment with non-empty text, ask the LLM which named entities
    appear. Returns {chunk_id: [entity_name, ...]}.

    Ported from Module 20's graph_builder.py::extract_moment_entities_batched
    -- keyed by chunk_id instead of an integer moment_index, since 25's
    `moments` are (chunk_id, text) tuples with no separate index concept.
    Batched to amortize LLM call overhead: O(ceil(N / batch_size)) LLM calls
    instead of O(N). This is what lets 25 discover entities the video-level
    summary extraction missed entirely (Module 20's moment_derived_entities);
    the existing substring/fuzzy MENTIONS matching above can only ever find
    entities the summary already named.

    Batches run concurrently, bounded by graph_extraction_batch_concurrency
    (20's own version of this loop is sequential -- ported that way at first
    here too, but a long video means many batches, and the whole point of
    graph_extraction_max_concurrency=1 is to keep only one video's LLM calls
    in flight at a time; a video that takes minutes to grind through 85
    sequential batches spends that whole time as a window another upload's
    extraction could land in and queue up behind. Each batch computes its own
    partial dict and returns it -- results are merged after gather(), not
    written to a shared dict from inside concurrent tasks, so there's no race
    on `results` itself."""
    targets = [(cid, text) for cid, text in moments if (text or "").strip()]
    if not targets:
        return {}

    semaphore = asyncio.Semaphore(settings.graph_extraction_batch_concurrency)

    async def _run_batch(chunk: List[Tuple[str, str]], start: int) -> Dict[str, List[str]]:
        moments_block = "\n\n".join(
            f"[{i}] {text[:600]}" for i, (_cid, text) in enumerate(chunk)
        )
        prompt = _BATCH_MOMENT_PROMPT.format(moments_block=moments_block)
        partial: Dict[str, List[str]] = {}
        async with semaphore:
            try:
                content = await _call_llm_raw([
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ], timeout=120.0)
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                data = json.loads(content)

                for entry in data.get("moments", []):
                    idx = entry.get("moment_index")
                    names = entry.get("entity_names", []) or []
                    if isinstance(idx, int) and 0 <= idx < len(chunk) and isinstance(names, list):
                        cid = chunk[idx][0]
                        partial[cid] = [str(n).strip() for n in names if str(n).strip()][:5]
            except Exception as exc:
                logger.warning("[graph_extractor] batched moment entity extraction failed for "
                              "batch %d (%s: %s)", start, type(exc).__name__, exc)
        return partial

    batch_tasks = [
        _run_batch(targets[start:start + batch_size], start)
        for start in range(0, len(targets), batch_size)
    ]
    partials = await asyncio.gather(*batch_tasks)

    results: Dict[str, List[str]] = {}
    for partial in partials:
        results.update(partial)

    return results


def _resolve_moment_entities_to_ids(
    moment_entity_names: Dict[str, List[str]],
    entity_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """Convert {chunk_id: [entity_name, ...]} into {chunk_id: {entity_id, ...}}.

    Names matched against entity_map (exact + substring, case-insensitive).
    Names NOT already known are minted as new entities (PERSON type by
    default -- most commonly missed items in ASR/moment text are people) and
    added into entity_map in place, so the caller's existing write/link logic
    picks them up too. Ported from Module 20's resolve_moment_entities_to_ids
    (graph_builder.py); no separate "new_entities" return value here since
    entity_map is mutated directly instead."""
    name_to_id: Dict[str, str] = {}
    for eid, ent in entity_map.items():
        name = (ent.get("name") or "").strip().lower()
        if name:
            name_to_id[name] = eid

    moment_entity_ids: Dict[str, Set[str]] = defaultdict(set)
    for chunk_id, names in moment_entity_names.items():
        for name in names:
            lower = name.lower()
            eid = name_to_id.get(lower)

            if not eid:
                for known_name, known_id in name_to_id.items():
                    if lower in known_name or known_name in lower:
                        eid = known_id
                        break

            if not eid:
                new_id = _slugify(name)
                if new_id not in entity_map:
                    entity_map[new_id] = {
                        "id": new_id, "name": name, "type": "PERSON", "description": "",
                    }
                    name_to_id[lower] = new_id
                eid = new_id

            moment_entity_ids[chunk_id].add(eid)

    return moment_entity_ids


async def extract_entities_and_relations(
    text: str, max_entities: int = 20, max_relations: int = 30, retries: int = 0,
) -> Dict[str, Any]:
    prompt = _EXTRACTION_PROMPT.format(text=text[:4000])
    extracted = await _call_llm(prompt, retries=retries)

    entities = [e for e in extracted.get("entities", [])[:max_entities] if (e.get("name") or "").strip()]
    relations = extracted.get("relations", [])[:max_relations]

    # entity_id is always derived here, from `name`, never trusted from the
    # LLM. Earlier this asked the LLM to invent a romanized slug id itself
    # (e.g. "川普" -> "chuan_pu") -- but that's a free-text transliteration,
    # not a deterministic function, so the *same* name extracted from two
    # different calls (the video-level summary vs. a later per-moment call)
    # could come back with two different ids, splitting one real entity into
    # several Entity nodes (confirmed live: "川普" landed as both "chuan_pu"
    # and "chuang_pu" in one video). _slugify() keeps CJK characters as-is
    # (Python's \w matches them), so the same name always yields the same id
    # with no transliteration step at all.
    name_to_id: Dict[str, str] = {}
    for ent in entities:
        name = ent["name"].strip()
        ent["id"] = _slugify(name)
        name_to_id[name] = ent["id"]

    filtered_relations = []
    for r in relations:
        subj, obj = r.get("subject"), r.get("object")
        if subj in name_to_id and obj in name_to_id:
            r["subject_id"] = name_to_id[subj]
            r["object_id"] = name_to_id[obj]
            filtered_relations.append(r)
        else:
            logger.warning(
                "[graph_extractor] dropping relation with mismatched names: subject=%r object=%r",
                subj, obj)

    # Fix 1B (ported from Module 20): the LLM often leaves time_start=null
    # even when the text clearly states a date -- backfill it deterministically
    # from the source text before returning, so more relations become
    # TemporalFacts than the LLM alone would produce.
    filled = fill_relation_dates(text, entities, filtered_relations)
    if filled:
        logger.info("[graph_extractor] backfilled time_start on %d/%d relation(s)",
                    filled, len(filtered_relations))

    logger.info("[graph_extractor] extracted %d entities, %d relations",
                len(entities), len(filtered_relations))
    return {"entities": entities, "relations": filtered_relations}


async def _write_entities_and_relations(
    client: ArcadeDBClient, branch_id: str, version_id: str,
    entities: List[Dict[str, Any]], relations: List[Dict[str, Any]],
    anchor_chunk_id: Optional[str],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Writes entities (anchored to anchor_chunk_id) and relations. Returns
    (id -> entity dict map, relation count) for reuse by the temporal-fact step."""
    entity_map: Dict[str, Dict[str, Any]] = {}
    for ent in entities:
        eid, name = ent.get("id"), ent.get("name")
        if not eid or not name:
            continue
        entity_map[eid] = ent
        try:
            await graph_indexer.index_entity(client, branch_id, EntityIndexRequest(
                entity_id=eid, name=name, type=ent.get("type") or "OTHER",
                description=ent.get("description"), source_chunk_id=anchor_chunk_id,
            ))
        except Exception as exc:
            logger.warning("[graph_extractor] failed to index entity %s: %s", eid, exc)

    relation_count = 0
    for rel in relations:
        subj, obj, predicate = rel.get("subject_id"), rel.get("object_id"), rel.get("predicate")
        if not subj or not obj or not predicate:
            continue
        try:
            await graph_indexer.index_relationship(client, branch_id, RelationshipIndexRequest(
                from_entity_id=subj, to_entity_id=obj, relation=predicate,
                confidence=rel.get("confidence"), source_version_id=version_id,
            ))
            relation_count += 1
        except Exception as exc:
            logger.warning("[graph_extractor] failed to index relation %s->%s: %s", subj, obj, exc)

    return entity_map, relation_count


async def _write_temporal_facts(
    client: ArcadeDBClient, branch_id: str, version_id: str,
    entity_map: Dict[str, Dict[str, Any]], relations: List[Dict[str, Any]],
    chunk_id: Optional[str],
) -> int:
    count = 0
    for rel in relations:
        time_start, time_end = rel.get("time_start"), rel.get("time_end")
        if not time_start and not time_end:
            continue
        subj_id = rel.get("subject_id")
        subject = entity_map.get(subj_id, {})
        obj = entity_map.get(rel.get("object_id"), {})
        value = obj.get("name", rel.get("object_id"))
        fact_id = _stable_fact_id(subj_id, rel["predicate"], value, time_start)
        try:
            await graph_indexer.index_temporal_fact(client, branch_id, TemporalFactIndexRequest(
                fact_id=fact_id, entity=subject.get("name", subj_id), relation=rel["predicate"],
                value=value, entity_id=subj_id, time_start=time_start, time_end=time_end,
                confidence=rel.get("confidence"), chunk_id=chunk_id, source_version_id=version_id,
            ))
            count += 1
        except Exception as exc:
            logger.warning("[graph_extractor] failed to index temporal fact %s: %s", fact_id, exc)
    return count


async def extract_and_index_video(
    client: ArcadeDBClient,
    branch_id: str,
    version_id: str,
    summary_text: str,
    moments: List[Tuple[str, str]],
) -> Dict[str, int]:
    """moments: [(chunk_id, text), ...] for this video's per-moment chunks,
    already resolved by the caller to the same chunk_id used when those
    moments were indexed as Chunk rows (required for MENTIONS/OBSERVED_IN
    edges to resolve). Never raises.

    summary_text is no longer anchored to a fake summary Chunk (that Chunk no
    longer gets created at all -- see indexer.index_source_summary()); the
    entities/relations extracted from it get a MENTIONED_IN edge straight to
    the Source (version_id), matching Module 20's create_mentioned_in()
    (pipeline.py:127, same summary-level extraction step)."""
    stats = {"entities": 0, "relations": 0, "temporal_facts": 0, "mentions": 0, "co_occurs": 0}
    # entity_id set per moment chunk -- feeds the CO_OCCURS_WITH pairing pass
    # at the end. Only real per-moment entities go in here, never the
    # summary-level ones (those get MENTIONED_IN, not MENTIONS, and were
    # never chunk-anchored to begin with now): Module 20's CO_OCCURS_WITH is
    # "same-Moment co-appearance" (neo4j_writer.py:9) -- pairing every entity
    # mentioned anywhere in a video's summary would be a much looser,
    # video-wide "appeared in the same video" relationship, not what this
    # edge type is meant to represent.
    moment_entity_ids: Dict[str, Set[str]] = defaultdict(set)
    entity_map: Dict[str, Dict[str, Any]] = {}
    relations: List[Dict[str, Any]] = []
    try:
        # Summary-level extraction -- matches Module 20's `if summary:` (step 3
        # of pipeline.py's run_ingest_pipeline()): entities/relations stay []
        # if there's no summary text at all, not an early return. Everything
        # below that depends on entities is itself gated (matching 20's own
        # step 4/5/6 gates below), but per-moment temporal facts is NOT --
        # same as 20, which runs write_moment_temporal_facts() regardless of
        # whether the summary-level extraction found anything, as long as
        # there are moments. This used to be a hard `if not entities: return
        # stats` here, which made 25 lose per-moment temporal facts too on a
        # summary-extraction failure -- something 20 never did.
        entities: List[Dict[str, Any]] = []
        if summary_text.strip():
            # retries=2: this is the one call in the whole function whose
            # failure previously zeroed out everything downstream -- worth a
            # couple of retries with backoff before giving up, unlike the
            # moment-batch/per-moment-temporal calls below, which already
            # tolerate a single failure without losing the whole video.
            extracted = await extract_entities_and_relations(summary_text, retries=2)
            entities, relations = extracted["entities"], extracted["relations"]

        # Step 4/5 (Module 20's pipeline.py numbering): entity nodes +
        # MENTIONED_IN, only if the summary extraction actually found something.
        if entities:
            entity_map, relation_count = await _write_entities_and_relations(
                client, branch_id, version_id, entities, relations, None)
            stats["entities"] = len(entity_map)
            stats["relations"] = relation_count

            for eid in entity_map:
                try:
                    await graph_indexer.index_mentioned_in(client, branch_id, eid, version_id)
                except Exception as exc:
                    logger.warning("[graph_extractor] failed to link entity %s to source %s: %s",
                                   eid, version_id, exc)

            stats["temporal_facts"] += await _write_temporal_facts(
                client, branch_id, version_id, entity_map, relations, None)

        # Step 6: entity-dependent moment enrichment -- matches Module 20's
        # `if entities and moments:` exactly (both the hybrid MENTIONS/
        # APPEARS_IN matching and the LLM-based moment-derived entity
        # discovery are nested inside that same gate in graph_builder.py /
        # pipeline.py, not independently gated).
        if entities and moments:
            # Hybrid MENTIONS: for each entity, link it to any moment whose
            # text contains its name (exact substring) or is a close fuzzy
            # match -- matches Module 20's hybrid (substring + fuzzy)
            # APPEARS_IN matching in match_entities_to_moments()
            # (graph_builder.py), same threshold (75). Only catches entities
            # the summary-level extraction already named; the LLM-based
            # discovery just below catches entities it missed entirely.
            _FUZZY_THRESHOLD = 75
            for eid, ent in entity_map.items():
                name = (ent.get("name") or "").strip().lower()
                if not name:
                    continue
                for chunk_id, text in moments:
                    if not text:
                        continue
                    search_text = text.lower()
                    matched = name in search_text
                    if not matched and _USE_FUZZY:
                        matched = fuzz.partial_ratio(name, search_text) >= _FUZZY_THRESHOLD
                    if matched:
                        try:
                            await graph_indexer.index_entity(client, branch_id, EntityIndexRequest(
                                entity_id=eid, name=ent["name"], type=ent.get("type") or "OTHER",
                                source_chunk_id=chunk_id,
                            ))
                            stats["mentions"] += 1
                            moment_entity_ids[chunk_id].add(eid)
                        except Exception as exc:
                            logger.warning("[graph_extractor] failed to link entity %s to chunk %s: %s",
                                           eid, chunk_id, exc)

            # General moment-level entity discovery: LLM-based, across ALL
            # moments with text -- catches entities the summary-level
            # extraction missed entirely, same as Module 20's
            # moment_derived_entities (extract_moment_entities_batched() +
            # resolve_moment_entities_to_ids() in graph_builder.py).
            moment_entity_names = await extract_moment_entities_batched(moments)
            if moment_entity_names:
                ids_before = set(entity_map.keys())
                discovered = _resolve_moment_entities_to_ids(moment_entity_names, entity_map)
                new_ids = set(entity_map.keys()) - ids_before

                for eid in new_ids:
                    ent = entity_map[eid]
                    try:
                        await graph_indexer.index_entity(client, branch_id, EntityIndexRequest(
                            entity_id=eid, name=ent["name"], type=ent.get("type") or "OTHER",
                        ))
                    except Exception as exc:
                        logger.warning("[graph_extractor] failed to index moment-derived entity %s: %s",
                                       eid, exc)

                for chunk_id, eids in discovered.items():
                    moment_entity_ids[chunk_id].update(eids)
                    for eid in eids:
                        ent = entity_map.get(eid)
                        if not ent:
                            continue
                        try:
                            await graph_indexer.index_entity(client, branch_id, EntityIndexRequest(
                                entity_id=eid, name=ent["name"], type=ent.get("type") or "OTHER",
                                source_chunk_id=chunk_id,
                            ))
                            stats["mentions"] += 1
                        except Exception as exc:
                            logger.warning("[graph_extractor] failed to link moment-derived entity "
                                          "%s to chunk %s: %s", eid, chunk_id, exc)

                if new_ids:
                    stats["entities"] += len(new_ids)
                    for eid in new_ids:
                        try:
                            await graph_indexer.index_mentioned_in(client, branch_id, eid, version_id)
                        except Exception as exc:
                            logger.warning("[graph_extractor] failed to link moment-derived entity "
                                          "%s to source %s: %s", eid, version_id, exc)

        # Per-moment temporal facts: NOT gated on entities/summary success at
        # all -- matches Module 20's `if enable_moment_temporal and moments:`
        # (pipeline.py step 7), which only checks moments exist. Only for
        # temporal signal, capped at graph_extraction_max_moments LLM calls.
        #
        # Entities discovered here are used ONLY to resolve names for the
        # TemporalFact record itself -- NOT written as Entity vertices, NOT
        # linked via MENTIONS, NOT included in CO_OCCURS_WITH pairing. This
        # matches Module 20's write_moment_temporal_facts() (tkg.py) exactly:
        # its merged_entities list is local to that function, never passed to
        # neo4j.upsert_entities()/create_appears_in()/create_co_occurs_with().
        # (Previously this also wrote real entities/MENTIONS/CO_OCCURS_WITH
        # here -- redundant with the general moment-level discovery pass
        # above, which already covers every moment with text, a superset of
        # "moments with a temporal signal", and a real divergence from 20.)
        # Concurrent, bounded by the same graph_extraction_batch_concurrency
        # as extract_moment_entities_batched() above -- same rationale (this
        # loop is capped at graph_extraction_max_moments=10 candidates so it
        # was never going to be a huge burst like the moment-batch loop can
        # be, but running it after that loop finishes serially was still
        # unnecessary added wall-clock time for a single video's extraction).
        # entity_map is only read here (dict(entity_map) copies it), never
        # mutated by this loop's tasks, so concurrent reads are race-free.
        temporal_semaphore = asyncio.Semaphore(settings.graph_extraction_batch_concurrency)

        async def _process_temporal_candidate(chunk_id: str, text: str) -> int:
            async with temporal_semaphore:
                try:
                    moment_extracted = await extract_entities_and_relations(
                        text, max_entities=5, max_relations=5)
                except Exception as exc:
                    logger.warning("[graph_extractor] moment %s extraction failed: %s", chunk_id, exc)
                    return 0
            moment_entities = moment_extracted["entities"]
            moment_relations = moment_extracted["relations"]
            if not moment_entities:
                return 0
            local_entity_map = dict(entity_map)
            for ent in moment_entities:
                eid = ent.get("id")
                if eid and eid not in local_entity_map:
                    local_entity_map[eid] = ent
            return await _write_temporal_facts(
                client, branch_id, version_id, local_entity_map, moment_relations, chunk_id)

        candidates = [(cid, text) for cid, text in moments if _has_temporal_signal(text)]
        candidates = candidates[:settings.graph_extraction_max_moments]
        temporal_counts = await asyncio.gather(*[
            _process_temporal_candidate(chunk_id, text) for chunk_id, text in candidates
        ])
        stats["temporal_facts"] += sum(temporal_counts)

        # CO_OCCURS_WITH: every pair of entities that share a moment, same
        # semantics as Module 20's create_co_occurs_with() (neo4j_writer.py:
        # 284, called from pipeline.py right after the equivalent of the
        # MENTIONS-writing step above). A moment with only one entity
        # contributes no pairs.
        for chunk_id, eids in moment_entity_ids.items():
            if len(eids) < 2:
                continue
            for eid_a, eid_b in combinations(sorted(eids), 2):
                try:
                    await graph_indexer.index_co_occurrence(client, branch_id, eid_a, eid_b)
                    stats["co_occurs"] += 1
                except Exception as exc:
                    logger.warning("[graph_extractor] failed to link co-occurrence %s<->%s: %s",
                                   eid_a, eid_b, exc)

        logger.info("[graph_extractor] branch=%s version=%s: %s", branch_id, version_id, stats)
    except Exception as exc:
        logger.error("[graph_extractor] extraction failed for branch=%s version=%s: %s",
                     branch_id, version_id, exc)
    return stats
