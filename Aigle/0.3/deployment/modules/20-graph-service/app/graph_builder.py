# kafka/services/video_graph_service/graph_builder.py
#
# Entity extraction from video summary (1 LLM call per video)
# + Hybrid keyword matching of entities against each Moment's ASR/LVLM text

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import httpx
from moment_parser import extract_search_text

logger = logging.getLogger(__name__)

_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434/v1")
_CHAT_MODEL   = os.getenv("CHAT_MODEL_NAME", "qwen2.5:7b-instruct")

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
    {{"id": "lowercase_slug", "name": "Entity Name", "type": "PERSON|ORG|PLACE|EVENT|CONCEPT|OTHER", "description": "one sentence"}}
  ],
  "relations": [
    {{"subject_id": "slug_a", "predicate": "VERB_PHRASE", "object_id": "slug_b", "time_start": null, "time_end": null, "confidence": 0.9}}
  ]
}}

Rules:
- id: lowercase underscore slug derived from the entity name
  (e.g. "Doey the Dough" → "doey_the_dough"; "川普" → "chuan_pu"; "習近平" → "xi_jin_ping")
- CRITICAL: subject_id and object_id MUST be copied EXACTLY from an id in the "entities" list
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


async def extract_entities_and_relations(
    text: str,
    max_entities: int = 20,
    max_relations: int = 30,
) -> Dict[str, Any]:
    """
    Call LLM to extract entities and relations from the video summary.
    Falls back to empty lists on failure.
    """
    prompt = _EXTRACTION_PROMPT.format(text=text[:4000])

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{_LLM_BASE_URL}/chat/completions",
                json={
                    "model": _CHAT_MODEL,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            extracted = json.loads(content)

    except Exception as e:
        logger.warning(f"LLM extraction failed ({e}), returning empty results")
        extracted = {"entities": [], "relations": []}

    entities  = extracted.get("entities", [])[:max_entities]
    relations = extracted.get("relations", [])[:max_relations]

    for ent in entities:
        if not ent.get("id"):
            ent["id"] = _slugify(ent.get("name", ""))

    valid_ids = {e["id"] for e in entities}
    filtered_relations = []
    for r in relations:
        subj, obj = r.get("subject_id"), r.get("object_id")
        if subj in valid_ids and obj in valid_ids:
            filtered_relations.append(r)
        else:
            logger.warning(
                f"Dropping relation with mismatched IDs: "
                f"subject_id={subj!r} object_id={obj!r} "
                f"(valid ids: {sorted(valid_ids)})"
            )
    relations = filtered_relations

    logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations from summary")
    return {"entities": entities, "relations": relations}


def match_entities_to_moments(
    entities: List[Dict[str, Any]],
    moments: List[Dict[str, Any]],
    threshold: int = 75,
) -> List[Dict[str, Any]]:
    """
    Hybrid matching: for each entity, search its name in each moment's
    ASR + LVLM text using substring and fuzzy matching.

    Returns list of matches:
      [{entity_id, entity_name, moment_id, confidence, match_type}]
    """
    try:
        from rapidfuzz import fuzz
        use_fuzzy = True
    except ImportError:
        use_fuzzy = False
        logger.warning("rapidfuzz not installed, falling back to substring matching only")

    matches = []
    for moment in moments:
        search_text = extract_search_text(moment.get("raw_text", "")).lower()
        if not search_text:
            continue

        for entity in entities:
            name = entity["name"].lower()
            if not name:
                continue

            # Exact substring match (fastest, highest confidence)
            if name in search_text:
                matches.append({
                    "entity_id":   entity["id"],
                    "entity_name": entity["name"],
                    "moment_id":   moment["moment_id"],
                    "confidence":  1.0,
                    "match_type":  "exact",
                })
                continue

            # Fuzzy match as fallback
            if use_fuzzy:
                score = fuzz.partial_ratio(name, search_text)
                if score >= threshold:
                    matches.append({
                        "entity_id":   entity["id"],
                        "entity_name": entity["name"],
                        "moment_id":   moment["moment_id"],
                        "confidence":  round(score / 100, 2),
                        "match_type":  "fuzzy",
                    })

    logger.info(
        f"Hybrid match: {len(entities)} entities × {len(moments)} moments "
        f"→ {len(matches)} APPEARS_IN candidates"
    )
    return matches


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


async def extract_moment_entities_batched(
    moments: List[Dict[str, Any]],
    batch_size: int = 8,
) -> Dict[int, List[str]]:
    """
    For each moment with non-empty contextual_text, ask the LLM which named
    entities appear. Returns {moment_index: [entity_name, ...]}.

    Batched to amortize LLM call overhead — `batch_size` moments per call.
    Big O: O(ceil(N / batch_size)) LLM calls instead of O(N).
    """
    targets = [
        m for m in moments
        if (m.get("contextual_text") or "").strip()
    ]
    if not targets:
        return {}

    results: Dict[int, List[str]] = {}

    for start in range(0, len(targets), batch_size):
        chunk = targets[start:start + batch_size]
        moments_block = "\n\n".join(
            f"[{m['moment_index']}] {(m.get('contextual_text') or '')[:600]}"
            for m in chunk
        )
        prompt = _BATCH_MOMENT_PROMPT.format(moments_block=moments_block)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{_LLM_BASE_URL}/chat/completions",
                    json={
                        "model": _CHAT_MODEL,
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                data = json.loads(content)

            for entry in data.get("moments", []):
                idx = entry.get("moment_index")
                names = entry.get("entity_names", []) or []
                if isinstance(idx, int) and isinstance(names, list):
                    results[idx] = [str(n).strip() for n in names if str(n).strip()][:5]

        except Exception as e:
            logger.warning(f"Batched moment extraction failed for batch {start}: {e}")
            continue

    logger.info(
        f"Per-moment LLM extraction: {len(targets)} moments → "
        f"{sum(len(v) for v in results.values())} entity mentions, "
        f"{(len(targets) + batch_size - 1) // batch_size} LLM calls"
    )
    return results


def resolve_moment_entities_to_ids(
    moment_entity_names: Dict[int, List[str]],
    entities: List[Dict[str, Any]],
    moments: List[Dict[str, Any]],
) -> tuple:
    """
    Convert {moment_index: [entity_name]} into APPEARS_IN match dicts.

    Names matched against known entities (exact + substring).
    Names NOT in the known list are auto-created as moment-derived entities
    (PERSON type by default — most commonly missed items in news ASR are people).

    Returns (matches, new_entities):
      matches      — list of {entity_id, entity_name, moment_id, confidence, match_type}
      new_entities — list of new entity dicts to be upserted by the caller
    """
    name_to_id: Dict[str, str] = {}
    for e in entities:
        name = (e.get("name") or "").strip().lower()
        if name and e.get("id"):
            name_to_id[name] = e["id"]

    moment_index_to_id: Dict[int, str] = {
        m["moment_index"]: m["moment_id"] for m in moments
    }

    matches: List[Dict[str, Any]] = []
    new_entity_map: Dict[str, Dict[str, Any]] = {}  # lower_name → entity dict

    for moment_index, names in moment_entity_names.items():
        moment_id = moment_index_to_id.get(moment_index)
        if not moment_id:
            continue
        seen_in_moment: set = set()

        for name in names:
            lower = name.lower()
            entity_id = name_to_id.get(lower)

            if not entity_id:
                for known_name, known_id in name_to_id.items():
                    if lower in known_name or known_name in lower:
                        entity_id = known_id
                        break

            if not entity_id:
                # First time seeing this name — mint a new entity
                if lower not in new_entity_map:
                    new_id = _slugify(name)
                    new_entity_map[lower] = {
                        "id":          new_id,
                        "name":        name,
                        "type":        "PERSON",
                        "description": "",
                    }
                    name_to_id[lower] = new_id
                entity_id = new_entity_map[lower]["id"]

            if entity_id and entity_id not in seen_in_moment:
                seen_in_moment.add(entity_id)
                is_new = lower in new_entity_map
                matches.append({
                    "entity_id":   entity_id,
                    "entity_name": name,
                    "moment_id":   moment_id,
                    "confidence":  0.8 if is_new else 0.9,
                    "match_type":  "llm_moment_derived" if is_new else "llm_contextual",
                })

    new_entities = list(new_entity_map.values())
    logger.info(f"resolve_moment_entities: {len(matches)} matches, {len(new_entities)} new entities minted")
    return matches, new_entities


async def post_temporal_facts(
    source_id: str,
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    temporal_service_url: str,
    moment_id: Optional[str] = None,
) -> int:
    """
    Push relations that carry real-world time_start/time_end to temporal_model_service.
    Video-relative timestamps (seconds) are NOT posted — only ISO 8601 dates.

    moment_id (optional): if these relations were extracted from a specific moment's
    contextual_text, supply the moment_id so the temporal fact can be back-anchored
    to a precise position in the video. Default None = video-level fact.
    """
    temporal_relations = [
        r for r in relations
        if r.get("time_start") or r.get("time_end")
    ]
    if not temporal_relations:
        return 0

    entity_map = {e["id"]: e for e in entities}
    count = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        for rel in temporal_relations:
            subject  = entity_map.get(rel["subject_id"], {})
            obj      = entity_map.get(rel["object_id"], {})
            body = {
                "entity":              subject.get("name", rel["subject_id"]),
                "relation":            rel["predicate"],
                "value":               obj.get("name", rel["object_id"]),
                "time_start":          rel.get("time_start"),
                "time_end":            rel.get("time_end"),
                "confidence":          rel.get("confidence", 1.0),
                "source_document_id":  source_id,
                "entity_id":           rel["subject_id"],
                "moment_id":           moment_id,
            }
            try:
                resp = await client.post(f"{temporal_service_url}/temporal/facts", json=body)
                resp.raise_for_status()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to post temporal fact: {e}")

    logger.info(
        f"Posted {count}/{len(temporal_relations)} temporal facts for {source_id}"
        f"{f' (moment={moment_id})' if moment_id else ''}"
    )
    return count


# ---------------------------------------------------------------------------
# Per-moment temporal extraction — opt-in, gated by content heuristics
# ---------------------------------------------------------------------------

# Keywords that suggest a moment may contain real-world temporal references.
# We skip LLM extraction unless a moment's contextual_text matches at least one.
_TEMPORAL_KEYWORDS = (
    # Chinese
    "年", "月", "日", "世紀", "年代", "之前", "之後", "以前", "以後",
    "去年", "今年", "明年", "昨日", "昨天", "今天", "明天",
    "最近", "後來", "當時", "那時",
    # English
    "year", "month", "day", "century", "decade", "before", "after",
    "yesterday", "today", "tomorrow", "ago", "later", "recently",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    # Numeric year patterns are caught by regex below
)


def _has_temporal_signal(text: str) -> bool:
    """Cheap heuristic — does this text look like it might mention real-world time?"""
    if not text:
        return False
    if any(kw in text for kw in _TEMPORAL_KEYWORDS):
        return True
    # 4-digit year (1900-2099)
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        return True
    return False


async def extract_and_post_moment_temporal_facts(
    source_id: str,
    entities: List[Dict[str, Any]],
    moments: List[Dict[str, Any]],
    temporal_service_url: str,
    max_moments: int = 10,
) -> int:
    """
    For moments whose contextual_text mentions real-world time, run LLM
    relation extraction and POST any time-anchored relations to TKG.

    Caps at `max_moments` to keep LLM cost bounded — we use heuristic
    `_has_temporal_signal` to pick the most promising candidates.

    Returns total temporal facts posted.
    """
    candidates = [
        m for m in moments
        if _has_temporal_signal(m.get("contextual_text") or "")
    ][:max_moments]

    if not candidates:
        logger.info("No moments with temporal signals — skipping moment-level TKG extraction")
        return 0

    logger.info(
        f"Per-moment TKG: {len(candidates)} moments have temporal signals; running LLM extraction"
    )

    total = 0
    for m in candidates:
        try:
            extracted = await extract_entities_and_relations(
                text=m["contextual_text"],
                max_entities=5,
                max_relations=5,
            )
        except Exception as e:
            logger.warning(f"Moment {m['moment_id']} relation extraction failed: {e}")
            continue

        # merge moment-extracted entities into the supplied entity list (so name
        # lookup works in post_temporal_facts)
        merged_entities = list(entities)
        known_ids = {e["id"] for e in entities}
        for new_e in extracted.get("entities", []):
            if new_e.get("id") and new_e["id"] not in known_ids:
                merged_entities.append(new_e)
                known_ids.add(new_e["id"])

        posted = await post_temporal_facts(
            source_id=source_id,
            entities=merged_entities,
            relations=extracted.get("relations", []),
            temporal_service_url=temporal_service_url,
            moment_id=m["moment_id"],
        )
        total += posted

    logger.info(f"Per-moment TKG: posted {total} moment-anchored temporal facts")
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = slug[:64]
    return slug or str(uuid.uuid4())[:8]
