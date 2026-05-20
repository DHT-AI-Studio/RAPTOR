# TKG models, helpers, and write functions.
# Used by main.py (API endpoints) and pipeline.py (direct ingest writes).
# Requires the neo4j AsyncDriver — callers pass it in; no global state here.

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("graph_service.tkg")

_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434/v1")
_CHAT_MODEL   = os.getenv("CHAT_MODEL_NAME", "qwen2.5:7b")

_EXTRACT_PROMPT = """\
Extract every temporal fact from the text. A temporal fact has:
- entity (subject)
- relation (uppercase verb phrase)
- value (object)
- time_start / time_end (ISO 8601 string or null if not stated)
- confidence (0..1)

Return ONLY valid JSON (no markdown):
{{
  "facts": [
    {{"entity": "...", "relation": "...", "value": "...",
      "time_start": "2024-03-01" | null, "time_end": null, "confidence": 0.9}}
  ]
}}

Only include facts that have at least a time_start.
Cap at {max_facts} facts.

Text:
{text}
"""

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TemporalFactIn(BaseModel):
    entity: str = Field(..., description="Subject entity name")
    relation: str = Field(..., description="Predicate / relation type")
    value: str = Field(..., description="Object / value")
    time_start: Optional[str] = Field(None, description="ISO 8601 or null")
    time_end: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source_document_id: Optional[str] = None
    entity_id: Optional[str] = Field(None, description="Neo4j Entity.id")
    moment_id: Optional[str] = Field(None, description="Neo4j Moment.id")


class TemporalFactOut(BaseModel):
    id: str
    entity: str
    relation: str
    value: str
    time_start: Optional[str]
    time_end: Optional[str]
    confidence: Optional[float]
    source_document_id: Optional[str]
    entity_id: Optional[str]
    moment_id: Optional[str]
    created_at: Optional[str]


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Free-form text to extract temporal facts from")
    source_document_id: Optional[str] = None
    moment_id: Optional[str] = None
    entity_id: Optional[str] = None
    max_facts: int = Field(8, ge=1, le=30)


class ExtractResponse(BaseModel):
    extracted: int
    written: int
    facts: List[TemporalFactOut]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_json_safe(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_json_safe(val) for k, val in v.items()}
    if hasattr(v, "isoformat") and callable(v.isoformat):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)


def _stable_fact_id(entity_id: str, relation: str, value: str, time_start: Optional[str]) -> str:
    raw = f"{entity_id or ''}|{relation}|{value}|{time_start or ''}"
    return f"tf-{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _serialize_tf(node) -> Dict[str, Any]:
    if node is None:
        return {}
    props = dict(node.items())

    def _str(k):
        v = props.get(k)
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            try:
                return v.isoformat()
            except Exception:
                return str(v)
        return str(v) if not isinstance(v, str) else v

    return {
        "id":                 props.get("id"),
        "entity":             props.get("entity"),
        "relation":           props.get("relation"),
        "value":              props.get("value"),
        "time_start":         _str("time_start"),
        "time_end":           _str("time_end"),
        "confidence":         props.get("confidence"),
        "source_document_id": props.get("source_document_id"),
        "entity_id":          props.get("entity_id"),
        "moment_id":          props.get("moment_id"),
        "created_at":         _str("created_at"),
    }


# ---------------------------------------------------------------------------
# Core write
# ---------------------------------------------------------------------------

_UPSERT_CYPHER = """
MERGE (tf:TemporalFact {id: $tf_id})
ON CREATE SET
    tf.entity              = $entity,
    tf.entity_id           = $entity_id,
    tf.relation            = $relation,
    tf.value               = $value,
    tf.time_start          = $time_start,
    tf.time_end            = $time_end,
    tf.confidence          = $confidence,
    tf.source_document_id  = $source_document_id,
    tf.moment_id           = $moment_id,
    tf.created_at          = datetime()
ON MATCH SET
    tf.confidence          = $confidence,
    tf.time_end            = COALESCE($time_end, tf.time_end),
    tf.updated_at          = datetime()
WITH tf
OPTIONAL MATCH (e:Entity {id: $entity_id})
FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
    MERGE (e)-[:HAS_TEMPORAL_FACT]->(tf)
)
WITH tf
OPTIONAL MATCH (m:Moment {id: $moment_id})
FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
    MERGE (tf)-[:OBSERVED_IN]->(m)
)
RETURN tf
"""


async def upsert_fact(driver, fact: TemporalFactIn) -> Dict[str, Any]:
    tf_id = _stable_fact_id(
        fact.entity_id or fact.entity, fact.relation, fact.value, fact.time_start
    )
    async with driver.session() as session:
        result = await session.run(
            _UPSERT_CYPHER,
            tf_id=tf_id, entity=fact.entity, entity_id=fact.entity_id,
            relation=fact.relation, value=fact.value, time_start=fact.time_start,
            time_end=fact.time_end, confidence=fact.confidence,
            source_document_id=fact.source_document_id, moment_id=fact.moment_id,
        )
        rec = await result.single()
        if not rec:
            raise RuntimeError("Neo4j returned no record for upsert")
        return _serialize_tf(rec["tf"])


# ---------------------------------------------------------------------------
# Pipeline helpers — direct writes used by ingest pipeline
# ---------------------------------------------------------------------------

async def write_relations_as_temporal_facts(
    driver,
    source_id: str,
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    moment_id: Optional[str] = None,
) -> int:
    """Write LLM-extracted relations that carry real-world timestamps as TemporalFacts."""
    temporal_relations = [r for r in relations if r.get("time_start") or r.get("time_end")]
    if not temporal_relations:
        return 0

    entity_map = {e["id"]: e for e in entities}
    count = 0
    for rel in temporal_relations:
        subject = entity_map.get(rel["subject_id"], {})
        obj     = entity_map.get(rel["object_id"], {})
        fact = TemporalFactIn(
            entity=subject.get("name", rel["subject_id"]),
            relation=rel["predicate"],
            value=obj.get("name", rel["object_id"]),
            time_start=rel.get("time_start"),
            time_end=rel.get("time_end"),
            confidence=float(rel.get("confidence", 1.0)),
            source_document_id=source_id,
            entity_id=rel["subject_id"],
            moment_id=moment_id,
        )
        try:
            await upsert_fact(driver, fact)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to upsert temporal fact: {e}")
    return count


_TEMPORAL_KEYWORDS = (
    "年", "月", "日", "世紀", "年代", "之前", "之後", "以前", "以後",
    "去年", "今年", "明年", "昨日", "昨天", "今天", "明天", "最近", "後來", "當時", "那時",
    "year", "month", "day", "century", "decade", "before", "after",
    "yesterday", "today", "tomorrow", "ago", "later", "recently",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _has_temporal_signal(text: str) -> bool:
    if not text:
        return False
    if any(kw in text for kw in _TEMPORAL_KEYWORDS):
        return True
    return bool(re.search(r"\b(?:19|20)\d{2}\b", text))


async def write_moment_temporal_facts(
    driver,
    source_id: str,
    entities: List[Dict[str, Any]],
    moments: List[Dict[str, Any]],
    max_moments: int = 10,
) -> int:
    """For moments with temporal signals, run LLM extraction and write TemporalFacts directly."""
    from graph_builder import extract_entities_and_relations

    candidates = [
        m for m in moments
        if _has_temporal_signal(m.get("contextual_text") or "")
    ][:max_moments]

    if not candidates:
        return 0

    total = 0
    for m in candidates:
        try:
            extracted = await extract_entities_and_relations(
                text=m["contextual_text"], max_entities=5, max_relations=5,
            )
        except Exception as e:
            logger.warning(f"Moment {m['moment_id']} extraction failed: {e}")
            continue

        merged_entities = list(entities)
        known_ids = {e["id"] for e in entities}
        for new_e in extracted.get("entities", []):
            if new_e.get("id") and new_e["id"] not in known_ids:
                merged_entities.append(new_e)
                known_ids.add(new_e["id"])

        posted = await write_relations_as_temporal_facts(
            driver,
            source_id=source_id,
            entities=merged_entities,
            relations=extracted.get("relations", []),
            moment_id=m["moment_id"],
        )
        total += posted

    return total


# ---------------------------------------------------------------------------
# LLM extraction — for POST /temporal/extract endpoint
# ---------------------------------------------------------------------------

async def llm_extract_temporal_facts(text: str, max_facts: int = 8) -> List[Dict[str, Any]]:
    prompt = _EXTRACT_PROMPT.format(text=text[:4000], max_facts=max_facts)
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{_LLM_BASE_URL}/chat/completions",
                json={
                    "model": _CHAT_MODEL,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": "You are a JSON-only information extractor."},
                        {"role": "user",   "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            return (json.loads(content).get("facts") or [])[:max_facts]
    except Exception as e:
        logger.warning(f"LLM extract failed: {e}")
        return []
