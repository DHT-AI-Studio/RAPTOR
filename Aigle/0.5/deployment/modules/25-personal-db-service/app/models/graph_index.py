"""Request models for graph/temporal indexing (PA-5) — unified `Chunk` model.

Entity is cross-media and carries NO source_id (0.3 search scopes via edges,
not source_id — verified). Content↔Entity linkage is the `MENTIONS(Chunk→Entity)`
edge, created when an entity index request names the chunk it was extracted from.
"""
from __future__ import annotations

from pydantic import BaseModel


class EntityIndexRequest(BaseModel):
    entity_id: str
    name: str
    type: str                                  # PERSON | ORG | PLACE | EVENT | CONCEPT | …
    description: str | None = None
    # if set, create MENTIONS(chunk -> entity) and refresh mention_count
    source_chunk_id: str | None = None
    modality: str | None = None             # text | visual | asr (on the MENTIONS edge)


class RelationshipIndexRequest(BaseModel):
    from_entity_id: str
    to_entity_id: str
    relation: str
    confidence: float | None = None
    source_version_id: str | None = None    # provenance, for delete-by-source only


class TemporalFactIndexRequest(BaseModel):
    fact_id: str
    entity: str                                # entity name/label the fact is about
    relation: str
    value: str
    entity_id: str | None = None            # link via HAS_TEMPORAL_FACT
    time_start: str | None = None
    time_end: str | None = None
    confidence: float | None = None
    chunk_id: str | None = None             # link via OBSERVED_IN (which chunk stated it)
    source_version_id: str | None = None    # provenance, for delete-by-source only
    # P12 (temporal plan §5.6/§5.8): the evidence an extracted fact rests on. Optional --
    # the v1 writers (Kafka temporal_facts entries, HTTP) supply none of them.
    date_text: str | None = None            # the date as the source wrote it
    source_sentence: str | None = None
    evidence_offsets: str | None = None     # JSON: {"date": [a, b], "entity": [a, b], "value": [a, b]} in the chunk text
    extent: str | None = None               # point | span
    anchor: str | None = None               # how the date was read: explicit | relative | anchored | since
    extractor_version: str | None = None
    links: str | None = None                # JSON [{relation, other}]: what the evidence says it is before / after / during
    media_start_ms: int | None = None       # where in a recording the claim was said (video/audio segments)
    media_end_ms: int | None = None
