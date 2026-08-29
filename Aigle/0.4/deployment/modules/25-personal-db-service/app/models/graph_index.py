"""Request models for graph/temporal indexing (PA-5) — unified `Chunk` model.

Entity is cross-media and carries NO source_id (0.3 search scopes via edges,
not source_id — verified). Content↔Entity linkage is the `MENTIONS(Chunk→Entity)`
edge, created when an entity index request names the chunk it was extracted from.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EntityIndexRequest(BaseModel):
    entity_id: str
    name: str
    type: str                                  # PERSON | ORG | PLACE | EVENT | CONCEPT | …
    description: Optional[str] = None
    # if set, create MENTIONS(chunk -> entity) and refresh mention_count
    source_chunk_id: Optional[str] = None
    modality: Optional[str] = None             # text | visual | asr (on the MENTIONS edge)


class RelationshipIndexRequest(BaseModel):
    from_entity_id: str
    to_entity_id: str
    relation: str
    confidence: Optional[float] = None
    source_version_id: Optional[str] = None    # provenance, for delete-by-source only


class TemporalFactIndexRequest(BaseModel):
    fact_id: str
    entity: str                                # entity name/label the fact is about
    relation: str
    value: str
    entity_id: Optional[str] = None            # link via HAS_TEMPORAL_FACT
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    confidence: Optional[float] = None
    chunk_id: Optional[str] = None             # link via OBSERVED_IN (which chunk stated it)
    source_version_id: Optional[str] = None    # provenance, for delete-by-source only
