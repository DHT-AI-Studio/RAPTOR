"""Publish an index-request onto the Kafka topic (PA-8 path — test / demo).

Simulates upstream workers 09-12: takes chunks / entities / relationships /
temporal facts, wraps them in the worker envelope, and publishes to
`personal-index-requests`. The PA-8 consumer then auto-creates the user's
ArcadeDB, embeds content with the service's BGE-M3, and indexes everything —
so the full Kafka -> ArcadeDB pipeline can be driven from Swagger and the
result searched, all in one place.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.kafka_producer import build_envelope, publish_index_request

router = APIRouter(prefix="/personal/publish", tags=["Publish (Kafka ingest)"])


class PublishIndexRequest(BaseModel):
    branch_id: str = Field(
        ..., description="Target user branch; the consumer routes to database user_<branch_id>.")
    version_id: Optional[str] = Field(
        None, description="Asset version id (optional; used for delete-by-version).")
    asset_path: Optional[str] = Field(
        None, description="Source asset path. With version_id and branch_id it forms the "
                          "event_id the consumer dedups on — two publishes with the same "
                          "three values are treated as the same event.")
    chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chunk entries. Each needs `id`, `type` (documents|videos|images|audios), "
                    "and `text`/`summary`. No embedding needed — the consumer embeds it.")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Entity entries: entity_id, name, type, source_chunk_id. "
                    "Repeat an entity with different source_chunk_id for cross-media mentions.")
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relationship entries: from_entity_id, to_entity_id, relation, confidence.")
    temporal_facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Temporal-fact entries: fact_id, relation, value, entity_id, time_start, chunk_id.")


@router.post("/index-request", summary="Publish an index-request to Kafka (drives the PA-8 consumer)")
async def publish_index(req: PublishIndexRequest) -> Dict[str, Any]:
    envelope = build_envelope(
        req.branch_id, req.version_id, req.chunks,
        req.entities, req.relationships, req.temporal_facts,
        asset_path=req.asset_path, source_module="25-publish-api")
    await publish_index_request(envelope)
    return {
        "status": "published",
        "topic": settings.kafka_topic,
        "event_id": envelope["event_id"],
        "branch_id": req.branch_id,
        "counts": {
            "chunks": len(req.chunks),
            "entities": len(req.entities),
            "relationships": len(req.relationships),
            "temporal_facts": len(req.temporal_facts),
        },
    }
