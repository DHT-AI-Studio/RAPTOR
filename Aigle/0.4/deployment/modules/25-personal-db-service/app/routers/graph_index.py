"""Entity / Relationship / TemporalFact indexing endpoints (PA-5).

Mounted under the same /personal/index prefix as PA-4's document/moment routes.
Auth: branch_id from the X-Branch-ID header (gateway injects it from the JWT sub).
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.models.graph_index import (EntityIndexRequest, RelationshipIndexRequest,
                                     TemporalFactIndexRequest)
from app.services import graph_indexer
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import DatabaseNotInitializedError

router = APIRouter(prefix="/personal/index", tags=["graph indexing"])
client = ArcadeDBClient()


def _require_branch(x_branch_id: str | None) -> str:
    if not x_branch_id:
        raise HTTPException(status_code=400, detail="Missing X-Branch-ID header")
    return x_branch_id


class IndexResponse(BaseModel):
    rid: str
    status: str = "indexed"


async def _run(fn, x_branch_id, req):
    branch_id = _require_branch(x_branch_id)
    try:
        return IndexResponse(rid=await fn(client, branch_id, req))
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/entity", response_model=IndexResponse, status_code=201)
async def index_entity(req: EntityIndexRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    """Upsert an Entity vertex (idempotent on entity_id)."""
    return await _run(graph_indexer.index_entity, x_branch_id, req)


@router.post("/relationship", response_model=IndexResponse, status_code=201)
async def index_relationship(req: RelationshipIndexRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    """Create a RELATION edge between two entities -- deduped per source
    (from/to/relation/source_version_id together), not just from/to/relation,
    so two independent sources asserting the same fact each get their own
    edge instead of the second silently no-op'ing onto the first's. See
    graph_indexer.index_relationship()'s own docstring."""
    return await _run(graph_indexer.index_relationship, x_branch_id, req)


@router.post("/temporal-fact", response_model=IndexResponse, status_code=201)
async def index_temporal_fact(req: TemporalFactIndexRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    """Upsert a TemporalFact vertex and optionally link it to an entity/moment."""
    return await _run(graph_indexer.index_temporal_fact, x_branch_id, req)
