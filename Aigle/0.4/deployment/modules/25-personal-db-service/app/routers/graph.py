"""Graph / TKG / GraphRAG search endpoints (PA-7).

Scoped to the caller's ArcadeDB database via the X-Branch-ID header:
  POST /personal/search/graph      — entity-graph traversal ({entities, edges, paths})
  POST /personal/search/tkg        — temporal facts, sorted by confidence
  POST /personal/search/graphrag   — vector + MENTIONS entity context
  GET  /personal/graph/entities    — paginated entity list
  GET  /personal/graph/entities/{name} — entity + in/out RELATION edges
  GET  /personal/graph/entities/{name}/co-occurring — same-moment CO_OCCURS_WITH neighbors
  POST /personal/graph/query       — raw read-only SELECT (power users)
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query

from app.models.graph_search import (GraphRAGRequest, GraphRAGResponse,
                                      GraphSearchRequest, GraphSearchResponse,
                                      RawGraphQueryRequest, RawGraphQueryResponse,
                                      TKGRequest, TKGResponse)
from app.services import searcher
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import DatabaseNotInitializedError

search_router = APIRouter(prefix="/personal/search", tags=["graph search"])
graph_router = APIRouter(prefix="/personal/graph", tags=["graph"])
client = ArcadeDBClient()


def _require_branch(x_branch_id: str | None) -> str:
    if not x_branch_id:
        raise HTTPException(status_code=400, detail="Missing X-Branch-ID header")
    return x_branch_id


async def _guard(coro):
    """Map service errors to HTTP: uninitialized DB -> 404, bad input -> 400."""
    try:
        return await coro
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@search_router.post("/graph", response_model=GraphSearchResponse,
                    summary="Entity-graph traversal (entities, edges, paths)")
async def graph(req: GraphSearchRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    branch_id = _require_branch(x_branch_id)
    return await _guard(searcher.graph_search(client, branch_id, req))


@search_router.post("/tkg", response_model=TKGResponse,
                    summary="Temporal knowledge graph query (facts by confidence)")
async def tkg(req: TKGRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    branch_id = _require_branch(x_branch_id)
    return await _guard(searcher.tkg_search(client, branch_id, req))


@search_router.post("/graphrag", response_model=GraphRAGResponse,
                    summary="GraphRAG: vector + linked entity context")
async def graphrag(req: GraphRAGRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    branch_id = _require_branch(x_branch_id)
    return await _guard(searcher.graphrag_search(client, branch_id, req))


@graph_router.get("/entities", summary="List entities (paginated) with type + mention count")
async def entities(type: str | None = Query(None), limit: int = Query(50), offset: int = Query(0),
                   x_branch_id: str = Header(None, alias="X-Branch-ID")):
    branch_id = _require_branch(x_branch_id)
    return await _guard(searcher.list_entities(
        client, branch_id, type=type, limit=limit, offset=offset))


@graph_router.get("/entities/{name}", summary="Entity + outgoing/incoming RELATION edges")
async def entity(name: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    branch_id = _require_branch(x_branch_id)
    result = await _guard(searcher.get_entity(client, branch_id, name))
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity not found: {name}")
    return result


@graph_router.get("/entities/{name}/co-occurring",
                  summary="Entities that most often co-occur with this one (same-moment CO_OCCURS_WITH)")
async def co_occurring(name: str, limit: int = Query(20, ge=1, le=200),
                       x_branch_id: str = Header(None, alias="X-Branch-ID")):
    branch_id = _require_branch(x_branch_id)
    result = await _guard(searcher.co_occurring_entities(client, branch_id, name, limit=limit))
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity not found: {name}")
    return result


@graph_router.post("/query", response_model=RawGraphQueryResponse,
                   summary="Raw ArcadeDB SQL (read-only SELECT only)")
async def raw_query(req: RawGraphQueryRequest, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    branch_id = _require_branch(x_branch_id)
    rows = await _guard(searcher.raw_graph_query(client, branch_id, req.query))
    return RawGraphQueryResponse(result=rows)
