"""Per-user chunk indexing endpoint (PA-4, unified `Chunk`).

Auth: branch_id from the `X-Branch-ID` header (gateway injects it from the JWT sub).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.models.index import ChunkIndexRequest, CloneVersionRequest
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import (DatabaseNotInitializedError, clone_version, delete_by_version,
                                  index_chunk, set_status_by_version, version_exists)

router = APIRouter(prefix="/personal/index", tags=["chunk indexing"])
client = ArcadeDBClient()


def _require_branch(x_branch_id: str | None) -> str:
    if not x_branch_id:
        raise HTTPException(status_code=400, detail="Missing X-Branch-ID header")
    return x_branch_id


class IndexResponse(BaseModel):
    rid: str
    status: str = "indexed"


@router.post("/chunk", response_model=IndexResponse, status_code=201)
async def index_chunk_endpoint(
    req: ChunkIndexRequest,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Index (upsert) one content Chunk of any media type into the user's ArcadeDB."""
    branch_id = _require_branch(x_branch_id)
    try:
        rid = await index_chunk(client, branch_id, req)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return IndexResponse(rid=rid)


class ExistsResponse(BaseModel):
    version_id: str
    indexed: bool


@router.get("/{version_id}/exists", response_model=ExistsResponse,
           summary="Whether any Chunk has been indexed for this asset version")
async def version_exists_endpoint(
    version_id: str,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Module 04's search_sync.check_indexed() equivalent -- replaces its old
    Qdrant-only check (module 17 retired) so the "content already active
    elsewhere, skip re-analysis" decision actually reflects Module 25's own
    index instead of a store nothing writes to any more."""
    branch_id = _require_branch(x_branch_id)
    try:
        indexed = await version_exists(client, branch_id, version_id)
    except DatabaseNotInitializedError:
        return ExistsResponse(version_id=version_id, indexed=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return ExistsResponse(version_id=version_id, indexed=indexed)


@router.post("/clone", summary="Clone one asset version's index onto a new version_id/asset_path")
async def clone_version_endpoint(
    req: CloneVersionRequest,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Module 04's content-dedup optimization (client.py: identical MD5 found
    at a different, already-archived asset_path) -- clones the source
    version's Source/Chunks/RELATION(the source's own)/TemporalFacts onto
    the target instead of re-running analysis. See indexer.clone_version()'s
    own docstring for exactly what is/isn't duplicated (Entity is shared,
    not cloned; CO_OCCURS_WITH is neither cloned nor incremented, see that
    docstring for why)."""
    branch_id = _require_branch(x_branch_id)
    try:
        return await clone_version(client, branch_id, req)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


class StatusRequest(BaseModel):
    status: Literal["active", "archived"]


@router.post("/{version_id}/status", summary="Archive/reactivate everything indexed from one asset version")
async def set_status_endpoint(
    version_id: str,
    req: StatusRequest,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Module 25's equivalent of Module 20's POST /source/set_status --
    called by Module 04's search_sync fan-out on archive/reactivate, same
    trigger as delete_version_endpoint below is for delfile/destroy."""
    branch_id = _require_branch(x_branch_id)
    try:
        return await set_status_by_version(client, branch_id, version_id, req.status)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{version_id}", summary="Delete everything indexed from one asset version")
async def delete_version_endpoint(
    version_id: str,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Remove all Chunks/Source/relationships/temporal-facts for an asset version
    (idempotent — unknown version_id just returns zero counts)."""
    branch_id = _require_branch(x_branch_id)
    try:
        return await delete_by_version(client, branch_id, version_id)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
