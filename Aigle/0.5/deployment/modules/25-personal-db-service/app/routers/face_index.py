"""Face trace / group write endpoint (dev plan v1.4 §4.5, "Trace-and-cluster write").

Internal: called by Module 11's face-detection service once per video. Thin router,
thick service -- the logic lives in services/identity_store.py.

Auth: branch_id from the `X-Branch-ID` header, same as the other index endpoints.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException

from app.models.face import FaceTracesRequest, FaceTracesResponse
from app.routers.index import _require_branch
from app.services.arcadedb_client import ArcadeDBClient
from app.services.identity_store import write_face_traces
from app.services.indexer import DatabaseNotInitializedError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/personal/index", tags=["face indexing"])
client = ArcadeDBClient()


@router.post("/face-traces", response_model=FaceTracesResponse, status_code=201,
             summary="Store one video's face traces and groups, and link groups to Identities")
async def face_traces_endpoint(
    req: FaceTracesRequest,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Replaces any earlier face write for this version_id. Each group is linked to the
    Identity of the other video's group or seed photo it matches; a match on another
    video's still-unassigned group makes the two one new pending Identity; no match
    leaves the group unassigned, on the video's pending list."""
    branch_id = _require_branch(x_branch_id)
    try:
        return await write_face_traces(client, branch_id, req)
    except DatabaseNotInitializedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- write_face_traces() spans ArcadeDB calls + identity-matching logic; converts any failure to 500
        logger.error(f"face_index failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
