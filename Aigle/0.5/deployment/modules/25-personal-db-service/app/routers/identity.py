"""Identity API (dev plan v1.4 §4.5): identity management, creation, seed photos,
pending persons, bind (a trace or a whole group)/unbind, merge, and undo/redo of merges and deletes.

Thin router, thick service -- this file only validates headers and maps errors to
status codes; the logic lives in services/identity_store.py.

Auth: branch_id from the `X-Branch-ID` header, same as the other endpoints. Fixed
paths (/identities, /upload, /file/..., /merge/...) are declared before /{identity_id}.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Response

from app.models.identity import (
    BindGroupIn,
    BindGroupResult,
    BindResult,
    BindTraceIn,
    DeleteIdentityResult,
    FaceImage,
    HistoryReplay,
    IdentityList,
    IdentityOut,
    IdentityPatch,
    IdentityTraces,
    IdentityUpload,
    MergeIn,
    MergeResult,
    PendingPersonIn,
    PendingPersonOut,
    PendingPersons,
    ProfileImageIn,
    UnbindIn,
)
from app.routers.index import _require_branch
from app.services import identity_store as store
from app.services.arcadedb_client import ArcadeDBClient
from app.services.indexer import DatabaseNotInitializedError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/personal/identity", tags=["identity"])
client = ArcadeDBClient()


async def _call(coro):
    try:
        return await coro
    except (DatabaseNotInitializedError, store.IdentityNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except store.IdentityConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- wraps identity_store/ArcadeDB calls; converts any failure to 500, logged
        logger.error(f"identity endpoint failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/identities", response_model=IdentityList, summary="List identities, most recently updated first")
async def list_identities_endpoint(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Literal["pending", "active", "merged", "deleted"] | None = None,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Merged and deleted identities are left out unless `status=merged` / `status=deleted` is asked for."""
    return await _call(store.list_identities(client, _require_branch(x_branch_id), limit, offset, status))


@router.post("/upload", response_model=IdentityOut, summary="Create an identity, or update one by identity_id")
async def upload_identity_endpoint(
    req: IdentityUpload,
    response: Response,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    identity, created = await _call(store.upload_identity(client, _require_branch(x_branch_id), req))
    response.status_code = 201 if created else 200
    return identity


@router.get("/file/{version_id}/pending-persons", response_model=PendingPersons,
            summary="A video's unassigned face groups and pending identities")
async def pending_persons_endpoint(version_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _call(store.pending_persons(client, _require_branch(x_branch_id), version_id))


@router.post("/file/{version_id}/pending-person", response_model=PendingPersonOut, status_code=201,
             summary="Create a pending identity from a video, optionally binding one of its traces")
async def pending_person_endpoint(
    version_id: str,
    req: PendingPersonIn,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    return await _call(store.create_pending_person(client, _require_branch(x_branch_id), version_id, req))


@router.post("/merge/{merge_id}/undo", response_model=HistoryReplay,
             summary="Undo a merge or delete within 24 hours; items changed since are skipped")
async def undo_endpoint(merge_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    """merge_id is the history_id returned by mergeinto or DELETE."""
    return await _call(store.replay_history(client, _require_branch(x_branch_id), merge_id, "undo"))


@router.post("/merge/{merge_id}/redo", response_model=HistoryReplay,
             summary="Redo an undone merge or delete within 24 hours; items changed since are skipped")
async def redo_endpoint(merge_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _call(store.replay_history(client, _require_branch(x_branch_id), merge_id, "redo"))


@router.get("/{identity_id}", response_model=IdentityOut)
async def get_identity_endpoint(identity_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _call(store.get_identity(client, _require_branch(x_branch_id), identity_id))


@router.patch("/{identity_id}", response_model=IdentityOut,
              summary="Rename, change aliases, or confirm a pending identity")
async def patch_identity_endpoint(
    identity_id: str,
    req: IdentityPatch,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    return await _call(store.patch_identity(client, _require_branch(x_branch_id), identity_id, req))


@router.delete("/{identity_id}", response_model=DeleteIdentityResult,
               summary="Delete an identity; its groups become unassigned. Undoable for 24 hours")
async def delete_identity_endpoint(identity_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _call(store.delete_identity(client, _require_branch(x_branch_id), identity_id))


@router.post("/{identity_id}/bind/trace", response_model=BindResult, summary="Move one trace under this identity")
async def bind_trace_endpoint(
    identity_id: str,
    req: BindTraceIn,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    return await _call(store.bind_trace(client, _require_branch(x_branch_id), identity_id, req.version_id, req.trace_id))


@router.post("/{identity_id}/bind/group", response_model=BindGroupResult,
             summary="Put a whole face group of a video under this identity")
async def bind_group_endpoint(
    identity_id: str,
    req: BindGroupIn,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    return await _call(store.bind_group(client, _require_branch(x_branch_id), identity_id, req.version_id, req.group_id))


@router.post("/{identity_id}/unbind", response_model=BindResult, summary="Take one trace out of this identity")
async def unbind_endpoint(
    identity_id: str,
    req: UnbindIn,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    return await _call(store.unbind_trace(client, _require_branch(x_branch_id), identity_id, req.version_id, req.trace_id))


@router.get("/{identity_id}/traces", response_model=IdentityTraces, summary="Every trace under this identity")
async def identity_traces_endpoint(identity_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _call(store.identity_traces(client, _require_branch(x_branch_id), identity_id))


@router.post("/{identity_id}/mergeinto", response_model=MergeResult,
             summary="Merge this identity into another; undoable for 24 hours")
async def mergeinto_endpoint(
    identity_id: str,
    req: MergeIn,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    return await _call(store.merge_identity(client, _require_branch(x_branch_id), identity_id,
                                            req.target_id, req.keep_source))


@router.post("/{identity_id}/profile-image", response_model=IdentityOut,
             summary="Set the seed photo (stored by the Gateway, embedded by Module 11) as this identity's avatar")
async def set_profile_image_endpoint(
    identity_id: str,
    req: ProfileImageIn,
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    return await _call(store.set_profile_image(client, _require_branch(x_branch_id), identity_id, req))


@router.get("/{identity_id}/profile-image", response_model=FaceImage, summary="This identity's picture")
async def get_profile_image_endpoint(identity_id: str, x_branch_id: str = Header(None, alias="X-Branch-ID")):
    return await _call(store.get_profile_image(client, _require_branch(x_branch_id), identity_id))
