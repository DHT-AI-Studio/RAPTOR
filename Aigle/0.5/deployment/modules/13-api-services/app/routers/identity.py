"""Identity gateway routes — `/api/{version}/identity/*` (dev plan v1.4 §4.5).

Proxies Module 25's `/personal/identity/*` one-to-one: people management, pending
persons, binding traces and groups, merge, and undo/redo of merges and deletes. The one route
that does more is the seed-photo upload, which runs the three-module pipeline of
1.4 §4.5 itself (see set_profile_image). Same rules as
the other Module 25 routes here: authenticated at the router mount, and the
caller's JWT `sub` is the only identity sent downstream (PersonalDBService), so no
route can reach another user's people.

Face images in the responses (avatars, representative frames) come back from
Module 25 as {image_path, image_version_id}; each also gets a presigned
`image_url` here, the way search results get their `asset_url`.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import httpx
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.routers.search import get_personal_db_service
from app.services.consent_check import ConsentCheck, get_consent_check
from app.services.face_urls import add_image_urls
from app.services.personal_db_service import PersonalDBService
from app.services.storage_service import StorageService, User

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["Identity"])


async def _call(
    svc: PersonalDBService,
    storage: StorageService,
    user: dict[str, Any],
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_id = user["sub"]
    await svc.ensure_database(user_id)
    data = await svc.identity(method, path, user_id, json=body, params=params)
    await add_image_urls(data, storage, user_id)
    return data


@router.get("/identities", summary="List people, most recently updated first")
async def list_identities(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status_: Literal["pending", "active", "merged", "deleted"] | None = Query(None, alias="status"),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Merged and deleted people are left out unless `status=merged` / `status=deleted` is asked for."""
    params = {"limit": limit, "offset": offset, **({"status": status_} if status_ else {})}
    return await _call(svc, storage, user, "GET", "/identities", params=params)


@router.post("/upload", summary="Create a person, or update one by identity_id")
async def upload_identity(
    body: dict[str, Any] = Body(..., examples=[{"name": "Peter", "aliases": {"zh-TW": "彼得"}}]),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Body: `name` (required), optional `aliases` ({locale: name}), `identity_type`
    (default `person`). With `identity_id`, updates that person -- or creates it under
    that id, so a retried request never makes a second one. A new person is `active`
    (someone named them on purpose). Returns the person."""
    return await _call(svc, storage, user, "POST", "/upload", body)


@router.get("/file/{version_id}/pending-persons", summary="A video's unassigned face groups and pending people")
async def pending_persons(
    version_id: str,
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """For one video (`version_id` from upload): `unassigned_groups` -- its face groups
    no person holds yet, each with its traces (`start_sec`/`end_sec`) and a
    representative frame -- and `pending_identities`: unconfirmed people holding one
    of its groups (e.g. the same face found in another video) or created from it with
    POST pending-person. Name and confirm a pending person with PATCH."""
    return await _call(svc, storage, user, "GET", f"/file/{version_id}/pending-persons")


@router.post("/file/{version_id}/pending-person", status_code=status.HTTP_201_CREATED,
             summary="Create a pending person from a video, optionally binding traces or a whole group")
async def pending_person(
    version_id: str,
    body: dict[str, Any] = Body(..., examples=[{"name": "", "trace_id": "..."}, {"name": "", "trace_ids": ["...", "..."]},
                                               {"name": "", "group_id": "..."}]),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Starts a `pending` person from this video (`name` may be empty), optionally
    binding right away -- at most one of:
    - `trace_id`: one of this video's traces (returned as `bind`);
    - `trace_ids`: a list of them (`binds`, one per trace);
    - `group_id`: a whole group from `unassigned_groups` (`bind_group`, as bind/group).
    Every ID is checked first: a bad one creates nothing (404)."""
    return await _call(svc, storage, user, "POST", f"/file/{version_id}/pending-person", body)


@router.post("/merge/{merge_id}/undo", summary="Undo a merge or delete within 24 hours")
async def undo(
    merge_id: str,
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """`merge_id` is the id returned by mergeinto or DELETE. Items changed again
    since are left alone and listed in `skipped`."""
    return await _call(svc, storage, user, "POST", f"/merge/{merge_id}/undo")


@router.post("/merge/{merge_id}/redo", summary="Redo an undone merge or delete within 24 hours")
async def redo(
    merge_id: str,
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Re-applies an undone merge or delete, within 24 hours of the original. Items
    changed again since are left alone and listed in `skipped`. 409 past the deadline
    or if it isn't currently undone."""
    return await _call(svc, storage, user, "POST", f"/merge/{merge_id}/redo")


@router.get("/{identity_id}", summary="One person")
async def get_identity(
    identity_id: str,
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """One person: `name`, `aliases`, `identity_status` (pending | active | merged |
    deleted), `source` (auto | manual), `avatar` (with a presigned `image_url`),
    `video_count`, `group_count`."""
    return await _call(svc, storage, user, "GET", f"/{identity_id}")


@router.patch("/{identity_id}", summary="Rename, change aliases, or confirm a pending person")
async def patch_identity(
    identity_id: str,
    body: dict[str, Any] = Body(..., examples=[{"name": "Peter Parker", "identity_status": "active"}]),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Body (all optional): `name`; `aliases` ({locale: name}, a null value removes
    that locale); `identity_status: "active"` confirms a pending person -- the only
    status change allowed here (`merged` comes only from mergeinto, `deleted` only from
    DELETE). 409 on a merged or deleted person. Returns the person."""
    return await _call(svc, storage, user, "PATCH", f"/{identity_id}", body)


@router.delete("/{identity_id}", summary="Delete a person; their face groups become unassigned. Undoable for 24 hours")
async def delete_identity(
    identity_id: str,
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """The person's face groups go back, unassigned, onto their videos' pending lists;
    their traces, frames and seed photo are kept. The person is marked `deleted`: left
    out of listings and People Search, and their seed photo stops matching new videos.
    Returns `history_id` (pass it to POST /merge/{history_id}/undo before
    `undo_deadline`, 24 hours) and `groups_unassigned`."""
    return await _call(svc, storage, user, "DELETE", f"/{identity_id}")


@router.post("/{identity_id}/bind/trace", summary="Move one trace under this person")
async def bind_trace(
    identity_id: str,
    body: dict[str, Any] = Body(..., examples=[{"version_id": "...", "trace_id": "..."}]),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Body: `version_id`, `trace_id`. Moves that trace into this person's group for
    its video (created if there's none); the group it left is recomputed, or removed
    if now empty. Returns `group_id`, `previous_group_id`, and any `removed_groups` /
    `removed_identities` (system-created pending people left empty)."""
    return await _call(svc, storage, user, "POST", f"/{identity_id}/bind/trace", body)


@router.post("/{identity_id}/bind/group", summary="Put a whole face group of a video under this person")
async def bind_group(
    identity_id: str,
    body: dict[str, Any] = Body(..., examples=[{"version_id": "...", "group_id": "..."}]),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Body: `version_id`, `group_id` (from pending-persons' `unassigned_groups`, or
    another person's group). Puts every trace of that group under this person in one
    call, instead of bind/trace once per trace; the group itself is kept. A group
    another person held moves here. Returns `previous_identity_id` and any
    `removed_identities` (system-created pending people left empty). 404 if the group
    isn't in that video."""
    return await _call(svc, storage, user, "POST", f"/{identity_id}/bind/group", body)


@router.post("/{identity_id}/unbind", summary="Take one trace out of this person")
async def unbind(
    identity_id: str,
    body: dict[str, Any] = Body(..., examples=[{"version_id": "...", "trace_id": "..."}]),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Body: `version_id`, `trace_id`. Takes the trace out of this person, leaving it in
    no group. 409 if the trace isn't bound to this person."""
    return await _call(svc, storage, user, "POST", f"/{identity_id}/unbind", body)


@router.get("/{identity_id}/traces", summary="Every segment this person appears in")
async def identity_traces(
    identity_id: str,
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Every continuous appearance of this person across all videos: `trace_id`,
    `version_id`, `start_sec`/`end_sec`, `representative` frame (with `image_url`), and
    `source_status` (the video's status; unregistered/archived videos are listed here
    but left out of People Search)."""
    return await _call(svc, storage, user, "GET", f"/{identity_id}/traces")


@router.post("/{identity_id}/mergeinto", summary="Merge this person into another; undoable for 24 hours")
async def mergeinto(
    identity_id: str,
    body: dict[str, Any] = Body(..., examples=[{"target_id": "...", "keep_source": True}]),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Body: `target_id`, `keep_source` (default true). Folds this person into the
    target: their face groups and seed photos move to the target, their aliases fill
    locales the target doesn't have (never overwriting), and the target takes their
    avatar if it has none. This person is kept as `merged` (or removed with
    `keep_source: false`). Returns `merge_id` -- POST /merge/{merge_id}/undo restores
    both people within `undo_deadline` (24 hours) -- `groups_moved` and the `target`.
    409 when merging into itself, or either person is merged/deleted."""
    return await _call(svc, storage, user, "POST", f"/{identity_id}/mergeinto", body)


async def _embed_face(client: httpx.AsyncClient, settings: Settings, photo: UploadFile, data: bytes) -> dict[str, Any]:
    """Module 11's POST /embed-face: {embedding, bbox, confidence, face_count}. Its 400
    (no face found, or not an image) reaches the caller as-is."""
    try:
        resp = await client.post(
            f"{settings.face_embed_url.rstrip('/')}/embed-face",
            files={"file": (photo.filename or "photo", data, photo.content_type or "application/octet-stream")},
            timeout=60.0,
        )
    except httpx.RequestError as exc:
        _logger.error("face embedding service unreachable: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Face embedding service unreachable")
    if resp.status_code == 400:
        raise HTTPException(status_code=400, detail=resp.json().get("detail", "No face detected"))
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Face embedding service error: {resp.text}")
    return resp.json()


@router.post("/{identity_id}/profile-image", summary="Set this person's photo; later videos will match it")
async def set_profile_image(
    identity_id: str,
    photo: UploadFile = File(..., description="A photo with this person's face (JPEG, PNG or WebP)"),
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    consent: ConsentCheck = Depends(get_consent_check),
    user: dict[str, Any] = Depends(get_current_user),
):
    """Seed-photo pipeline (1.4 §4.5): Module 11 embeds the photo (400 if it has no
    face) -> Module 04 stores it -> Module 25 records the embedding and makes the photo
    this person's avatar. The person is checked first and the photo stored only once
    a face is found, so a failed request leaves nothing behind. `photo.face_count` > 1
    means the largest, clearest face was used.

    Requires the caller's `face_detection` consent (GD-11): without it, 403
    PERMISSION_DENIED before the photo is read or embedded; 503 when it can't be checked."""
    user_id = user["sub"]
    await consent.require(user_id, "face_detection")
    data = await photo.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty photo")
    await svc.ensure_database(user_id)
    current = await svc.identity("GET", f"/{identity_id}", user_id)
    if current.get("identity_status") == "merged":
        raise HTTPException(status_code=409, detail=f"identity {identity_id} was merged into another identity")
    if current.get("identity_status") == "deleted":
        raise HTTPException(status_code=409, detail=f"identity {identity_id} was deleted")

    face = await _embed_face(client, settings, photo, data)
    stored = await storage.upload_identity_photo(identity_id, photo.filename, data, photo.content_type,
                                                 User(user_id=user_id, branch_id=user_id))
    result = await svc.identity("POST", f"/{identity_id}/profile-image", user_id, json={
        "image_path": stored["image_path"], "image_version_id": stored.get("image_version_id"),
        "embedding": face["embedding"]})
    await add_image_urls(result, storage, user_id)
    result["photo"] = {"face_count": face.get("face_count"), "confidence": face.get("confidence")}
    return result


@router.get("/{identity_id}/profile-image", summary="This person's picture, with a presigned URL")
async def get_profile_image(
    identity_id: str,
    svc: PersonalDBService = Depends(get_personal_db_service),
    storage: StorageService = Depends(get_storage_service),
    user: dict[str, Any] = Depends(get_current_user),
):
    """The person's avatar: their seed photo if one was set, otherwise a
    representative face frame -- with a presigned `image_url`."""
    return await _call(svc, storage, user, "GET", f"/{identity_id}/profile-image")
