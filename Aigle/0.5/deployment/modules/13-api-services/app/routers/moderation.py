"""CM-5 / VIE01-408: admin review API for the automated flagging system.

GET .../flags, GET .../flags/{flag_id} and PATCH .../flags/{flag_id} proxy to
Module 23's internal /moderation/flags/* (app/routers/moderation_flags.py) --
the `moderation-admin` Keycloak role check happens here, before ever reaching
that internal network, same pattern as gdpr.py's `_is_gdpr_admin` gate.

POST (flag creation) is deliberately not exposed here: a flag is created by
whichever of modules 09-12 holds the content id, the instant its own
moderation verdict crosses the flagging threshold -- not by an admin action.

`asset_url` is this router's own enrichment (Module 23 has no notion of it) --
resolved via Module 04's presign-url, the same call search_service.py/
face_urls.py already make, so an admin can actually see the flagged asset
instead of just its asset_path/version_id (or, worse, just a moment/chunk
sub-reference that means nothing on its own). Every flag has asset_path set
now, regardless of content type, so this is attempted for every flag -- it
can still come back null on a presign failure, which must never break the
review response itself.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.services.storage_service import StorageService, User

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Moderation"])

# Mirrors module 23's app/models/moderation_flag.py, for Swagger typing only.
FlagStatus = Literal["flagged", "cleared", "removed"]


class ModerationFlagRecord(BaseModel):
    id: UUID
    branch_id: str
    version_id: str
    asset_path: str
    moment_id: str | None
    chunk_id: str | None
    category: str
    score: float | None
    model: str
    policy_version: str | None
    status: FlagStatus
    event_id: UUID | None = Field(None, description="Linked CF-4 audit_events row for this flag's creation")
    created_at: datetime
    updated_at: datetime
    asset_url: str | None = Field(None, description="Presigned download URL, resolved via Module 04 -- null if "
                                                       "the presign call itself failed")


class ModerationFlagListResponse(BaseModel):
    items: list[ModerationFlagRecord]
    total: int = Field(..., description="Total matching rows across all pages")
    page: int
    page_size: int


class ModerationFlagStatusUpdate(BaseModel):
    status: FlagStatus
    note: str | None = Field(None, max_length=2000, description="Optional reviewer note; audited, never stored in the flag row itself")


def _is_moderation_admin(payload: dict[str, Any]) -> bool:
    roles = payload.get("realm_access", {}).get("roles", [])
    return "moderation-admin" in roles


def _require_moderation_admin(current_user: dict[str, Any]) -> None:
    if not _is_moderation_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the moderation-admin role")


async def _forward(client: httpx.AsyncClient, settings: Settings, method: str, path: str, **kwargs) -> dict[str, Any]:
    try:
        resp = await client.request(
            method, f"{settings.guardrail_url.rstrip('/')}/moderation{path}",
            timeout=settings.guardrail_timeout, **kwargs,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                             detail="Could not connect to guardrail service.") from exc


async def _attach_asset_url(record: dict[str, Any], storage: StorageService, current_user: dict[str, Any]) -> None:
    """Best-effort: a presign failure must not break the admin review, it just means no
    preview link for this one flag. X-Branch-ID is the flag's own branch_id (whose data
    this actually is), not the admin's -- Module 04's /presign-url does no ownership
    check of its own (same trust boundary as search_service.py's resolve_version_urls()),
    so this is purely about sending the semantically correct value, not bypassing a check."""
    asset_path = record["asset_path"]
    user = User(user_id=current_user["sub"], branch_id=record["branch_id"])
    try:
        record["asset_url"] = await storage.get_presign_url(asset_path, record["version_id"], user)
    except Exception as exc:  # noqa: BLE001 -- a presign failure must not block the admin review
        logger.warning("[moderation] presign failed for %s: %s", asset_path, exc)


@router.get("/flags", response_model=ModerationFlagListResponse, summary="List moderation flags (moderation-admin only)")
async def list_flags(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status_filter: str | None = Query(None, alias="status"),
    category: str | None = None,
    branch_id: str | None = None,
    version_id: str | None = None,
    asset_path: str | None = None,
    moment_id: str | None = None,
    chunk_id: str | None = None,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage_service),
):
    _require_moderation_admin(current_user)
    params = {k: v for k, v in {
        "page": page, "page_size": page_size, "status": status_filter, "category": category,
        "branch_id": branch_id, "version_id": version_id, "asset_path": asset_path,
        "moment_id": moment_id, "chunk_id": chunk_id,
        "from": from_.isoformat() if from_ else None, "to": to.isoformat() if to else None,
    }.items() if v is not None}
    body = await _forward(client, settings, "GET", "/flags", params=params)
    for item in body.get("items", []):
        await _attach_asset_url(item, storage, current_user)
    return body


@router.get("/flags/{flag_id}", response_model=ModerationFlagRecord, summary="Get one moderation flag (moderation-admin only)")
async def get_flag(
    flag_id: UUID,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage_service),
):
    _require_moderation_admin(current_user)
    record = await _forward(client, settings, "GET", f"/flags/{flag_id}")
    await _attach_asset_url(record, storage, current_user)
    return record


@router.patch("/flags/{flag_id}", response_model=ModerationFlagRecord,
              summary="Transition a flag's review status (moderation-admin only)")
async def update_flag_status(
    flag_id: UUID,
    body: ModerationFlagStatusUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage_service),
):
    _require_moderation_admin(current_user)
    record = await _forward(client, settings, "PATCH", f"/flags/{flag_id}", json=body.model_dump(mode="json"))
    await _attach_asset_url(record, storage, current_user)
    return record
