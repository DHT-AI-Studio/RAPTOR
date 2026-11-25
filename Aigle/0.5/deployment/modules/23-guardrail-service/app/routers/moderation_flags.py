"""Router for CM-5 (VIE01-408): the automated flagging system.

POST /moderation/flags is called by whichever module holds the content id
(10/11/12) once its own moderation verdict (app/models/moderation.py) crosses
the flagging threshold -- module 23 itself has no asset/moment/chunk identity
of its own, it only classifies bytes it's handed.

GET /moderation/flags and PATCH /moderation/flags/{id} are the admin review
API's backend: Module 13's own /api/0.5/moderation/* proxies here, doing the
`moderation-admin` Keycloak role check itself (same pattern as
13-api-services/app/routers/gdpr.py's `_is_gdpr_admin` gate) before ever
reaching this internal network -- these two routes trust their caller, same
as every other inter-module call in this repo.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.moderation_flag import (
    FLAG_STATUSES,
    ModerationFlagCreate,
    ModerationFlagListResponse,
    ModerationFlagRecord,
    ModerationFlagStatusUpdate,
)
from app.services import moderation_flags

router = APIRouter(tags=["Moderation Flags"])


@router.post("/flags", response_model=ModerationFlagRecord, status_code=201,
            summary="Create a moderation flag for one asset/moment/chunk")
async def create_flag(flag: ModerationFlagCreate) -> ModerationFlagRecord:
    return await moderation_flags.create_flag(flag)


@router.get("/flags", response_model=ModerationFlagListResponse, summary="List moderation flags")
async def list_flags(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: str | None = Query(None, description=f"One of {FLAG_STATUSES}"),
    category: str | None = None,
    branch_id: str | None = None,
    version_id: str | None = None,
    asset_path: str | None = None,
    moment_id: str | None = None,
    chunk_id: str | None = None,
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
) -> ModerationFlagListResponse:
    if status is not None and status not in FLAG_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {FLAG_STATUSES}")
    rows, total = await moderation_flags.list_flags(
        page=page, page_size=page_size, status=status, category=category, branch_id=branch_id,
        version_id=version_id, asset_path=asset_path, moment_id=moment_id, chunk_id=chunk_id, from_ts=from_, to_ts=to,
    )
    return ModerationFlagListResponse(
        items=[ModerationFlagRecord(**dict(r)) for r in rows],
        total=total, page=page, page_size=page_size,
    )


@router.get("/flags/{flag_id}", response_model=ModerationFlagRecord, summary="Get one moderation flag")
async def get_flag(flag_id: UUID) -> ModerationFlagRecord:
    record = await moderation_flags.get_flag(flag_id)
    if record is None:
        raise HTTPException(status_code=404, detail="flag not found")
    return record


@router.patch("/flags/{flag_id}", response_model=ModerationFlagRecord,
              summary="Transition a flag's review status (cleared / removed)")
async def update_flag_status(flag_id: UUID, update: ModerationFlagStatusUpdate) -> ModerationFlagRecord:
    record = await moderation_flags.update_status(flag_id, update.status, update.note)
    if record is None:
        raise HTTPException(status_code=404, detail="flag not found")
    return record
