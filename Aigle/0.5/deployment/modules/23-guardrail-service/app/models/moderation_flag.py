"""Request/response models for the CM-5 (VIE01-408) automated flagging system.

A `moderation_flags` row is distinct from `guardrail_violations` (app/models/violation.py):
the violations log is a write-once audit trail of every check call, with no notion of
"this specific asset/moment/chunk"; a flag is a mutable, admin-reviewable record tied to
one Module 04 asset (asset_path + version_id, always), optionally narrowed to one moment
or chunk within it, created by the caller (module 09/10/11/12 -- whichever one holds the
content id) once a moderation verdict (app/models/moderation.py) crosses the flagging
threshold.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

FlagStatus = Literal["flagged", "cleared", "removed"]
FLAG_STATUSES: tuple[FlagStatus, ...] = ("flagged", "cleared", "removed")


class ModerationFlagCreate(BaseModel):
    """POST /moderation/flags body. asset_path + version_id identify the real Module 04
    asset this content belongs to -- always required, so an admin reviewing the flag can
    actually see what it is (not just an opaque moment_id/chunk_id). moment_id/chunk_id
    are an optional, finer-grained sub-reference within that asset -- at most one of the
    two, never both; neither set means the flag is about the whole asset (e.g. an image)."""
    branch_id: str = Field(..., min_length=1, description="Tenant this content belongs to -- Module 25's search "
                                                            "exclusion is keyed on it, one ArcadeDB per branch")
    version_id: str = Field(..., min_length=1, description="The asset/source version this content belongs to, "
                                                             "paired with asset_path below")
    asset_path: str = Field(..., min_length=1, description="The underlying Module 04 asset's path -- without it "
                                                              "(and version_id above) a flag can't be traced back "
                                                              "to a real, queryable asset at all")
    moment_id: str | None = None
    chunk_id: str | None = None
    category: str = Field(..., min_length=1, description="Moderation taxonomy category, e.g. 'nsfw'")
    score: float | None = Field(None, ge=0.0, le=1.0)
    model: str = Field(..., min_length=1, description="Guard model that produced the verdict, e.g. 'shieldgemma-2'")
    policy_version: str | None = None

    @model_validator(mode="after")
    def _at_most_one_sub_reference(self) -> ModerationFlagCreate:
        if self.moment_id is not None and self.chunk_id is not None:
            raise ValueError("at most one of moment_id, chunk_id may be set")
        return self


class ModerationFlagStatusUpdate(BaseModel):
    """PATCH /moderation/flags/{id} body -- an admin review action."""
    status: FlagStatus
    note: str | None = Field(None, max_length=2000, description="Optional reviewer note; audited, never stored in the flag row itself")


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


class ModerationFlagListResponse(BaseModel):
    items: list[ModerationFlagRecord]
    total: int = Field(..., description="Total matching rows across all pages")
    page: int
    page_size: int
