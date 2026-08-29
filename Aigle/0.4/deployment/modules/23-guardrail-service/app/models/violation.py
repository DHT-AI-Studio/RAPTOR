"""Response models for the guardrail_violations audit log (GB-6)."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ViolationRecord(BaseModel):
    id: UUID
    policy_id: UUID | None
    module: str | None
    direction: str
    category: str | None
    action_taken: str
    request_id: str | None
    created_at: datetime


class ViolationListResponse(BaseModel):
    items: list[ViolationRecord]
    total: int = Field(..., description="Total matching rows across all pages")
    page: int
    page_size: int


class ViolationSummaryEntry(BaseModel):
    category: str | None
    action_taken: str
    count: int


class ViolationSummaryResponse(BaseModel):
    since: datetime
    until: datetime
    counts: list[ViolationSummaryEntry]
