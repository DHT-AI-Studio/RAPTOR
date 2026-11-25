"""GD-3 / VIE01-248: retention policy models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RetentionPolicy(BaseModel):
    data_category: str
    retention_days: int
    auto_delete: bool
    legal_basis: str | None = None
    notes: str | None = None
    updated_at: datetime
    updated_by: str | None = None


class RetentionPolicyUpdate(BaseModel):
    """One row's worth of PUT /retention/policy changes -- data_category
    identifies which row; every other field is optional, so a caller only
    supplies what it's actually changing."""
    data_category: str
    retention_days: int | None = None
    auto_delete: bool | None = None
    legal_basis: str | None = None
    notes: str | None = None


class RetentionPolicyListResponse(BaseModel):
    policies: list[RetentionPolicy]


class RetentionDryRunResponse(BaseModel):
    """data_category -> count of rows/sessions that WOULD be deleted if
    enforcement ran right now. Read-only: no deletion happens. Only
    auto_delete=True categories with a real enforcement target are
    included (see retention_engine.ENFORCEABLE_CATEGORIES)."""
    counts: dict[str, int]
    as_of: datetime


class RetentionLogEntry(BaseModel):
    run_at: datetime
    data_category: str
    rows_deleted: int
    duration_ms: int
