"""GD-4 / VIE01-249: Data Subject Access Request (DSAR) report shape.

Field names and nesting follow the ticket's JSON structure exactly. A count
this codebase has no real data behind (e.g. total_bytes -- Module 04
records no file size) is null, never a fabricated zero, and is listed with
the reason in `unavailable_fields` -- same omit-rather-than-fake policy as
GD-2's manifest.json `omitted` section. `stores_failed` names the stores
that should have answered but didn't (a transient outage, not a gap in the
codebase), so a partial report is never mistaken for a complete one.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.job import JobStatus


class IdentityCategory(BaseModel):
    email: str | None = None
    username: str | None = None
    created_at: str | None = None


class UploadedFilesCategory(BaseModel):
    count: int | None = None
    total_bytes: int | None = None
    file_types: list[str] | None = None


class IndexedContentCategory(BaseModel):
    documents: int | None = None
    entities: int | None = None
    moments: int | None = None
    pii_flagged: int | None = None


class MemoryCategory(BaseModel):
    sessions: int | None = None
    long_term_facts: int | None = None
    oldest_session: str | None = None


class BenchmarkCategory(BaseModel):
    schemas_owned: int | None = None
    runs: int | None = None


class SafetyViolationsCategory(BaseModel):
    count: int | None = None


class DataCategories(BaseModel):
    identity: IdentityCategory = Field(default_factory=IdentityCategory)
    uploaded_files: UploadedFilesCategory = Field(default_factory=UploadedFilesCategory)
    indexed_content: IndexedContentCategory = Field(default_factory=IndexedContentCategory)
    memory: MemoryCategory = Field(default_factory=MemoryCategory)
    benchmark: BenchmarkCategory = Field(default_factory=BenchmarkCategory)
    safety_violations: SafetyViolationsCategory = Field(default_factory=SafetyViolationsCategory)


class ConsentStatus(BaseModel):
    """GD-5's four purposes (the ticket's example shows the first two), plus MTR-24's face_detection
    and GD-11's voice_clone.
    Null only when there's no Keycloak account to read them from."""
    memory: bool | None = None
    analytics: bool | None = None
    benchmark: bool | None = None
    voice_recording: bool | None = None
    face_detection: bool | None = None
    voice_clone: bool | None = None


class RetentionSummary(BaseModel):
    files: str
    memory: str
    violations: str


class Rights(BaseModel):
    erasure_endpoint: str
    export_endpoint: str
    consent_endpoint: str | None = None


class DSARReport(BaseModel):
    user_id: str
    generated_at: datetime
    data_categories: DataCategories
    consent_status: ConsentStatus
    retention_summary: RetentionSummary
    ccpa_category: str
    ccpa_source: str
    ccpa_sells_data: bool
    rights: Rights
    # Dotted field path -> why this deployment can't back it.
    unavailable_fields: dict[str, str] = Field(default_factory=dict)
    stores_failed: list[str] = Field(default_factory=list)


class DSARPendingResponse(BaseModel):
    job_id: str
    status: str


class DSARJob(BaseModel):
    job_id: str
    user_id: str
    requested_at: datetime
    requested_by: str
    status: JobStatus
    completed_at: datetime | None = None
    notes: str | None = None
    # The DSARReport itself once completed; null while pending/in_progress,
    # and again after dsar_result_ttl_hours (see config.py).
    result: dict | None = None
    expires_at: datetime | None = None
