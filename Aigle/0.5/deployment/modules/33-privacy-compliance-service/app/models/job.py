"""GD-1 / VIE01-246: erasure job state.

Eight stores are fanned out to, plus Keycloak (Module 06), which is
handled before and after the fan-out rather than as a store status.
module_35 deletes the user's voice-clone speaker embeddings. module_34
deletes the user's voice recordings; it is `skipped` while GD_MODULE_34_URL
is empty, as module_35 is while GD_MODULE_35_URL is. module_23 (CM-5 /
VIE01-408) deletes the user's moderation_flags rows -- distinct from
guardrail_violations, which has no user column and is never erasable.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

# Keyed to match the ticket's store_statuses shape (module_XX).
STORE_KEYS = (
    "module_04", "module_25", "module_26", "module_03", "module_23", "module_02", "module_34", "module_35",
)


class StoreStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELETED = "deleted"
    FAILED = "failed"
    # A store whose module has no GDPR endpoint configured (GD_MODULE_34_URL
    # / GD_MODULE_35_URL empty) -- nothing exists there to delete. Used by
    # GD-5's consent-withdrawal cascade and GD-1's erasure fan-out; counts
    # as done when deciding whether a job is COMPLETED.
    SKIPPED = "skipped"


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL_FAILURE = "partial_failure"
    ALREADY_ERASED = "already_erased"
    # PRIV-X4 / VIE01-358: an erasure requested while the user is under a
    # legal hold. Nothing has been deleted; the daily expiry job sets it back
    # to pending and runs it once the hold ends.
    DEFERRED = "deferred"


class JobType(str, Enum):
    """GD-2 / VIE01-247: gdpr_erasure_jobs is shared between erasure (GD-1)
    and export (GD-2) jobs, distinguished by this column, per the ticket's
    explicit instruction to reuse the table rather than add a new one."""
    ERASURE = "erasure"
    EXPORT = "export"
    # GD-4 / VIE01-249: only large-account (or overrunning) DSARs ever get a
    # job row -- a normal DSAR is answered synchronously and stores nothing.
    DSAR = "dsar"
    # GD-5 / VIE01-250: partial erasure of one purpose's data after its
    # consent is withdrawn -- never touches the Keycloak account.
    CONSENT_WITHDRAWAL = "consent_withdrawal"


class ErasureJob(BaseModel):
    job_id: str
    user_id: str
    requested_at: datetime
    requested_by: str
    status: JobStatus
    store_statuses: dict[str, StoreStatus] = Field(default_factory=dict)
    completed_at: datetime | None = None
    notes: str | None = None
    estimated_completion_at: datetime | None = None


class ErasureSubmitResponse(BaseModel):
    job_id: str
    status: str
    estimated_completion_at: datetime | None = None
    # PRIV-X4 / VIE01-358: set when the erasure is DEFERRED by a legal hold.
    reason: str | None = None
    hold_until: date | None = None
    # The ticket's ["audit_logs_7yr"] shape, each entry also citing its basis:
    # "<data>_<retention>: <basis>", e.g. "gdpr_access_log_1yr: Accountability: ...".
    exception_fields: list[str] = Field(default_factory=list)


class ExportJob(BaseModel):
    job_id: str
    user_id: str
    requested_at: datetime
    requested_by: str
    status: JobStatus
    notes: str | None = None
    completed_at: datetime | None = None
    # The ticket's single download_url: the export's only ZIP, or part 1 of
    # a split export (download_urls lists every part, in order).
    download_url: str | None = None
    download_urls: list[str] | None = None
    size_bytes: int | None = None
    expires_at: datetime | None = None


class ExportSubmitResponse(BaseModel):
    job_id: str
    status: str


def initial_store_statuses() -> dict[str, StoreStatus]:
    return {key: StoreStatus.PENDING for key in STORE_KEYS}
