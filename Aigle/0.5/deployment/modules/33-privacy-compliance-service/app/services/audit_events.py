"""V05-10 / VIE01-364: Module 33's CF-4 audit events.

Every finished erasure job -- a full GD-1 erasure or a GD-5 consent-
withdrawal partial erasure -- leaves one `gdpr.erasure` event in this
service's own audit_events table, through the shared raptor_audit library
(CF-4, deployment/modules/common/raptor_audit). Privacy-minimized by that
library: the data subject's id is stored only as actor_id_hash, and the
requester (when someone other than the subject, i.e. a gdpr-admin) only as
metadata.requested_by_hash. Unlike gdpr_erasure_jobs, the event carries no
raw user id, so it can outlive the job row as proof the erasure ran.

The other privacy actions -- export, access (DSAR), rectification,
restriction, objection, consent change, legal hold -- each leave one event
of their own type (privacy_event). The shared outcomes are guardrail words,
so these use `allow` for "the action was carried out" and `error` for "it
failed"; what was done is in metadata.action. They are written fail-open:
each action already has its own record (job row, consent history,
restriction/objection log, legal hold row), so a lost event is logged, never
a reason to undo or refuse the action. Free text (a restriction reason, a
hold's reason/authority, a rectified value) is never put in an event.
"""
from __future__ import annotations

from typing import Any, Literal

import asyncpg
from raptor_audit import (
    AuditEvent,
    Outcome,
    PostgresAuditWriter,
    build_event,
    hash_value,
)

from app.core.config import settings
from app.models.job import StoreStatus

SOURCE_MODULE = 33
SOURCE_SERVICE = "privacy-compliance-service"


def erasure_event(
    *, job_id: str, user_id: str, requested_by: str, job_type: str, outcome: Outcome,
    store_statuses: dict[str, StoreStatus], purpose: str | None = None,
) -> AuditEvent:
    """`outcome` is "erase" once every store is deleted or skipped, "error"
    while any store is still failed (PARTIAL_FAILURE)."""
    metadata: dict[str, Any] = {
        "job_type": job_type,
        "store_statuses": {k: StoreStatus(v).value for k, v in store_statuses.items()},
        "requested_by_subject": requested_by == user_id,
    }
    if requested_by != user_id:
        metadata["requested_by_hash"] = hash_value(requested_by)
    if purpose is not None:
        metadata["purpose"] = purpose
    return build_event(
        source_module=SOURCE_MODULE, source_service=SOURCE_SERVICE, event_type="gdpr.erasure",
        outcome=outcome, actor_id=user_id, resource_id=job_id, metadata=metadata,
        retention_days=settings.audit_retention_days,
    )


PrivacyEventType = Literal[
    "gdpr.export", "gdpr.access", "gdpr.rectification", "gdpr.restriction",
    "gdpr.objection", "gdpr.consent", "gdpr.legal_hold",
]


def privacy_event(
    event_type: PrivacyEventType, *, user_id: str, action: str, ok: bool,
    requested_by: str | None = None, resource_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """One finished privacy action. `requested_by` None means the system
    (e.g. the daily legal-hold expiry), not a person."""
    meta: dict[str, Any] = {"action": action, **(metadata or {})}
    if requested_by is None:
        meta["requested_by_system"] = True
    else:
        meta["requested_by_subject"] = requested_by == user_id
        if requested_by != user_id:
            meta["requested_by_hash"] = hash_value(requested_by)
    return build_event(
        source_module=SOURCE_MODULE, source_service=SOURCE_SERVICE, event_type=event_type,
        outcome="allow" if ok else "error", actor_id=user_id, resource_id=resource_id, metadata=meta,
        retention_days=settings.audit_retention_days,
    )


def audit_writer(pool: asyncpg.Pool) -> PostgresAuditWriter:
    async def _get_pool() -> asyncpg.Pool:
        return pool
    return PostgresAuditWriter(_get_pool)
