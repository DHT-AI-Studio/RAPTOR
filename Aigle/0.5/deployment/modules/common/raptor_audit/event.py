"""The CF-4 (VIE01-234) audit event: one normalized, privacy-minimized record
shape every v0.5 initiative (guardrail, GDPR/CCPA, gRPC, audio) publishes.

Privacy minimization is enforced here, not left to callers: `build_event()`
takes the raw `actor_id` (Keycloak sub / user_id) and `content` only to hash
them, and `AuditEvent` forbids extra fields, so neither can end up in a
stored record verbatim.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "guardrail.decision", "gdpr.erasure", "grpc.call", "audio.transcript",
    # V05-10 / VIE01-364: Module 33's other privacy actions.
    "gdpr.export", "gdpr.access", "gdpr.rectification", "gdpr.restriction",
    "gdpr.objection", "gdpr.consent", "gdpr.legal_hold",
    # CM-5 / VIE01-408: a moderation_flags row's lifecycle transition
    # (created, cleared, removed) -- which one is in metadata["status"],
    # same convention as guardrail.decision's non-CF-4 actions (metadata["decision"]).
    "moderation.flag",
]
Outcome = Literal["allow", "block", "redact", "erase", "error"]

DEFAULT_RETENTION_DAYS = 365


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_module: int = Field(..., ge=1, description="Emitting module number, e.g. 23")
    source_service: str = Field(..., min_length=1, description="Emitting service name, e.g. guardrail-service")
    event_type: EventType
    actor_id_hash: str | None = Field(None, description="SHA-256 hex of the user id — never the raw id")
    resource_id: str | None = Field(None, description="session_id or request_id the event is about")
    outcome: Outcome
    policy_version: str | None = None
    content_hash: str | None = Field(None, description="SHA-256 hex of the screened text — never raw content")
    metadata: dict[str, Any] = Field(default_factory=dict)
    retention_days: int = Field(DEFAULT_RETENTION_DAYS, ge=1)


def hash_value(value: str | None) -> str | None:
    """SHA-256 hex digest of `value` (UTF-8); None stays None."""
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_event(*, source_module: int, source_service: str, event_type: EventType, outcome: Outcome,
                actor_id: str | None = None, resource_id: str | None = None, content: str | None = None,
                policy_version: str | None = None, metadata: dict[str, Any] | None = None,
                retention_days: int = DEFAULT_RETENTION_DAYS) -> AuditEvent:
    """Build an event; `actor_id` and `content` are only ever stored as hashes."""
    return AuditEvent(
        source_module=source_module, source_service=source_service, event_type=event_type,
        actor_id_hash=hash_value(actor_id), resource_id=resource_id, outcome=outcome,
        policy_version=policy_version, content_hash=hash_value(content),
        metadata=dict(metadata or {}), retention_days=retention_days,
    )
