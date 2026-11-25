"""raptor_audit — the shared CF-4 (VIE01-234) audit-event library.

See README.md in this directory for the schema and how a module adopts it.
"""
from .event import (
    DEFAULT_RETENTION_DAYS,
    AuditEvent,
    EventType,
    Outcome,
    build_event,
    hash_value,
)
from .postgres import (
    AUDIT_EVENTS_DDL,
    AuditWriteError,
    PostgresAuditWriter,
    insert_event,
)

__all__ = [
    "AUDIT_EVENTS_DDL",
    "DEFAULT_RETENTION_DAYS",
    "AuditEvent",
    "AuditWriteError",
    "EventType",
    "Outcome",
    "PostgresAuditWriter",
    "build_event",
    "hash_value",
    "insert_event",
]
