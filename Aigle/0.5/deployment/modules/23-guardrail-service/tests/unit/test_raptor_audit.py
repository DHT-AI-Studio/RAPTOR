"""The shared CF-4 audit-event library (deployment/modules/common/raptor_audit),
tested here because module 23 is its first adopter and CI only runs a
module's own tests/unit/."""
from __future__ import annotations

import hashlib
import json
import logging
from uuid import UUID

import pytest
from pydantic import ValidationError
from raptor_audit import (
    AUDIT_EVENTS_DDL,
    AuditEvent,
    AuditWriteError,
    PostgresAuditWriter,
    build_event,
    hash_value,
    insert_event,
)

pytestmark = pytest.mark.asyncio

_RAW_CONTENT = "my phone number is 0912-345-678"
_RAW_USER = "keycloak-sub-6f1c"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event(**overrides) -> AuditEvent:
    kwargs = {"source_module": 23, "source_service": "guardrail-service", "event_type": "guardrail.decision",
              "outcome": "block", "actor_id": _RAW_USER, "resource_id": "req-1", "content": _RAW_CONTENT,
              "policy_version": "1.0.0", "metadata": {"decision": "block"}}
    kwargs.update(overrides)
    return build_event(**kwargs)


# ── schema & hashing ─────────────────────────────────────────────────────────────

async def test_raw_content_and_actor_never_appear_only_their_sha256():
    event = _event()
    dumped = event.model_dump_json()
    assert _RAW_CONTENT not in dumped
    assert _RAW_USER not in dumped
    assert event.content_hash == _sha256(_RAW_CONTENT)
    assert event.actor_id_hash == _sha256(_RAW_USER)


async def test_hash_value_keeps_none():
    assert hash_value(None) is None
    assert build_event(source_module=23, source_service="s", event_type="grpc.call",
                       outcome="allow").content_hash is None


async def test_event_carries_every_cf4_field():
    event = _event()
    assert set(json.loads(event.model_dump_json())) == {
        "event_id", "timestamp", "source_module", "source_service", "event_type", "actor_id_hash",
        "resource_id", "outcome", "policy_version", "content_hash", "metadata", "retention_days",
    }
    assert isinstance(event.event_id, UUID)
    assert event.timestamp.tzinfo is not None


async def test_retention_days_defaults_to_365_and_can_be_overridden():
    assert _event().retention_days == 365
    assert _event(retention_days=30).retention_days == 30
    with pytest.raises(ValidationError):
        _event(retention_days=0)


@pytest.mark.parametrize("field, value", [("outcome", "warn"), ("event_type", "guardrail.check")])
async def test_values_outside_the_schema_are_rejected(field, value):
    with pytest.raises(ValidationError):
        _event(**{field: value})


async def test_extra_fields_cannot_be_smuggled_in():
    with pytest.raises(ValidationError):
        AuditEvent(source_module=23, source_service="s", event_type="guardrail.decision",
                   outcome="block", content=_RAW_CONTENT)


async def test_each_event_gets_its_own_id():
    assert _event().event_id != _event().event_id


async def test_ddl_creates_the_audit_events_table():
    assert "CREATE TABLE IF NOT EXISTS audit_events" in AUDIT_EVENTS_DDL


# ── PostgreSQL sink ──────────────────────────────────────────────────────────────

class _Conn:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.executed: list[tuple] = []

    async def execute(self, query, *params):
        if self.fail:
            raise ConnectionError("postgres down")
        self.executed.append((query, params))


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _Acquire(self._conn)


def _writer(conn: _Conn) -> PostgresAuditWriter:
    async def _get_pool():
        return _Pool(conn)
    return PostgresAuditWriter(_get_pool)


async def test_insert_event_sends_every_column():
    conn = _Conn()
    event = _event()
    await insert_event(conn, event)
    query, params = conn.executed[0]
    assert "INSERT INTO audit_events" in query
    assert "$11::text::jsonb" in query                          # never re-encoded by a jsonb codec
    assert params == (
        event.event_id, event.timestamp, 23, "guardrail-service", "guardrail.decision",
        _sha256(_RAW_USER), "req-1", "block", "1.0.0", _sha256(_RAW_CONTENT),
        json.dumps({"decision": "block"}), 365,
    )


async def test_emit_writes_the_event():
    conn = _Conn()
    await _writer(conn).emit(_event())
    assert len(conn.executed) == 1


async def test_emit_fail_open_logs_a_warning_and_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING, logger="raptor_audit.postgres"):
        await _writer(_Conn(fail=True)).emit(_event())           # must not raise
    assert "not written" in caplog.text


async def test_emit_fail_closed_raises_audit_write_error():
    with pytest.raises(AuditWriteError):
        await _writer(_Conn(fail=True)).emit(_event(event_type="gdpr.erasure", outcome="erase"),
                                             fail_closed=True)


async def test_emit_fail_closed_also_covers_an_unavailable_pool():
    async def _no_pool():
        raise RuntimeError("pool not initialized")

    with pytest.raises(AuditWriteError):
        await PostgresAuditWriter(_no_pool).emit(_event(), fail_closed=True)
