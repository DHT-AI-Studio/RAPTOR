"""
GD-7 / VIE01-252: unit tests for the data access audit log routes
(app/routers/access.py) -- record write, admin filters and paging, and the
data subject's own view.

Mocked: the asyncpg pool is a fake that records the SQL and arguments it
receives and filters an in-memory row list the same way, so the filter
logic is checked against real rows rather than just SQL text.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_audit_log.py -v
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from app.core.config import settings
from app.main import app
from fastapi.testclient import TestClient

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(user_id, accessor_id, minutes, endpoint="/api/0.4/memory/stats"):
    return {"id": uuid.uuid4(), "user_id": user_id, "accessor_id": accessor_id, "endpoint": endpoint,
            "method": "GET", "data_categories": ["memory"], "purpose": "service_provision",
            "timestamp": T0 + timedelta(minutes=minutes), "ip_hash": "h"}


class _FakeConn:
    """Understands exactly the statements access.py issues."""

    def __init__(self, db):
        self.db = db

    def _match(self, sql, args):
        rows = self.db.rows
        conds = sql.split("WHERE", 1)[1].split("ORDER BY")[0] if "WHERE" in sql else ""
        for i, arg in enumerate(args, start=1):
            placeholder = f"${i}"
            if f"user_id = {placeholder}" in conds:
                rows = [r for r in rows if r["user_id"] == arg]
            elif f"timestamp >= {placeholder}" in conds:
                rows = [r for r in rows if r["timestamp"] >= arg]
            elif f"timestamp <= {placeholder}" in conds:
                rows = [r for r in rows if r["timestamp"] <= arg]
        return sorted(rows, key=lambda r: r["timestamp"], reverse=True)

    async def execute(self, sql, *args):
        self.db.executed.append((sql, args))

    async def fetchval(self, sql, *args):
        return len(self._match(sql, args))

    async def fetch(self, sql, *args):
        limit, offset = args[-2], args[-1]
        return self._match(sql, args[:-2])[offset:offset + limit]


class _FakePool:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executed: list = []

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _FakeConn(pool)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


@pytest.fixture
def api():
    def _make(rows=()):
        app.state.pool = _FakePool(rows)
        return TestClient(app, headers={"X-Internal-Token": settings.internal_token}), app.state.pool
    return _make


# ── Write ───────────────────────────────────────────────────────────────────

def test_write_inserts_the_record(api):
    client, pool = api()
    record = {"user_id": "alice", "accessor_id": "dpo-1", "endpoint": "/api/0.4/gdpr/dsar/alice",
              "method": "GET", "data_categories": ["identity", "memory"], "purpose": "right_of_access",
              "ip_hash": "a" * 64, "timestamp": "2026-01-01T00:00:00+00:00"}
    assert client.post("/internal/gdpr/access-log", json=record).status_code == 204
    sql, args = pool.executed[0]
    assert "INSERT INTO gdpr_access_log" in sql
    assert args[:7] == ("alice", "dpo-1", "/api/0.4/gdpr/dsar/alice", "GET",
                        ["identity", "memory"], "right_of_access", "a" * 64)
    assert args[7] == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_routes_require_internal_token(api):
    api()
    bare = TestClient(app)
    assert bare.get("/audit-log").status_code == 401
    assert bare.get("/audit-log/my", headers={"X-User-ID": "alice"}).status_code == 401
    assert bare.post("/internal/gdpr/access-log", json={}).status_code == 401


# ── Admin listing: filters and paging ───────────────────────────────────────

ROWS = [_row("alice", "alice", 0), _row("alice", "dpo-1", 10), _row("bob", "bob", 20), _row("alice", "alice", 30)]


def test_unfiltered_listing_is_newest_first(api):
    client, _ = api(ROWS)
    body = client.get("/audit-log").json()
    assert body["total"] == 4
    assert [i["timestamp"] for i in body["items"]] == sorted((i["timestamp"] for i in body["items"]), reverse=True)


def test_filter_by_user_id(api):
    client, _ = api(ROWS)
    body = client.get("/audit-log", params={"user_id": "alice"}).json()
    assert body["total"] == 3 and {i["user_id"] for i in body["items"]} == {"alice"}


def test_filter_by_time_window(api):
    client, _ = api(ROWS)
    params = {"from": (T0 + timedelta(minutes=5)).isoformat(), "to": (T0 + timedelta(minutes=25)).isoformat()}
    body = client.get("/audit-log", params=params).json()
    assert body["total"] == 2
    assert {(i["user_id"], i["accessor_id"]) for i in body["items"]} == {("alice", "dpo-1"), ("bob", "bob")}


def test_filters_combine(api):
    client, _ = api(ROWS)
    params = {"user_id": "alice", "from": (T0 + timedelta(minutes=5)).isoformat()}
    body = client.get("/audit-log", params=params).json()
    assert body["total"] == 2


def test_paging(api):
    client, _ = api(ROWS)
    first = client.get("/audit-log", params={"page": 1, "page_size": 3}).json()
    second = client.get("/audit-log", params={"page": 2, "page_size": 3}).json()
    assert (len(first["items"]), len(second["items"])) == (3, 1)
    assert first["total"] == second["total"] == 4


def test_page_size_capped_at_1000(api):
    client, _ = api(ROWS)
    assert client.get("/audit-log", params={"page_size": 1000}).status_code == 200
    assert client.get("/audit-log", params={"page_size": 1001}).status_code == 422


def test_inverted_time_window_rejected(api):
    client, _ = api(ROWS)
    params = {"from": (T0 + timedelta(minutes=30)).isoformat(), "to": T0.isoformat()}
    assert client.get("/audit-log", params=params).status_code == 422


# ── The data subject's own view ─────────────────────────────────────────────

def test_my_log_shows_only_own_data_and_hashes_other_accessors(api):
    client, _ = api(ROWS)
    body = client.get("/audit-log/my", headers={"X-User-ID": "alice"}).json()
    assert body["total"] == 3
    items = body["items"]
    assert set(items[0]) == {"accessor_id", "accessed_by_self", "endpoint", "timestamp"}
    others = [i for i in items if not i["accessed_by_self"]]
    assert [i["accessor_id"] for i in others] == [hashlib.sha256(b"dpo-1").hexdigest()]
    assert all(i["accessor_id"] == "alice" for i in items if i["accessed_by_self"])
    assert "dpo-1" not in str(body)


def test_my_log_requires_user(api):
    client, _ = api(ROWS)
    assert client.get("/audit-log/my").status_code == 400
