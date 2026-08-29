"""Unit tests for app/db/repo.update_policy_content — direct by-id content
edits used by PUT /guardrail/{policy_id}."""
import pytest

from app.db import repo

pytestmark = pytest.mark.asyncio


class FakeConn:
    def __init__(self, row=None):
        self._row = row
        self.executed: list[tuple] = []

    async def fetchrow(self, query, *params):
        self.executed.append((query, params))
        return self._row


class FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return FakeAcquire(self._conn)


async def test_update_single_field_builds_correct_set_clause():
    conn = FakeConn(row={"id": "p1"})
    pool = FakePool(conn)

    await repo.update_policy_content(pool, "p1", {"content_llama_guard": "new prompt"})

    query, params = conn.executed[-1]
    assert "SET content_llama_guard = $1" in query
    assert "WHERE id = $2" in query
    assert params == ("new prompt", "p1")


async def test_update_multiple_fields_builds_all_set_clauses_in_order():
    conn = FakeConn(row={"id": "p1"})
    pool = FakePool(conn)

    await repo.update_policy_content(
        pool, "p1", {"raw_content": "orig", "content_granite_guardian": "granite prompt"},
    )

    query, params = conn.executed[-1]
    assert "raw_content = $1" in query
    assert "content_granite_guardian = $2" in query
    assert "WHERE id = $3" in query
    assert params == ("orig", "granite prompt", "p1")


async def test_update_returns_none_when_policy_not_found():
    conn = FakeConn(row=None)
    pool = FakePool(conn)

    result = await repo.update_policy_content(pool, "missing", {"raw_content": "x"})

    assert result is None
