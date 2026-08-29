"""Unit tests for app/db/repo.upsert_policy_content — the (name, version, target)
upsert that lets one policy accumulate per-guard-model content across
multiple uploads."""
import pytest

from app.db import repo

pytestmark = pytest.mark.asyncio


class FakeConn:
    def __init__(self, existing_id=None):
        self.existing_id = existing_id
        self.executed: list[tuple] = []

    async def fetchrow(self, query, *params):
        self.executed.append((query, params))
        if "SELECT id FROM guardrail_policies WHERE name" in query:
            return {"id": self.existing_id} if self.existing_id is not None else None
        # INSERT/UPDATE ... RETURNING id, name, version, created_at
        return {"id": self.existing_id or "new-id", "name": params[0] if "INSERT" in query else "n",
                "version": params[1] if "INSERT" in query else "v", "created_at": "now"}


class FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeAcquiredConn(FakeConn):
    def transaction(self):
        return FakeTxn()


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


async def test_upsert_original_on_new_policy_inserts_raw_content_only():
    conn = FakeAcquiredConn(existing_id=None)
    pool = FakePool(conn)

    await repo.upsert_policy_content(pool, "my-policy", "1.0", "original", "hello")

    insert_query, insert_params = conn.executed[-1]
    assert "INSERT INTO guardrail_policies (name, version, raw_content)" in insert_query
    assert insert_params == ("my-policy", "1.0", "hello")


async def test_upsert_model_override_on_new_policy_sets_raw_content_empty_and_target_column():
    conn = FakeAcquiredConn(existing_id=None)
    pool = FakePool(conn)

    await repo.upsert_policy_content(pool, "my-policy", "1.0", "llama-guard", "llama prompt")

    insert_query, insert_params = conn.executed[-1]
    assert "content_llama_guard" in insert_query
    assert "raw_content" in insert_query
    assert insert_params == ("my-policy", "1.0", "", "llama prompt")


async def test_upsert_on_existing_policy_updates_only_the_target_column():
    conn = FakeAcquiredConn(existing_id="existing-uuid")
    pool = FakePool(conn)

    await repo.upsert_policy_content(pool, "my-policy", "1.0", "gpt-oss-safeguard", "gpt prompt")

    update_query, update_params = conn.executed[-1]
    assert "UPDATE guardrail_policies SET content_gpt_oss_safeguard" in update_query
    assert update_params == ("gpt prompt", "existing-uuid")


async def test_upsert_original_on_existing_policy_updates_raw_content():
    conn = FakeAcquiredConn(existing_id="existing-uuid")
    pool = FakePool(conn)

    await repo.upsert_policy_content(pool, "my-policy", "1.0", "original", "updated content")

    update_query, update_params = conn.executed[-1]
    assert "UPDATE guardrail_policies SET raw_content" in update_query
    assert update_params == ("updated content", "existing-uuid")
