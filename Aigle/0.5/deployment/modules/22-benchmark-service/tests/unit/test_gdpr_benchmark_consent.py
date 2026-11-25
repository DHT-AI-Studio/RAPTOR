"""
GD-5 / VIE01-250: DELETE /internal/gdpr/benchmark/user/{user_id} -- what a
benchmark consent withdrawal deletes.

The fake connection applies the three statements of
schema_store.delete_user_benchmark_data to in-memory tables (the real SQL
runs in Module 33's live tests/test_consent_e2e.py), so these pin down the
rule itself: the user's runs go, their schemas go unless another user still
has runs on one -- that one is kept and detached instead -- and nobody
else's data is touched.

Run from the 22-benchmark-service/ directory:
    python -m pytest tests/unit/test_gdpr_benchmark_consent.py -v
"""
from __future__ import annotations

import pytest
from app.core import config
from app.core.db import db
from app.routers import gdpr as gdpr_router
from app.services import schema_store
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _Tables:
    def __init__(self, schemas, runs):
        self.schemas = {sid: owner for sid, owner in schemas}  # id -> owner_id
        self.runs = list(runs)  # (branch_id, schema_id)


class _FakeConn:
    def __init__(self, t: _Tables):
        self.t = t

    def transaction(self):
        class _Tx:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *exc):
                return False

        return _Tx()

    async def execute(self, sql, user_id):
        sql = " ".join(sql.split())
        if sql.startswith("DELETE FROM benchmark_runs WHERE branch_id"):
            before = len(self.t.runs)
            self.t.runs = [r for r in self.t.runs if r[0] != user_id]
            return f"DELETE {before - len(self.t.runs)}"
        if sql.startswith("DELETE FROM benchmark_schemas"):
            used = {sid for _, sid in self.t.runs}
            doomed = [sid for sid, owner in self.t.schemas.items() if owner == user_id and sid not in used]
            for sid in doomed:
                del self.t.schemas[sid]
            return f"DELETE {len(doomed)}"
        if sql.startswith("UPDATE benchmark_schemas SET owner_id = NULL"):
            mine = [sid for sid, owner in self.t.schemas.items() if owner == user_id]
            for sid in mine:
                self.t.schemas[sid] = None
            return f"UPDATE {len(mine)}"
        raise AssertionError(f"unexpected SQL: {sql}")


class _FakePool:
    def __init__(self, t: _Tables):
        self.t = t

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _FakeConn(pool.t)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


@pytest.fixture
def tables(monkeypatch):
    t = _Tables(
        schemas=[("s-alice-only", "alice"), ("s-alice-shared", "alice"), ("s-bob", "bob"), ("s-legacy", None)],
        runs=[("alice", "s-alice-only"), ("alice", "s-alice-shared"), ("bob", "s-alice-shared"),
              ("bob", "s-bob"), ("alice", "s-bob"), ("alice", "s-legacy")],
    )
    monkeypatch.setattr(db, "pool", _FakePool(t))
    return t


@pytest.mark.asyncio
async def test_withdrawal_deletes_runs_and_unshared_schemas_and_detaches_shared(tables):
    counts = await schema_store.delete_user_benchmark_data("alice")
    assert counts == {"runs_deleted": 4, "schemas_deleted": 1, "schemas_detached": 1}
    assert all(branch != "alice" for branch, _ in tables.runs)
    # Kept because Bob's run depends on it, but no longer attributed to Alice.
    assert tables.schemas == {"s-alice-shared": None, "s-bob": "bob", "s-legacy": None}
    # Bob's own runs -- including the one on Alice's schema -- are untouched.
    assert sorted(tables.runs) == [("bob", "s-alice-shared"), ("bob", "s-bob")]


@pytest.mark.asyncio
async def test_withdrawal_for_user_without_benchmark_data(tables):
    counts = await schema_store.delete_user_benchmark_data("carol")
    assert counts == {"runs_deleted": 0, "schemas_deleted": 0, "schemas_detached": 0}
    assert len(tables.runs) == 6 and len(tables.schemas) == 4


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(config.get_settings(), "gdpr_internal_token", "test-token")
    app = FastAPI()
    app.include_router(gdpr_router.router)
    return TestClient(app)


def test_endpoint_requires_internal_token(tables, monkeypatch):
    r = _client(monkeypatch).delete("/internal/gdpr/benchmark/user/alice")
    assert r.status_code == 401
    assert len(tables.runs) == 6


def test_endpoint_fails_closed_without_configured_token(tables, monkeypatch):
    monkeypatch.setattr(config.get_settings(), "gdpr_internal_token", "")
    app = FastAPI()
    app.include_router(gdpr_router.router)
    r = TestClient(app).delete("/internal/gdpr/benchmark/user/alice", headers={"X-Internal-Token": ""})
    assert r.status_code == 401


def test_endpoint_returns_counts(tables, monkeypatch):
    r = _client(monkeypatch).delete("/internal/gdpr/benchmark/user/alice", headers={"X-Internal-Token": "test-token"})
    assert r.status_code == 200
    assert r.json() == {"user_id": "alice", "runs_deleted": 4, "schemas_deleted": 1, "schemas_detached": 1}
