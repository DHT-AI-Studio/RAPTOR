"""Shared fixtures for the integration tests (PA-4/5/6).

These hit a REAL ArcadeDB (they exercise the actual SQL), so they need a running
server. Point them at it with env vars and run:

    PD_ARCADEDB_URL=http://localhost:2480 PD_ARCADEDB_PASSWORD=$ARCADEDB_ROOT_PASSWORD pytest

If ArcadeDB is unreachable the integration tests skip (so the pure-unit suite
still runs anywhere). Each test gets its own throwaway `user_<branch>` database,
created + schema-initialized on entry and dropped on exit — no shared state, so
the isolation guarantee is exercised for real.
"""
from __future__ import annotations

import hashlib
from typing import List

import pytest
import pytest_asyncio

from app.core.config import settings
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.schema_init import initialize_schema


def fake_vector(text: str, dim: int | None = None) -> List[float]:
    """Deterministic pseudo-embedding from a text hash — same text -> same vector,
    different text -> different vector. Not semantic; enough to exercise ranking."""
    dim = dim or settings.vector_dim
    digest = hashlib.sha256(text.encode()).digest()
    return [((digest[i % len(digest)] / 255.0) - 0.5) for i in range(dim)]


async def _reachable(client: ArcadeDBClient) -> bool:
    try:
        await client.list_databases()
        return True
    except Exception:
        return False


@pytest.fixture
def client() -> ArcadeDBClient:
    return ArcadeDBClient()


@pytest_asyncio.fixture
async def arcade(client: ArcadeDBClient) -> ArcadeDBClient:
    """A reachable ArcadeDB client, or skip. For tests that manage their own DB."""
    if not await _reachable(client):
        pytest.skip("ArcadeDB not reachable (set PD_ARCADEDB_URL)")
    return client


@pytest_asyncio.fixture
async def make_db(client: ArcadeDBClient):
    """Factory: init a fresh per-branch database, auto-dropped at test teardown."""
    created: list[str] = []

    branches: list[str] = []

    async def _init(branch: str) -> str:
        if not await _reachable(client):
            pytest.skip("ArcadeDB not reachable (set PD_ARCADEDB_URL)")
        name = db_name_for(branch)
        if await client.database_exists(name):
            await client.drop_database(name)
        await client.create_database(name)
        await initialize_schema(client, name)
        await _clear_index_claims(branch)
        created.append(name)
        branches.append(branch)
        return branch

    yield _init

    for name in created:
        try:
            await client.drop_database(name)
        except Exception:
            pass
    for branch in branches:
        await _clear_index_claims(branch)


async def _clear_index_claims(branch: str) -> None:
    """Forget this branch's VIE01-190 event claims.

    The ArcadeDB database is recreated per test, but `personal_index_events` is
    permanent by design. Tests reuse fixed branch and version ids, so without
    this the *second* run of the suite would see every event as an already-seen
    duplicate and index nothing — the tests would pass once and then fail
    forever. PostgreSQL being absent is fine: then there are no claims to clear.
    """
    try:
        from app.services.audit import get_pool
        pool = await get_pool()
        await pool.execute("DELETE FROM personal_index_events WHERE user_id = $1", branch)
    except Exception:
        pass


@pytest.fixture
def mock_embed(monkeypatch):
    """Patch the search layer's query embedder with the deterministic fake."""
    async def _embed(texts: List[str]) -> List[List[float]]:
        return [fake_vector(t) for t in texts]
    monkeypatch.setattr("app.services.searcher.embed_texts", _embed)
    return _embed
