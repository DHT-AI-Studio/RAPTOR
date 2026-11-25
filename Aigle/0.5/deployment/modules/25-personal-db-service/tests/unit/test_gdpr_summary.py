"""GD-4 / VIE01-249: GET /internal/gdpr/user/{user_id}/summary -- the
count-only indexed-content summary Module 33's DSAR assembler calls.

Pure unit test: ArcadeDB is replaced by a fake client (database_exists +
the handful of queries the summary issues), so it runs anywhere -- unlike
this directory's integration tests, which need a live ArcadeDB.
"""
from __future__ import annotations

import pytest
from app.core.config import settings
from app.routers import database as database_router
from app.routers import gdpr as gdpr_router
from fastapi import FastAPI
from fastapi.testclient import TestClient

TOKEN = "test-internal-token"


class _FakeArcade:
    """chunks: list of (type, version_id); sources: list of (media_type, version_id)."""

    def __init__(self, exists: bool, chunks=(), sources=(), entities: int = 0, no_source_type: bool = False,
                 pii_flagged: int = 0, pii_query_fails: bool = False):
        self.exists = exists
        self.pii_flagged = pii_flagged
        self.pii_query_fails = pii_query_fails
        self.chunks = list(chunks)
        self.sources = list(sources)
        self.entities = entities
        self.no_source_type = no_source_type
        self.queries: list[str] = []

    async def database_exists(self, db):
        return self.exists

    async def query(self, db, sql):
        self.queries.append(sql)
        if "FROM Source WHERE media_type = 'documents'" in sql:
            if self.no_source_type:
                raise RuntimeError("Type with name 'Source' was not found")
            return [{"version_id": v} for t, v in self.sources if t == "documents"]
        if "DISTINCT(version_id)" in sql:
            return [{"version_id": v} for v in {v for t, v in self.chunks if t == "documents"}]
        if "GROUP BY type" in sql:
            by_type: dict = {}
            for t, _ in self.chunks:
                by_type[t] = by_type.get(t, 0) + 1
            return [{"type": t, "c": c} for t, c in by_type.items()]
        if "pii_flag = true" in sql:
            if self.pii_query_fails:
                raise RuntimeError("ArcadeDB unavailable")
            return [{"n": self.pii_flagged}]
        if "FROM Entity" in sql:
            return [{"c": self.entities}]
        if "FROM Chunk" in sql:
            return [{"c": len(self.chunks)}]
        return [{"c": 0}]


@pytest.fixture
def client_for(monkeypatch):
    monkeypatch.setattr(settings, "gdpr_internal_token", TOKEN)

    def _make(fake: _FakeArcade) -> TestClient:
        # _collect_counts lives in database.py and closes over that module's
        # own `client`; gdpr.py imported the same object by name -- patch both.
        monkeypatch.setattr(database_router, "client", fake)
        monkeypatch.setattr(gdpr_router, "client", fake)
        app = FastAPI()
        app.include_router(gdpr_router.router)
        return TestClient(app)

    return _make


def _get(client: TestClient, user_id: str = "u1"):
    return client.get(f"/internal/gdpr/user/{user_id}/summary", headers={"X-Internal-Token": TOKEN})


def test_summary_requires_internal_token(client_for):
    r = client_for(_FakeArcade(True)).get("/internal/gdpr/user/u1/summary")
    assert r.status_code == 401


def test_documents_counts_documents_not_chunks(client_for):
    """3 PDFs split into 40 chunks are 3 documents. Moments stay segments."""
    chunks = [("documents", f"v{i % 3}") for i in range(40)]
    chunks += [("videos", "vid1")] * 7 + [("images", "img1")] * 2 + [("audios", "aud1")]
    sources = [("documents", "v0"), ("documents", "v1"), ("documents", "v2"), ("videos", "vid1")]
    fake = _FakeArcade(True, chunks=chunks, sources=sources, entities=13, pii_flagged=4)
    r = _get(client_for(fake))
    assert r.status_code == 200
    assert r.json() == {
        "user_id": "u1", "db_exists": True,
        "documents": 3, "moments": 10, "entities": 13, "pii_flagged": 4,
        # Internal threshold input: all 50 chunks + 13 entities, not documents.
        "indexed_items": 63,
    }
    # Never a row-content query: counts, or version_id only.
    for q in fake.queries:
        assert "count(" in q.lower() or "version_id" in q, q
        assert "SELECT *" not in q and "text" not in q.lower()


def test_documents_is_union_of_tagged_sources_and_chunk_versions(client_for):
    """Live data has document chunks with no tagged Source, and a Source can
    exist before its chunks -- each side alone undercounts."""
    chunks = [("documents", "v1"), ("documents", "v1"), ("documents", "v2")]  # v2: no Source
    sources = [("documents", "v1"), ("documents", "v3")]                      # v3: no chunks yet
    r = _get(client_for(_FakeArcade(True, chunks=chunks, sources=sources)))
    assert r.json()["documents"] == 3


def test_missing_vertex_type_counts_as_none(client_for):
    fake = _FakeArcade(True, chunks=[("documents", "v1")], no_source_type=True)
    assert _get(client_for(fake)).json()["documents"] == 1


def test_summary_for_unprovisioned_user_is_zero_not_404(client_for):
    fake = _FakeArcade(False)
    r = _get(client_for(fake), "ghost")
    assert r.status_code == 200
    assert r.json() == {
        "user_id": "ghost", "db_exists": False,
        "documents": 0, "moments": 0, "entities": 0, "pii_flagged": 0, "indexed_items": 0,
    }
    assert fake.queries == []


def test_pii_flagged_count_failure_is_null_not_zero(client_for):
    """GD-6: a failed count must not read as "no PII"."""
    fake = _FakeArcade(True, chunks=[("documents", "v1")], pii_query_fails=True)
    assert _get(client_for(fake)).json()["pii_flagged"] is None
