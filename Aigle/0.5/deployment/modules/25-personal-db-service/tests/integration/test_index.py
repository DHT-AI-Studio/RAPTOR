"""PA-10 — indexing: chunk, entity+relation, temporal fact; verify vertex/edge counts.

Runs against a real ArcadeDB (skips if unreachable). The query embedder is not
needed here — chunks are indexed with deterministic fake vectors.
"""
from __future__ import annotations

import pytest
from app.models.graph_index import (
    EntityIndexRequest,
    RelationshipIndexRequest,
    TemporalFactIndexRequest,
)
from app.models.index import ChunkIndexRequest
from app.services import graph_indexer, indexer
from app.services.arcadedb_client import db_name_for

from tests.integration.conftest import fake_vector

pytestmark = pytest.mark.asyncio


def _chunk(chunk_id: str, text: str, **extra) -> ChunkIndexRequest:
    return ChunkIndexRequest(chunk_id=chunk_id, type="documents", embedding_type="text",
                             embedding=fake_vector(text), text=text, **extra)


async def _count(client, db, vtype: str) -> int:
    rows = await client.query(db, f"SELECT count(*) AS c FROM {vtype}")
    return rows[0]["c"] if rows else 0


# ─────────────────────────── chunk indexing (PA-4) ──────────────────────────
async def test_index_chunk_upsert_is_idempotent(client, make_db):
    branch = await make_db("ittest_chunk")
    db = db_name_for(branch)
    await indexer.index_chunk(client, branch, _chunk("c1", "hello world"))
    await indexer.index_chunk(client, branch, _chunk("c1", "hello again"))    # same id
    assert await _count(client, db, "Chunk") == 1


async def test_chunk_with_version_links_source(client, make_db):
    branch = await make_db("ittest_source")
    db = db_name_for(branch)
    await indexer.index_chunk(client, branch, _chunk("c1", "a", version_id="v1", filename="a.pdf"))
    await indexer.index_chunk(client, branch, _chunk("c2", "b", version_id="v1", filename="a.pdf"))
    assert await _count(client, db, "Source") == 1
    assert await _count(client, db, "HAS_CHUNK") == 2


async def test_delete_by_version_prunes_orphan_entity(client, make_db):
    branch = await make_db("ittest_delete")
    db = db_name_for(branch)
    await indexer.index_chunk(client, branch, _chunk("c1", "x", version_id="v1"))
    await graph_indexer.index_entity(client, branch, EntityIndexRequest(
        entity_id="e1", name="Acme", type="ORG", source_chunk_id="c1"))

    result = await indexer.delete_by_version(client, branch, "v1")
    assert (result["chunks"], result["sources"], result["orphan_entities"]) == (1, 1, 1)
    assert await _count(client, db, "Chunk") == 0
    assert await _count(client, db, "Entity") == 0
    assert (await indexer.delete_by_version(client, branch, "v1"))["chunks"] == 0   # idempotent


# ──────────────── version_exists() -- Module 04's check_indexed() backend ───
async def test_version_exists_true_for_any_status_by_default(client, make_db):
    branch = await make_db("ittest_exists_any")
    await indexer.index_chunk(client, branch, _chunk("c1", "x", version_id="v1", status="archived"))
    assert await indexer.version_exists(client, branch, "v1") is True


async def test_version_exists_false_for_missing_version(client, make_db):
    branch = await make_db("ittest_exists_missing")
    assert await indexer.version_exists(client, branch, "nonexistent-v") is False


async def test_version_exists_with_status_filter_ignores_other_states(client, make_db):
    """A non-active row (e.g. a stale clone_point() leftover) must not count
    as proof of a genuinely-active version when a caller asks specifically
    for status="active" -- this is what update_status()'s double-check
    relies on."""
    branch = await make_db("ittest_exists_status")
    await indexer.index_chunk(client, branch, _chunk("c1", "x", version_id="v1", status="cloned"))
    assert await indexer.version_exists(client, branch, "v1", status="active") is False
    assert await indexer.version_exists(client, branch, "v1", status="cloned") is True
    assert await indexer.version_exists(client, branch, "v1") is True   # status-blind default


# ──────────────── entity / relationship / temporal fact (PA-5) ──────────────
async def test_entity_mentions_and_count(client, make_db):
    branch = await make_db("ittest_entity")
    db = db_name_for(branch)
    await indexer.index_chunk(client, branch, _chunk("c1", "Samsung news"))
    await indexer.index_chunk(client, branch, _chunk("c2", "more Samsung"))
    for chunk in ("c1", "c2"):
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(
            entity_id="samsung", name="Samsung", type="ORG", source_chunk_id=chunk))

    assert await _count(client, db, "Entity") == 1               # deduped across chunks
    assert await _count(client, db, "MENTIONS") == 2
    cnt = await client.query(db, "SELECT mention_count AS c FROM Entity WHERE entity_id='samsung'")
    assert cnt[0]["c"] == 2


async def test_relationship_dedup(client, make_db):
    branch = await make_db("ittest_rel")
    db = db_name_for(branch)
    for eid, name in [("a", "A"), ("b", "B")]:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(
            entity_id=eid, name=name, type="ORG"))
    req = RelationshipIndexRequest(from_entity_id="a", to_entity_id="b",
                                   relation="partners_with", confidence=0.8)
    await graph_indexer.index_relationship(client, branch, req)
    await graph_indexer.index_relationship(client, branch, req)   # same edge twice
    assert await _count(client, db, "RELATION") == 1


async def test_temporal_fact_edges(client, make_db):
    branch = await make_db("ittest_tkg")
    db = db_name_for(branch)
    await indexer.index_chunk(client, branch, _chunk("c1", "event happened"))
    await graph_indexer.index_entity(client, branch, EntityIndexRequest(
        entity_id="e1", name="E1", type="ORG", source_chunk_id="c1"))
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="tf1", entity="E1", relation="did", value="something",
        entity_id="e1", time_start="2026-01", chunk_id="c1", confidence=0.9))

    assert await _count(client, db, "TemporalFact") == 1
    assert await _count(client, db, "HAS_TEMPORAL_FACT") == 1     # entity -> fact
    assert await _count(client, db, "OBSERVED_IN") == 1          # fact -> chunk


# ── StatusRequest Literal (MTR-3 regression guard) ───────────────────────────
# Pure Pydantic-boundary check, no ArcadeDB needed -- isolates exactly what
# VIE01-255 widened (StatusRequest used to reject "failed"/"unregistered",
# which Module 04's new /update-status?status=failed call would trigger).

def test_status_request_accepts_failed_and_unregistered():
    from app.routers.index import StatusRequest
    assert StatusRequest(status="failed").status == "failed"
    assert StatusRequest(status="unregistered").status == "unregistered"


async def test_search_scoped_to_several_versions(client, make_db):
    from app.models.search import SearchRequest
    from app.services import searcher
    branch = await make_db("ittest_p21_search_scope")
    for cid, version in (("c1", "v1"), ("c2", "v2"), ("c3", "v3")):
        await indexer.index_chunk(client, branch, _chunk(cid, "Android history notes", version_id=version))
    got = await searcher.bm25_search(client, branch, SearchRequest(query="Android", top_k=10, version_id=["v1", "v3"]))
    assert {r.payload.get("version_id") for r in got.results} == {"v1", "v3"}
    one = await searcher.bm25_search(client, branch, SearchRequest(query="Android", top_k=10, version_id="v2"))
    assert {r.payload.get("version_id") for r in one.results} == {"v2"}
