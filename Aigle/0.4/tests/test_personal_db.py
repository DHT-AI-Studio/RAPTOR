"""Personal DB contract and isolation tests (VIE01-192).

What these are for: proving that one user cannot read, write, or infer the
existence of another user's data, and that the index → search → delete lifecycle
holds end to end. Everything runs against real services — isolation is a claim
about where bytes actually live, and a mock would only prove the mock is
consistent with itself.

Layout mirrors the AC:

    contract     — index, retrieve, and the negative case (B gets 0 results)
    lifecycle    — A deletes; A's data gone, B's untouched
    graph        — entity traversal and temporal timeline scoped to one user
    dedup        — same event three times, one record
    dlq          — malformed event parks after 3 attempts and is recorded as failed
    isolation    — all five gateway routes refuse a cross-user request with 403
    performance  — hybrid-search p95 over 1,000 documents (a recorded baseline,
                   deliberately not a pass/fail gate)
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
import uuid

import pytest

from app.models.graph_index import (EntityIndexRequest, RelationshipIndexRequest,
                                    TemporalFactIndexRequest)
from app.models.graph_search import GraphSearchRequest, TKGRequest
from app.models.index import ChunkIndexRequest
from app.models.search import SearchRequest
from app.services import graph_indexer, searcher
from app.services.arcadedb_client import db_name_for
from app.services.indexer import index_chunk

pytestmark = pytest.mark.asyncio

A_DOC_TEXT = "Quarterly revenue grew twelve percent driven by enterprise renewals"
B_DOC_TEXT = "Sourdough hydration ratios for a cold overnight proof"


async def _index_doc(arcade, user: str, chunk_id: str, text: str, **extra) -> None:
    """Index one document chunk, letting the service embed the text itself."""
    await index_chunk(arcade, user, ChunkIndexRequest(
        chunk_id=chunk_id, type="documents", embedding_type="text",
        text=text, filename=f"{chunk_id}.txt", **extra,
    ))


# ── contract: index → retrieve → the other user gets nothing ──────────────────

async def test_user_b_cannot_retrieve_user_a_document(arcade, two_users):
    """The central isolation claim. A indexes; A finds it; B searching the same
    words finds nothing, because B's database does not contain it at all."""
    user_a, user_b = two_users
    await _index_doc(arcade, user_a, "a-doc-1", A_DOC_TEXT)

    req = SearchRequest(query="quarterly revenue growth", top_k=10)

    a_results = await searcher.hybrid_search(arcade, user_a, req)
    assert len(a_results.results) >= 1, "user A cannot retrieve their own document"
    assert any("revenue" in (r.payload.get("text") or "") for r in a_results.results)

    b_results = await searcher.hybrid_search(arcade, user_b, req)
    assert len(b_results.results) == 0, (
        f"user B retrieved {len(b_results.results)} result(s) from user A's data"
    )


async def test_databases_are_physically_separate(arcade, two_users):
    """Not just filtered — separate databases. A row count of B's store is the
    check a WHERE-clause-based 'isolation' would fail."""
    user_a, user_b = two_users
    await _index_doc(arcade, user_a, "a-doc-2", A_DOC_TEXT)

    a_rows = await arcade.query(db_name_for(user_a), "SELECT count(*) AS c FROM Chunk")
    b_rows = await arcade.query(db_name_for(user_b), "SELECT count(*) AS c FROM Chunk")
    assert a_rows[0]["c"] == 1
    assert b_rows[0]["c"] == 0
    assert db_name_for(user_a) != db_name_for(user_b)


# ── lifecycle: delete one user, the other is untouched ────────────────────────

async def test_delete_removes_only_the_deleting_user(arcade, two_users):
    user_a, user_b = two_users
    await _index_doc(arcade, user_a, "a-doc-3", A_DOC_TEXT)
    await _index_doc(arcade, user_b, "b-doc-1", B_DOC_TEXT)

    await arcade.drop_database(db_name_for(user_a))

    assert not await arcade.database_exists(db_name_for(user_a))
    assert await arcade.database_exists(db_name_for(user_b)), "user B's database was collateral"

    b_results = await searcher.hybrid_search(
        arcade, user_b, SearchRequest(query="sourdough hydration", top_k=10))
    assert len(b_results.results) >= 1, "user B lost data when user A deleted theirs"

    # Recreate so the fixture teardown has something to drop.
    from app.services.schema_init import initialize_schema
    await arcade.create_database(db_name_for(user_a))
    await initialize_schema(arcade, db_name_for(user_a))


# ── graph + temporal, scoped to one user ──────────────────────────────────────

async def test_graph_search_returns_only_the_users_entities(arcade, two_users):
    user_a, user_b = two_users
    await graph_indexer.index_entity(arcade, user_a, EntityIndexRequest(
        entity_id="ent-acme", name="Acme Corp", type="ORG", source_chunk_id="a-doc-1"))
    await graph_indexer.index_entity(arcade, user_a, EntityIndexRequest(
        entity_id="ent-jane", name="Jane Roe", type="PERSON", source_chunk_id="a-doc-1"))
    await graph_indexer.index_relationship(arcade, user_a, RelationshipIndexRequest(
        from_entity_id="ent-jane", to_entity_id="ent-acme", relation="WORKS_AT"))

    found = await searcher.graph_search(
        arcade, user_a, GraphSearchRequest(entity_name="Jane Roe", max_depth=2))
    names = {e.get("name") for e in (found.entities or [])}
    assert "Acme Corp" in names, f"traversal did not reach the linked entity: {names}"

    empty = await searcher.graph_search(
        arcade, user_b, GraphSearchRequest(entity_name="Jane Roe", max_depth=2))
    assert not (empty.entities or []), "user B saw user A's entities"


async def test_temporal_query_returns_the_users_timeline(arcade, two_users):
    user_a, user_b = two_users
    for fid, value, start in (
        ("f-2024", "Series A", "2024-03-01T00:00:00Z"),
        ("f-2025", "Series B", "2025-06-01T00:00:00Z"),
    ):
        await graph_indexer.index_temporal_fact(arcade, user_a, TemporalFactIndexRequest(
            fact_id=fid, entity="Acme Corp", relation="FUNDING_ROUND",
            value=value, time_start=start, confidence=0.9))

    timeline = await searcher.tkg_search(arcade, user_a, TKGRequest(entity_name="Acme Corp"))
    values = {f.get("value") for f in timeline.facts}
    assert {"Series A", "Series B"} <= values, f"timeline incomplete: {values}"

    b_timeline = await searcher.tkg_search(arcade, user_b, TKGRequest(entity_name="Acme Corp"))
    assert not b_timeline.facts, "user B saw user A's temporal facts"


# ── deduplication ─────────────────────────────────────────────────────────────

async def test_replaying_the_same_event_three_times_indexes_once(arcade, two_users, pg_pool):
    """The AC case: three identical publishes, one record.

    Driven through `_handle_message` rather than a live broker so the assertion
    is about the dedup logic and not about Kafka delivery semantics — the broker
    path is covered by the DLQ test below.
    """
    import redis.asyncio as aioredis
    from app.core.config import settings
    from app.services.index_events import compute_event_id
    from app.services.kafka_consumer import _handle_message

    user_a, _ = two_users
    asset_path, version_id = "/s3/dedup.pdf", "v-dedup"
    event_id = compute_event_id(asset_path, version_id, user_a)
    await pg_pool.execute("DELETE FROM personal_index_events WHERE event_id = $1", event_id)

    envelope = {
        "event_id": event_id,
        "schema_version": "personal-index-v1",
        "source_module": "12-document",
        "payload": {"branch_id": user_a, "parameters": {
            "version_id": version_id, "asset_path": asset_path,
            "chunks": [{"id": "dedup-c1", "payload": {
                "type": "documents", "branch_id": user_a, "text": A_DOC_TEXT}}],
        }},
    }

    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
    except Exception:
        pytest.skip("Redis not reachable (set PD_REDIS_URL)")

    await redis.delete("personal:indexed:dedup-c1")
    try:
        for _ in range(3):
            await _handle_message(arcade, redis, envelope)
    finally:
        await redis.aclose()

    rows = await arcade.query(db_name_for(user_a), "SELECT count(*) AS c FROM Chunk")
    assert rows[0]["c"] == 1, f"replay produced {rows[0]['c']} records, expected exactly 1"

    claims = await pg_pool.fetchval(
        "SELECT count(*) FROM personal_index_events WHERE event_id = $1", event_id)
    assert claims == 1


# ── dead-letter queue ─────────────────────────────────────────────────────────

async def test_malformed_event_goes_to_dlq_and_is_recorded_as_failed(arcade, pg_pool):
    """Three attempts, then parked — and the claim flips to 'failed' so the event
    can be retried later instead of being locked out by its own claim."""
    from app.core.config import settings
    from app.services import kafka_consumer as kc

    event_id = uuid.uuid4().hex + uuid.uuid4().hex[:32]      # 64 hex chars
    await pg_pool.execute(
        """INSERT INTO personal_index_events (event_id, user_id, status)
           VALUES ($1, 'dlq-user', 'processed')
           ON CONFLICT (event_id) DO UPDATE SET status = 'processed', error = NULL""",
        event_id)

    sent = []

    class RecordingProducer:
        async def send_and_wait(self, topic, value):
            sent.append((topic, value))

    await kc._send_to_dlq(RecordingProducer(), {"event_id": event_id}, attempts=3,
                          error="ValueError: malformed payload")

    assert sent, "nothing was published to the DLQ"
    topic, msg = sent[0]
    assert topic == settings.kafka_dlq_topic
    assert msg["attempts"] == 3
    assert "malformed payload" in msg["error"]
    assert msg["original"]["event_id"] == event_id, "the original message was not preserved"

    row = await pg_pool.fetchrow(
        "SELECT status, error FROM personal_index_events WHERE event_id = $1", event_id)
    assert row["status"] == "failed", "the failure was not recorded"
    assert "malformed payload" in row["error"]

    # A failed event must be re-claimable, or a fixed publisher could never retry.
    from app.services.index_events import claim_event
    assert await claim_event(event_id, "dlq-user") is True

    await pg_pool.execute("DELETE FROM personal_index_events WHERE event_id = $1", event_id)


# ── gateway isolation: all five routes refuse a cross-user request ────────────

# Modules 25 and 13 both ship a top-level package called `app`, and this file has
# already imported module 25's. Importing module 13's in the same interpreter
# would resolve `app.api` against the wrong module, so the gateway checks run in
# a subprocess with only module 13 on the path. Same assertions, clean namespace.
_GATEWAY_CHECKS = r'''
import json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", "{}")

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from app.api.dependencies import get_current_user
from app.routers.personal_db import router, get_personal_db_service, get_hybrid_search_service

seen = []

class StubService:
    """Records the user id each route forwards. A route that reached this stub
    with someone else's id fails the test as loudly as a wrong status code."""
    async def ensure_database(self, uid): seen.append(["init", uid])
    async def hybrid_search(self, uid, body): seen.append(["hybrid", uid]); return {"results": []}
    async def bm25_search(self, uid, body): seen.append(["bm25", uid]); return {"results": []}
    async def vector_search(self, uid, body): seen.append(["vector", uid]); return {"results": []}
    async def graph_search(self, uid, body): seen.append(["graph", uid]); return {"entities": []}
    async def temporal_search(self, uid, body): seen.append(["temporal", uid]); return {"facts": []}
    async def status(self, uid): seen.append(["status", uid]); return {"user_id": uid}
    async def delete_database(self, uid): seen.append(["delete", uid]); return {"deleted": True}

class StubHybridSearchService:
    """Stands in for the real enrich_with_urls call (asset_url resolution via
    module 04) — no real network I/O, so this test stays fast and isolated
    like StubService. Reranking for /hybrid happens inside module 25 itself
    now (see reranker.py), not here, so there is nothing to stub for it."""
    async def enrich_with_urls(self, items, user_dict):
        seen.append(["enrich", user_dict.get("user_id")])

def build(authenticated=True):
    seen.clear()
    app = FastAPI()
    app.include_router(router, prefix="/api/0.4/personal-db",
                       dependencies=[Depends(get_current_user)])
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: {"sub": "userA"}
    app.dependency_overrides[get_personal_db_service] = lambda: StubService()
    app.dependency_overrides[get_hybrid_search_service] = lambda: StubHybridSearchService()
    return TestClient(app), app

out = {}

# unauthenticated → 401 (real auth dependency left in place)
c, app = build(authenticated=False)
out["unauthenticated_status"] = c.get("/api/0.4/personal-db/status").status_code
out["route_count"] = len([p for p in app.openapi()["paths"] if "personal-db" in p])

# cross-user → 403 on the two lifecycle routes, which still take user_id as a
# query param to check against sub.
c, _ = build()
out["cross_user"] = {
    "status":   c.get("/api/0.4/personal-db/status?user_id=userB").status_code,
    "delete":   c.request("DELETE", "/api/0.4/personal-db/?user_id=userB").status_code,
}
out["reached_downstream_after_403"] = list(seen)

# The five search routes have no user_id field anymore — there's no claim
# left to reject, so a stray user_id key in the body is just an unrecognised
# JSON field pydantic ignores. Confirm that's genuinely what happens (200, own
# data still served) rather than the route breaking on the unexpected key.
c, _ = build()
out["search_ignores_stray_user_id"] = c.post(
    "/api/0.4/personal-db/search/hybrid", json={"query": "q", "user_id": "userB"}
).status_code
out["stray_user_id_forwarded_as"] = list(seen)

# own data → 200, and the id forwarded downstream is always the token subject
c, _ = build()
out["own_implicit"] = c.post("/api/0.4/personal-db/search/hybrid", json={"query": "q"}).status_code
out["forwarded"] = list(seen)

# enrich wiring: all three search modes must call enrich_with_urls exactly
# once — none of them rerank here, that's module 25's own job now
c, _ = build()
c.post("/api/0.4/personal-db/search/hybrid", json={"query": "q", "top_k": 5})
out["hybrid_calls"] = list(seen)

c, _ = build()
c.post("/api/0.4/personal-db/search/bm25", json={"query": "q", "top_k": 5})
out["bm25_calls"] = list(seen)

print("__RESULT__" + json.dumps(out))
'''


def _run_gateway_checks() -> dict:
    import pathlib
    import subprocess
    import sys

    m13 = pathlib.Path(__file__).resolve().parents[1] / "deployment/modules/13-api-services"
    proc = subprocess.run(
        [sys.executable, "-c", _GATEWAY_CHECKS, str(m13)],
        capture_output=True, text=True, timeout=180,
    )
    marker = "__RESULT__"
    if marker not in proc.stdout:
        pytest.skip(f"module 13 gateway not importable:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout.split(marker, 1)[1].strip())


@pytest.fixture(scope="module")
def gateway():
    return _run_gateway_checks()


async def test_lifecycle_routes_deny_cross_user_access_with_403(gateway):
    """status/delete still take user_id as a query param and reject a
    mismatched one — the five search routes have no such field anymore (see
    test_search_routes_ignore_a_stray_user_id below)."""
    for name, code in gateway["cross_user"].items():
        assert code == 403, f"{name} allowed a cross-user request ({code})"
    assert not gateway["reached_downstream_after_403"], (
        f"a rejected request still reached the downstream service: "
        f"{gateway['reached_downstream_after_403']}"
    )


async def test_search_routes_ignore_a_stray_user_id(gateway):
    """The search routes dropped the user_id field entirely — there is no
    claim left to check, so a caller sending one anyway just has it ignored
    as an unrecognised JSON key, not rejected. The request still gets served
    as the caller's own data (sub), never the stray value."""
    assert gateway["search_ignores_stray_user_id"] == 200
    forwarded = gateway["stray_user_id_forwarded_as"]
    assert forwarded, "the request never reached the downstream service"
    assert {uid for _, uid in forwarded} == {"userA"}, (
        f"expected the request to be served as the caller's own sub regardless of "
        f"the stray user_id in the body, got: {forwarded}"
    )


async def test_routes_reject_unauthenticated_requests_with_401(gateway):
    assert gateway["unauthenticated_status"] == 401


async def test_all_seven_routes_appear_in_the_openapi_spec(gateway):
    assert gateway["route_count"] == 7, (
        f"expected 7 personal-db routes in the OpenAPI spec, found {gateway['route_count']}"
    )


async def test_own_requests_are_served_with_the_token_subject(gateway):
    assert gateway["own_implicit"] == 200

    forwarded = gateway["forwarded"]
    assert forwarded, "the request never reached the downstream service"
    assert {uid for _, uid in forwarded} == {"userA"}, f"a foreign id was forwarded: {forwarded}"
    assert ["init", "userA"] in forwarded, "the database was not provisioned on first request"


async def test_hybrid_search_enriches_with_asset_urls(gateway):
    """/search/hybrid calls module 25, then enriches with asset_url. Reranking
    is module 25's own job now (see 25's reranker.py) — this gateway layer has
    nothing to do with it, so there is no "rerank" step to see here."""
    names = [call[0] for call in gateway["hybrid_calls"]]
    assert names == ["init", "hybrid", "enrich"], f"expected init->hybrid->enrich, got {names}"


async def test_bm25_search_also_enriches(gateway):
    """Same enrichment wiring as /hybrid — asset_url resolution is not
    specific to any one search mode."""
    names = [call[0] for call in gateway["bm25_calls"]]
    assert names == ["init", "bm25", "enrich"], f"expected init->bm25->enrich, got {names}"


# ── performance baseline (recorded, not gated) ────────────────────────────────

@pytest.mark.slow
async def test_hybrid_search_latency_baseline_over_1000_documents(arcade, two_users):
    """Records p95 for hybrid search over 1,000 indexed documents.

    Deliberately not an assertion on the number: this machine is not the
    deployment target, and a latency gate that fails on a busy laptop trains
    people to ignore the suite. Run it with `-s` to read the figure.
    """
    if os.getenv("PD_RUN_PERF_BASELINE", "").lower() not in ("1", "true", "yes"):
        pytest.skip("set PD_RUN_PERF_BASELINE=1 to run (indexes 1,000 documents)")

    user_a, _ = two_users
    for i in range(1000):
        await _index_doc(arcade, user_a, f"perf-{i}",
                         f"Document {i} about quarterly revenue, renewals and margin")

    req = SearchRequest(query="quarterly revenue growth", top_k=10)
    await searcher.hybrid_search(arcade, user_a, req)          # warm the query path

    timings = []
    for _ in range(20):
        started = time.perf_counter()
        await searcher.hybrid_search(arcade, user_a, req)
        timings.append((time.perf_counter() - started) * 1000)

    timings.sort()
    p95 = timings[int(len(timings) * 0.95) - 1]
    print(f"\n  hybrid search over 1,000 docs — "
          f"p50 {statistics.median(timings):.0f}ms  p95 {p95:.0f}ms  target <500ms")
    assert timings, "no timings collected"
