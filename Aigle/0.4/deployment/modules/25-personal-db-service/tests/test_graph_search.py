"""Unit tests for PA-7 graph traversal, TKG query and the read-only SQL guard.

The service functions talk to ArcadeDB only through ArcadeDBClient, so we swap in
a FakeClient that dispatches canned results by SQL substring — no live DB needed.
"""
import pytest

from app.models.graph_search import GraphSearchRequest, TKGRequest
from app.services import searcher


class FakeClient:
    """Minimal ArcadeDBClient stand-in: routes query() by SQL substring."""
    def __init__(self, routes):
        self.routes = routes            # list[(substr, result_rows)]
        self.calls = []                 # (sql, params) seen, for assertions

    async def database_exists(self, db):
        return True

    async def query(self, db, sql, params=None):
        self.calls.append((sql, params))
        for substr, rows in self.routes:
            if substr in sql:
                return rows
        return []


# ----------------------------------------------------------- read-only guard
@pytest.mark.parametrize("sql", [
    "SELECT FROM Entity",
    "  select name from Entity where type = 'ORG'  ",
    "SELECT out('MENTIONS').name FROM Chunk",
    "SELECT FROM RELATION;",                       # trailing semicolon tolerated
])
def test_select_is_allowed(sql):
    assert searcher.is_read_only_select(sql) is True


@pytest.mark.parametrize("sql", [
    "INSERT INTO Entity SET name = 'x'",
    "UPDATE Entity SET name = 'x' WHERE name = 'y'",
    "DELETE FROM Entity WHERE name = 'x'",
    "CREATE VERTEX TYPE Foo",
    "DROP DATABASE user_x",
    "TRUNCATE TYPE Entity",
    "ALTER TYPE Entity",
    "SELECT FROM Entity; DROP DATABASE user_x",     # statement chaining
    "SELECT FROM Entity WHERE x IN (DELETE FROM y)", # DML hidden in subquery
    "",
    "   ",
])
def test_non_select_is_rejected(sql):
    assert searcher.is_read_only_select(sql) is False


# ----------------------------------------------------------- graph traversal
@pytest.mark.asyncio
async def test_graph_search_dedupes_entities_and_builds_edges_and_paths():
    client = FakeClient([
        # TRAVERSE returns the seed twice (dupe) + two neighbours
        ("TRAVERSE both('RELATION')", [
            {"name": "Samsung", "entity_id": "samsung", "type": "ORG", "mention_count": 4},
            {"name": "Samsung", "entity_id": "samsung", "type": "ORG", "mention_count": 4},
            {"name": "Court", "entity_id": "court", "type": "ORG", "mention_count": 1},
            {"name": "Labor Union", "entity_id": "union", "type": "ORG", "mention_count": 3},
        ]),
        ("FROM RELATION WHERE", [
            {"relation": "ruled_on", "from_name": "Court", "from_id": "court",
             "to_name": "Samsung", "to_id": "samsung", "confidence": 0.95, "@props": "x"},
            {"relation": "negotiates_with", "from_name": "Samsung", "from_id": "samsung",
             "to_name": "Labor Union", "to_id": "union", "confidence": 0.9},
        ]),
        ("shortestPath", [{"path": ["Samsung", "Court"], "@props": "path:9"}]),
    ])
    req = GraphSearchRequest(entity_name="Samsung", max_depth=2)
    resp = await searcher.graph_search(client, "demo", req)

    # seed appears once despite the duplicate row
    ids = [e["entity_id"] for e in resp.entities]
    assert ids == ["samsung", "court", "union"]
    # metadata keys are stripped from projected rows
    assert all("@props" not in e for e in resp.entities)
    # edges parsed into GraphEdge and @props dropped
    assert len(resp.edges) == 2
    assert resp.edges[0].relation == "ruled_on"
    assert resp.edges[0].from_name == "Court" and resp.edges[0].to_name == "Samsung"
    # a shortest path was collected for each non-seed target (Court, Labor Union)
    assert ["Samsung", "Court"] in resp.paths


@pytest.mark.asyncio
async def test_graph_search_clamps_depth_into_maxdepth_literal():
    client = FakeClient([("TRAVERSE both('RELATION')", [])])
    await searcher.graph_search(client, "demo", GraphSearchRequest(entity_name="X", max_depth=99))
    traverse_sql = client.calls[0][0]
    assert "MAXDEPTH 5" in traverse_sql      # clamped to 5, inlined as an int literal


@pytest.mark.asyncio
async def test_graph_search_query_override_must_be_select():
    client = FakeClient([])
    req = GraphSearchRequest(entity_name="X", query="DROP DATABASE user_x")
    with pytest.raises(ValueError):
        await searcher.graph_search(client, "demo", req)


# ----------------------------------------------------------- TKG query
@pytest.mark.asyncio
async def test_tkg_search_applies_filters_and_orders_by_confidence():
    facts = [
        {"fact_id": "tf1", "entity": "Samsung", "relation": "strike_ruling",
         "value": "production must continue", "time_start": "2026-05",
         "confidence": 0.95, "@props": "confidence:4"},
    ]
    client = FakeClient([("FROM TemporalFact", facts)])
    req = TKGRequest(entity_name="Samsung", time_start="2026-01", time_end="2026-12", top_k=10)
    resp = await searcher.tkg_search(client, "demo", req)

    sql, params = client.calls[0]
    assert "entity = :en" in sql
    assert "time_start IS NULL OR time_start >= :ts" in sql
    assert "time_end IS NULL OR time_end <= :te" in sql
    assert "ORDER BY confidence DESC" in sql
    assert "LIMIT 10" in sql
    assert params == {"en": "Samsung", "ts": "2026-01", "te": "2026-12"}
    # returned facts are cleaned of record metadata
    assert resp.facts[0]["fact_id"] == "tf1"
    assert "@props" not in resp.facts[0]


@pytest.mark.asyncio
async def test_tkg_search_no_filters_has_no_where():
    client = FakeClient([("FROM TemporalFact", [])])
    await searcher.tkg_search(client, "demo", TKGRequest())
    sql = client.calls[0][0]
    assert "WHERE" not in sql
    assert "ORDER BY confidence DESC" in sql
