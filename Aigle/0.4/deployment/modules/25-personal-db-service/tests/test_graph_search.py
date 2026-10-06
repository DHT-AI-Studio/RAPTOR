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
async def _stub_entity_search(monkeypatch, entities):
    """tkg_search() = entity fulltext search -> subgraph -> TemporalFact SQL. Only the last step is
    under test here, so stub the first two and let FakeClient record the TemporalFact query."""
    async def fulltext_search_entities(client, branch_id, query, limit=10, score_threshold=None):
        return entities

    async def get_subgraph(client, branch_id, entity_id, max_depth=2, limit=50):
        return {"nodes": [], "edges": []}

    async def fulltext_search_moments(client, branch_id, query, limit=10, score_threshold=None):
        return []

    monkeypatch.setattr(searcher.graph_query, "fulltext_search_entities", fulltext_search_entities)
    monkeypatch.setattr(searcher.graph_query, "get_subgraph", get_subgraph)
    monkeypatch.setattr(searcher.graph_query, "fulltext_search_moments", fulltext_search_moments)


def _temporal_fact_sql(client):
    return next((sql, params) for sql, params in client.calls if "FROM TemporalFact" in sql)


@pytest.mark.asyncio
async def test_tkg_search_applies_time_window_and_orders_by_time_start(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "e1", "name": "Samsung", "type": "ORG"}])
    facts = [{"fact_id": "tf1", "entity": "Samsung", "entity_id": "e1", "relation": "strike_ruling",
              "value": "production must continue", "time_start": "2026-05",
              "confidence": 0.95, "@props": "confidence:4"}]
    client = FakeClient([("FROM TemporalFact", facts)])
    req = TKGRequest(query="Samsung", time_start="2026-01", time_end="2026-12")
    resp = await searcher.tkg_search(client, "demo", req)

    sql, params = _temporal_fact_sql(client)
    assert "entity_id IN :eids" in sql
    assert "time_start IS NULL OR time_start >= :ts" in sql
    assert "time_end IS NULL OR time_end <= :te" in sql
    assert "ORDER BY time_start ASC" in sql
    assert params == {"eids": ["e1"], "ts": "2026-01", "te": "2026-12"}
    # returned facts are cleaned of record metadata
    assert resp.temporal_facts[0]["fact_id"] == "tf1"
    assert "@props" not in resp.temporal_facts[0]


@pytest.mark.asyncio
async def test_tkg_search_without_window_adds_no_time_clauses(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "e1", "name": "Samsung", "type": "ORG"}])
    client = FakeClient([("FROM TemporalFact", [])])
    await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung"))
    sql, params = _temporal_fact_sql(client)
    assert ":ts" not in sql and ":te" not in sql
    assert params == {"eids": ["e1"]}


@pytest.mark.asyncio
async def test_tkg_search_without_matched_entities_skips_the_fact_query(monkeypatch):
    await _stub_entity_search(monkeypatch, [])
    client = FakeClient([("FROM TemporalFact", [{"fact_id": "never"}])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="nobody"))
    assert resp.temporal_facts == []
    assert not any("FROM TemporalFact" in sql for sql, _ in client.calls)


# --- window semantics: run the generated WHERE fragment for real (SQLite speaks the same
# --- `IS NULL` / string-comparison subset as ArcadeDB SQL, and time_* are STRING properties)
_FACTS = [  # (fact_id, time_start, time_end)
    ("inside", "2005-03-01", "2005-09-30"),
    ("exactly_window", "2005-01-01", "2005-12-31"),
    ("straddles_start", "2004-06-01", "2005-06-01"),
    ("straddles_end", "2005-06-01", "2006-06-01"),
    ("before", "2001-01-01", "2002-01-01"),
    ("after", "2009-01-01", "2010-01-01"),
    ("open_started_inside", "2005-06-01", None),
    ("open_started_after", "2008-01-01", None),
    ("open_started_before", "2001-01-01", None),
    ("no_start_ended_inside", None, "2005-05-01"),
    ("no_start_ended_before", None, "2003-01-01"),
    ("no_start_ended_after", None, "2009-01-01"),
    ("no_bounds", None, None),
]


def _facts_in_window(clauses, params):
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE TemporalFact (fact_id TEXT, time_start TEXT, time_end TEXT)")
    con.executemany("INSERT INTO TemporalFact VALUES (?, ?, ?)", _FACTS)
    where = " AND ".join(clauses) or "1 = 1"
    return {r[0] for r in con.execute(f"SELECT fact_id FROM TemporalFact WHERE {where}", params)}


def test_time_window_returns_only_facts_inside_the_window():
    got = _facts_in_window(*searcher._temporal_fact_window("2005-01-01", "2005-12-31"))
    assert got == {"inside", "exactly_window", "open_started_inside", "no_start_ended_inside", "no_bounds"}


def test_open_ended_fact_starting_after_the_window_is_not_returned():
    got = _facts_in_window(*searcher._temporal_fact_window("2005-01-01", "2005-12-31"))
    assert "open_started_after" not in got          # no time_end used to bypass the upper bound
    assert "open_started_inside" in got             # ...but one that starts inside the window still counts


def test_fact_without_start_that_ended_before_the_window_is_not_returned():
    got = _facts_in_window(*searcher._temporal_fact_window("2005-01-01", "2005-12-31"))
    assert "no_start_ended_before" not in got       # no time_start used to bypass the lower bound
    assert "no_start_ended_inside" in got


def test_only_time_start_bound():
    got = _facts_in_window(*searcher._temporal_fact_window("2005-01-01", None))
    assert got == {"inside", "exactly_window", "straddles_end", "after", "open_started_inside", "open_started_after",
                   "no_start_ended_inside", "no_start_ended_after", "no_bounds"}


def test_only_time_end_bound():
    got = _facts_in_window(*searcher._temporal_fact_window(None, "2005-12-31"))
    assert got == {"inside", "exactly_window", "straddles_start", "before", "open_started_inside",
                   "open_started_before", "no_start_ended_inside", "no_start_ended_before", "no_bounds"}


def test_no_window_returns_everything():
    assert _facts_in_window(*searcher._temporal_fact_window(None, None)) == {f[0] for f in _FACTS}


def test_previous_clauses_leaked_facts_through_a_null_endpoint():
    """Documents the bug: the original two clauses let a NULL endpoint bypass the opposite bound."""
    legacy = ["(time_start IS NULL OR time_start >= :ts)", "(time_end IS NULL OR time_end <= :te)"]
    got = _facts_in_window(legacy, {"ts": "2005-01-01", "te": "2005-12-31"})
    assert {"open_started_after", "no_start_ended_before"} <= got
