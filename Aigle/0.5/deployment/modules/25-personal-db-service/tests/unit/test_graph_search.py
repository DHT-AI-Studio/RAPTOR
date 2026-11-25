"""Unit tests for PA-7 graph traversal, TKG query and the read-only SQL guard.

The service functions talk to ArcadeDB only through ArcadeDBClient, so we swap in
a FakeClient that dispatches canned results by SQL substring — no live DB needed.
"""
import pytest
from app.models.graph_search import GraphSearchRequest, TKGRequest
from app.services import searcher
from app.services.temporal_normalize import DateParseError, normalize_fact_dates


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
        # _resolve_entity_id(): the seed is found through the Entity(name) full-text index
        ("SEARCH_INDEX('Entity[name]'", [{"entity_id": "samsung", "name": "Samsung"}]),
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
    client = FakeClient([
        ("SEARCH_INDEX('Entity[name]'", [{"entity_id": "x", "name": "X"}]),
        ("TRAVERSE both('RELATION')", []),
    ])
    await searcher.graph_search(client, "demo", GraphSearchRequest(entity_name="X", max_depth=99))
    traverse_sql = next(sql for sql, _ in client.calls if "TRAVERSE" in sql)   # after the seed lookup
    assert "MAXDEPTH 5" in traverse_sql      # clamped to 5, inlined as an int literal


@pytest.mark.asyncio
async def test_graph_search_unknown_seed_returns_nothing():
    """Full-text matching is over-inclusive (bigrams), so the seed is only accepted on
    an exact name match; no exact match means no traversal at all."""
    client = FakeClient([("SEARCH_INDEX('Entity[name]'", [{"entity_id": "x2", "name": "X2"}])])
    resp = await searcher.graph_search(client, "demo", GraphSearchRequest(entity_name="X", max_depth=2))
    assert resp.entities == [] and resp.edges == [] and resp.paths == []
    assert not any("TRAVERSE" in sql for sql, _ in client.calls)


@pytest.mark.asyncio
async def test_graph_search_query_override_must_be_select():
    client = FakeClient([])
    req = GraphSearchRequest(entity_name="X", query="DROP DATABASE user_x")
    with pytest.raises(ValueError):
        await searcher.graph_search(client, "demo", req)


# ----------------------------------------------------------- TKG query
# tkg_search() with an explicit `entity`: entity full-text search -> subgraph
# expansion -> TemporalFacts of the matched entities (TMP-4: the gate applies only then). The full-text search and the
# subgraph expansion live in graph_query and are stubbed here; FakeClient only
# answers the TemporalFact query this test is about.
@pytest.fixture
def tkg_graph(monkeypatch):
    calls = {"moments": 0}

    async def _entities(client, branch_id, query_text, limit=10, score_threshold=None):
        return [{"entity_id": "samsung", "name": "Samsung"}] if "Samsung" in query_text else []

    async def _subgraph(client, branch_id, entity_id, max_depth=2, limit=50):
        return {"nodes": [], "edges": []}

    async def _moments(client, branch_id, query_text, limit=10, strategy="hybrid", score_threshold=None):
        calls["moments"] += 1
        return []

    monkeypatch.setattr(searcher.graph_query, "fulltext_search_entities", _entities)
    monkeypatch.setattr(searcher.graph_query, "get_subgraph", _subgraph)
    monkeypatch.setattr(searcher.graph_query, "fulltext_search_moments", _moments)
    return calls


@pytest.mark.asyncio
async def test_tkg_search_filters_facts_of_matched_entities_by_time_window(tkg_graph):
    facts = [
        {"fact_id": "tf1", "entity": "Samsung", "entity_id": "samsung", "relation": "strike_ruling",
         "value": "production must continue", "time_start": "2026-05",
         "confidence": 0.95, "@props": "confidence:4"},
    ]
    client = FakeClient([("FROM TemporalFact", facts)])
    req = TKGRequest(query="Samsung strike", entity="Samsung", time_start="2026-01", time_end="2026-12")
    resp = await searcher.tkg_search(client, "demo", req)

    sql, params = next((sql, p) for sql, p in client.calls if "FROM TemporalFact" in sql)
    assert "entity_id IN :eids" in sql
    assert "(status IS NULL OR status = 'active')" in sql      # pre-migration facts count as active
    assert "time_start IS NULL OR time_start >= :ts" in sql
    assert "time_end IS NULL OR time_end <= :te" in sql
    assert "ORDER BY time_start ASC" in sql
    # TMP-3: the window is parsed -- "2026-01".."2026-12" is all of Jan..Dec 2026, as UTC ms for
    # normalised rows and as calendar dates for rows not yet migrated
    assert params == {"eids": ["samsung"], "ts": "2026-01-01", "te": "2026-12-31",
                      "ts_after_ms": _ms("2026-01-01T00:00:00+08:00") - 1, "te_ms": _ms("2027-01-01T00:00:00+08:00") - 1}
    assert [e["id"] for e in resp.matched_entities] == ["samsung"]
    # returned facts are cleaned of record metadata
    assert resp.temporal_facts[0]["fact_id"] == "tf1"
    assert "@props" not in resp.temporal_facts[0]


@pytest.mark.asyncio
async def test_tkg_search_without_time_window_has_no_time_clauses(tkg_graph):
    client = FakeClient([("FROM TemporalFact", [])])
    await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung", entity="Samsung"))
    sql = next(sql for sql, _ in client.calls if "FROM TemporalFact" in sql)
    assert "time_start >=" not in sql and "time_end <=" not in sql
    assert "ORDER BY time_start ASC" in sql


@pytest.mark.asyncio
async def test_tkg_search_with_no_matched_entity_skips_facts(tkg_graph):
    """A named entity that matches nothing: no TemporalFact query at all (TMP-4: never
    silently widened to every fact), and moments are searched directly by the query text
    instead (Module 20's fallback)."""
    client = FakeClient([("FROM TemporalFact", [{"fact_id": "should-not-appear"}])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="nothing known", entity="LG"))
    assert not any("FROM TemporalFact" in sql for sql, _ in client.calls)
    assert resp.temporal_facts == [] and resp.matched_entities == []
    assert tkg_graph["moments"] == 1

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
    req = TKGRequest(query="Samsung", entity="Samsung", time_start="2026-01", time_end="2026-12")
    resp = await searcher.tkg_search(client, "demo", req)

    sql, params = _temporal_fact_sql(client)
    assert "entity_id IN :eids" in sql
    assert "time_start IS NULL OR time_start >= :ts" in sql
    assert "time_end IS NULL OR time_end <= :te" in sql
    assert "ORDER BY time_start ASC" in sql
    assert params == {"eids": ["e1"], "ts": "2026-01-01", "te": "2026-12-31",
                      "ts_after_ms": _ms("2026-01-01T00:00:00+08:00") - 1, "te_ms": _ms("2027-01-01T00:00:00+08:00") - 1}
    # returned facts are cleaned of record metadata
    assert resp.temporal_facts[0]["fact_id"] == "tf1"
    assert "@props" not in resp.temporal_facts[0]


@pytest.mark.asyncio
async def test_tkg_search_without_window_adds_no_time_clauses(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "e1", "name": "Samsung", "type": "ORG"}])
    client = FakeClient([("FROM TemporalFact", [])])
    await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung", entity="Samsung"))
    sql, params = _temporal_fact_sql(client)
    assert ":ts" not in sql and ":te" not in sql
    assert params == {"eids": ["e1"]}


@pytest.mark.asyncio
async def test_tkg_search_without_matched_entities_skips_the_fact_query(monkeypatch):
    await _stub_entity_search(monkeypatch, [])
    client = FakeClient([("FROM TemporalFact", [{"fact_id": "never"}])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="nobody", entity="nobody"))
    assert resp.temporal_facts == [] and resp.retrieval == "entity"
    assert not any("FROM TemporalFact" in sql for sql, _ in client.calls)


# --- TMP-4 (D13): without a named entity, facts are not gated on an entity match

@pytest.mark.asyncio
async def test_no_entity_means_temporal_only_retrieval(monkeypatch):
    await _stub_entity_search(monkeypatch, [])                    # the query text matches no entity
    facts = [{"fact_id": "b", "entity": "Android", "relation": "r", "value": "v", "time_start": "2013-01-01"},
             {"fact_id": "a", "entity": "Google", "relation": "r", "value": "v", "time_start": "2012-05-01"}]
    client = FakeClient([("s_state = 'known'", facts)])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="What happened", time_start="2012-01-01"))
    assert resp.retrieval == "temporal" and resp.complete is True
    assert [f["fact_id"] for f in resp.temporal_facts] == ["a", "b"]          # by time_start
    sqls = [sql for sql, _ in client.calls if "FROM TemporalFact" in sql and "LIMIT 1" not in sql]
    assert len(sqls) == 2 and not any("entity_id IN" in q for q in sqls)   # no untimelined rows: no 3rd read
    indexed = next(q for q in sqls if "s_state = 'known'" in q and "timeline_id" in q)
    assert "s_lo > :ts_after_ms" in indexed and "s_lo >=" not in indexed     # exclusive: ArcadeDB index bug


@pytest.mark.asyncio
async def test_query_that_names_an_entity_is_gated_on_it(monkeypatch):
    """No `entity` parameter, but the query text fulltext-matches one: its facts only."""
    await _stub_entity_search(monkeypatch, [{"entity_id": "samsung", "name": "三星電子", "type": "ORG"}])
    client = FakeClient([("entity_id IN :eids", [{"fact_id": "s1", "time_start": "2010-06-01"}])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="三星電子發生了什麼事"))
    assert resp.retrieval == "entity" and [f["fact_id"] for f in resp.temporal_facts] == ["s1"]
    _, params = _temporal_fact_sql(client)
    assert params["eids"] == ["samsung"]


@pytest.mark.asyncio
async def test_matched_entity_also_gets_facts_that_mention_its_name(monkeypatch):
    """"Samsung 2010-2016": Samsung's own facts, plus other entities' facts whose evidence names Samsung."""
    await _stub_entity_search(monkeypatch, [{"entity_id": "samsung", "name": "Samsung", "type": "ORG"}])
    own = [{"fact_id": "own", "entity": "Samsung", "entity_id": "samsung", "relation": "released",
            "value": "Galaxy S", "time_start": "2010-06-01"}]
    window = [{"fact_id": "own", "entity": "Samsung", "relation": "released", "value": "Galaxy S",
               "time_start": "2010-06-01", "evidence": []},                                 # already Samsung's: not twice
              {"fact_id": "s4", "entity": "Google", "relation": "announced",
               "value": "a special version of the Samsung Galaxy S4", "time_start": "2013-05-01", "evidence": []},
              {"fact_id": "passage", "entity": "Android", "relation": "partner", "value": "OEM",
               "time_start": "2009-01-01", "evidence": ["Samsung shipped it first"]},
              {"fact_id": "other", "entity": "HTC", "relation": "released", "value": "Dream",
               "time_start": "2008-10-22", "evidence": []}]
    client = FakeClient([("entity_id IN :eids", own), ("s_state = 'known'", window)])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung 2010-2016", keywords=["Samsung"]))
    assert resp.retrieval == "entity" and resp.complete is True
    assert [f["fact_id"] for f in resp.temporal_facts] == ["passage", "own", "s4"]        # by time_start
    assert all("evidence" not in f for f in resp.temporal_facts)


@pytest.mark.asyncio
async def test_matched_entity_without_a_name_to_look_for_keeps_its_own_facts(monkeypatch):
    """Only the query text matched: no keywords and no `entity`, so nothing names what to look for."""
    await _stub_entity_search(monkeypatch, [{"entity_id": "samsung", "name": "Samsung", "type": "ORG"}])
    client = FakeClient([("entity_id IN :eids", [{"fact_id": "own", "time_start": "2010-06-01"}]),
                         ("s_state = 'known'", [{"fact_id": "s4", "entity": "Google", "value": "Samsung Galaxy S4"}])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung"))
    assert [f["fact_id"] for f in resp.temporal_facts] == ["own"]
    assert not any("s_state = 'known'" in sql for sql, _ in client.calls)   # no window-candidate read


@pytest.mark.asyncio
async def test_keywords_must_all_appear_in_the_evidence(monkeypatch):
    await _stub_entity_search(monkeypatch, [])
    facts = [{"fact_id": "lg", "entity": "LG", "relation": "released", "value": "G2 phone", "evidence": []},
             {"fact_id": "htc", "entity": "HTC", "relation": "released", "value": "Dream phone",
              "evidence": ["The HTC Dream was the first Android phone"]},
             {"fact_id": "lg_passage", "entity": "Android", "relation": "partner", "value": "OEM",
              "evidence": ["LG built a phone running it"]}]
    client = FakeClient([("s_state = 'known'", facts)])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="LG phone", keywords=["lg", "Phone"]))
    assert resp.retrieval == "keywords"
    assert {f["fact_id"] for f in resp.temporal_facts} == {"lg", "lg_passage"}
    assert all("evidence" not in f for f in resp.temporal_facts)              # not part of the response


@pytest.mark.asyncio
async def test_temporal_only_results_are_limited_after_filtering_and_flag_incomplete(monkeypatch):
    await _stub_entity_search(monkeypatch, [])
    facts = [{"fact_id": f"f{i}", "entity": "X", "relation": "r", "value": "v", "time_start": f"20{10 + i}-01-01"}
             for i in range(5)]
    client = FakeClient([("s_state = 'known'", facts)])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="q", limit=3))
    assert [f["fact_id"] for f in resp.temporal_facts] == ["f0", "f1", "f2"] and resp.complete is False


# --- window semantics: run the generated WHERE fragment for real (SQLite speaks the same
# --- `IS NULL` / comparison subset as ArcadeDB SQL; time_* are STRING properties, the TMP-3
# --- ranges LONG). Every matrix test runs twice: on rows written before TMP-3 (no *_state:
# --- the string fallback) and on the same rows normalised (UTC ms ranges).
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
_COLUMNS = ("fact_id", "time_start", "time_end", "s_state", "e_state", "s_lo", "s_hi", "e_lo", "e_hi")


def _ms(iso: str) -> int:
    from datetime import datetime
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def _row(fact_id, time_start, time_end, normalised):
    if not normalised:
        return (fact_id, time_start, time_end) + (None,) * 6
    d = normalize_fact_dates(time_start, time_end, "Asia/Taipei")
    return (fact_id,) + tuple(d[c] for c in _COLUMNS[1:])


def _facts_in_window(clauses, params, facts=_FACTS, normalised=False):
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE TemporalFact ({', '.join(_COLUMNS)})")
    con.executemany(f"INSERT INTO TemporalFact VALUES ({', '.join('?' * len(_COLUMNS))})",
                    [_row(*f, normalised) for f in facts])
    where = " AND ".join(clauses) or "1 = 1"
    return {r[0] for r in con.execute(f"SELECT fact_id FROM TemporalFact WHERE {where}", params)}


@pytest.fixture(params=[False, True], ids=["legacy_rows", "normalised_rows"])
def in_window(request):
    """_facts_in_window over the 13-fact matrix, on legacy or on normalised rows."""
    return lambda window: _facts_in_window(*window, normalised=request.param)


def test_time_window_returns_only_facts_inside_the_window(in_window):
    got = in_window(searcher._temporal_fact_window("2005-01-01", "2005-12-31"))
    assert got == {"inside", "exactly_window", "open_started_inside", "no_start_ended_inside", "no_bounds"}


def test_open_ended_fact_starting_after_the_window_is_not_returned(in_window):
    got = in_window(searcher._temporal_fact_window("2005-01-01", "2005-12-31"))
    assert "open_started_after" not in got          # no time_end used to bypass the upper bound
    assert "open_started_inside" in got             # ...but one that starts inside the window still counts


def test_fact_without_start_that_ended_before_the_window_is_not_returned(in_window):
    got = in_window(searcher._temporal_fact_window("2005-01-01", "2005-12-31"))
    assert "no_start_ended_before" not in got       # no time_start used to bypass the lower bound
    assert "no_start_ended_inside" in got


def test_only_time_start_bound(in_window):
    got = in_window(searcher._temporal_fact_window("2005-01-01", None))
    assert got == {"inside", "exactly_window", "straddles_end", "after", "open_started_inside", "open_started_after",
                   "no_start_ended_inside", "no_start_ended_after", "no_bounds"}


def test_only_time_end_bound(in_window):
    got = in_window(searcher._temporal_fact_window(None, "2005-12-31"))
    assert got == {"inside", "exactly_window", "straddles_start", "before", "open_started_inside",
                   "open_started_before", "no_start_ended_inside", "no_start_ended_before", "no_bounds"}


def test_no_window_returns_everything(in_window):
    assert in_window(searcher._temporal_fact_window(None, None)) == {f[0] for f in _FACTS}


def test_previous_clauses_leaked_facts_through_a_null_endpoint():
    """Documents the bug: the original two clauses let a NULL endpoint bypass the opposite bound."""
    legacy = ["(time_start IS NULL OR time_start >= :ts)", "(time_end IS NULL OR time_end <= :te)"]
    got = _facts_in_window(legacy, {"ts": "2005-01-01", "te": "2005-12-31"})
    assert {"open_started_after", "no_start_ended_before"} <= got


# --- TMP-3: canonical dates (D3). Normalised rows are compared as UTC ms ranges, never as strings.

def test_window_sent_as_a_timestamp_still_matches_a_fact_starting_that_day():
    """D3: Module 18 sends "2011-01-01T00:00:00"; as strings, "2011-01-01" < that, so a fact
    starting on the window's first day failed its own lower bound."""
    facts = [("starts_day_one", "2011-01-01", "2011-03-01")]
    window = searcher._temporal_fact_window("2011-01-01T00:00:00", "2011-12-31T23:59:59")
    assert _facts_in_window(*window, facts=facts, normalised=True) == {"starts_day_one"}


def test_year_only_fact_is_the_whole_year():
    """D4: "2011" is all of 2011 -- inside a 2011 window, outside a window that ends mid-year."""
    facts = [("year_2011", "2011", "2011")]
    assert _facts_in_window(*searcher._temporal_fact_window("2011", "2011"), facts=facts, normalised=True) == {"year_2011"}
    assert _facts_in_window(*searcher._temporal_fact_window("2011-01-01", "2011-06-30"),
                            facts=facts, normalised=True) == set()


def test_mixed_date_forms_compare_by_instant_not_by_string():
    facts = [("month", "2011-05", None), ("datetime_utc", "2011-05-03T10:00:00Z", None), ("slashes", "2011/05/03", None)]
    got = _facts_in_window(*searcher._temporal_fact_window("2011-05-01", "2011-05-31"), facts=facts, normalised=True)
    assert got == {"month", "datetime_utc", "slashes"}


def test_window_bound_in_another_zone_is_converted():
    # 2011-05-03 23:30 in Taipei is 15:30 UTC: inside a UTC window ending 16:00, not one ending 15:00
    facts = [("late_evening", "2011-05-03T23:30:00+08:00", None)]
    assert _facts_in_window(*searcher._temporal_fact_window("2011-05-03T00:00:00Z", "2011-05-03T16:00:00Z"),
                            facts=facts, normalised=True) == {"late_evening"}
    assert _facts_in_window(*searcher._temporal_fact_window("2011-05-03T00:00:00Z", "2011-05-03T15:00:00Z"),
                            facts=facts, normalised=True) == set()


def test_unparseable_window_bound_is_rejected_not_dropped():
    with pytest.raises(DateParseError):
        searcher._temporal_fact_window("2011-13-01", None)
    assert issubclass(DateParseError, ValueError)        # graph router maps ValueError -> 400


@pytest.mark.asyncio
async def test_tkg_search_rejects_a_bad_window_before_querying(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "e1", "name": "Samsung", "type": "ORG"}])
    client = FakeClient([("FROM TemporalFact", [])])
    with pytest.raises(ValueError):
        await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung", time_end="sometime"))
    assert not any("FROM TemporalFact" in sql for sql, _ in client.calls)


def test_module18_window_covers_whole_days():
    """Module 18 sends "2011" as 2011-01-01T00:00:00.000+08:00 .. 2011-12-31T23:59:59.999+08:00."""
    facts = [("year_only", "2011", "2011"), ("ends_new_years_eve", "2011-06-01", "2011-12-31"),
             ("ends_next_day", "2011-06-01", "2012-01-01")]
    window = searcher._temporal_fact_window("2011-01-01T00:00:00.000+08:00", "2011-12-31T23:59:59.999+08:00")
    assert _facts_in_window(*window, facts=facts, normalised=True) == {"year_only", "ends_new_years_eve"}


@pytest.mark.parametrize("keyword, found", [
    ("releases", True),     # plural -> "release" in "released"
    ("phones", True),
    ("Android", True),
    ("glass", False),       # "ss" is not a plural
    ("LG", False),
])
def test_keyword_match_tolerates_english_plurals(keyword, found):
    fact = {"entity": "HTC", "relation": "released", "value": "Dream phone", "evidence": ["the first Android device"]}
    assert searcher._mentions_all(fact, [keyword]) is found
