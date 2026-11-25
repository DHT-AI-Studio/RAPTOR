"""P07 (§5.9): TKG facts selected by their Allen relations to the query window.

The SQL prefilter is run for real on SQLite (same `IS NULL` / comparison subset as ArcadeDB
SQL) and checked exhaustively to be a necessary condition: it never drops a fact the exact
Allen filter keeps.
"""
import sqlite3
from itertools import product

import pytest
from app.models.graph_search import TKGRequest
from app.services import searcher, temporal_match
from app.services.temporal_normalize import normalize_fact_dates
from pydantic import ValidationError

from .test_graph_search import FakeClient, _stub_entity_search

TZ = "Asia/Taipei"


def _fact(fact_id, time_start, time_end, **extra):
    return {"fact_id": fact_id, "entity": "X", "relation": "r", "value": fact_id,
            **normalize_fact_dates(time_start, time_end, TZ), **extra}


def _judge(fact, ts, te, match=None, relations=None, certainty="possible"):
    w0, w1 = temporal_match.window_ms(ts, te, TZ)
    return temporal_match.judge(fact, w0, w1, temporal_match.relation_set(match, relations), certainty, TZ)


# ----------------------------------------------------------- exact filter

def test_tenure_spanning_the_window_overlaps_it_but_is_not_inside():
    """"What was Hugo Barra doing in 2010": served 2008-2013 contains the year."""
    barra = _fact("barra", "2008", "2013")
    assert _judge(barra, "2010", "2010", match="inside")[0] is False
    keep, allen = _judge(barra, "2010", "2010", match="overlaps")
    assert keep and allen == {"vs_window": ["contains"], "certainty": "certain"}
    assert _judge(barra, "2010", "2010", match="contains")[0] is True


def test_year_fact_in_its_own_year_is_certainly_inside_though_several_relations_remain():
    keep, allen = _judge(_fact("y", "2011", "2011"), "2011", "2011", match="inside")
    assert keep and allen["certainty"] == "certain"
    assert set(allen["vs_window"]) == {"starts", "during", "finishes", "equals"}


def test_certain_drops_facts_that_only_may_match():
    # started some time in 2010, ended inside the 2011 window: may overlap, may be inside
    straddle = _fact("s", "2010", "2011-06-30")
    keep, allen = _judge(straddle, "2010-07-01", "2011-12-31", match="inside")
    assert keep and allen["certainty"] == "possible"
    assert "overlaps" in allen["vs_window"] and "during" in allen["vs_window"]
    assert _judge(straddle, "2010-07-01", "2011-12-31", match="inside", certainty="certain")[0] is False


def test_explicit_relations_override_match_and_after_is_strict():
    window = ("2011", "2011")
    assert _judge(_fact("a", "2012-03-01", "2012-03-01"), *window, match="inside", relations=["after"])[0]
    assert not _judge(_fact("b", "2011-12-31", "2011-12-31"), *window, relations=["after"])[0]
    assert not _judge(_fact("c", "2011-06-01", "2012-06-01"), *window, relations=["after"])[0]


def test_one_sided_window_after_2011():
    """"After 2011" = [2012-01-01, unbounded): inside keeps what starts from 2012 on."""
    assert _judge(_fact("in", "2013-05-01", "2014-01-01"), "2012-01-01", None, match="inside") == \
        (True, {"vs_window": ["during"], "certainty": "certain"})
    assert _judge(_fact("out", "2011-05-01", "2011-06-01"), "2012-01-01", None, match="inside")[0] is False
    assert _judge(_fact("cross", "2011-05-01", "2013-01-01"), "2012-01-01", None, match="inside")[0] is False
    assert _judge(_fact("cross", "2011-05-01", "2013-01-01"), "2012-01-01", None, match="overlaps")[0] is True


def test_open_window_side_is_beyond_an_unknown_endpoint():
    """An unknown end is still some instant: it can't pass the end of "from 2012 on". So a fact that
    started in 2013 is certainly inside it -- and one with an unknown start certainly inside "up to 2012"."""
    assert _judge(_fact("ongoing", "2013-05-01", None), "2012-01-01", None, match="inside") == \
        (True, {"vs_window": ["during"], "certainty": "certain"})
    assert _judge(_fact("no_start", None, "2010-05-01"), None, "2012-12-31", match="inside") == \
        (True, {"vs_window": ["during"], "certainty": "certain"})
    keep, allen = _judge(_fact("no_start", None, "2013-05-01"), "2012-01-01", None, match="inside")
    assert keep and allen == {"vs_window": ["overlaps", "starts", "during"], "certainty": "possible"}


def test_open_window_matches_brute_force_on_a_small_line():
    """Classification against [w0, unbounded) equals the relations of every concrete fact on a line long
    enough to put the window's end past all of them (exhaustive, n = 6)."""
    from app.temporal.allen import EndpointRange, IntervalDomain, relate
    n, far = 6, 100
    sides = [(None, None)] + [(lo, hi) for lo in range(n) for hi in range(lo, n)]
    for (sl, sh), (el, eh) in product(sides, repeat=2):
        domain = IntervalDomain(EndpointRange(sl, sh), EndpointRange(el, eh))
        for w0 in range(n):
            want = {relate(s, e, w0, far) for s in range(-1 if sl is None else sl, (n + 1 if sh is None else sh) + 1)
                    for e in range(-1 if el is None else el, (n + 1 if eh is None else eh) + 1) if s <= e}
            assert temporal_match.classify(domain, w0, None) == want, (sl, sh, el, eh, w0)
            mirror = {relate(s, e, -far, w0) for s in range(-1 if sl is None else sl, (n + 1 if sh is None else sh) + 1)
                      for e in range(-1 if el is None else el, (n + 1 if eh is None else eh) + 1) if s <= e}
            assert temporal_match.classify(domain, None, w0) == mirror, (sl, sh, el, eh, w0)


def test_unknown_endpoint_may_match_but_never_certainly():
    no_end = _fact("n", "2010-06-01", None)
    keep, allen = _judge(no_end, "2010", "2010", match="inside")
    assert keep and allen["certainty"] == "possible" and "contains" not in allen["vs_window"]
    assert _judge(no_end, "2010", "2010", match="inside", certainty="certain")[0] is False


def test_point_on_the_window_boundary_overlaps_it():
    point = _fact("p", "2011-12-31", "2011-12-31", extent="point")
    assert _judge(point, "2011", "2011", match="overlaps")[0] is True
    assert _judge(point, "2012", "2012", match="overlaps")[0] is False


def test_legacy_row_is_classified_from_its_strings():
    legacy = {"fact_id": "l", "time_start": "2008", "time_end": "2013"}           # no *_state yet
    assert _judge(legacy, "2010", "2010", match="overlaps")[1] == {"vs_window": ["contains"], "certainty": "certain"}
    unreadable = {"fact_id": "u", "time_start": "sometime", "time_end": None}
    assert _judge(unreadable, "2010", "2010", match="overlaps") == (False, None)
    assert _judge(unreadable, "2010", "2010", match="any") == (True, None)


def test_unknown_relation_name_is_rejected():
    with pytest.raises(ValidationError):
        TKGRequest(query="q", relations=["sideways"])


# ----------------------------------------------------------- SQL prefilter is necessary

_COLUMNS = ("fact_id", "time_start", "time_end", *temporal_match.STATE_COLUMNS)


def _sql_keeps(rows, clauses, params):
    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE TemporalFact ({', '.join(_COLUMNS)})")
    con.executemany(f"INSERT INTO TemporalFact VALUES ({', '.join('?' * len(_COLUMNS))})",
                    [tuple(r.get(c) for c in _COLUMNS) for r in rows])
    where = " AND ".join(clauses) or "1 = 1"
    return {r[0] for r in con.execute(f"SELECT fact_id FROM TemporalFact WHERE {where}", params)}  # nosec B608


def _grid_facts(n=5):
    """Every fact on an n-point line: each endpoint known (a range lo..hi) or unknown."""
    sides = [("unknown", None, None)] + [("known", lo, hi) for lo in range(n) for hi in range(lo, n)]
    rows = []
    for (ss, sl, sh), (es, el, eh) in product(sides, repeat=2):
        for extent in ("span", "point"):
            rows.append({"fact_id": f"{ss}{sl}{sh}-{es}{el}{eh}-{extent}", "s_state": ss, "s_lo": sl, "s_hi": sh,
                         "e_state": es, "e_lo": el, "e_hi": eh, "extent": extent})
    return rows


def test_intersection_prefilter_never_drops_an_accepted_fact():
    rows = _grid_facts()
    clauses, _ = searcher._intersection_window("2010", "2010")
    windows = [(w0, w1) for w0 in range(-1, 6) for w1 in range(w0, 6)] + \
              [(w0, None) for w0 in range(6)] + [(None, w1) for w1 in range(6)]
    checked = 0
    for w0, w1 in windows:
        use = [c for c in clauses if (":te_ms" in c and w1 is not None) or (":ts_after_ms" in c and w0 is not None)]
        params = {k: v for k, v in (("te_ms", w1), ("ts_after_ms", None if w0 is None else w0 - 1)) if v is not None}
        passed = _sql_keeps(rows, use, params)
        for match in ("inside", "contains", "overlaps"):
            accepted = temporal_match.MATCH_RELATIONS[match]
            for row in rows:
                keep, _ = temporal_match.judge(row, w0, w1, accepted, "possible", TZ)
                if keep:
                    assert row["fact_id"] in passed, (row, w0, w1, match)
                    checked += 1
    assert checked > 10_000


def test_intersection_prefilter_lets_legacy_rows_through():
    clauses, params = searcher._intersection_window("2010", "2010")
    legacy = [{"fact_id": "l", "time_start": "2001", "time_end": "2002"}]
    assert _sql_keeps(legacy, clauses, params) == {"l"}          # classified from its strings afterwards


def test_needs_intersection_only_without_before_or_after():
    assert temporal_match.needs_intersection(temporal_match.MATCH_RELATIONS["overlaps"])
    assert not temporal_match.needs_intersection(temporal_match.MATCH_RELATIONS["any"])
    assert not temporal_match.needs_intersection(frozenset({"after"}))


# ----------------------------------------------------------- tkg_search

@pytest.mark.asyncio
async def test_overlaps_reads_by_intersection_and_returns_allen(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "hb", "name": "Hugo Barra", "type": "PERSON"}])
    rows = [{**_fact("barra", "2008", "2013"), "@props": "x"}, _fact("later", "2014", "2016")]
    client = FakeClient([("entity_id IN :eids", rows)])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(
        query="Hugo Barra", entity="Hugo Barra", time_start="2010", time_end="2010", match="overlaps"))
    sql, params = next((s, p) for s, p in client.calls if "entity_id IN :eids" in s)
    assert "s_lo <= :te_ms" in sql and "e_hi > :ts_after_ms" in sql
    assert "time_start >=" not in sql and ":ts)" not in sql                       # not containment
    assert set(params) == {"eids", "te_ms", "ts_after_ms"}
    assert [f["fact_id"] for f in resp.temporal_facts] == ["barra"]
    fact = resp.temporal_facts[0]
    assert fact["allen"] == {"vs_window": ["contains"], "certainty": "certain"}
    assert not set(temporal_match.STATE_COLUMNS) & set(fact) and "@props" not in fact


@pytest.mark.asyncio
async def test_default_keeps_containment_and_annotates(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "e1", "name": "Samsung", "type": "ORG"}])
    client = FakeClient([("entity_id IN :eids", [_fact("s", "2010-06-01", "2010-06-01")])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(
        query="Samsung", entity="Samsung", time_start="2010", time_end="2010"))
    sql, _ = next((s, p) for s, p in client.calls if "entity_id IN :eids" in s)
    assert "time_start IS NULL OR time_start >= :ts" in sql                       # today's containment SQL
    assert resp.temporal_facts[0]["allen"] == {"vs_window": ["during"], "certainty": "certain"}


@pytest.mark.asyncio
async def test_no_window_means_no_allen_and_no_filter(monkeypatch):
    await _stub_entity_search(monkeypatch, [{"entity_id": "e1", "name": "Samsung", "type": "ORG"}])
    client = FakeClient([("entity_id IN :eids", [_fact("s", "2010", "2010")])])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(query="Samsung", entity="Samsung", match="inside"))
    assert [f["fact_id"] for f in resp.temporal_facts] == ["s"] and "allen" not in resp.temporal_facts[0]


@pytest.mark.asyncio
async def test_temporal_only_limit_applies_after_the_exact_filter(monkeypatch):
    await _stub_entity_search(monkeypatch, [])
    rows = [_fact(f"out{i}", "2001", "2002") for i in range(3)] + [_fact("in", "2010-05-01", "2010-06-01")]
    client = FakeClient([("s_state = 'known'", rows)])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(
        query="q", time_start="2010", time_end="2010", match="overlaps", limit=1))
    assert [f["fact_id"] for f in resp.temporal_facts] == ["in"] and resp.complete is True
    read = next(s for s, _ in client.calls if "s_state = 'known'" in s and "LIMIT 1" not in s)
    assert "(e_state <> 'known' OR e_hi > :ts_after_ms)" in read and "s_lo <= :te_ms" in read


@pytest.mark.asyncio
async def test_strict_after_reads_without_a_window_predicate(monkeypatch):
    await _stub_entity_search(monkeypatch, [])
    rows = [_fact("y2011", "2011-12-31", "2011-12-31"), _fact("y2012", "2012-01-01", "2012-02-01")]
    client = FakeClient([("s_state = 'known'", rows)])
    resp = await searcher.tkg_search(client, "demo", TKGRequest(
        query="q", time_start="2011", time_end="2011", relations=["after"]))
    assert [f["fact_id"] for f in resp.temporal_facts] == ["y2012"]
    read = next(s for s, _ in client.calls if "s_state = 'known'" in s and "LIMIT 1" not in s)
    assert ":te_ms" not in read and ":ts_after_ms" not in read
