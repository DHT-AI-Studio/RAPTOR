"""TKG time window, executed by a REAL ArcadeDB.

`tkg_search` restricts TemporalFacts to a [time_start, time_end] window: `time_start` is a lower
bound on fact.time_start, `time_end` an upper bound on fact.time_end, a NULL endpoint imposes no
constraint on its own side -- but must not let the fact escape the opposite bound of the window
(an open-ended fact that starts after the window, a fact with no start that ended before it).

test_graph_search.py covers the generated SQL and, with an in-memory SQLite table, the semantics of
the predicate offline. These tests run it where it matters -- ArcadeDB's own SQL dialect, its NULL
handling and its STRING comparison -- through the real indexer and `searcher.tkg_search()`. They need
a running ArcadeDB (see conftest.py) and skip otherwise.
"""
from __future__ import annotations

import pytest

from app.models.graph_index import EntityIndexRequest, TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher
from app.services.arcadedb_client import db_name_for

pytestmark = pytest.mark.asyncio

# (fact_id, time_start, time_end); ISO date strings, one format throughout (STRING compare)
_FACTS = [
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

_ALL = {f[0] for f in _FACTS}

# window (time_start, time_end) -> the facts that must come back (containment; NULL = unconstrained
# on its own side, never a way around the other bound)
_CASES = {
    "both bounds": (("2005-01-01", "2005-12-31"),
                    {"inside", "exactly_window", "open_started_inside", "no_start_ended_inside", "no_bounds"}),
    "only time_start": (("2005-01-01", None),
                        {"inside", "exactly_window", "straddles_end", "after", "open_started_inside", "open_started_after",
                         "no_start_ended_inside", "no_start_ended_after", "no_bounds"}),
    "only time_end": ((None, "2005-12-31"),
                      {"inside", "exactly_window", "straddles_start", "before", "open_started_inside",
                       "open_started_before", "no_start_ended_inside", "no_start_ended_before", "no_bounds"}),
    "no window": ((None, None), _ALL),
}


async def _seed(client, make_db) -> str:
    branch = await make_db("ittest_tkg_window")
    # distractor entities keep the tiny corpus's BM25 score for "Android" above zero
    for eid, name in [("e1", "Android"), ("e2", "Nexus"), ("e3", "Pixel")]:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id=eid, name=name, type="PRODUCT"))
    for fid, ts, te in _FACTS:
        await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
            fact_id=fid, entity="Android", entity_id="e1", relation="r", value=fid, time_start=ts, time_end=te))
    # another entity's fact lying inside the window must never show up in an "Android" query
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="other_entity", entity="Nexus", entity_id="e2", relation="r", value="x",
        time_start="2005-03-01", time_end="2005-09-30"))
    return branch


async def _tkg(client, branch, window) -> list[str]:
    ts, te = window
    resp = await searcher.tkg_search(client, branch, TKGRequest(
        query="Android", time_start=ts, time_end=te, score_threshold=0))
    return [f["fact_id"] for f in resp.temporal_facts]


async def test_tkg_search_window_semantics_in_arcadedb(client, make_db):
    branch = await _seed(client, make_db)
    for name, (window, expected) in _CASES.items():
        got = await _tkg(client, branch, window)
        assert set(got) == expected, (
            f"{name} {window}: unexpected={sorted(set(got) - expected)} missing={sorted(expected - set(got))}")
        assert len(got) == len(set(got)), f"{name}: a fact was returned twice"
        assert "other_entity" not in got, f"{name}: another entity's fact leaked into the result"


async def test_open_ended_and_unknown_start_facts_do_not_escape_the_window(client, make_db):
    """Regression for the NULL-endpoint leak, named after the two facts that used to come back."""
    branch = await _seed(client, make_db)
    got = set(await _tkg(client, branch, ("2005-01-01", "2005-12-31")))
    assert "open_started_after" not in got       # no time_end used to bypass the window's upper bound
    assert "no_start_ended_before" not in got    # no time_start used to bypass the window's lower bound
    assert "open_started_inside" in got          # an open-ended fact that starts inside the window still counts
    assert "no_start_ended_inside" in got


async def test_previous_predicate_leaks_in_arcadedb(client, make_db):
    """Guard against a vacuous test: with the previous two-clause predicate, ArcadeDB itself returns
    the two facts that lie completely outside the window, so the data above does exercise the bug."""
    branch = await _seed(client, make_db)
    rows = await client.query(
        db_name_for(branch),
        "SELECT fact_id FROM TemporalFact WHERE entity_id = :eid "
        "AND (time_start IS NULL OR time_start >= :ts) AND (time_end IS NULL OR time_end <= :te)",
        params={"eid": "e1", "ts": "2005-01-01", "te": "2005-12-31"})
    leaked = {r["fact_id"] for r in rows}
    assert {"open_started_after", "no_start_ended_before"} <= leaked
