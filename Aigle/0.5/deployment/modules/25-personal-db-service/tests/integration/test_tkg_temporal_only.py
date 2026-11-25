"""TMP-4 (D13) temporal-only retrieval, executed by a REAL ArcadeDB.

A small Android-history tenant (the plan's Appendix A queries, no LG facts in it) is
indexed through the real write path. Covers: a generic question with no entity returns
facts, keywords filter on the evidence (fact + source passage), the entity gate applies
only to a named entity, the (timeline_id, s_lo) index is what serves the scan, and the
scan's latency on 10k facts.
"""
from __future__ import annotations

import json
import statistics
import time

import httpx
import pytest
from app.models.graph_index import EntityIndexRequest, TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher
from app.services.arcadedb_client import db_name_for
from app.services.temporal_normalize import normalize_fact_dates

pytestmark = pytest.mark.asyncio

_ENTITIES = [("android", "Android", "PRODUCT"), ("google", "Google", "ORG"), ("htc", "HTC", "ORG"),
             ("samsung", "Samsung", "ORG"), ("motorola", "Motorola Mobility", "ORG"),
             ("hugo", "Hugo Barra", "PERSON")]
# (fact_id, entity_id, relation, value, time_start, time_end, source passage)
_HISTORY = [
    ("founded", "android", "founded_by", "Andy Rubin", "2003-10", None,
     "Android Inc. was founded in Palo Alto in October 2003."),
    ("acquired", "google", "acquired", "Android Inc.", "2005-07", "2005-08",
     "Google acquired Android Inc. in July 2005."),
    ("oha", "google", "announced", "Open Handset Alliance", "2007-11-05", None,
     "On November 5, 2007 the Open Handset Alliance was unveiled."),
    ("g1", "htc", "released", "HTC Dream (T-Mobile G1) phone", "2008-10-22", None,
     "The first commercially available Android phone was the HTC Dream."),
    ("barra", "hugo", "served_as", "Android product spokesperson", "2008", "2013",
     "Hugo Barra served as the product spokesperson for Android."),
    ("galaxy_s", "samsung", "released", "Galaxy S phone", "2010-06", None,
     "Samsung launched the Galaxy S smartphone in June 2010."),
    ("ics", "android", "released", "Android 4.0 Ice Cream Sandwich", "2011-10-18", None,
     "Ice Cream Sandwich unified the phone and tablet versions."),
    ("motorola_buy", "google", "acquired", "Motorola Mobility", "2012-05-22", None,
     "Google completed its purchase of Motorola Mobility in May 2012."),
    ("motorola_sale", "motorola", "sold_to", "Lenovo", "2014", None,
     "Motorola Mobility was sold to Lenovo in 2014, two years after Google bought it."),
    ("lollipop", "android", "released", "Android 5.0 Lollipop", "2014-11-03", None,
     "Lollipop introduced Material Design."),
]
_NAMES = {eid: name for eid, name, _ in _ENTITIES}

# What Module 18 sends (TMP-3 + TMP-4): "after 2011" is [2012-01-01, unbounded)
AFTER_2011 = "2012-01-01T00:00:00.000+08:00"
FROM_2010, TO_2012 = "2010-01-01T00:00:00.000+08:00", "2012-12-31T23:59:59.999+08:00"


async def _android_history(client, make_db, name="ittest_tmp4_history"):
    branch = await make_db(name)
    db = db_name_for(branch)
    for eid, ename, etype in _ENTITIES:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id=eid, name=ename, type=etype))
    for fid, eid, rel, value, ts, te, passage in _HISTORY:
        await client.command(db, "INSERT INTO Chunk SET chunk_id = :c, text = :t, status = 'active'",
                             params={"c": f"chunk_{fid}", "t": passage})
        await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
            fact_id=fid, entity=_NAMES[eid], entity_id=eid, relation=rel, value=value,
            time_start=ts, time_end=te, chunk_id=f"chunk_{fid}"))
    return branch


async def _tkg(client, branch, **kw):
    return await searcher.tkg_search(client, branch, TKGRequest(score_threshold=0, **kw))


async def test_what_happened_after_2011_without_an_entity(client, make_db):
    """D13: no entity in the question used to mean no facts at all."""
    branch = await _android_history(client, make_db)
    resp = await _tkg(client, branch, query="What happened after 2011?", time_start=AFTER_2011)
    got = [f["fact_id"] for f in resp.temporal_facts]
    assert resp.retrieval == "temporal" and resp.complete
    assert got == ["motorola_buy", "motorola_sale", "lollipop"]          # >= 1, every one after 2011, by start
    assert {f["entity_id"] for f in resp.temporal_facts} == {"google", "motorola", "android"}   # no entity gate


async def test_lg_phone_finds_nothing_when_no_fact_mentions_lg(client, make_db):
    """Keywords must all appear in the evidence; a phone fact that isn't LG's is not an answer."""
    branch = await _android_history(client, make_db)
    resp = await _tkg(client, branch, query="LG phone 2010-2012", keywords=["LG", "phone"],
                      time_start=FROM_2010, time_end=TO_2012)
    assert resp.retrieval == "keywords" and resp.temporal_facts == []
    # the same window does hold phone facts -- the keywords, not an empty window, excluded them
    phones = await _tkg(client, branch, query="phone 2010-2012", keywords=["phone"],
                        time_start=FROM_2010, time_end=TO_2012)
    assert {f["fact_id"] for f in phones.temporal_facts} == {"galaxy_s", "ics"}   # ics: "phone" only in its passage


async def test_keyword_found_only_in_the_source_passage(client, make_db):
    branch = await _android_history(client, make_db)
    resp = await _tkg(client, branch, query="Palo Alto", keywords=["palo alto"])
    assert [f["fact_id"] for f in resp.temporal_facts] == ["founded"]


async def test_entity_named_in_the_question_gates_the_facts(client, make_db):
    """AC "entity gate only when an entity is provided in the query": a question that names one
    still gets only that entity's facts -- Chinese included, where Module 18 derives no keywords."""
    branch = await make_db("ittest_tmp4_zh")
    for eid, name in [("samsung", "三星電子"), ("tsmc", "台積電"), ("apple", "蘋果公司")]:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id=eid, name=name, type="ORG"))
    for fid, eid, ts in [("s1", "samsung", "2010-06"), ("s2", "samsung", "2024-07"), ("t1", "tsmc", "2020-05"),
                         ("a1", "apple", "2007-01")]:
        await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
            fact_id=fid, entity=eid, entity_id=eid, relation="r", value="v", time_start=ts))
    resp = await _tkg(client, branch, query="三星電子發生了什麼事")
    assert resp.retrieval == "entity" and [f["fact_id"] for f in resp.temporal_facts] == ["s1", "s2"]
    history = await _android_history(client, make_db)
    google = await _tkg(client, history, query="What did Google acquire after 2011", time_start=AFTER_2011)
    assert google.retrieval == "entity" and [f["fact_id"] for f in google.temporal_facts] == ["motorola_buy"]


async def test_named_entity_also_finds_facts_that_mention_it(client, make_db):
    """"Samsung 2010-2016": the Galaxy S4 fact belongs to Google but answers it. Matching the
    entity "Samsung" used to stop there, so it was missed."""
    branch = await _android_history(client, make_db)
    await client.command(db_name_for(branch), "INSERT INTO Chunk SET chunk_id = 'chunk_s4', text = :t, status = 'active'",
                         params={"t": "Google announced a special version of the Samsung Galaxy S4."})
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="s4", entity="Google", entity_id="google", relation="announced",
        value="a special version of the Galaxy S4", time_start="2013-05", chunk_id="chunk_s4"))   # "Samsung": passage only
    window = {"time_start": "2010-01-01T00:00:00.000+08:00", "time_end": "2016-12-31T23:59:59.999+08:00"}
    resp = await _tkg(client, branch, query="Samsung 2010-2016", keywords=["Samsung"], **window)
    assert resp.retrieval == "entity" and "Samsung" in [e["name"] for e in resp.matched_entities]
    assert [f["fact_id"] for f in resp.temporal_facts] == ["galaxy_s", "s4"]                 # Samsung's own + the mention
    plain = await _tkg(client, branch, query="Samsung 2010-2016", **window)                  # no name to look for
    assert [f["fact_id"] for f in plain.temporal_facts] == ["galaxy_s"]


async def test_entity_gate_applies_only_to_a_named_entity(client, make_db):
    branch = await _android_history(client, make_db)
    named = await _tkg(client, branch, query="Motorola after 2011", entity="Motorola Mobility",
                       time_start=AFTER_2011)
    assert named.retrieval == "entity"
    # Motorola's own sale, and Google's purchase whose value names Motorola Mobility
    assert [f["fact_id"] for f in named.temporal_facts] == ["motorola_buy", "motorola_sale"]
    unknown = await _tkg(client, branch, query="LG after 2011", entity="LG", time_start=AFTER_2011)
    assert unknown.retrieval == "entity" and unknown.temporal_facts == []   # never widened to every fact


async def test_unknown_start_and_unmigrated_facts_are_candidates_too(client, make_db):
    branch = await _android_history(client, make_db)
    db = db_name_for(branch)
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id="no_start", entity="Android", entity_id="android", relation="r", value="v",
        time_start=None, time_end="2013-01-01"))
    await client.command(db, "INSERT INTO TemporalFact SET fact_id = 'legacy', entity = 'Android', "
                             "entity_id = 'android', relation = 'r', value = 'v', time_start = '2013-02-01'")
    resp = await _tkg(client, branch, query="after 2011", time_start=AFTER_2011)
    assert {"no_start", "legacy"} <= {f["fact_id"] for f in resp.temporal_facts}


@pytest.mark.parametrize("migrated", [True, False], ids=["normalised", "not_yet_backfilled"])
async def test_temporal_only_window_matches_the_tmp1_matrix(client, make_db, migrated):
    """The temporal-only reads use their own simplified predicates; they must give exactly the
    TMP-1/TMP-3 window semantics, on normalised rows and on rows the backfill hasn't reached."""
    from .test_tkg_window import _CASES, _FACTS
    branch = await make_db("ittest_tmp4_matrix")
    db = db_name_for(branch)
    await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id="e1", name="Android", type="PRODUCT"))
    for fid, ts, te in _FACTS:
        if migrated:
            await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
                fact_id=fid, entity="Android", entity_id="e1", relation="r", value=fid, time_start=ts, time_end=te))
        else:
            await client.command(db, "INSERT INTO TemporalFact SET fact_id = :f, entity = 'Android', entity_id = 'e1', "
                                     "relation = 'r', value = :f, time_start = :ts, time_end = :te",
                                 params={"f": fid, "ts": ts, "te": te})
    for name, ((ts, te), expected) in _CASES.items():
        resp = await _tkg(client, branch, query="what happened", time_start=ts, time_end=te)
        got = {f["fact_id"] for f in resp.temporal_facts}
        assert resp.retrieval == "temporal"
        assert got == expected, f"{name}: unexpected={sorted(got - expected)} missing={sorted(expected - got)}"


async def _explain(client, db, sql, params):
    async with httpx.AsyncClient(auth=client.auth, timeout=client.timeout) as h:
        r = await h.post(f"{client.base}/api/v1/command/{db}",
                         json={"language": "sql", "command": f"EXPLAIN {sql}", "params": params})
        r.raise_for_status()
        return r.json().get("explain", "")


async def test_temporal_scan_is_served_by_the_timeline_start_index(client, make_db, monkeypatch):
    branch = await _android_history(client, make_db)
    db = db_name_for(branch)
    seen = []
    real_query = client.query

    async def record(database, sql, params=None, **kw):
        seen.append((sql, params))
        return await real_query(database, sql, params=params, **kw)

    monkeypatch.setattr(client, "query", record)
    for window in [{"time_start": AFTER_2011}, {"time_end": TO_2012}, {"time_start": FROM_2010, "time_end": TO_2012}]:
        seen.clear()
        await _tkg(client, branch, query="what happened", **window)
        sql, params = next((s, p) for s, p in seen if "s_state = 'known'" in s and "timeline_id" in s)
        plan = await _explain(client, db, sql, params)
        assert "FETCH FROM INDEX TemporalFact[timeline_id,s_lo]" in plan, (window, plan)


async def test_index_range_scan_bug_is_avoided(client, make_db):
    """ArcadeDB 26.10: a NOTUNIQUE index range scan with an inclusive lower bound drops rows that
    share a key. Many facts of one year share s_lo; all must come back from the indexed scan."""
    branch = await make_db("ittest_tmp4_samekey")
    db = db_name_for(branch)
    for i in range(12):
        await client.command(db, "INSERT INTO TemporalFact SET fact_id = :f", params={"f": f"y{i}"})
    for i in range(12):                                      # UPDATEd onto a shared key, like the backfill
        props = normalize_fact_dates("2012", None, "Asia/Taipei")
        await client.command(db, "UPDATE TemporalFact SET " + ", ".join(f"{k} = :{k}" for k in props)
                             + ", entity = 'X', entity_id = 'x' WHERE fact_id = :f", params={**props, "f": f"y{i}"})
    resp = await _tkg(client, branch, query="after 2011", time_start=AFTER_2011)
    assert len(resp.temporal_facts) == 12


async def test_temporal_scan_p95_under_500ms_on_10k_facts(client, make_db):
    branch = await make_db("ittest_tmp4_perf")
    db = db_name_for(branch)
    for start in range(0, 10_000, 500):
        rows = []
        for i in range(start, start + 500):
            year = 1990 + i % 35
            props = normalize_fact_dates(f"{year}-{1 + i % 12:02d}-{1 + i % 28:02d}", None, "Asia/Taipei")
            rows.append({**props, "fact_id": f"p{i}", "entity": f"E{i % 300}", "entity_id": f"e{i % 300}",
                         "relation": "r", "value": "v", "status": "active"})
        await client.command(db, "INSERT INTO TemporalFact CONTENT " + json.dumps(rows))
    assert (await client.query(db, "SELECT count(*) AS n FROM TemporalFact"))[0]["n"] == 10_000

    windows = [{"time_start": f"{y}-01-01T00:00:00.000+08:00", "time_end": f"{y}-12-31T23:59:59.999+08:00"}
               for y in range(1995, 2020)] + [{"time_start": AFTER_2011}, {"time_end": "1999-12-31"}]
    timings = []
    for _ in range(2):
        for w in windows:
            t0 = time.perf_counter()
            resp = await _tkg(client, branch, query="what happened", **w)
            timings.append((time.perf_counter() - t0) * 1000)
            assert resp.temporal_facts
    p95 = statistics.quantiles(timings, n=20)[18]
    print(f"temporal-only tkg_search on 10k facts: p50={statistics.median(timings):.0f}ms p95={p95:.0f}ms")
    assert p95 < 500


async def test_facts_normalised_by_tmp3_but_without_a_timeline_are_found(client, make_db):
    """TMP-3 deployed and backfilled before TMP-4: rows have ranges but no timeline_id yet.
    They are not on the indexed path until the backfill runs again, but must still be found."""
    from .test_tkg_window import _CASES, _FACTS
    branch = await make_db("ittest_tmp4_tmp3rows")
    db = db_name_for(branch)
    await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id="e1", name="Android", type="PRODUCT"))
    for fid, ts, te in _FACTS:
        props = {k: v for k, v in normalize_fact_dates(ts, te, "Asia/Taipei").items() if k != "timeline_id"}
        await client.command(db, "INSERT INTO TemporalFact SET fact_id = :f, entity = 'Android', entity_id = 'e1', "
                                 "relation = 'r', value = :f, " + ", ".join(f"{k} = :{k}" for k in props),
                             params={**props, "f": fid})
    assert not await client.query(db, "SELECT FROM TemporalFact WHERE timeline_id IS NOT NULL")
    for name, ((ts, te), expected) in _CASES.items():
        got = {f["fact_id"] for f in (await _tkg(client, branch, query="q", time_start=ts, time_end=te)).temporal_facts}
        assert got == expected, f"{name}: unexpected={sorted(got - expected)} missing={sorted(expected - got)}"
