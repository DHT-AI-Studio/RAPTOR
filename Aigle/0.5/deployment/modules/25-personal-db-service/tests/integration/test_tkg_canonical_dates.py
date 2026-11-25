"""Canonical TemporalFact dates (TMP-3), executed by a REAL ArcadeDB.

test_tkg_window.py runs TMP-1's window matrix on facts written through the indexer, which now
normalises them (UTC ms ranges, compared as LONG). These tests cover what that file can't:
what the indexer stores, rows written before TMP-3 (no *_state, string fallback) answering the
same matrix, and the D3 bug itself -- a window sent as a timestamp against date / year facts.
"""
from __future__ import annotations

import pytest
from app.models.graph_index import EntityIndexRequest, TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher
from app.services.arcadedb_client import db_name_for

from .test_tkg_window import _CASES, _FACTS

pytestmark = pytest.mark.asyncio


async def _entities(client, branch):
    # distractor entities keep the tiny corpus's BM25 score for "Android" above zero
    for eid, name in [("e1", "Android"), ("e2", "Nexus"), ("e3", "Pixel")]:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id=eid, name=name, type="PRODUCT"))


async def _fact(client, branch, fid, ts, te):
    await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
        fact_id=fid, entity="Android", entity_id="e1", relation="r", value=fid, time_start=ts, time_end=te))


async def _tkg(client, branch, ts, te) -> set[str]:
    resp = await searcher.tkg_search(client, branch, TKGRequest(
        query="Android", time_start=ts, time_end=te, score_threshold=0))
    return {f["fact_id"] for f in resp.temporal_facts}


async def test_indexer_stores_ranges_and_keeps_the_raw_values(client, make_db):
    branch = await make_db("ittest_tmp3_store")
    await _entities(client, branch)
    await _fact(client, branch, "f", "2011", "2013-06-15T10:00:00Z")
    row = (await client.query(db_name_for(branch), "SELECT FROM TemporalFact WHERE fact_id = 'f'"))[0]
    assert (row["time_start_raw"], row["time_end_raw"]) == ("2011", "2013-06-15T10:00:00Z")
    assert (row["time_start"], row["time_end"]) == ("2011-01-01", "2013-06-15")
    assert (row["s_state"], row["e_state"], row["precision"]) == ("known", "known", "year")
    assert row["s_lo"] < row["s_hi"] < row["e_lo"] <= row["e_hi"]
    assert row["quarantine_reason"] is None and row["migration_version"]
    assert row["requires_reextraction"] is False


async def test_reindexing_clears_a_previous_range(client, make_db):
    branch = await make_db("ittest_tmp3_reindex")
    await _entities(client, branch)
    await _fact(client, branch, "f", "2011", "2012")
    await _fact(client, branch, "f", "2011", None)                  # the end is no longer known
    row = (await client.query(db_name_for(branch), "SELECT FROM TemporalFact WHERE fact_id = 'f'"))[0]
    assert row["e_state"] == "unknown"
    assert row.get("e_lo") is None and row.get("e_hi") is None and row.get("time_end") is None


async def test_bad_date_is_stored_quarantined_not_dropped(client, make_db):
    branch = await make_db("ittest_tmp3_quarantine")
    await _entities(client, branch)
    await _fact(client, branch, "f", "2023-13-01", "2024")
    row = (await client.query(db_name_for(branch), "SELECT FROM TemporalFact WHERE fact_id = 'f'"))[0]
    assert row["s_state"] == "unknown" and row["time_start_raw"] == "2023-13-01"
    assert "2023-13-01" in row["quarantine_reason"]
    assert row["requires_reextraction"] is True and row["precision"] == "unknown"


async def test_window_as_timestamps_matches_day_and_year_facts(client, make_db):
    """D3: Module 18 sends timestamps; facts hold dates or years. Compared as strings, a fact
    starting on the window's first day ("2011-01-01" < "2011-01-01T00:00:00") and a year-only
    fact ("2011" < "2011-01-01...") both fell outside a window that plainly contains them."""
    branch = await make_db("ittest_tmp3_d3")
    await _entities(client, branch)
    await _fact(client, branch, "starts_day_one", "2011-01-01", "2011-03-01")
    await _fact(client, branch, "year_only", "2011", "2011")
    await _fact(client, branch, "next_year", "2012-01-01", "2012-02-01")
    # what Module 18 sends for "2011"
    assert await _tkg(client, branch, "2011-01-01T00:00:00.000+08:00",
                      "2011-12-31T23:59:59.999+08:00") == {"starts_day_one", "year_only"}
    # a timestamp is a point (TMP-5): a window ending at 23:59:59.000 does not contain a year
    # that ends at 23:59:59.999
    assert await _tkg(client, branch, "2011-01-01T00:00:00", "2011-12-31T23:59:59") == {"starts_day_one"}
    assert await _tkg(client, branch, "2011", "2011") == {"starts_day_one", "year_only"}
    # a year-only fact is the whole year: not contained in its first half
    assert await _tkg(client, branch, "2011-01-01", "2011-06-30") == {"starts_day_one"}


async def test_rows_written_before_tmp3_answer_the_same_window_matrix(client, make_db):
    """Facts written before TMP-3 have no *_state until the backfill reaches them; the window
    falls back to comparing their strings against the window's calendar dates."""
    branch = await make_db("ittest_tmp3_legacy")
    await _entities(client, branch)
    db = db_name_for(branch)
    for fid, ts, te in _FACTS:
        await client.command(db, "INSERT INTO TemporalFact SET fact_id = :f, entity = 'Android', entity_id = 'e1', "
                                 "relation = 'r', value = :f, time_start = :ts, time_end = :te",
                             params={"f": fid, "ts": ts, "te": te})
    assert not await client.query(db, "SELECT FROM TemporalFact WHERE s_state IS NOT NULL")
    for name, ((ts, te), expected) in _CASES.items():
        got = await _tkg(client, branch, ts, te)
        assert got == expected, f"{name}: unexpected={sorted(got - expected)} missing={sorted(expected - got)}"


async def test_bad_window_is_a_value_error(client, make_db):
    branch = await make_db("ittest_tmp3_badwindow")
    await _entities(client, branch)
    with pytest.raises(ValueError):
        await _tkg(client, branch, "2011-02-30", None)


async def test_facts_sharing_a_start_are_all_returned(client, make_db):
    """Every fact of one year has the same s_lo. Once rows are UPDATEd to that value -- by the
    backfill, or by re-indexing an existing fact -- a NOTUNIQUE index on s_lo made ArcadeDB 26.10's
    range scan return only the last of them. Guards against bringing that index back."""
    from app.services import temporal_backfill
    branch = await make_db("ittest_tmp3_samekey")
    await _entities(client, branch)
    db = db_name_for(branch)
    for fid in ("legacy_a", "legacy_b"):                          # pre-TMP-3 rows, migrated by UPDATE
        await client.command(db, "INSERT INTO TemporalFact SET fact_id = :f, entity = 'Android', entity_id = 'e1', "
                                 "relation = 'r', value = :f, time_start = '2011'", params={"f": fid})
    await temporal_backfill.backfill_database(client, db)
    for fid in ("reindexed_a", "reindexed_b"):                    # indexed, then indexed again
        await _fact(client, branch, fid, "2012", None)
        await _fact(client, branch, fid, "2011", None)
    expected = {"legacy_a", "legacy_b", "reindexed_a", "reindexed_b"}
    assert await _tkg(client, branch, "2011", "2011") == expected
    assert await _tkg(client, branch, "2011-01-01T00:00:00.000+08:00", None) == expected


async def test_padded_dates_from_a_writer_without_precision(client, make_db):
    """The Temporal Demo writes "2012" as 2012-01-01T00:00:00Z .. 2012-12-31T00:00:00Z. Stored as
    written, but doubtful: precision unknown, marked for re-extraction. The end's midnight is the
    whole of 31 December, so a fact on that day is inside a window ending then."""
    branch = await make_db("ittest_tmp8_padded")
    await _entities(client, branch)
    await _fact(client, branch, "padded_year", "2012-01-01T00:00:00Z", "2012-12-31T00:00:00Z")
    await _fact(client, branch, "new_years_eve", "2012-12-31T00:00:00Z", None)
    await _fact(client, branch, "exact", "2012-05-03", None)
    rows = {r["fact_id"]: r for r in await client.query(
        db_name_for(branch), "SELECT fact_id, precision, requires_reextraction, reextraction_reason FROM TemporalFact")}
    assert rows["padded_year"]["precision"] == "unknown" and rows["padded_year"]["requires_reextraction"] is True
    assert "whole years" in rows["padded_year"]["reextraction_reason"]
    assert rows["exact"]["requires_reextraction"] is False and rows["exact"].get("reextraction_reason") is None
    # a window that a client ends at midnight ("...T00:00:00Z") used to stop at 00:00 and drop the day itself
    got = await _tkg(client, branch, "2012-12-31T00:00:00Z", "2012-12-31T00:00:00Z")
    assert got == {"new_years_eve"}
