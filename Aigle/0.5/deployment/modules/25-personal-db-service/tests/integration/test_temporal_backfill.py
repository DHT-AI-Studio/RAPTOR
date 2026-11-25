"""TMP-3 legacy backfill, executed by a REAL ArcadeDB.

Rows are inserted the way facts were stored before TMP-3 (raw strings, no ranges), then
temporal_backfill migrates them. Covers: raw values kept, year-only -> whole year,
unparseable -> quarantined not dropped, idempotence,
checkpoint/resume after an interruption, dry run, and the TKG window afterwards.
"""
from __future__ import annotations

import pytest
from app.models.graph_index import EntityIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher, temporal_backfill
from app.services.arcadedb_client import db_name_for
from app.services.temporal_normalize import NORMALIZER_VERSION

from .test_tkg_window import _CASES, _FACTS

pytestmark = pytest.mark.asyncio

_LEGACY = [  # (fact_id, time_start, time_end) as the old extractor stored them
    ("year", "2011", None),
    ("day", "2011-05-03", "2011-05-20"),
    ("padded_month", "2011-05-01", None),
    ("timestamp", "2011-05-03T10:00:00", None),
    ("bad_month", "2023-13-01", "2024"),
    ("placeholder", "null", "2012"),
    ("reversed", "2015", "2012"),
    ("no_dates", None, None),
    ("zh_year", "2011年", None),        # TMP-5 grammar: quarantined by the ISO-only parser, now resolved
]


async def _legacy_db(client, make_db, name, facts=_LEGACY):
    branch = await make_db(name)
    for eid, ename in [("e1", "Android"), ("e2", "Nexus"), ("e3", "Pixel")]:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id=eid, name=ename, type="PRODUCT"))
    db = db_name_for(branch)
    for fid, ts, te in facts:
        await client.command(db, "INSERT INTO TemporalFact SET fact_id = :f, entity = 'Android', entity_id = 'e1', "
                                 "relation = 'r', value = :f, time_start = :ts, time_end = :te",
                             params={"f": fid, "ts": ts, "te": te})
    return branch, db


async def _rows(client, db):
    return {r["fact_id"]: r for r in await client.query(db, "SELECT FROM TemporalFact")}


async def test_backfill_normalises_and_keeps_every_row(client, make_db):
    _, db = await _legacy_db(client, make_db, "ittest_bf_rows")
    stats = await temporal_backfill.backfill_database(client, db, page_size=3)
    assert (stats.pending_before, stats.migrated, stats.pending_after, stats.errors) == (len(_LEGACY), len(_LEGACY), 0, [])
    rows = await _rows(client, db)
    assert set(rows) == {f[0] for f in _LEGACY}                       # nothing dropped
    for fid, ts, te in _LEGACY:                                        # raw values preserved as written
        assert (rows[fid].get("time_start_raw"), rows[fid].get("time_end_raw")) == (ts, te), fid
        assert rows[fid]["migration_version"] == NORMALIZER_VERSION

    year = rows["year"]                                                # all of 2011, not 1 January
    assert (year["precision"], year["time_start"], year["s_state"], year["e_state"]) == ("year", "2011-01-01", "known", "unknown")
    assert year["s_hi"] - year["s_lo"] == 365 * 86_400_000 - 1

    assert rows["day"]["precision"] == "day" and rows["day"]["time_end"] == "2011-05-20"
    zh = rows["zh_year"]
    assert (zh["precision"], zh["time_start"], zh["s_hi"] - zh["s_lo"]) == ("year", "2011-01-01", 365 * 86_400_000 - 1)
    assert zh.get("quarantine_reason") is None
    assert rows["timestamp"]["precision"] == "second" and rows["timestamp"]["time_start"] == "2011-05-03"

    assert rows["padded_month"]["precision"] == "day"                 # stored as written

    bad = rows["bad_month"]                                            # quarantined, with a reason, not dropped
    assert bad["s_state"] == "unknown" and bad.get("s_lo") is None and "2023-13-01" in bad["quarantine_reason"]
    assert bad["precision"] == "unknown" and bad["requires_reextraction"] is True     # P03
    assert bad["e_state"] == "known"
    assert "after time_end" in rows["reversed"]["quarantine_reason"]
    assert rows["reversed"]["requires_reextraction"] is True
    assert sorted(f for f, r in rows.items() if r["requires_reextraction"]) == ["bad_month", "reversed"]
    assert rows["placeholder"]["s_state"] == "unknown" and rows["placeholder"].get("quarantine_reason") is None
    assert rows["no_dates"]["s_state"] == rows["no_dates"]["e_state"] == "unknown"
    assert stats.quarantined == 2


async def test_backfill_is_idempotent(client, make_db):
    _, db = await _legacy_db(client, make_db, "ittest_bf_idem")
    await temporal_backfill.backfill_database(client, db)
    before = await _rows(client, db)
    again = await temporal_backfill.backfill_database(client, db)
    assert (again.pending_before, again.migrated) == (0, 0)
    after = await _rows(client, db)
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("@")}
    assert {f: strip(r) for f, r in before.items()} == {f: strip(r) for f, r in after.items()}


async def test_interrupted_backfill_resumes_where_it_stopped(client, make_db, monkeypatch):
    _, db = await _legacy_db(client, make_db, "ittest_bf_resume")
    real_command, calls = client.command, {"n": 0}

    async def crash_after_three(database, sql, params=None, **kw):
        if sql.startswith("UPDATE TemporalFact"):
            calls["n"] += 1
            if calls["n"] > 3:
                raise KeyboardInterrupt("simulated crash")
        return await real_command(database, sql, params=params, **kw)

    monkeypatch.setattr(client, "command", crash_after_three)
    with pytest.raises(KeyboardInterrupt):
        await temporal_backfill.backfill_database(client, db, page_size=2)
    monkeypatch.setattr(client, "command", real_command)

    done = [r for r in (await _rows(client, db)).values() if r.get("migration_version") == NORMALIZER_VERSION]
    assert len(done) == 3                                              # the checkpoint: three rows committed
    resumed = await temporal_backfill.backfill_database(client, db, page_size=2)
    assert (resumed.pending_before, resumed.migrated, resumed.pending_after) == (len(_LEGACY) - 3, len(_LEGACY) - 3, 0)


async def test_failed_row_is_reported_and_the_rest_still_migrate(client, make_db, monkeypatch):
    _, db = await _legacy_db(client, make_db, "ittest_bf_fail")
    real_command = client.command

    async def fail_one(database, sql, params=None, **kw):
        if sql.startswith("UPDATE TemporalFact") and (params or {}).get("fact_id") == "day":
            raise RuntimeError("boom")
        return await real_command(database, sql, params=params, **kw)

    monkeypatch.setattr(client, "command", fail_one)
    stats = await temporal_backfill.backfill_database(client, db, page_size=2)
    assert stats.migrated == len(_LEGACY) - 1 and stats.pending_after == 1
    assert len(stats.errors) == 1 and stats.errors[0].startswith("day:")


async def test_dry_run_changes_nothing(client, make_db):
    _, db = await _legacy_db(client, make_db, "ittest_bf_dry")
    stats = await temporal_backfill.backfill_database(client, db, dry_run=True)
    assert (stats.pending_before, stats.migrated, stats.pending_after) == (len(_LEGACY), 0, len(_LEGACY))
    assert not await client.query(db, "SELECT FROM TemporalFact WHERE s_state IS NOT NULL")


async def test_rows_from_an_older_normaliser_are_redone_from_their_raw_values(client, make_db):
    _, db = await _legacy_db(client, make_db, "ittest_bf_oldver", facts=[("year", "2011", None)])
    await temporal_backfill.backfill_database(client, db)
    # as if an older normaliser had written a wrong projection: the raw value is the source of truth
    await client.command(db, "UPDATE TemporalFact SET migration_version = 'tmp3-0', time_start = '1999-01-01' "
                             "WHERE fact_id = 'year'")
    stats = await temporal_backfill.backfill_database(client, db)
    row = (await _rows(client, db))["year"]
    assert stats.migrated == 1
    assert (row["time_start"], row["time_start_raw"], row["precision"]) == ("2011-01-01", "2011", "year")


async def test_window_matrix_holds_after_backfill(client, make_db):
    branch, db = await _legacy_db(client, make_db, "ittest_bf_window", facts=_FACTS)
    await temporal_backfill.backfill_database(client, db)
    assert not await client.query(db, "SELECT FROM TemporalFact WHERE s_state IS NULL")
    for name, ((ts, te), expected) in _CASES.items():
        resp = await searcher.tkg_search(client, branch, TKGRequest(query="Android", time_start=ts, time_end=te,
                                                                     score_threshold=0))
        got = {f["fact_id"] for f in resp.temporal_facts}
        assert got == expected, f"{name}: unexpected={sorted(got - expected)} missing={sorted(expected - got)}"


async def test_cli_runs_a_branch(client, make_db):
    branch, db = await _legacy_db(client, make_db, "ittest_bf_cli")
    assert await temporal_backfill.main(["--branch", branch, "--dry-run"]) == 0
    assert await temporal_backfill.main(["--branch", branch]) == 0
    assert not await client.query(db, "SELECT FROM TemporalFact WHERE s_state IS NULL")


async def test_row_without_fact_id_is_reported_not_looped_on(client, make_db):
    _, db = await _legacy_db(client, make_db, "ittest_bf_nofid", facts=[("ok", "2011", None)])
    await client.command(db, "INSERT INTO TemporalFact SET entity_id = 'e1', time_start = '2012'")   # no fact_id
    stats = await temporal_backfill.backfill_database(client, db, page_size=1)
    assert stats.migrated == 1 and stats.pending_after == 1
    assert len(stats.errors) == 1 and stats.errors[0].startswith("None:")
