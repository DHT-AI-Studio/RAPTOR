"""TMP-8 TemporalFact v2 schema + migration, executed by a REAL ArcadeDB.

Covers: every v2 property and the three indexes exist (and applying the schema again is
harmless), the two new indexes do not lose rows that share a key (the ArcadeDB 26.10
range-scan bug, see schema_init.py), the migration upgrades a database that predates v2
without touching its rows' identity, and it runs the same with PD_TEMPORAL_V2 on or off.
"""
from __future__ import annotations

import pytest
from app.core.config import settings
from app.models.graph_index import EntityIndexRequest, TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher, temporal_backfill
from app.services.arcadedb_client import db_name_for
from app.services.schema_init import _ddl, initialize_schema
from app.services.temporal_normalize import NORMALIZER_VERSION

from .test_temporal_backfill import _LEGACY, _legacy_db, _rows

pytestmark = pytest.mark.asyncio

V2_PROPERTIES = {
    "s_state": "STRING", "e_state": "STRING", "s_lo": "LONG", "s_hi": "LONG", "e_lo": "LONG", "e_hi": "LONG",
    "timeline_id": "STRING", "axis": "STRING", "time_unit": "STRING", "timezone": "STRING",
    "normalizer_version": "STRING", "extent": "STRING", "precision": "STRING", "anchor": "STRING",
    "date_text": "STRING", "source_sentence": "STRING", "evidence_offsets": "STRING",
    "confirmed_active_at": "LONG", "derived": "BOOLEAN", "support_ids": "LIST", "network_version": "STRING",
    "media_start_ms": "LONG", "media_end_ms": "LONG",
    "time_start": "STRING", "time_end": "STRING",                      # v1 projections, retained
}
V2_INDEXES = {("timeline_id", "s_lo"), ("timeline_id", "e_hi"), ("entity_id", "s_lo")}
_V2_ONLY = ("axis", "time_unit", "extent", "anchor", "date_text", "source_sentence", "evidence_offsets",
            "confirmed_active_at", "derived", "support_ids", "network_version", "media_start_ms", "media_end_ms",
            "(timeline_id, e_hi)", "(entity_id, s_lo)")


async def _fact_type(client, db) -> dict:
    return (await client.query(db, "SELECT FROM schema:types WHERE name = 'TemporalFact'"))[0]


async def test_v2_properties_and_indexes_exist(client, make_db):
    db = db_name_for(await make_db("ittest_tmp8_schema"))
    await initialize_schema(client, db)                                 # a second time: must change nothing
    fact = await _fact_type(client, db)
    types = {p["name"]: p["type"] for p in fact["properties"]}
    assert {name: types.get(name) for name in V2_PROPERTIES} == V2_PROPERTIES
    indexes = {tuple(i["properties"]) for i in fact["indexes"]}
    assert V2_INDEXES <= indexes
    assert len(fact["indexes"]) == len(indexes)                         # no index created twice


async def test_new_indexes_return_every_fact_sharing_a_key(client, make_db):
    """Thirty facts of one entity, all "2011": one (entity_id, s_lo) key, one
    (timeline_id, e_hi) key. Each is written twice -- the bug shows on rows UPDATEd to a
    shared key -- and a window starting exactly on that key must still return all of them."""
    branch = await make_db("ittest_tmp8_shared_key")
    db = db_name_for(branch)
    await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id="e1", name="Android", type="PRODUCT"))
    n = 30
    for _ in range(2):
        for i in range(n):
            await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
                fact_id=f"f{i:02d}", entity="Android", entity_id="e1", relation="r", value=f"v{i}",
                time_start="2011", time_end="2011"))
    row = (await _rows(client, db))["f00"]
    s_lo, e_hi = row["s_lo"], row["e_hi"]

    async def count(where: str, **params) -> int:
        return (await client.query(db, f"SELECT count(*) AS n FROM TemporalFact WHERE {where}", params=params))[0]["n"]

    assert await count("entity_id = 'e1' AND s_lo > :x", x=s_lo - 1) == n
    assert await count("entity_id = 'e1' AND s_lo <= :x", x=s_lo) == n
    assert await count("entity_id = 'e1' AND s_lo > :x", x=s_lo) == 0
    assert await count("timeline_id = 'world' AND e_hi > :x", x=e_hi - 1) == n
    assert await count("timeline_id = 'world' AND e_hi <= :x", x=e_hi) == n
    assert await count("timeline_id = 'world' AND e_hi > :x", x=e_hi) == 0

    by_entity = await searcher.tkg_search(client, branch, TKGRequest(
        query="Android", entity="Android", time_start="2011", time_end="2011", limit=100, score_threshold=0))
    assert (by_entity.retrieval, len(by_entity.temporal_facts)) == ("entity", n)
    by_time = await searcher.tkg_search(client, branch, TKGRequest(
        query="what happened", time_start="2011", time_end="2011", limit=100))
    assert (by_time.retrieval, len(by_time.temporal_facts)) == ("temporal", n)


async def test_migration_upgrades_a_database_that_predates_v2(arcade, client):
    """A database holding only what existed before TMP-8: legacy rows, none of the new
    properties or indexes. The migration adds the schema and normalises the rows; the v1
    projections stay, and nothing a legacy row cannot establish is invented."""
    branch = "ittest_tmp8_predates"
    db = db_name_for(branch)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)
    await arcade.create_database(db)
    try:
        for stmt in _ddl(settings.vector_dim, settings.vector_similarity):
            if not any(new in stmt for new in _V2_ONLY):
                await arcade.command(db, stmt, ignore_exists=True)
        before = await _fact_type(client, db)
        assert not {"axis", "extent"} & {p["name"] for p in before["properties"]}
        for fid, ts, te in _LEGACY:
            await client.command(db, "INSERT INTO TemporalFact SET fact_id = :f, entity = 'Android', entity_id = 'e1', "
                                     "relation = 'r', value = :f, time_start = :ts, time_end = :te",
                                 params={"f": fid, "ts": ts, "te": te})

        stats = await temporal_backfill.backfill_database(client, db, page_size=4)
        assert (stats.migrated, stats.pending_after, stats.errors) == (len(_LEGACY), 0, [])

        fact = await _fact_type(client, db)
        assert set(V2_PROPERTIES) <= {p["name"] for p in fact["properties"]}
        assert V2_INDEXES <= {tuple(i["properties"]) for i in fact["indexes"]}
        rows = await _rows(client, db)
        assert set(rows) == {f[0] for f in _LEGACY}
        for row in rows.values():
            assert (row["timeline_id"], row["axis"], row["time_unit"]) == ("world", "world", "ms")
            for guessed in ("extent", "anchor", "date_text", "source_sentence", "confirmed_active_at", "derived"):
                assert row.get(guessed) is None, guessed

        year = rows["year"]                         # "2011": a year range, and a lossy v1 projection of it
        assert (year["precision"], year["time_start"], year["time_start_raw"]) == ("year", "2011-01-01", "2011")
        assert year["s_hi"] - year["s_lo"] == 365 * 86_400_000 - 1
        assert (year["e_state"], year.get("e_lo"), year.get("time_end")) == ("unknown", None, None)   # NULL: unknown
        bad = rows["bad_month"]
        assert bad["requires_reextraction"] is True and bad["precision"] == "unknown" and bad["quarantine_reason"]

        again = await temporal_backfill.backfill_database(client, db)
        assert (again.pending_before, again.migrated) == (0, 0)
    finally:
        await arcade.drop_database(db)


@pytest.mark.parametrize("flag", [False, True])
async def test_migration_does_not_depend_on_the_v2_flag(client, make_db, monkeypatch, flag):
    monkeypatch.setattr(settings, "temporal_v2", flag)
    _, db = await _legacy_db(client, make_db, f"ittest_tmp8_flag_{int(flag)}")
    stats = await temporal_backfill.backfill_database(client, db)
    assert (stats.migrated, stats.pending_after, stats.errors) == (len(_LEGACY), 0, [])
    rows = await _rows(client, db)
    assert all(r["migration_version"] == NORMALIZER_VERSION and r["s_state"] in ("known", "unknown")
               for r in rows.values())
    assert (rows["year"]["precision"], rows["year"]["time_start"]) == ("year", "2011-01-01")
