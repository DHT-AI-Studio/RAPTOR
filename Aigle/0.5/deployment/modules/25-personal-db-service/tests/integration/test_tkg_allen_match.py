"""P07 (§5.9) Allen match modes, executed by a REAL ArcadeDB.

The Android-history tenant from test_tkg_temporal_only.py, queried for 2010 the way Module 18
sends a year. Facts are indexed through the real write path, so the prefilter runs on the
stored UTC ms ranges and allen.py classifies what it reads.
"""
from __future__ import annotations

import pytest
from app.services.arcadedb_client import db_name_for

from .test_tkg_temporal_only import _android_history, _tkg

pytestmark = pytest.mark.asyncio

Y2010 = {"time_start": "2010-01-01T00:00:00.000+08:00", "time_end": "2010-12-31T23:59:59.999+08:00"}
Y2011 = {"time_start": "2011-01-01T00:00:00.000+08:00", "time_end": "2011-12-31T23:59:59.999+08:00"}


async def test_what_was_hugo_barra_doing_in_2010(client, make_db):
    """Served 2008-2013: not inside 2010, but it contains the year, so it overlaps it."""
    branch = await _android_history(client, make_db, "ittest_p07_barra")
    inside = await _tkg(client, branch, query="Hugo Barra", entity="Hugo Barra", match="inside", **Y2010)
    assert inside.temporal_facts == []
    resp = await _tkg(client, branch, query="Hugo Barra", entity="Hugo Barra", match="overlaps", **Y2010)
    assert [(f["fact_id"], f["allen"]) for f in resp.temporal_facts] == \
        [("barra", {"vs_window": ["contains"], "certainty": "certain"})]


async def test_what_was_active_in_2010(client, make_db):
    branch = await _android_history(client, make_db, "ittest_p07_active")
    resp = await _tkg(client, branch, query="What was going on in 2010?", match="overlaps", **Y2010)
    got = {f["fact_id"]: f["allen"]["certainty"] for f in resp.temporal_facts}
    # started before 2010 with no known end: may still have been going on
    assert got == {"founded": "possible", "oha": "possible", "g1": "possible",
                   "barra": "certain", "galaxy_s": "certain"}
    certain = await _tkg(client, branch, query="What was going on in 2010?", match="overlaps",
                         certainty="certain", **Y2010)
    assert {f["fact_id"] for f in certain.temporal_facts} == {"barra", "galaxy_s"}


async def test_strict_after_2011(client, make_db):
    branch = await _android_history(client, make_db, "ittest_p07_after")
    resp = await _tkg(client, branch, query="What happened after 2011?", relations=["after"], **Y2011)
    assert [f["fact_id"] for f in resp.temporal_facts] == ["motorola_buy", "motorola_sale", "lollipop"]
    assert all(f["allen"] == {"vs_window": ["after"], "certainty": "certain"} for f in resp.temporal_facts)


async def test_row_not_normalised_yet_is_classified_from_its_strings(client, make_db):
    branch = await _android_history(client, make_db, "ittest_p07_legacy")
    await client.command(db_name_for(branch), "INSERT INTO TemporalFact SET fact_id = 'legacy', entity = 'Android', "
                         "entity_id = 'android', relation = 'r', value = 'v', "
                         "time_start = '2009', time_end = '2010-03-01'")
    resp = await _tkg(client, branch, query="What was going on in 2010?", match="overlaps", **Y2010)
    legacy = next(f for f in resp.temporal_facts if f["fact_id"] == "legacy")
    assert legacy["allen"] == {"vs_window": ["overlaps"], "certainty": "certain"}


async def test_default_request_is_unchanged_but_annotated(client, make_db):
    branch = await _android_history(client, make_db, "ittest_p07_default")
    resp = await _tkg(client, branch, query="What happened in 2010?", **Y2010)
    assert {f["fact_id"] for f in resp.temporal_facts} == {"galaxy_s"}       # containment, as before
    assert resp.temporal_facts[0]["allen"]["certainty"] == "possible"        # no known end: may run past 2010
