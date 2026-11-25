"""D8 `when`, executed by a REAL ArcadeDB: facts written through the indexer come back from the TKG
search with their dates at the precision they were given, not as the padded day."""
from __future__ import annotations

import pytest
from app.models.graph_index import EntityIndexRequest, TemporalFactIndexRequest
from app.models.graph_search import TKGRequest
from app.services import graph_indexer, searcher

pytestmark = pytest.mark.asyncio


async def test_tkg_facts_carry_their_dates_at_their_precision(client, make_db):
    branch = await make_db("ittest_d8_when")
    for eid, name in [("samsung", "Samsung"), ("x1", "Nexus"), ("x2", "Pixel")]:   # distractors keep BM25 > 0
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(entity_id=eid, name=name, type="ORG"))
    for fid, ts, te in [("strike", "2024-07", None), ("tenure", "2008", "2013"), ("launch", "2010-06-01", "2010-06-01")]:
        await graph_indexer.index_temporal_fact(client, branch, TemporalFactIndexRequest(
            fact_id=fid, entity="Samsung", entity_id="samsung", relation="r", value=fid, time_start=ts, time_end=te))
    resp = await searcher.tkg_search(client, branch, TKGRequest(query="Samsung", score_threshold=0))
    got = {f["fact_id"]: (f["time_start"], f["when"], f["precision"]) for f in resp.temporal_facts}
    assert got == {"strike": ("2024-07-01", "2024-07 (no end date stated)", "month"),
                   "tenure": ("2008-01-01", "2008 to 2013", "year"),
                   "launch": ("2010-06-01", "2010-06-01", "day")}
