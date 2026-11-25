"""CM-5 / VIE01-408: graphrag/tkg's own Chunk-touching queries
(fulltext_search_moments/get_subgraph/get_moment_subgraph) must exclude a
flagged or removed Chunk/Source the same way searcher.py's
_filters()/_source_filters() already do for the main hybrid/vector/BM25
search path -- otherwise that content stays reachable through the graph API
even though plain search hides it. 'cleared' content is not excluded. Pure
SQL/Cypher-string assertions against a FakeClient, no live ArcadeDB -- this
file had no direct unit coverage before (status='active' itself was only
ever verified live, per its own docstrings), so these also close that gap.
"""
from __future__ import annotations

import pytest
from app.services import graph_query


class FakeClient:
    def __init__(self):
        self.calls: list[dict] = []

    async def database_exists(self, db):
        return True

    async def query(self, db, sql, params=None, language=None):
        self.calls.append({"sql": sql, "params": params, "language": language})
        return []


@pytest.mark.asyncio
async def test_fulltext_search_moments_excludes_flagged_chunks():
    client = FakeClient()
    await graph_query.fulltext_search_moments(client, "b1", "hello")
    sql = client.calls[0]["sql"]
    assert "AND status = 'active'" in sql
    assert "AND (review_status IS NULL OR (review_status != 'flagged' AND review_status != 'removed'))" in sql


@pytest.mark.asyncio
async def test_get_subgraph_status_ok_excludes_flagged_nodes():
    client = FakeClient()
    await graph_query.get_subgraph(client, "b1", "ent-1")
    # node_rows and edge_rows (calls[1], calls[2]) both carry _status_ok
    for call in client.calls[1:]:
        assert "n.status IS NULL OR n.status = 'active'" in call["sql"]
        assert "n.review_status IS NULL OR (n.review_status <> 'flagged' AND n.review_status <> 'removed')" in call["sql"]


@pytest.mark.asyncio
async def test_get_moment_subgraph_excludes_a_flagged_chunk():
    client = FakeClient()
    await graph_query.get_moment_subgraph(client, "b1", "chunk-1")
    sql = client.calls[0]["sql"]
    assert "status: 'active'" in sql
    assert "c.review_status IS NULL OR (c.review_status <> 'flagged' AND c.review_status <> 'removed')" in sql
