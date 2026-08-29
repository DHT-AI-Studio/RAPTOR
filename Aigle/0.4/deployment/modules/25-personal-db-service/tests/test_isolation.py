"""PA-10 — isolation: index as user A, search as user B; user B gets 0 results.

Isolation is physical: each branch_id maps to its own `user_<branch>` ArcadeDB
database, so B's database simply never contains A's data.
"""
from __future__ import annotations

import pytest

from app.models.index import ChunkIndexRequest
from app.models.search import SearchRequest
from app.services import indexer, searcher
from tests.conftest import fake_vector

pytestmark = pytest.mark.asyncio


async def test_user_b_cannot_see_user_a_data(client, make_db, mock_embed):
    branch_a = await make_db("ittest_iso_a")
    branch_b = await make_db("ittest_iso_b")                 # distinct, empty database

    await indexer.index_chunk(client, branch_a, ChunkIndexRequest(
        chunk_id="c1", type="documents", embedding_type="text",
        embedding=fake_vector("secret data for user A"), text="secret data for user A"))

    resp_a = await searcher.vector_search(client, branch_a, SearchRequest(query="secret", top_k=5))
    resp_b = await searcher.vector_search(client, branch_b, SearchRequest(query="secret", top_k=5))

    assert len(resp_a.results) == 1
    assert len(resp_b.results) == 0                          # no cross-contamination
