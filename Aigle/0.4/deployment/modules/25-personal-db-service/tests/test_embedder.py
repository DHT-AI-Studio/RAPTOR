"""PA-10 — the REAL local embedder (module 25's SentenceTransformer BGE-M3).

The rest of the suite mocks `embed_texts` for speed, which leaves the actual
embedding path (`app/services/embedder.py`) uncovered. This test exercises it
for real, so the local-embedding change is actually validated: correct vector
count, 1024 dims, normalised, and distinct vectors for distinct text — plus a
real (un-mocked) vector search end to end.

It loads the ~2GB BGE-M3 model, so it's the slow one — skips if
sentence-transformers (or the model) is unavailable.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("sentence_transformers")

from app.core.config import settings                       # noqa: E402
from app.models.index import ChunkIndexRequest             # noqa: E402
from app.models.search import SearchRequest                # noqa: E402
from app.services import indexer, searcher                 # noqa: E402
from app.services.arcadedb_client import db_name_for       # noqa: E402
from app.services.embedder import embed_texts              # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_embedder_returns_normalised_1024d_vectors():
    vecs = await embed_texts(["hello world", "a completely different sentence"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == settings.vector_dim               # 1024
        assert all(isinstance(x, float) for x in v)
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-2                       # normalize_embeddings=True
    assert vecs[0] != vecs[1]                               # distinct text -> distinct vector


async def test_real_embedding_vector_search_ranks_relevant_first(client, make_db):
    """End-to-end with the REAL embedder (no mock): the chunk about cats should
    rank above the one about tax law for a cat query."""
    branch = await make_db("ittest_real_embed")
    db = db_name_for(branch)

    texts = {"cat": "The cat sat on the warm windowsill grooming its fur.",
             "tax": "Quarterly corporate tax filings are due at the end of the month."}
    vectors = await embed_texts(list(texts.values()))
    for (cid, text), vec in zip(texts.items(), vectors):
        await indexer.index_chunk(client, branch, ChunkIndexRequest(
            chunk_id=cid, type="documents", embedding_type="text", embedding=vec, text=text))

    # real query embedding (embed_texts NOT mocked here)
    resp = await searcher.vector_search(client, branch, SearchRequest(query="kitten playing", top_k=2))
    assert resp.results, "expected results from a real vector search"
    assert resp.results[0].payload["chunk_id"] == "cat"     # semantically closest ranks first
