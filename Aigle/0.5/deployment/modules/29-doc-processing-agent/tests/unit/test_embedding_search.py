"""Unit tests for DA-6 EmbeddingSearchTool.

Run fully offline: Module 07 is mocked with ``httpx.MockTransport`` (real
request/response shape) or a deterministic keyword vectorizer, and Qdrant runs
in embedded ``:memory:`` mode -- no services required.

The DA-6 acceptance-criterion integration test (real bge-m3 embeddings against
a live Module 07) is in ``tests/integration/test_embedding_search.py``.
"""
from __future__ import annotations

import json
import math

import httpx
from app.chunking import chunk_text
from app.tools.embedding_search import EmbeddingSearchConfig, EmbeddingSearchTool
from qdrant_client import QdrantClient

# ── keyword vectorizer (deterministic stand-in for a real embedding model) ────
_VOCAB = ["scope", "supplier", "emissions", "transport", "energy", "water",
          "carbon", "reduction"]


def _kw_vector(text: str) -> list[float]:
    t = text.lower()
    v = [float(t.count(w)) for w in _VOCAB]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _mock_module07(dim_vocab: bool = True) -> httpx.MockTransport:
    """MockTransport that answers Module 07's ``/inference/infer`` embedding call."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/inference/infer")
        body = json.loads(request.content)
        assert body["task"] == "embedding"
        assert body["model_name"]
        inputs = body["data"]["inputs"]
        vecs = [_kw_vector(t) for t in inputs]
        return httpx.Response(200, json={"success": True, "result": {"embeddings": vecs}})

    return httpx.MockTransport(handler)


def _tool(vector_size: int) -> EmbeddingSearchTool:
    cfg = EmbeddingSearchConfig(
        embed_model="bge-m3-test",
        collection="da6_unit",
        vector_size=vector_size,
        chunk_size=180,
        chunk_overlap=30,
    )
    http = httpx.Client(transport=_mock_module07(), base_url="http://module07.test")
    qdrant = QdrantClient(location=":memory:")
    return EmbeddingSearchTool(config=cfg, qdrant_client=qdrant, http_client=http)


# ── chunker ───────────────────────────────────────────────────────────────────
def test_chunk_text_basic():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    text = "a" * 500
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) == 4  # step=150 -> starts 0,150,300,450
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_overlap_clamped():
    # overlap >= size must not deadlock
    chunks = chunk_text("hello world " * 20, size=50, overlap=999)
    assert len(chunks) >= 1


# ── embedding request shape ────────────────────────────────────────────────────
def test_embed_request_shape_and_parse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": {"embeddings": [[0.1] * 4, [0.2] * 4]}})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://m07.test")
    tool = EmbeddingSearchTool(
        config=EmbeddingSearchConfig(embed_model="bge-m3", collection="c", vector_size=4),
        qdrant_client=QdrantClient(location=":memory:"),
        http_client=http,
    )
    out = tool._embed(["one", "two"])
    assert out == [[0.1] * 4, [0.2] * 4]
    assert captured["body"]["task"] == "embedding"          # canonical, not "text-embedding"
    assert captured["body"]["model_name"] == "bge-m3"
    assert captured["body"]["data"]["inputs"] == ["one", "two"]


# ── index + search plumbing ─────────────────────────────────────────────────────
def test_index_and_search_ranks_expected_chunk():
    tool = _tool(vector_size=len(_VOCAB))
    doc = (
        "Scope 3 supplier emissions dominate the value chain footprint. "
        "Water usage and energy intensity are tracked separately per site. "
        "Transport logistics contribute a smaller share of total carbon."
    )
    n = tool.index_document(doc, source="doc-a")
    assert n >= 2

    results = tool.search("scope supplier emissions", top_k=3)
    assert results, "search returned no hits"
    top = results[0]
    assert {"id", "score", "text", "source"} <= set(top)
    assert top["source"] == "doc-a"
    assert "scope 3 supplier emissions" in top["text"].lower()
    assert 0.0 <= top["score"] <= 1.0000001


def test_reindex_same_document_is_idempotent():
    tool = _tool(vector_size=len(_VOCAB))
    doc = "Scope 3 supplier emissions and carbon reduction targets. " * 4
    first = tool.index_document(doc, source="dup")
    tool.index_document(doc, source="dup")  # same content, same ids
    count = tool._qdrant.count(collection_name=tool.cfg.collection).count
    assert count == first  # no duplicate points


def test_forward_indexes_then_searches():
    tool = _tool(vector_size=len(_VOCAB))
    out = tool.forward(
        query="supplier emissions",
        document_text="Scope 3 supplier emissions reporting overview. Carbon reduction levers.",
        source="upload-1",
        top_k=5,
    )
    assert out["indexed_chunks"] >= 1
    assert out["results"]
    assert out["results"][0]["source"] == "upload-1"


def test_search_only_when_no_document_text():
    tool = _tool(vector_size=len(_VOCAB))
    tool.index_document("Scope 3 emissions from suppliers.", source="pre")
    out = tool.forward(query="scope emissions", top_k=3)
    assert out["indexed_chunks"] == 0
    assert out["results"]
