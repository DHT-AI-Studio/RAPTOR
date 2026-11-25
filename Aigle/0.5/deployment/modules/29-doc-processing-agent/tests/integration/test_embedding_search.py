"""DA-6 acceptance-criterion test: real v314 bge-m3 embeddings against a live
Module 07, asserting the EmbeddingSearchTool returns a top score > 0.7 for the
scope3_emissions.pdf fixture.

Hits the **v314 dev** AI-Lifecycle API (``DA_TEST_INFERENCE_URL``, default
``http://localhost:9998``). Skipped automatically when the endpoint or
reportlab/pymupdf are unavailable.

Mocked, fully-offline unit coverage of the same tool is in
``tests/unit/test_embedding_search.py``.
"""
from __future__ import annotations

import os

import httpx
import pytest
from app.tools.embedding_search import EmbeddingSearchConfig, EmbeddingSearchTool
from qdrant_client import QdrantClient


def _read_pdf_text(path: str) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


@pytest.mark.integration
def test_scope3_pdf_search_scores_above_0_7():
    pytest.importorskip("fitz", reason="PyMuPDF required for the PDF read")
    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    pdf = os.path.join(fixtures, "scope3_emissions.pdf")
    if not os.path.exists(pdf):
        pytest.skip("scope3_emissions.pdf missing — run make_scope3_pdf.py")

    inference_url = os.getenv("DA_TEST_INFERENCE_URL", "http://localhost:9998")
    embed_model = os.getenv("DA_EMBED_MODEL", "bge-m3")
    try:
        httpx.get(inference_url + "/inference/health", timeout=3)
    except Exception:  # noqa: BLE001 -- reachability probe feeding pytest.skip -- skip reason itself communicates it
        pytest.skip(f"Module 07 (v314) not reachable at {inference_url}")

    cfg = EmbeddingSearchConfig(
        ai_lifecycle_url=inference_url,
        embed_model=embed_model,
        collection="da6_integration",
        vector_size=1024,
        chunk_size=400,
        chunk_overlap=60,
    )
    tool = EmbeddingSearchTool(config=cfg, qdrant_client=QdrantClient(location=":memory:"))

    text = _read_pdf_text(pdf)
    assert "scope 3" in text.lower()
    indexed = tool.index_document(text, source="scope3_emissions.pdf")
    assert indexed >= 1

    results = tool.search("scope 3 supplier emissions", top_k=5)
    assert results, "no results returned"
    assert results[0]["score"] > 0.7, f"top score {results[0]['score']:.3f} <= 0.7"
    assert results[0]["source"] == "scope3_emissions.pdf"
