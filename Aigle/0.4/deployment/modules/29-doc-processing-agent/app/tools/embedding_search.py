"""EmbeddingSearchTool — semantic document search for DocAgent (DA-6).

Pipeline
--------
1. **Index** an uploaded document's extracted text: chunk → embed via **Module 07**
   (``POST /inference/infer``, ``task="embedding"``, model ``bge-m3``, 1024-dim) →
   upsert into the shared Qdrant collection ``doc_agent_embeddings`` (cosine).
2. **Search**: embed the query string → nearest-neighbour lookup → ranked
   ``[{id, score, text, source}]``.

Both happen in a single ``/api/v1/docagent/process?task=search`` request: the
just-uploaded document is indexed, then searched together with everything
previously indexed — so DocAgent powers ad-hoc RAG with no separate pipeline.

Decoupling from DA-1..DA-5
--------------------------
This tool operates on **already-extracted text** (``document_text``); turning an
uploaded PDF/DOCX/… into text is DA-2's ``PDFRenderTool`` / reader chain. That
keeps DA-6 independently testable and buildable while DA-1..DA-5 are owned by
others. Config is read from ``DA_*`` env vars via ``EmbeddingSearchConfig`` (the
same vars DA-1's ``app/core/config.py`` will expose) or injected directly, so
there is no import-time dependency on the not-yet-existing scaffold.

Notes on spec vs. reality
-------------------------
* The ticket wrote ``task: "text-embedding"``; Module 07's canonical task family
  is ``embedding`` (see ``22-benchmark-service/app/services/judge.py``), used here.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.chunking import chunk_text
from app.core.doc_collection import ensure_doc_collection

logger = logging.getLogger(__name__)

# smolagents is optional at import time so the tool (and its tests) run without
# the heavy agent stack installed. When present, we subclass the real Tool so an
# agent can discover this as a callable tool (DA-2 AC: "all tools are
# smolagents.Tool subclasses").
try:  # pragma: no cover - trivial import guard
    from smolagents import Tool as _ToolBase
except Exception:  # pragma: no cover
    class _ToolBase:  # minimal shim mirroring the smolagents.Tool surface
        def __init__(self, *args, **kwargs):
            pass


# Stable namespace so the same (source, chunk_idx, content) always maps to the
# same point id — re-uploading an identical document overwrites instead of
# duplicating (content-hash idempotency; see plan open-question #2).
_POINT_NS = uuid.UUID("6ba7b814-9dad-11d1-80b4-00c04fd430c8")


@dataclass
class EmbeddingSearchConfig:
    """DA-6 settings, sourced from ``DA_*`` env vars (or constructed directly)."""

    ai_lifecycle_url: str = "http://raptor-ai-lifecycle-api:8010"
    embed_model: str = "bge-m3"
    qdrant_url: str = "http://raptor-qdrant:6333"
    collection: str = "doc_agent_embeddings"
    vector_size: int = 1024
    embed_timeout: float = 30.0
    chunk_size: int = 512
    chunk_overlap: int = 64
    embed_batch_size: int = 64

    @classmethod
    def from_env(cls) -> "EmbeddingSearchConfig":
        d = cls()
        return cls(
            ai_lifecycle_url=os.getenv("DA_AI_LIFECYCLE_URL", d.ai_lifecycle_url),
            embed_model=os.getenv("DA_EMBED_MODEL", d.embed_model),
            qdrant_url=os.getenv("DA_QDRANT_URL", d.qdrant_url),
            collection=os.getenv("DA_QDRANT_DOC_COLLECTION", d.collection),
            vector_size=int(os.getenv("DA_QDRANT_VECTOR_SIZE", d.vector_size)),
            embed_timeout=float(os.getenv("DA_EMBED_TIMEOUT", d.embed_timeout)),
            chunk_size=int(os.getenv("DA_CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(os.getenv("DA_CHUNK_OVERLAP", d.chunk_overlap)),
            embed_batch_size=int(os.getenv("DA_EMBED_BATCH_SIZE", d.embed_batch_size)),
        )


class EmbeddingSearchTool(_ToolBase):
    name = "embedding_search"
    description = (
        "Semantically search previously uploaded documents. Optionally index a "
        "just-uploaded document's text first, then return the passages most "
        "similar to the query as a ranked list of {id, score, text, source}."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural-language search query.",
        },
        "document_text": {
            "type": "string",
            "description": "Full extracted text of a just-uploaded document to "
            "index before searching. Omit to search the existing store only.",
            "nullable": True,
        },
        "source": {
            "type": "string",
            "description": "Identifier for the indexed document (filename or asset id).",
            "nullable": True,
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return (default 10).",
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(
        self,
        config: Optional[EmbeddingSearchConfig] = None,
        qdrant_client: Optional[QdrantClient] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        super().__init__()
        self.cfg = config or EmbeddingSearchConfig.from_env()
        self._qdrant = qdrant_client or QdrantClient(url=self.cfg.qdrant_url)
        self._http = http_client  # injected for tests; else a client per call
        ensure_doc_collection(self._qdrant, self.cfg.collection, self.cfg.vector_size)

    # ── Module 07 embedding ──────────────────────────────────────────────────
    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` via Module 07 (batched). Mirrors judge.py `_infer_embeddings`."""
        texts = list(texts)
        if not texts:
            return []
        url = self.cfg.ai_lifecycle_url.rstrip("/") + "/inference/infer"
        client = self._http or httpx.Client(timeout=self.cfg.embed_timeout)
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.cfg.embed_batch_size):
                batch = texts[start : start + self.cfg.embed_batch_size]
                body = {
                    "task": "embedding",
                    "model_name": self.cfg.embed_model,
                    "data": {"inputs": batch},
                }
                resp = client.post(url, json=body)
                if resp.status_code >= 400:
                    logger.warning(
                        "Module 07 embedding call failed: HTTP %s — %s",
                        resp.status_code, resp.text[:500],
                    )
                    resp.raise_for_status()
                result = resp.json().get("result") or {}
                batch_vecs = result.get("embeddings", [])
                if len(batch_vecs) != len(batch):
                    raise RuntimeError(
                        f"Module 07 returned {len(batch_vecs)} vectors for {len(batch)} inputs"
                    )
                vectors.extend([float(x) for x in v] for v in batch_vecs)
            return vectors
        finally:
            if self._http is None:
                client.close()

    # ── indexing ─────────────────────────────────────────────────────────────
    def index_document(
        self,
        text: str,
        source: str,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> int:
        """Chunk → embed → upsert. Returns the number of chunks indexed."""
        chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            return 0
        vectors = self._embed(chunks)
        now = time.time()
        points: list[qmodels.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {"text": chunk, "source": source, "chunk_idx": idx, "indexed_at": now}
            if extra_payload:
                payload.update(extra_payload)
            pid = str(uuid.uuid5(_POINT_NS, f"{source}::{idx}::{chunk}"))
            points.append(qmodels.PointStruct(id=pid, vector=vector, payload=payload))
        self._qdrant.upsert(collection_name=self.cfg.collection, points=points, wait=True)
        logger.info("Indexed %d chunk(s) from source=%r", len(points), source)
        return len(points)

    # ── search ───────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Nearest-neighbour search; returns ranked {id, score, text, source}."""
        qvec = self._embed([query])[0]
        # query_points is the current API (qdrant-client >= 1.10; .search() was
        # removed in 1.14+). Passing a raw vector as `query` does a kNN lookup.
        hits = self._qdrant.query_points(
            collection_name=self.cfg.collection,
            query=qvec,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "id": str(h.id),
                "score": float(h.score),
                "text": (h.payload or {}).get("text", ""),
                "source": (h.payload or {}).get("source"),
            }
            for h in hits
        ]

    # ── smolagents entrypoint ────────────────────────────────────────────────
    def forward(
        self,
        query: str,
        document_text: Optional[str] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        top_k = top_k or 10
        indexed = 0
        if document_text:
            indexed = self.index_document(document_text, source or "inline-upload")
        results = self.search(query, top_k)
        return {"query": query, "indexed_chunks": indexed, "results": results}
