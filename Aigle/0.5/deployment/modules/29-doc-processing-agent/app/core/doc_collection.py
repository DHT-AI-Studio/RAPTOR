"""Qdrant collection bootstrap for the DocAgent embedding store (DA-6).

The ``doc_agent_embeddings`` collection (cosine, 1024-dim by default) is created
idempotently. ``ensure_doc_collection`` is safe to call repeatedly — from the
FastAPI ``lifespan`` at startup (DA-1's ``app/main.py``) and from
``EmbeddingSearchTool`` on first use — so DA-6 works standalone even before the
lifespan wiring lands.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


def ensure_doc_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    """Create ``collection`` (cosine, ``vector_size``-dim) if it does not exist."""
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
    )
    logger.info(
        "Created Qdrant collection %r (size=%d, distance=%s)",
        collection, vector_size, distance,
    )
