"""Local cross-encoder reranker for Module 25.

Loads the configured CrossEncoder model once (lazy, thread-safe) — reuses the
sentence-transformers dependency already required by embedder.py, no new
requirement. Self-contained by design: hybrid_search() must keep reranking
even if module 17 is retired, since the whole point of this service is to be
able to replace it.

Scoring matches module 17's own /api/v1/search/rerank endpoint (sigmoid over
the raw cross-encoder logit, same temperature default) so result ordering is
comparable between the two backends.

Settings (env prefix PD_):
  PD_RERANKER_MODEL        — HuggingFace cross-encoder model name/path
                              (default: BAAI/bge-reranker-v2-m3)
  PD_RERANKER_DEVICE       — 'cpu', 'cuda', or 'auto' (default: auto)
  PD_RERANKER_TEMPERATURE  — sigmoid temperature scaling (default: 0.25)
"""
from __future__ import annotations

import asyncio
import math
from functools import partial
from typing import Any, Dict, List

from app.core.config import settings
from app.models.search import RerankDocument, RerankResult, SearchResult

_model = None
_model_lock = asyncio.Lock()


def _load_model():
    from sentence_transformers import CrossEncoder
    device = None if settings.reranker_device in (None, "", "auto") else settings.reranker_device
    return CrossEncoder(settings.reranker_model, device=device)


async def _get_model():
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                loop = asyncio.get_event_loop()
                _model = await loop.run_in_executor(None, _load_model)
    return _model


def _extract_text(payload: Dict[str, Any]) -> str:
    return payload.get("text") or payload.get("summary") or ""


async def rerank(query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
    """Rerank in place (score overwritten, sigmoid-scaled) and truncate to top_k."""
    if not results:
        return results
    model = await _get_model()
    pairs = [[query, _extract_text(r.payload)] for r in results]
    loop = asyncio.get_event_loop()
    predict = partial(model.predict, pairs, show_progress_bar=False)
    scores = await loop.run_in_executor(None, predict)
    t = max(settings.reranker_temperature, 1e-6)
    for r, score in zip(results, scores):
        r.score = round(1.0 / (1.0 + math.exp(-float(score) / t)), 6)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


async def rerank_documents(
    query: str, documents: List[RerankDocument], top_k: int | None,
) -> List[RerankResult]:
    """Generic rerank over caller-supplied (id, text) pairs -- the personal-db
    equivalent of Module 17's /api/v1/search/rerank utility endpoint (exposed
    via POST /personal/search/rerank), not tied to any one user's database.
    Same sigmoid-scaled scoring as rerank() above, reusing the same model
    singleton, but over an arbitrary caller-supplied document list instead of
    this service's own SearchResult/payload shape -- e.g. video_search.py's
    personal-db equivalent needs this to rerank RRF-fused candidates pooled
    from four different endpoints, not a single hybrid_search() call."""
    if not documents:
        return []
    model = await _get_model()
    pairs = [[query, d.text] for d in documents]
    loop = asyncio.get_event_loop()
    predict = partial(model.predict, pairs, show_progress_bar=False)
    scores = await loop.run_in_executor(None, predict)
    t = max(settings.reranker_temperature, 1e-6)
    results = [
        RerankResult(id=d.id, score=round(1.0 / (1.0 + math.exp(-float(score) / t)), 6), payload=d.payload)
        for d, score in zip(documents, scores)
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k] if top_k is not None else results
