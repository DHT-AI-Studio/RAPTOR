"""Local embedding client for Module 25.

Loads the configured SentenceTransformer model once (lazy, thread-safe) and
produces 1024-dim normalised dense vectors.  No HTTP dependency on Module 07.

Settings (env prefix PD_):
  PD_EMBEDDING_MODEL   — HuggingFace model name/path  (default: BAAI/bge-m3)
  PD_EMBEDDING_DEVICE  — 'cpu', 'cuda', or 'auto'     (default: auto)
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import List

from app.core.config import settings

logger = logging.getLogger("personal_db.embedder")

_model = None
_model_lock = asyncio.Lock()


def _load_model():
    from sentence_transformers import SentenceTransformer
    # sentence-transformers/torch reject the literal "auto" — pass None so it
    # auto-selects (cuda if available, else cpu). 'cpu'/'cuda[:n]' pass through.
    device = None if settings.embedding_device in (None, "", "auto") else settings.embedding_device
    logger.info("[embedder] loading model %s on device=%s", settings.embedding_model, device or "auto")
    model = SentenceTransformer(settings.embedding_model, device=device)
    logger.info("[embedder] model ready")
    return model


async def _get_model():
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                loop = asyncio.get_event_loop()
                _model = await loop.run_in_executor(None, _load_model)
    return _model


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return one 1024-dim normalised float vector per input text."""
    model = await _get_model()
    loop = asyncio.get_event_loop()
    encode = partial(model.encode, texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = await loop.run_in_executor(None, encode)
    return vectors.tolist()
