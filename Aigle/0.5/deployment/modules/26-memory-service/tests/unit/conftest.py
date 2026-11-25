"""
Shared fixtures for tests/unit/.

Every file here except test_latency.py constructs real MemvidStore/service
objects without mocking the embedder, so every frame write (session turns,
summaries, boundaries, facts, media entries) was paying for a real
sentence-transformers model load + CPU inference (BAAI/bge-m3, ~1s+ per
call). Across ~200+ test functions that adds up to 47min-1h+ -- looked
like a hang from CI, but it's just real embedding inference with nothing
timing it out.

Patches services.memvid_store._get_embedder for the whole session with a
deterministic, dependency-free stub -- same target test_latency.py already
patches for its own benchmark, but using feature-hashing instead of pure
random vectors so text with overlapping words still embeds as more similar
than text that doesn't (test_search.py's test_session_search_returns_hits
asserts the "Raptor" turn ranks above the unrelated "weather" turn, which a
pure-random embedder would only satisfy by chance).
"""
import hashlib
import os
import re
import sys

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

import numpy as np
import pytest
from memvid_sdk.embeddings import EmbeddingProvider

_DIM = 128


def _embed_text(text: str) -> list[float]:
    # Character trigrams instead of whitespace-delimited word tokens --
    # CJK text (this service handles Chinese content) has no spaces between
    # words, so an ASCII/word tokenizer sees it as unrecognizable and loses
    # all signal. Trigrams over raw characters capture substring overlap
    # for any script without needing language-specific segmentation.
    s = re.sub(r"\s+", " ", text.lower()).strip()
    n = 3
    grams = [s[i : i + n] for i in range(len(s) - n + 1)] or ([s] if s else [])
    vec = np.zeros(_DIM)
    for g in grams:
        bucket = int(hashlib.md5(g.encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % _DIM
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).tolist()
    # No recognizable tokens (empty/punctuation-only text) -- fall back to a
    # stable pseudo-random unit vector so cosine similarity stays
    # well-defined instead of comparing against an all-zero vector.
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.standard_normal(_DIM)
    return (vec / np.linalg.norm(vec)).tolist()


class _FastLexicalEmbedder(EmbeddingProvider):
    """Deterministic feature-hashing embedder -- no model to load, and
    shared-vocabulary texts still embed closer together than unrelated
    ones, unlike a pure-random stub."""

    @property
    def dimension(self) -> int:
        return _DIM

    @property
    def model_name(self) -> str:
        return "fast-lexical-stub"

    def embed_documents(self, texts):
        return [_embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _embed_text(text)


_stub = _FastLexicalEmbedder()


@pytest.fixture(scope="session", autouse=True)
def _fast_embedder():
    from services import memvid_store

    original = memvid_store._get_embedder
    memvid_store._get_embedder = lambda: _stub
    try:
        yield
    finally:
        memvid_store._get_embedder = original
