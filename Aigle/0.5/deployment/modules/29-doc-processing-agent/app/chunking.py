"""Text chunking for DocAgent embedding search (DA-6).

Sentence-aware, dependency-free chunking. Whitespace (including the hard line
breaks PDF extraction injects mid-sentence) is collapsed first, then text is
split on sentence boundaries and greedily packed into ~``size``-char windows
with ``overlap`` chars of trailing-sentence context carried into the next chunk.

Keeping chunks aligned to sentence boundaries (rather than slicing mid-word)
materially improves retrieval quality: a focused passage stays intact instead of
being diluted across a cut, which raises its cosine similarity to a matching
query. A single sentence longer than ``size`` falls back to a hard char window so
the function always terminates. bge-m3 handles 8192 tokens, so a ~512-char window
stays well within limits.

Token-aware chunking (model tokenizer / tiktoken) is a straightforward future
swap — kept char-based here so DA-6 pulls in no tokenizer dependency.
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_BOUNDARY = re.compile(r"\s")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Fixed char window for a single over-long segment (no mid-run infinite loop)."""
    step = max(1, size - overlap)
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        window = text[i : i + size].strip()
        if window:
            out.append(window)
        i += step
    return out


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Return the trailing <=``overlap`` chars of ``chunk``, snapped to a word start."""
    if overlap <= 0 or len(chunk) <= overlap:
        return chunk if overlap > 0 else ""
    tail = chunk[-overlap:]
    m = _WORD_BOUNDARY.search(tail)
    return tail[m.end():].strip() if m else tail.strip()


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Split ``text`` into overlapping, sentence-aligned windows of ~``size`` chars.

    Empty/blank input yields ``[]``; ``size <= 0`` returns the whole text as one
    chunk. Chunks are non-empty and stripped.
    """
    text = (text or "").strip()
    if not text:
        return []
    if size <= 0:
        return [text]

    overlap = max(0, min(overlap, size - 1))
    sentences = _split_sentences(text)

    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if len(sent) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(sent, size, overlap))
            continue
        if cur and len(cur) + 1 + len(sent) > size:
            chunks.append(cur)
            tail = _overlap_tail(cur, overlap)
            cur = f"{tail} {sent}".strip() if tail else sent
        else:
            cur = f"{cur} {sent}".strip() if cur else sent
    if cur:
        chunks.append(cur)
    return chunks
