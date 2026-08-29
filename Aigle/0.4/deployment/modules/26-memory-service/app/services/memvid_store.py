from __future__ import annotations

import contextlib
import fcntl
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

# Suppress HuggingFace tokenizers warning when worker processes are forked
# after the parent has already loaded the tokenizer.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import memvid_sdk
from memvid_sdk.embeddings import EmbeddingProvider
from sentence_transformers import SentenceTransformer

from core.config import settings

# Per-path locks. Search runs in forked worker processes (see get_search_pool),
# so a threading.Lock alone can't serialise across them — flock on a sidecar
# .lock file is kernel-held and works across processes.
_write_locks: dict[str, threading.Lock] = {}
_write_lock_map = threading.Lock()


@contextlib.contextmanager
def _lock_for(path: str) -> Iterator[None]:
    with _write_lock_map:
        if path not in _write_locks:
            _write_locks[path] = threading.Lock()
        thread_lock = _write_locks[path]

    with thread_lock:
        lock_path = f"{path}.lock"
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def normalize_scores(raw_scores: list[float]) -> list[float]:
    """Min-max normalize a batch of raw hybrid-search scores to [0, 1].

    memvid-sdk's absolute score scale is undocumented and varies by query, so a
    fixed-midpoint sigmoid saturates to ~1.0 for realistic inputs. Normalizing
    relative to the other candidates in the same result batch keeps scores
    comparable within one response regardless of the underlying raw scale.
    """
    if not raw_scores:
        return []
    lo, hi = min(raw_scores), max(raw_scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in raw_scores]
    return [round((s - lo) / (hi - lo), 4) for s in raw_scores]

# ── Process pool for GIL-free fan-out searches ────────────────────────────────
# memvid-sdk's find() is GIL-bound (pure-Python BM25 + HNSW), so concurrent
# asyncio.to_thread calls serialise on the GIL. A fork-based ProcessPoolExecutor
# bypasses this: workers inherit the already-loaded embedder singleton via
# copy-on-write, so no extra RAM is consumed for model weights.
_search_pool: ProcessPoolExecutor | None = None
_search_pool_lock = threading.Lock()


def get_search_pool() -> ProcessPoolExecutor:
    """Return (or lazily create) the shared process pool for concurrent searches."""
    global _search_pool
    if _search_pool is None:
        with _search_pool_lock:
            if _search_pool is None:
                # Pre-warm the embedder before forking so workers inherit it via
                # copy-on-write (avoids re-loading the model in each worker).
                _get_embedder()
                workers = min(8, (os.cpu_count() or 2) * 2)
                ctx = multiprocessing.get_context("fork")
                _search_pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    return _search_pool


class LocalEmbeddings(EmbeddingProvider):
    """In-process SentenceTransformer embedding provider (no external HTTP calls)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


@lru_cache(maxsize=1)
def _get_embedder() -> LocalEmbeddings:
    """Singleton — model is loaded once and reused across all MemvidStore instances."""
    return LocalEmbeddings(settings.embedding_model)


# ── Module-level sync helpers (run via asyncio.to_thread) ────────────────────
# These helpers automatically maintain a sidecar .meta.json index alongside
# every .mv2 file so that custom metadata (frame_type, session_id, timestamp,
# text) can be retrieved from timeline/search results — memvid-sdk hit objects
# only expose frame_id + text snippets, not arbitrary stored fields.

def sync_remove(path: str, frame_id: str | int) -> None:
    """Soft-delete a frame by removing it from the sidecar only.

    Calling _mv.remove() corrupts the HNSW vector index, causing find() to return
    no results on subsequent searches. Since sync_search and sync_timeline both
    filter by meta_index presence, removing the sidecar entry is sufficient —
    the frame becomes invisible to all API endpoints without touching the .mv2 file.
    """
    from services import meta_index
    meta_index.remove_entry(path, frame_id)


def sync_get_remaining_capacity(path: str) -> int:
    """Return remaining capacity bytes of an existing .mv2 file."""
    with _lock_for(path):
        store = MemvidStore()
        store.open(path, read_only=True)
        try:
            return store._mv.stats().get("remaining_capacity_bytes", 0)
        finally:
            store.close()


class StoreCapacityError(RuntimeError):
    """Raised when the .mv2 file has hit the SDK capacity limit (50 MB free tier)."""


def sync_append(path: str, frame: dict) -> str:
    """Open .mv2, append one frame, write sidecar, close. Returns frame_id."""
    from services import meta_index
    with _lock_for(path):
        store = MemvidStore()
        store.open(path)
        try:
            frame_id = str(store.append(frame))
        except memvid_sdk.CapacityExceededError as exc:
            raise StoreCapacityError(str(exc)) from exc
        finally:
            store.close()
        meta = {k: v for k, v in frame.items() if v is not None}
        meta["frame_id"] = frame_id
        meta["frame_type"] = frame.get("frame_type") or frame.get("label", "")
        meta_index.put(path, frame_id, meta)
        return frame_id


def sync_append_batch(path: str, frames: list[dict]) -> list[str]:
    """Open .mv2 once, append many frames (single embedding batch call), update sidecar, close."""
    from services import meta_index
    with _lock_for(path):
        store = MemvidStore()
        store.open(path)
        try:
            ids = [str(i) for i in store.append_batch(frames)]
        finally:
            store.close()
        for frame_id, frame in zip(ids, frames):
            meta = {k: v for k, v in frame.items() if v is not None}
            meta["frame_id"] = frame_id
            meta["frame_type"] = frame.get("frame_type") or frame.get("label", "")
            meta_index.put(path, frame_id, meta)
        return ids


def sync_timeline(path: str) -> list[dict]:
    """Return all timeline entries joined with sidecar metadata.

    When vector embeddings are enabled, put_many() creates internal SDK chunk
    frames alongside user frames. Only frames with a meta_index entry are user
    frames — internal frames are silently skipped.
    """
    from services import meta_index
    with _lock_for(path):
        store = MemvidStore()
        store.open(path, read_only=True)
        try:
            # Use a far-future ceiling so frames with timestamps slightly ahead
            # of wall clock (e.g. batch-seeded with future offsets) are included.
            raw = store.timeline(from_ts=0, to_ts=9_999_999_999)
        finally:
            store.close()
    index = meta_index.load(path)
    result = []
    for entry in raw:
        fid = str(entry.get("frame_id", ""))
        meta = index.get(fid, {})
        if not meta:
            continue  # internal SDK chunk frame — skip
        merged = {**entry, **meta}
        # Prefer stored timestamp (float) over SDK's int version
        if "timestamp" not in meta and entry.get("timestamp"):
            merged["timestamp"] = float(entry["timestamp"])
        result.append(merged)
    return result


def sync_search_counted(path: str, query: str, top_k: int, query_vec: list[float] | None = None) -> tuple[list[dict], int]:
    """Like sync_search but also returns the total frame count from the sidecar.

    Loads meta_index only once (sync_search would load it anyway), so callers
    that need the total can avoid a second meta_index.load thread task.
    """
    from services import meta_index
    with _lock_for(path):
        store = MemvidStore()
        store.open(path, read_only=True)
        try:
            hits = store.search(query, top_k=top_k * 3, query_vec=query_vec)
        finally:
            store.close()
    index = meta_index.load(path)
    result = []
    for hit in hits:
        fid = str(hit.get("frame_id", ""))
        meta = index.get(fid, {})
        if not meta:
            continue
        result.append({**hit, **meta})
    return result, len(index)


def sync_search(path: str, query: str, top_k: int, query_vec: list[float] | None = None) -> list[dict]:
    """Run hybrid search, join hits with sidecar metadata.

    Fetches top_k * 3 from the SDK to compensate for internal SDK chunk frames
    that will be filtered out (they have no meta_index entry).

    Pass a pre-computed query_vec to skip per-thread embedding inference — useful
    when fan-out searches run across many files concurrently (avoids N serialized
    embed_query calls on the same GIL-bound PyTorch model).
    """
    from services import meta_index
    with _lock_for(path):
        store = MemvidStore()
        store.open(path, read_only=True)
        try:
            hits = store.search(query, top_k=top_k * 3, query_vec=query_vec)
        finally:
            store.close()
    index = meta_index.load(path)
    result = []
    for hit in hits:
        fid = str(hit.get("frame_id", ""))
        meta = index.get(fid, {})
        if not meta:
            continue  # internal SDK chunk frame — skip
        result.append({**hit, **meta})
    return result


class MemvidStore:
    """Thin wrapper over memvid-sdk for Raptor memory operations.

    Lifecycle: open(path) → append/search/timeline → close()
    """

    def __init__(self) -> None:
        self._mv: memvid_sdk.Memvid | None = None
        self._embedder: LocalEmbeddings = _get_embedder()

    def open(self, path: str, read_only: bool = False) -> None:
        """Open (or create) a .mv2 file, creating parent dirs as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "open" if p.exists() else "create"
        self._mv = memvid_sdk.use(
            "basic",
            str(p),
            mode=mode,
            enable_lex=True,
            enable_vec=True,
            read_only=read_only,
        )

    def append(self, frame: dict[str, Any]) -> str:
        """Write one frame, storing a vector embedding when embedder is available.

        memvid-sdk v2 ID spaces: put_many() returns an internal chunk block ID
        that differs from the user-visible frame ID used by timeline() and find().
        The correct user frame ID = stats['frame_count'] BEFORE the put_many call,
        so we capture that and return it instead of the chunk block ID.
        """
        self._require_open()
        user_frame_id = str(self._mv.stats()["frame_count"])
        text = frame.get("text") or frame.get("title", "Untitled")
        vec = self._embedder.embed_query(text)
        put_frame = {
            "title": frame.get("title", "Untitled"),
            "label": frame.get("label", "text"),
            "text": frame.get("text", ""),
        }
        self._mv.put_many([put_frame], embeddings=[vec])
        return user_frame_id

    def append_batch(self, frames: list[dict[str, Any]]) -> list[str]:
        """Write multiple frames with batch embedding but individual puts.

        Batch-embedding all texts at once is efficient; writing them one at a
        time avoids a memvid-sdk BM25 indexing bug where large put_many calls
        corrupt the lex index, making the first portion of frames unsearchable.
        """
        self._require_open()
        texts = [f.get("text") or f.get("title", "Untitled") for f in frames]
        vecs = self._embedder.embed_documents(texts)
        ids: list[str] = []
        for frame, vec in zip(frames, vecs):
            user_frame_id = str(self._mv.stats()["frame_count"])
            put_frame = {
                "title": frame.get("title", "Untitled"),
                "label": frame.get("label", "text"),
                "text": frame.get("text", ""),
            }
            self._mv.put_many([put_frame], embeddings=[vec])
            ids.append(user_frame_id)
        return ids

    # SDK's HNSW uses ef_search=50; find() panics when k*2 > ef_search (i.e. k > 25).
    _HNSW_K_MAX = 25

    def search(self, query: str, top_k: int = 5, query_vec: list[float] | None = None) -> list[dict]:
        """Hybrid search (lexical + semantic when embedder available). Returns FindHit list."""
        self._require_open()
        qvec = query_vec if query_vec is not None else self._embedder.embed_query(query)
        result = self._mv.find(query, k=min(top_k, self._HNSW_K_MAX), query_embedding=qvec)
        return result.get("hits", [])

    # SDK's Memvid.timeline() defaults to limit=100 — a single .mv2 file can
    # hold far more frames than that (up to the ~50MB free-tier cap), so an
    # unbounded "give me everything in this range" call needs an explicit
    # limit well above what any one file could ever contain, or results
    # silently truncate to the newest/oldest 100 with no error.
    _TIMELINE_LIMIT = 1_000_000

    def timeline(self, from_ts: float, to_ts: float) -> list[dict]:
        """Return TimelineEntry list between two Unix timestamps."""
        self._require_open()
        return list(self._mv.timeline(
            since=int(from_ts), until=int(to_ts), limit=self._TIMELINE_LIMIT,
        ))

    def remove(self, frame_id: str | int) -> None:
        """Remove a frame by its integer frame_id.

        WARNING: calling _mv.remove() destroys the entire HNSW vector index,
        causing find() to return zero results. Use sync_remove() (meta-index-only
        soft-delete) instead of calling this method directly.
        """
        self._require_open()
        self._mv.remove(int(frame_id))

    def close(self) -> None:
        if self._mv is not None:
            self._mv.close()
            self._mv = None

    def _require_open(self) -> None:
        if self._mv is None:
            raise RuntimeError("MemvidStore: call open() first")
