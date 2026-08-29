"""
Latency benchmark for MV-3: long-term search must complete in < 200 ms
for a store with ≤ 10,000 frames.

10,000 frames are spread across multiple shards (~1,400 frames each) to
stay within the memvid-sdk free-tier 50 MB per-file cap. The service
fan-outs search across all shards concurrently and merges results.

Seeding uses an in-process stub embedder (64-dim random vectors) so the
fixture completes in seconds rather than minutes. The test measures search
*infrastructure* latency — HNSW fan-out, shard merge, date filtering —
not embedding-model throughput.

Run separately from unit tests:

    python -m pytest test_latency.py -v -m slow

Skip in CI by default:

    python -m pytest -m "not slow"
"""
import math
import os
import sys
import time

os.environ.setdefault("MEM_REDIS_HOST", "localhost")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import asyncio
from pathlib import Path
from typing import List, Sequence
from unittest.mock import patch

import numpy as np
import pytest
import pytest_asyncio

from memvid_sdk.embeddings import EmbeddingProvider
from services.long_term_memory import LongTermMemoryService, SearchRequest
from services.memvid_store import sync_append_batch

FRAME_COUNT = 10_000
FRAMES_PER_SHARD = 1_400
LATENCY_LIMIT_MS = 200
REPEATS = 5
_DIM = 64  # small dim keeps seeding fast; HNSW search latency is independent of dim


# ── Stub embedder ─────────────────────────────────────────────────────────────

class _FastEmbedder(EmbeddingProvider):
    """Deterministic random embedder — no model loading, guaranteed consistent dims."""

    @property
    def dimension(self) -> int:
        return _DIM

    @property
    def model_name(self) -> str:
        return "fast-stub"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(texts), _DIM)).tolist()

    def embed_query(self, text: str) -> List[float]:
        seed = int.from_bytes(text.encode()[:8].ljust(8, b"\x00"), "little") & 0xFFFF_FFFF
        return np.random.default_rng(seed).standard_normal(_DIM).tolist()


_stub = _FastEmbedder()

# Only memvid_store needs patching. long_term_memory.search() imports _get_embedder
# inside the method body, so the dynamic `from services.memvid_store import _get_embedder`
# re-resolves the module attribute on every call and picks up the patch automatically.
_PATCHES = [
    patch("services.memvid_store._get_embedder", new=lambda: _stub),
]


# ── Frame factory ─────────────────────────────────────────────────────────────

def _build_frames(n: int, base_ts: float) -> list[dict]:
    frame_types = ["conversation", "preference", "fact"]
    topics = ["AI assistant", "user preference", "known fact", "system event", "entity record"]
    return [
        {
            "title": f"{frame_types[i % 3]} {i}",
            "label": frame_types[i % 3],
            "text": (
                f"{topics[i % 5]} number {i}: "
                f"details about this {frame_types[i % 3]} stored in long-term memory."
            ),
            "timestamp": base_ts + i,
            "session_id": f"sess_{i % 100}",
            "frame_type": frame_types[i % 3],
        }
        for i in range(n)
    ]


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def seeded_svc(tmp_path_factory):
    """Seed 10,000 frames in-process with the stub embedder, then yield the service.

    The stub patch stays active across the yield so all tests in the module
    use the same 64-dim embedder for both index building and query embedding —
    eliminating the dimension mismatch that occurs when seeder and searcher
    run in separate processes with potentially different model configurations.
    """
    tmp = tmp_path_factory.mktemp("latency")
    svc = LongTermMemoryService(storage_root=str(tmp))

    user_dir = Path(str(tmp)) / "user_bench_user"
    (user_dir / "media").mkdir(parents=True, exist_ok=True)

    frames = _build_frames(FRAME_COUNT, base_ts=1_000_000.0)
    num_shards = math.ceil(FRAME_COUNT / FRAMES_PER_SHARD)
    loop = asyncio.get_running_loop()

    for p in _PATCHES:
        p.start()
    try:
        # Seed all shards concurrently in the default thread pool.
        # Each shard path is independent, so there are no write-lock conflicts.
        seed_tasks = [
            loop.run_in_executor(
                None,
                sync_append_batch,
                str(user_dir / f"long_term_{i}.mv2"),
                frames[i * FRAMES_PER_SHARD : min((i + 1) * FRAMES_PER_SHARD, FRAME_COUNT)],
            )
            for i in range(num_shards)
        ]
        await asyncio.gather(*seed_tasks)

        # Pre-warm the process pool so workers are forked before timing begins.
        # Without this, the first timed search includes ~500ms of fork overhead.
        await svc.search("bench_user", SearchRequest(query="warmup", top_k=1))

        yield svc
    finally:
        for p in _PATCHES:
            p.stop()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.asyncio
async def test_search_latency_under_200ms(seeded_svc):
    """Each of 5 search calls on a 10,000-frame store must complete within 200 ms."""
    queries = [
        "AI assistant details",
        "user preference memory",
        "known fact stored",
        "system event number",
        "entity long-term",
    ]
    assert len(queries) == REPEATS

    for query in queries:
        req = SearchRequest(query=query, top_k=10)
        t0 = time.perf_counter()
        await seeded_svc.search("bench_user", req)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < LATENCY_LIMIT_MS, (
            f"Search '{query}' took {elapsed_ms:.1f} ms — exceeds {LATENCY_LIMIT_MS} ms SLA"
        )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_search_latency_worst_case_top50(seeded_svc):
    """Worst-case: top_k=50 on 10,000-frame store must still be under 200 ms."""
    req = SearchRequest(query="memory details long-term", top_k=50)
    t0 = time.perf_counter()
    await seeded_svc.search("bench_user", req)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < LATENCY_LIMIT_MS, (
        f"top_k=50 search took {elapsed_ms:.1f} ms — exceeds {LATENCY_LIMIT_MS} ms SLA"
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_search_latency_with_date_filter(seeded_svc):
    """Date-filtered search on 10,000-frame store must stay under 200 ms."""
    # Frames span timestamps 1_000_000 to 1_009_999; filter to middle third
    req = SearchRequest(
        query="fact stored long-term",
        top_k=10,
        from_date=1_003_333.0,
        to_date=1_006_666.0,
    )
    t0 = time.perf_counter()
    await seeded_svc.search("bench_user", req)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < LATENCY_LIMIT_MS, (
        f"Date-filtered search took {elapsed_ms:.1f} ms — exceeds {LATENCY_LIMIT_MS} ms SLA"
    )
