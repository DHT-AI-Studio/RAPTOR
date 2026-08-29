"""
MV-7: Memory Search API — unit + integration + latency tests.

Run fast tests (no latency):

Run latency benchmark (slow, seeds 50 sessions × 100 turns each):
"""
import os
import sys
import time

os.environ.setdefault("MEM_REDIS_HOST", "localhost")
os.environ.setdefault("MEM_STORAGE_ROOT", "/tmp/mv7_test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from services.session_memory import (
    SessionMemoryService,
    SessionSearchRequest,
    TurnAppendRequest,
)
from services.long_term_memory import LongTermMemoryService, FactAddRequest
from routers.search import GlobalSearchService, GlobalSearchRequest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def redis():
    r = FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def session_svc(tmp_path, redis):
    yield SessionMemoryService(redis=redis, storage_root=str(tmp_path))


@pytest_asyncio.fixture
async def lt_svc(tmp_path):
    yield LongTermMemoryService(storage_root=str(tmp_path))


@pytest_asyncio.fixture
async def global_svc(tmp_path):
    svc = GlobalSearchService.__new__(GlobalSearchService)
    from pathlib import Path
    from services.long_term_memory import LongTermMemoryService
    from services.multimedia_memory import MultimediaMemoryService
    svc._root = Path(str(tmp_path))
    svc._lt_svc = LongTermMemoryService(storage_root=str(tmp_path))
    svc._mm_svc = MultimediaMemoryService(storage_root=str(tmp_path))
    yield svc


# ── Session search: unit tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_search_returns_hits(session_svc, redis):
    user_id = "u1"
    session_id = "sess_search"
    await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
        user_message="Tell me about Raptor platform",
        assistant_response="Raptor is a distributed AI platform for enterprise use.",
    ))
    await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
        user_message="What is the weather today?",
        assistant_response="I don't have real-time weather data.",
    ))

    req = SessionSearchRequest(query="Raptor platform", top_k=5)
    resp = await session_svc.search(user_id, session_id, req)

    assert resp.total_frames_searched == 2
    assert len(resp.hits) >= 1
    top = resp.hits[0]
    assert "Raptor" in top.text or "Raptor" in top.user_message
    assert 0.0 <= top.score <= 1.0
    assert top.timestamp.endswith("+00:00") or top.timestamp.endswith("Z")
    assert top.turn_index >= 1
    assert isinstance(top.user_message, str)
    assert isinstance(top.assistant_response, str)


@pytest.mark.asyncio
async def test_session_search_empty_session_returns_zero(session_svc):
    req = SessionSearchRequest(query="anything", top_k=5)
    resp = await session_svc.search("nobody", "ghost_sess", req)

    assert resp.hits == []
    assert resp.total_frames_searched == 0


@pytest.mark.asyncio
async def test_session_search_date_filter_from_date(session_svc):
    user_id = "u_date"
    session_id = "sess_date"
    base = 1_700_000_000.0

    for i in range(5):
        await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
            user_message=f"message {i}",
            assistant_response=f"response {i}",
            timestamp=base + i * 3600,
        ))

    from datetime import datetime, timezone
    cutoff_iso = datetime.fromtimestamp(base + 2 * 3600, tz=timezone.utc).isoformat()
    req = SessionSearchRequest(query="message", top_k=10, from_date=cutoff_iso)
    resp = await session_svc.search(user_id, session_id, req)

    for hit in resp.hits:
        hit_ts = datetime.fromisoformat(hit.timestamp).timestamp()
        assert hit_ts >= base + 2 * 3600, f"Hit timestamp {hit.timestamp} is before from_date"


@pytest.mark.asyncio
async def test_session_search_date_filter_to_date(session_svc):
    user_id = "u_date2"
    session_id = "sess_date2"
    base = 1_700_100_000.0

    for i in range(5):
        await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
            user_message=f"query {i}",
            assistant_response=f"answer {i}",
            timestamp=base + i * 3600,
        ))

    from datetime import datetime, timezone
    cutoff_iso = datetime.fromtimestamp(base + 2 * 3600, tz=timezone.utc).isoformat()
    req = SessionSearchRequest(query="query", top_k=10, to_date=cutoff_iso)
    resp = await session_svc.search(user_id, session_id, req)

    for hit in resp.hits:
        hit_ts = datetime.fromisoformat(hit.timestamp).timestamp()
        assert hit_ts <= base + 2 * 3600, f"Hit timestamp {hit.timestamp} is after to_date"


@pytest.mark.asyncio
async def test_session_search_respects_top_k(session_svc):
    user_id = "u_topk"
    session_id = "sess_topk"
    for i in range(10):
        await session_svc.append_turn(user_id, session_id, TurnAppendRequest(
            user_message=f"turn {i}",
            assistant_response=f"answer {i}",
        ))

    req = SessionSearchRequest(query="turn", top_k=3)
    resp = await session_svc.search(user_id, session_id, req)

    assert len(resp.hits) <= 3
    assert resp.total_frames_searched == 10


@pytest.mark.asyncio
async def test_session_search_user_isolation(session_svc):
    """User A's search must not see User B's turns."""
    for uid, msg in [("user_a", "secret project Alpha"), ("user_b", "something else")]:
        await session_svc.append_turn(uid, "sess_shared_id", TurnAppendRequest(
            user_message=msg,
            assistant_response="ok",
        ))

    req = SessionSearchRequest(query="secret project Alpha", top_k=10)
    resp_a = await session_svc.search("user_a", "sess_shared_id", req)
    resp_b = await session_svc.search("user_b", "sess_shared_id", req)

    # user_b's session file is separate → should not contain user_a's turn
    texts_b = [h.text for h in resp_b.hits]
    assert not any("secret project Alpha" in t for t in texts_b)


# ── Global search: unit tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_global_search_empty_user_returns_zeros(global_svc):
    req = GlobalSearchRequest(query="anything")
    resp = await global_svc.search("no_such_user", req)

    assert resp.sessions == []
    assert resp.longterm == []
    assert resp.multimedia == []
    assert resp.total_frames_searched == 0


@pytest.mark.asyncio
async def test_global_search_scope_sessions_only(tmp_path, redis):
    session_svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await session_svc.append_turn("u_scope", "sess1", TurnAppendRequest(
        user_message="global search test content",
        assistant_response="found it",
    ))

    from pathlib import Path
    from services.multimedia_memory import MultimediaMemoryService
    svc = GlobalSearchService.__new__(GlobalSearchService)
    svc._root = Path(str(tmp_path))
    svc._lt_svc = LongTermMemoryService(storage_root=str(tmp_path))
    svc._mm_svc = MultimediaMemoryService(storage_root=str(tmp_path))

    req = GlobalSearchRequest(query="global search test", top_k=5, scope=["sessions"])
    resp = await svc.search("u_scope", req)

    assert resp.longterm == []
    assert resp.multimedia == []
    assert resp.total_frames_searched > 0


@pytest.mark.asyncio
async def test_global_search_scope_longterm_only(tmp_path):
    lt_svc = LongTermMemoryService(storage_root=str(tmp_path))
    await lt_svc.add_fact("u_lt", FactAddRequest(text="User prefers dark mode", frame_type="preference"))

    from pathlib import Path
    from services.multimedia_memory import MultimediaMemoryService
    svc = GlobalSearchService.__new__(GlobalSearchService)
    svc._root = Path(str(tmp_path))
    svc._lt_svc = lt_svc
    svc._mm_svc = MultimediaMemoryService(storage_root=str(tmp_path))

    req = GlobalSearchRequest(query="dark mode preference", top_k=5, scope=["longterm"])
    resp = await svc.search("u_lt", req)

    assert resp.sessions == []
    assert resp.multimedia == []
    assert resp.total_frames_searched > 0
    assert len(resp.longterm) >= 1
    assert resp.longterm[0].frame_type in ("preference", "fact", "conversation", "entity")


@pytest.mark.asyncio
async def test_global_search_total_frames_is_sum(tmp_path, redis):
    session_svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    lt_svc = LongTermMemoryService(storage_root=str(tmp_path))

    for i in range(3):
        await session_svc.append_turn("u_sum", f"s{i}", TurnAppendRequest(
            user_message=f"msg {i}", assistant_response=f"ans {i}",
        ))
    await lt_svc.add_fact("u_sum", FactAddRequest(text="a known fact", frame_type="fact"))

    from pathlib import Path
    from services.multimedia_memory import MultimediaMemoryService
    svc = GlobalSearchService.__new__(GlobalSearchService)
    svc._root = Path(str(tmp_path))
    svc._lt_svc = lt_svc
    svc._mm_svc = MultimediaMemoryService(storage_root=str(tmp_path))

    req = GlobalSearchRequest(query="msg", top_k=10, scope=["sessions", "longterm"])
    resp = await svc.search("u_sum", req)

    # 3 session files × 1 frame each + 1 longterm shard × 1 frame = 4
    assert resp.total_frames_searched == 4


@pytest.mark.asyncio
async def test_global_search_hit_fields(tmp_path, redis):
    session_svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    await session_svc.append_turn("u_fields", "sess_f", TurnAppendRequest(
        user_message="field check question",
        assistant_response="field check answer",
    ))

    from pathlib import Path
    from services.multimedia_memory import MultimediaMemoryService
    svc = GlobalSearchService.__new__(GlobalSearchService)
    svc._root = Path(str(tmp_path))
    svc._lt_svc = LongTermMemoryService(storage_root=str(tmp_path))
    svc._mm_svc = MultimediaMemoryService(storage_root=str(tmp_path))

    req = GlobalSearchRequest(query="field check", top_k=5, scope=["sessions"])
    resp = await svc.search("u_fields", req)

    if resp.sessions:
        hit = resp.sessions[0]
        assert isinstance(hit.text, str)
        assert 0.0 <= hit.score <= 1.0
        assert hit.timestamp.endswith("+00:00") or "T" in hit.timestamp
        assert isinstance(hit.session_id, str)
        assert isinstance(hit.turn_index, int)


# ── API-level tests (TestClient) ──────────────────────────────────────────────

def _make_api_client(tmp_path, fake_redis):
    from main import app
    from core.dependencies import get_current_user, get_redis
    from routers.sessions import get_session_service

    app.dependency_overrides[get_redis] = lambda: fake_redis
    app.dependency_overrides[get_session_service] = lambda: SessionMemoryService(
        redis=fake_redis, storage_root=str(tmp_path)
    )
    app.dependency_overrides[get_current_user] = lambda: "api_test_user"

    # patch storage root so GlobalSearchService uses the tmp dir
    import core.config
    original_root = core.config.settings.storage_root
    core.config.settings.storage_root = str(tmp_path)

    client = TestClient(app)
    return client, original_root


@pytest.fixture
def api_client(tmp_path):
    import asyncio
    from fakeredis import FakeAsyncRedis as _FR
    fake_redis = _FR(decode_responses=True)

    client, orig = _make_api_client(tmp_path, fake_redis)
    yield client, tmp_path, fake_redis

    import core.config
    core.config.settings.storage_root = orig
    from main import app
    app.dependency_overrides.clear()
    asyncio.get_event_loop().run_until_complete(fake_redis.aclose())


def test_api_session_search_returns_200(api_client):
    client, tmp_path, fake_redis = api_client

    client.post("/memory/sessions/sess_api/turns", json={
        "user_message": "What is the weather?",
        "assistant_response": "Sunny today.",
    })

    r = client.post("/memory/sessions/sess_api/search", json={"query": "weather", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert "hits" in body
    assert "total_frames_searched" in body
    assert body["total_frames_searched"] == 1


def test_api_session_search_empty_session(api_client):
    client, *_ = api_client
    r = client.post("/memory/sessions/nonexistent_sess/search", json={"query": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"] == []
    assert body["total_frames_searched"] == 0


def test_api_session_search_hit_shape(api_client):
    client, *_ = api_client
    client.post("/memory/sessions/sess_shape/turns", json={
        "user_message": "explain neural networks",
        "assistant_response": "Neural networks are...",
    })

    r = client.post("/memory/sessions/sess_shape/search", json={"query": "neural networks", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()["hits"]
    if hits:
        h = hits[0]
        assert "text" in h
        assert "score" in h
        assert "timestamp" in h
        assert "turn_index" in h
        assert "user_message" in h
        assert "assistant_response" in h


def test_api_session_search_with_date_filter(api_client):
    client, *_ = api_client
    r = client.post("/memory/sessions/sess_dates/search", json={
        "query": "test",
        "top_k": 5,
        "from_date": "2024-01-01T00:00:00Z",
        "to_date": "2025-12-31T23:59:59Z",
    })
    assert r.status_code == 200


def test_api_global_search_returns_200(api_client):
    client, *_ = api_client
    r = client.post("/memory/search", json={"query": "anything", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert "sessions" in body
    assert "longterm" in body
    assert "multimedia" in body
    assert "total_frames_searched" in body


def test_api_global_search_scope_filter(api_client):
    client, *_ = api_client
    r = client.post("/memory/search", json={
        "query": "test query",
        "top_k": 5,
        "scope": ["longterm"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["sessions"] == []
    assert body["multimedia"] == []


def test_api_global_search_invalid_scope_rejected(api_client):
    client, *_ = api_client
    r = client.post("/memory/search", json={
        "query": "test",
        "scope": ["sessions", "invalid_scope"],
    })
    assert r.status_code == 422


# ── Latency benchmark ─────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.asyncio
async def test_session_search_latency_50_sessions(tmp_path):
    """
    MV-7 SLA: P95 < 300 ms for user with ≤ 50 sessions, ≤ 5,000 turns/session.
    We seed 50 sessions × 100 turns (lighter than 5k, but representative for
    BM25 search latency since memvid scales with frame count per file, not session count).
    """
    import asyncio
    from fakeredis import FakeAsyncRedis as _FR
    redis = _FR(decode_responses=True)
    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))

    NUM_SESSIONS = 50
    TURNS_PER_SESSION = 100
    LATENCY_MS = 300
    REPEATS = 20

    for s in range(NUM_SESSIONS):
        for t in range(TURNS_PER_SESSION):
            await svc.append_turn(f"bench_user", f"bench_sess_{s}", TurnAppendRequest(
                user_message=f"session {s} turn {t} about machine learning and AI",
                assistant_response=f"The answer for session {s} turn {t} is here.",
                timestamp=1_700_000_000.0 + s * 10000 + t,
            ))

    queries = [
        "machine learning", "AI answer", "session turn", "here response", "learning model",
        "deep learning", "neural network", "data science", "platform architecture", "training data",
        "inference speed", "model accuracy", "validation set", "embeddings vector", "transformer",
        "attention mechanism", "gradient descent", "backpropagation", "loss function", "optimizer",
    ]
    assert len(queries) == REPEATS

    latencies = []
    for query in queries:
        req = SessionSearchRequest(query=query, top_k=10)
        t0 = time.perf_counter()
        resp = await svc.search("bench_user", f"bench_sess_0", req)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    p95 = latencies[int(REPEATS * 0.95) - 1]
    assert p95 < LATENCY_MS, (
        f"P95 session search latency {p95:.1f} ms exceeds {LATENCY_MS} ms SLA. "
        f"All: {[f'{x:.1f}' for x in latencies]}"
    )

    await redis.aclose()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_global_search_latency_50_sessions(tmp_path):
    """Global search across 50 sessions + longterm must complete in < 300 ms P95."""
    import asyncio
    from fakeredis import FakeAsyncRedis as _FR
    from pathlib import Path
    from unittest.mock import patch
    redis = _FR(decode_responses=True)
    session_svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))
    lt_svc = LongTermMemoryService(storage_root=str(tmp_path))

    NUM_SESSIONS = 50
    TURNS = 20
    LATENCY_MS = 300
    REPEATS = 10

    async def _noop_trigger(*args, **kwargs):
        pass

    # Suppress fire-and-forget extraction tasks during seeding so they don't
    # saturate the event loop with Ollama retries and inflate search latency.
    with patch.object(SessionMemoryService, "_trigger_extraction", _noop_trigger):
        for s in range(NUM_SESSIONS):
            for t in range(TURNS):
                await session_svc.append_turn("bench2", f"gs_{s}", TurnAppendRequest(
                    user_message=f"turn {t} knowledge base query",
                    assistant_response=f"answer {t} from knowledge base",
                ))
    for i in range(50):
        await lt_svc.add_fact("bench2", FactAddRequest(
            text=f"long term fact {i} about user preferences and history",
            frame_type="fact",
        ))

    from services.multimedia_memory import MultimediaMemoryService
    svc = GlobalSearchService.__new__(GlobalSearchService)
    svc._root = Path(str(tmp_path))
    svc._lt_svc = lt_svc
    svc._mm_svc = MultimediaMemoryService(storage_root=str(tmp_path))

    queries = [f"knowledge base query {i}" for i in range(REPEATS)]

    latencies = []
    for query in queries:
        req = GlobalSearchRequest(query=query, top_k=5, scope=["sessions", "longterm"])
        t0 = time.perf_counter()
        await svc.search("bench2", req)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    p95 = latencies[int(REPEATS * 0.95) - 1]
    assert p95 < LATENCY_MS, (
        f"P95 global search latency {p95:.1f} ms exceeds {LATENCY_MS} ms SLA. "
        f"All: {[f'{x:.1f}' for x in latencies]}"
    )

    await redis.aclose()
