"""
API-level integration tests using FastAPI TestClient + dependency override.
No real Keycloak, Redis, or NFS needed.

Run:
    pip install fakeredis pytest pytest-asyncio httpx
        python -m pytest test_api.py -v
"""
import os
import shutil
import sys

os.environ.setdefault("MEM_REDIS_HOST", "localhost")
os.environ.setdefault("MEM_STORAGE_ROOT", "/tmp/mv_api_test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

# Import after env vars are set
from main import app
from core.dependencies import get_current_user, get_redis
from services.session_memory import SessionMemoryService
from services.compact_memory import CompactMemoryService
from routers.sessions import get_session_service
from routers.compact import get_compact_service

FAKE_USER = "test_user_42"

# ── Dependency overrides ──────────────────────────────────────────────────────

_fake_redis = FakeAsyncRedis(decode_responses=True)


def _override_redis():
    return _fake_redis


def _override_svc():
    return SessionMemoryService(redis=_fake_redis, storage_root="/tmp/mv_api_test")


def _override_compact_svc():
    return CompactMemoryService(storage_root="/tmp/mv_api_test")


async def _override_user():
    return FAKE_USER


app.dependency_overrides[get_redis] = _override_redis
app.dependency_overrides[get_session_service] = _override_svc
app.dependency_overrides[get_compact_service] = _override_compact_svc
app.dependency_overrides[get_current_user] = _override_user

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def clean_storage():
    shutil.rmtree("/tmp/mv_api_test", ignore_errors=True)
    yield
    shutil.rmtree("/tmp/mv_api_test", ignore_errors=True)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["memvid_version"], str) and body["memvid_version"]


# ── POST /memory/sessions/{session_id}/turns ──────────────────────────────────

def test_append_turn_returns_201():
    r = client.post(
        "/memory/sessions/sess_001/turns",
        json={
            "user_message": "What is Raptor?",
            "assistant_response": "Raptor is a platform...",
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["session_id"] == "sess_001"
    assert body["turn_index"] == 1
    assert "frame_id" in body


def test_append_turn_increments_turn_index():
    for i in range(1, 4):
        r = client.post(
            "/memory/sessions/sess_002/turns",
            json={"user_message": f"Q{i}", "assistant_response": f"A{i}"},
        )
        assert r.status_code == 201
        assert r.json()["turn_index"] == i


# ── GET /memory/sessions/{session_id}/recent ─────────────────────────────────

def test_get_recent_returns_turns():
    for i in range(5):
        client.post(
            "/memory/sessions/sess_003/turns",
            json={"user_message": f"msg{i}", "assistant_response": f"ans{i}"},
        )
    r = client.get("/memory/sessions/sess_003/recent?n=3")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_get_recent_missing_session_returns_empty():
    r = client.get("/memory/sessions/nonexistent_xyz/recent?n=5")
    assert r.status_code == 200
    assert r.json() == []


# ── GET /memory/sessions ──────────────────────────────────────────────────────

def test_list_sessions():
    r = client.get("/memory/sessions")
    assert r.status_code == 200
    sessions = r.json()
    assert isinstance(sessions, list)
    ids = {s["session_id"] for s in sessions}
    assert "sess_001" in ids


# ── DELETE /memory/sessions/{session_id} ──────────────────────────────────────

def test_delete_session():
    client.post(
        "/memory/sessions/sess_to_delete/turns",
        json={"user_message": "bye", "assistant_response": "goodbye"},
    )
    r = client.delete("/memory/sessions/sess_to_delete")
    assert r.status_code == 204

    ids = {s["session_id"] for s in client.get("/memory/sessions").json()}
    assert "sess_to_delete" not in ids
    assert client.get("/memory/sessions/sess_to_delete/recent?n=5").json() == []


def test_delete_nonexistent_session_returns_404():
    r = client.delete("/memory/sessions/does_not_exist_xyz")
    assert r.status_code == 404


# ── POST /memory/compact/evaluate ────────────────────────────────────────────

def test_compact_evaluate_under_threshold():
    r = client.post(
        "/memory/compact/evaluate",
        json={"messages": [{"role": "user", "content": "hi"}], "context_window": 128000},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["should_compact"] is False
    assert body["auto_compact_threshold"] == 95000


def test_compact_evaluate_over_threshold():
    r = client.post(
        "/memory/compact/evaluate",
        json={
            "messages": [{"role": "user", "content": "x" * (95000 * 4 + 100)}],
            "context_window": 128000,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["should_compact"] is True
    assert body["tokens_over_threshold"] > 0


def test_compact_evaluate_extra_tokens():
    r = client.post(
        "/memory/compact/evaluate",
        json={"messages": [], "context_window": 128000, "extra_tokens": 96000},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_count"] == 96000
    assert body["should_compact"] is True


def test_compact_evaluate_with_session_id_ignores_messages_uses_aggregate():
    # session_id present → messages list is ignored, aggregation kicks in
    # (session has no archived turns / facts / media → token_count 0)
    r = client.post(
        "/memory/compact/evaluate",
        json={
            "messages": [{"role": "user", "content": "x" * (95000 * 4)}],
            "session_id": "does_not_exist_eval_branch",
            "context_window": 128000,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_count"] == 0
    assert body["should_compact"] is False


def test_compact_evaluate_with_session_id_matches_session_scoped_endpoint():
    session_id = "compact_evaluate_branch_sess"
    client.post(
        f"/memory/sessions/{session_id}/turns",
        json={"user_message": "q", "assistant_response": "a" * (96000 * 4)},
    )
    r1 = client.post(
        "/memory/compact/evaluate",
        json={"session_id": session_id, "context_window": 128000},
    )
    r2 = client.post(
        f"/memory/sessions/{session_id}/compact/evaluate",
        json={"context_window": 128000},
    )
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    client.delete("/memory")


# ── POST /memory/sessions/{session_id}/compact (dry_run) ─────────────────────

def test_compact_session_no_session_returns_false():
    r = client.post(
        "/memory/sessions/nonexistent_sess/compact",
        json={"context_window": 128000, "dry_run": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["compacted"] is False
    assert body["source"] == "no_session"


def test_compact_session_under_budget_returns_false():
    # Create a session with 1 small turn (well under 95K threshold)
    client.post(
        "/memory/sessions/sess_compact_small/turns",
        json={"user_message": "tiny question", "assistant_response": "tiny answer"},
    )
    r = client.post(
        "/memory/sessions/sess_compact_small/compact",
        json={"context_window": 128000, "dry_run": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["compacted"] is False
    assert body["source"] == "under_budget"


# ── GET /memory/sessions/{session_id}/summaries ───────────────────────────────

def test_get_summaries_empty_when_none():
    client.post(
        "/memory/sessions/sess_nosummary/turns",
        json={"user_message": "q", "assistant_response": "a"},
    )
    r = client.get("/memory/sessions/sess_nosummary/summaries")
    assert r.status_code == 200
    assert r.json() == []


# ── User isolation ──────────────────────────────────────────────────────────
# get_current_user resolves user_id from the X-User-ID header (see
# core/dependencies.py). Swap the override mid-test to simulate two different
# callers hitting the same session_id and confirm storage (user_{user_id}/...)
# keeps them apart.

def _as_user(user_id: str):
    async def _override():
        return user_id
    return _override


def test_other_user_cannot_compact_or_list_summaries_for_session():
    app.dependency_overrides[get_current_user] = _as_user("owner_1")
    try:
        r = client.post(
            "/memory/sessions/sess_shared_id/turns",
            json={"user_message": "owner secret", "assistant_response": "owner secret answer"},
        )
        assert r.status_code == 201
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    app.dependency_overrides[get_current_user] = _as_user("intruder_1")
    try:
        # Same session_id, different user => resolves to a different storage
        # path, so from the intruder's perspective the session never existed.
        r = client.post(
            "/memory/sessions/sess_shared_id/compact",
            json={"context_window": 128000, "dry_run": True},
        )
        assert r.status_code == 200
        assert r.json()["source"] == "no_session"

        r = client.get("/memory/sessions/sess_shared_id/summaries")
        assert r.status_code == 200
        assert r.json() == []

        r = client.get("/memory/sessions/sess_shared_id/recent?n=10")
        assert r.status_code == 200
        assert r.json() == []
    finally:
        app.dependency_overrides[get_current_user] = _override_user
