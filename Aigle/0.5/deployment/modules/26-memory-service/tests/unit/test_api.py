"""
API-level integration tests using FastAPI TestClient + dependency override.
No real Keycloak, Redis, or NFS needed.

Run:
    pip install fakeredis pytest pytest-asyncio httpx
        python -m pytest test_api.py -v
"""
import asyncio
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MEM_REDIS_HOST", "localhost")
os.environ.setdefault("MEM_STORAGE_ROOT", "/tmp/mv_api_test")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

import pytest
from core.config import settings
from core.dependencies import get_current_user, get_redis
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

# Import after env vars are set
from main import app
from routers.compact import get_compact_service
from routers.management import ManagementService, get_management_service
from routers.sessions import get_session_service
from services.compact_memory import CompactMemoryService
from services.session_memory import SessionMemoryService

# core.config.settings is a module-level singleton read by ManagementService
# (routers/management.py) at construction time, not just at import time — so
# os.environ.setdefault above only takes effect if this module happens to be
# the first one in the whole pytest run to import core.config. Run this file
# alongside another test module that imports the app first (e.g.
# `pytest test_directory_permissions.py test_api.py`) and settings.storage_root
# is already frozen to the .env default (/mnt/memvid-storage) by the time we
# get here, silently pointing every /memory/stats, /memory/export and
# DELETE /memory call at a different directory than the one SessionMemoryService
# writes to below. Overwrite the singleton attribute directly so this module's
# storage root is authoritative regardless of import order.
settings.storage_root = "/tmp/mv_api_test"

FAKE_USER = "test_user_42"

# ── Dependency overrides ──────────────────────────────────────────────────────

_fake_redis = FakeAsyncRedis(decode_responses=True)


def _run(coro):
    # A throwaway loop instead of asyncio.get_event_loop(): once any
    # pytest-asyncio-managed test has run in this thread, get_event_loop()
    # raises "There is no current event loop" (set_event_loop(None) at its
    # teardown disables the auto-create fallback for the rest of the session).
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _override_redis():
    return _fake_redis


def _override_svc():
    return SessionMemoryService(redis=_fake_redis, storage_root="/tmp/mv_api_test")


def _override_compact_svc():
    return CompactMemoryService(storage_root="/tmp/mv_api_test")


def _override_management_svc():
    # ManagementService() takes no args -- it reads settings.storage_root
    # fresh at construction time, which by the time this file's own tests
    # actually run (not just at collection/import time) may have been
    # overwritten by another tests/unit/*.py file's own module-level
    # settings.storage_root assignment. Bypass __init__ and set _root
    # directly, same fix as the session/compact overrides above.
    svc = ManagementService.__new__(ManagementService)
    svc._root = Path("/tmp/mv_api_test")
    return svc


async def _override_user():
    return FAKE_USER


app.dependency_overrides[get_redis] = _override_redis
app.dependency_overrides[get_session_service] = _override_svc
app.dependency_overrides[get_compact_service] = _override_compact_svc
app.dependency_overrides[get_management_service] = _override_management_svc
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


# get_current_user resolves the caller's identity from X-User-ID, a header
# that Module 13 (API Gateway) injects only after it has itself verified the
# caller's JWT — this service never sees or parses a token. So "user B's JWT
# can't retrieve user A's memory" reduces, at this service's boundary, to:
# whatever identity the dependency override resolves to, every management/
# stats/export/delete endpoint must be scoped to *that* identity's directory
# and never leak or act on another identity's data. Swapping the override
# (as above) is the correct simulation of "two different verified callers"
# for a service that trusts its upstream gateway for authentication.

def test_other_user_cannot_see_stats_export_or_delete_owners_memory():
    app.dependency_overrides[get_current_user] = _as_user("owner_2")
    try:
        r = client.post(
            "/memory/sessions/sess_mgmt/turns",
            json={"user_message": "owner secret 2", "assistant_response": "answer"},
        )
        assert r.status_code == 201
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    app.dependency_overrides[get_current_user] = _as_user("intruder_2")
    try:
        r = client.get("/memory/stats")
        assert r.status_code == 200
        assert r.json()["session_count"] == 0

        r = client.get("/memory/export")
        assert r.status_code == 200
        body = r.text
        assert "owner secret 2" not in body
        assert '"sessions": [\n\n  ]' in body or '"sessions": [' in body

        # Deleting as the intruder must not find (let alone remove) owner_2's data.
        r = client.delete("/memory")
        assert r.status_code == 404
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    # Owner's data must still be intact after the intruder's no-op delete.
    app.dependency_overrides[get_current_user] = _as_user("owner_2")
    try:
        r = client.get("/memory/stats")
        assert r.status_code == 200
        assert r.json()["session_count"] == 1
    finally:
        app.dependency_overrides[get_current_user] = _override_user


def test_delete_all_for_one_user_leaves_other_users_files_and_caches_untouched():
    """A real (successful, non-404) DELETE /memory for one user must not
    touch another user's on-disk memvid files or Redis keys — the blast
    radius of GDPR erasure must be exactly one user, never a glob/scan
    pattern that accidentally sweeps in a neighbour."""
    app.dependency_overrides[get_current_user] = _as_user("owner_3")
    try:
        r = client.post(
            "/memory/sessions/sess_a/turns",
            json={"user_message": "owner_3 secret", "assistant_response": "answer"},
        )
        assert r.status_code == 201
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    app.dependency_overrides[get_current_user] = _as_user("owner_4")
    try:
        r = client.post(
            "/memory/sessions/sess_b/turns",
            json={"user_message": "owner_4 secret", "assistant_response": "answer"},
        )
        assert r.status_code == 201
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    # owner_4 also has a Module 15 short-term chat cache entry sitting in Redis.
    _run(_fake_redis.set("chat_memory:owner_4", "cached turns"))

    app.dependency_overrides[get_current_user] = _as_user("owner_3")
    try:
        r = client.delete("/memory")
        assert r.status_code == 204
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    app.dependency_overrides[get_current_user] = _as_user("owner_4")
    try:
        r = client.get("/memory/stats")
        assert r.status_code == 200
        assert r.json()["session_count"] == 1

        r = client.get("/memory/export")
        assert "owner_4 secret" in r.text
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    assert _run(_fake_redis.get("chat_memory:owner_4")) == "cached turns"


def test_delete_all_cleans_redis_only_user_with_no_memvid_directory():
    """Regression test: a user who only ever hit Module 15's short-term chat
    cache (chat_memory:{user_id}) and never archived a turn into Module 26
    has no on-disk memvid directory at all. DELETE /memory used to return
    False (-> 404) before even scanning Redis in that case, leaving the
    stale Redis-only record behind despite the GDPR erasure "succeeding"
    from the caller's point of view being impossible (404 looks like
    nothing to erase). It must now find and delete that Redis key."""
    app.dependency_overrides[get_current_user] = _as_user("redis_only_user")
    try:
        _run(_fake_redis.set("chat_memory:redis_only_user", "cached turns"))

        r = client.delete("/memory")
        assert r.status_code == 204

        assert _run(_fake_redis.get("chat_memory:redis_only_user")) is None
    finally:
        app.dependency_overrides[get_current_user] = _override_user


# ── session_id traversal safety ──────────────────────────────────────────────
# session_id is interpolated directly into a filename (session_shards.py:
# f"session_{session_id}_0.mv2"), not a directory component, so a value
# containing "/" or ".." would be a traversal vector if it ever reached the
# filesystem layer un-sanitized. FastAPI's default {session_id} path
# converter (no ":path") structurally cannot match a literal "/" within a
# single segment, so the routing layer itself is the first line of defense —
# this test proves that defense empirically rather than assuming it.

@pytest.mark.parametrize("raw_segment", [
    "..%2F..%2Fintruder_2%2Fsession_x",  # encoded slashes
    "%2e%2e",
])
def test_session_id_cannot_traverse_via_encoded_slashes(raw_segment):
    app.dependency_overrides[get_current_user] = _as_user("owner_2")
    try:
        r = client.post(
            f"/memory/sessions/{raw_segment}/turns",
            json={"user_message": "x", "assistant_response": "y"},
        )
        # Either rejected by routing/validation, or accepted as a literal
        # (harmless) filename fragment — never allowed to write outside
        # owner_2's own user_owner_2/ directory.
        assert r.status_code in (201, 404, 422)
    finally:
        app.dependency_overrides[get_current_user] = _override_user

    escaped_path = os.path.join("/tmp/mv_api_test", "session_x_0.mv2")
    assert not os.path.exists(escaped_path)
