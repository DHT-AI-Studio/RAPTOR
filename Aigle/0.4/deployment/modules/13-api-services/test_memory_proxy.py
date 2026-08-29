"""
Unit tests for the Module 26 memory compaction proxy (memory.py).

Run from the 13-api-services/ directory:
    conda run -n CIE python -m pytest test_memory_proxy.py -v

Uses FastAPI's dependency_overrides to bypass real JWT/Keycloak/UMA (get_current_user
returns a fixed fake sub), and monkeypatches httpx.AsyncClient inside the proxy
module so no real Module 26 is needed — verifies the proxy's own logic: requests
scope to the JWT sub, a mismatched caller-supplied X-User-ID is rejected with
403, and paths/methods forward correctly.
"""
import json
import os
import sys
from unittest import mock

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")

sys.path.insert(0, os.path.dirname(__file__))

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_current_user
import app.routers.memory as memory_module

FAKE_SUB = "jwt-verified-user-42"


async def _override_current_user():
    return {"sub": FAKE_SUB}


app.dependency_overrides[get_current_user] = _override_current_user

client = TestClient(app)


class _CapturingAsyncClient:
    """Drop-in replacement for httpx.AsyncClient that records the outbound
    request and returns a canned response instead of hitting Module 26."""

    captured: dict = {}
    response_json: dict = {"ok": True}
    response_status: int = 200

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, headers=None, params=None, content=None):
        _CapturingAsyncClient.captured = {
            "method": method, "url": url, "headers": dict(headers or {}),
            "params": dict(params or {}),
            "content": json.loads(content) if content else None,
        }
        return httpx.Response(
            _CapturingAsyncClient.response_status,
            json=_CapturingAsyncClient.response_json,
            request=httpx.Request(method, url),
        )


@pytest.fixture(autouse=True)
def _patch_httpx(monkeypatch):
    monkeypatch.setattr(memory_module.httpx, "AsyncClient", _CapturingAsyncClient)
    yield


def test_evaluate_compact_forwards_to_module26_path():
    r = client.post(
        "/api/0.3/memory/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/compact/evaluate"
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_x_user_id_mismatch_rejected_with_403():
    # Caller tries to spoof a different user via X-User-ID — must be denied,
    # not silently corrected to the caller's own identity.
    r = client.post(
        "/api/0.3/memory/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 403


def test_x_user_id_matching_jwt_sub_is_allowed():
    # Caller echoing their own sub back is not spoofing — must pass through.
    r = client.post(
        "/api/0.3/memory/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake", "X-User-ID": FAKE_SUB},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB


def test_no_x_user_id_header_uses_jwt_sub():
    # The common case — caller doesn't send X-User-ID at all.
    r = client.post(
        "/api/0.3/memory/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["headers"]["X-User-ID"] == FAKE_SUB


def test_compact_session_forwards_session_id_in_path():
    r = client.post(
        "/api/0.3/memory/sessions/my_sess/compact",
        json={"trigger": "manual", "context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/compact"
    )


def test_evaluate_session_compact_forwards_session_id():
    r = client.post(
        "/api/0.3/memory/sessions/my_sess/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/compact/evaluate"
    )


def test_get_session_summaries_is_get():
    r = client.get(
        "/api/0.3/memory/sessions/my_sess/summaries",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["method"] == "GET"
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/summaries"
    )


def test_delete_session_summary_forwards_both_ids():
    r = client.delete(
        "/api/0.3/memory/sessions/my_sess/summaries/sum-123",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["method"] == "DELETE"
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/summaries/sum-123"
    )


def test_upstream_502_on_connection_error(monkeypatch):
    class _FailingClient(_CapturingAsyncClient):
        async def request(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(memory_module.httpx, "AsyncClient", _FailingClient)
    r = client.post(
        "/api/0.3/memory/compact/evaluate",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 502


# ── Sessions ────────────────────────────────────────────────────────────────

def test_append_turn_forwards_session_id_and_body():
    r = client.post(
        "/api/0.3/memory/sessions/my_sess/turns",
        json={"user_message": "hi", "assistant_response": "hello"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["method"] == "POST"
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/turns"
    )
    assert _CapturingAsyncClient.captured["content"] == {
        "user_message": "hi", "assistant_response": "hello",
    }


def test_search_session_forwards_path():
    r = client.post(
        "/api/0.3/memory/sessions/my_sess/search",
        json={"query": "q"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/search"
    )


def test_get_timeline_forwards_query_params():
    r = client.get(
        "/api/0.3/memory/sessions/my_sess/timeline?page=2&page_size=10",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["method"] == "GET"
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/timeline"
    )
    assert _CapturingAsyncClient.captured["params"] == {"page": "2", "page_size": "10"}


def test_get_recent_turns_forwards_path():
    r = client.get(
        "/api/0.3/memory/sessions/my_sess/recent?n=5",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess/recent"
    )


def test_list_sessions_forwards_to_bare_sessions_path():
    r = client.get(
        "/api/0.3/memory/sessions",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/sessions"


def test_delete_session_forwards_session_id():
    r = client.delete(
        "/api/0.3/memory/sessions/my_sess",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["method"] == "DELETE"
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my_sess"
    )


# ── Long-term Memory ──────────────────────────────────────────────────────────

def test_search_longterm_forwards_path():
    r = client.post(
        "/api/0.3/memory/longterm/search",
        json={"query": "q"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/longterm/search"


def test_add_fact_forwards_path():
    r = client.post(
        "/api/0.3/memory/longterm/facts",
        json={"text": "偏好繁體中文", "frame_type": "preference"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/longterm/facts"


def test_get_facts_forwards_path():
    r = client.get(
        "/api/0.3/memory/longterm/facts",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["method"] == "GET"
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/longterm/facts"


def test_delete_fact_forwards_frame_id():
    r = client.delete(
        "/api/0.3/memory/longterm/facts/frame-1",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/longterm/facts/frame-1"
    )


def test_delete_longterm_forwards_bare_path():
    r = client.delete(
        "/api/0.3/memory/longterm",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/longterm"


def test_extract_longterm_is_not_proxied():
    r = client.post(
        "/api/0.3/memory/longterm/extract",
        json={"session_id": "s1", "turn": {}},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 404


# ── Multimedia Memory ─────────────────────────────────────────────────────────

def test_index_video_forwards_path():
    r = client.post(
        "/api/0.3/memory/multimedia/video",
        json={
            "asset_path": "a", "version_id": "v1",
            "start_sec": 0.0, "end_sec": 5.0, "transcription": "t",
        },
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/multimedia/video"


def test_index_audio_forwards_path():
    r = client.post(
        "/api/0.3/memory/multimedia/audio",
        json={"asset_path": "a", "version_id": "v1", "start_sec": 0.0, "end_sec": 5.0},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/multimedia/audio"


def test_index_image_forwards_path():
    r = client.post(
        "/api/0.3/memory/multimedia/image",
        json={"asset_path": "a", "version_id": "v1"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/multimedia/image"


def test_search_multimedia_forwards_path():
    r = client.post(
        "/api/0.3/memory/multimedia/search",
        json={"query": "q"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/multimedia/search"


# ── Global Search ──────────────────────────────────────────────────────────────

def test_global_search_forwards_path():
    r = client.post(
        "/api/0.3/memory/search",
        json={"query": "q"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/search"


# ── Management ─────────────────────────────────────────────────────────────────

def test_get_stats_forwards_path():
    r = client.get(
        "/api/0.3/memory/stats",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/stats"


def test_delete_all_memory_forwards_bare_path():
    r = client.delete(
        "/api/0.3/memory",
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory"


def test_delete_all_memory_x_user_id_mismatch_rejected():
    r = client.delete(
        "/api/0.3/memory",
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 403


# ── Export (streaming) ──────────────────────────────────────────────────────────

class _CapturingStreamClient:
    """Drop-in replacement for httpx.AsyncClient exercising the .stream() path
    used by /memory/export instead of the buffered .request() path."""

    captured: dict = {}
    chunks: list = [b'{"sessions": [], ', b'"longterm": []}']
    response_status: int = 200
    response_headers: dict = {"content-type": "application/json"}

    def __init__(self, *args, **kwargs):
        pass

    async def aclose(self):
        return None

    def stream(self, method, url, headers=None, params=None):
        _CapturingStreamClient.captured = {
            "method": method, "url": url, "headers": dict(headers or {}),
            "params": dict(params or {}),
        }
        return _StreamCtx()


class _StreamResponse:
    def __init__(self):
        self.status_code = _CapturingStreamClient.response_status
        self.headers = dict(_CapturingStreamClient.response_headers)

    async def aiter_bytes(self):
        for chunk in _CapturingStreamClient.chunks:
            yield chunk


class _StreamCtx:
    async def __aenter__(self):
        return _StreamResponse()

    async def __aexit__(self, *exc):
        return False


def test_export_memory_streams_response_body():
    import app.routers.memory as memory_module_local

    with mock.patch.object(
        memory_module_local.httpx, "AsyncClient", _CapturingStreamClient
    ):
        r = client.get(
            "/api/0.3/memory/export",
            headers={"Authorization": "Bearer fake"},
        )
    assert r.status_code == 200
    assert r.content == b'{"sessions": [], "longterm": []}'
    assert _CapturingStreamClient.captured["method"] == "GET"
    assert _CapturingStreamClient.captured["url"] == "http://raptor-memory:8026/memory/export"


def test_export_memory_x_user_id_mismatch_rejected():
    import app.routers.memory as memory_module_local

    with mock.patch.object(
        memory_module_local.httpx, "AsyncClient", _CapturingStreamClient
    ):
        r = client.get(
            "/api/0.3/memory/export",
            headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
        )
    assert r.status_code == 403


# ── Flat top-level aliases ───────────────────────────────────────────────────

def test_store_memory_forwards_to_longterm_facts():
    r = client.post(
        "/api/0.3/memory/store",
        json={"text": "偏好繁體中文", "frame_type": "preference"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/longterm/facts"
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_retrieve_memory_forwards_to_search_as_post_json():
    r = client.get(
        "/api/0.3/memory/retrieve",
        params={"query": "上季營收", "top_k": 3},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/search"
    assert _CapturingAsyncClient.captured["method"] == "POST"
    assert _CapturingAsyncClient.captured["content"] == {"query": "上季營收", "top_k": 3}


def test_retrieve_memory_x_user_id_mismatch_rejected():
    r = client.get(
        "/api/0.3/memory/retrieve",
        params={"query": "q"},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 403


def test_timeline_memory_forwards_to_user_timeline():
    r = client.get(
        "/api/0.3/memory/timeline",
        params={"page": 2, "page_size": 10},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == "http://raptor-memory:8026/memory/timeline"
    assert _CapturingAsyncClient.captured["method"] == "GET"
    assert dict(_CapturingAsyncClient.captured["params"])["page"] == "2"


def test_compact_memory_forwards_to_default_session():
    r = client.post(
        "/api/0.3/memory/compact",
        json={"trigger": "manual", "context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/default/compact"
    )
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_compact_memory_forwards_explicit_session_id():
    r = client.post(
        "/api/0.3/memory/compact",
        params={"session_id": "my-conversation-42"},
        json={"trigger": "manual", "context_window": 128000},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my-conversation-42/compact"
    )


def test_compact_memory_x_user_id_mismatch_rejected():
    r = client.post(
        "/api/0.3/memory/compact",
        json={"context_window": 128000},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 403


def test_archive_memory_forwards_to_default_session_turns():
    r = client.post(
        "/api/0.3/memory/archive",
        json={"user_message": "hi", "assistant_response": "hello"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/default/turns"
    )
    assert _CapturingAsyncClient.captured["method"] == "POST"


def test_archive_memory_forwards_explicit_session_id():
    r = client.post(
        "/api/0.3/memory/archive",
        params={"session_id": "my-conversation-42"},
        json={"user_message": "hi", "assistant_response": "hello"},
        headers={"Authorization": "Bearer fake"},
    )
    assert r.status_code == 200
    assert _CapturingAsyncClient.captured["url"] == (
        "http://raptor-memory:8026/memory/sessions/my-conversation-42/turns"
    )


def test_archive_memory_x_user_id_mismatch_rejected():
    r = client.post(
        "/api/0.3/memory/archive",
        json={"user_message": "hi", "assistant_response": "hello"},
        headers={"Authorization": "Bearer fake", "X-User-ID": "attacker-controlled-id"},
    )
    assert r.status_code == 403
