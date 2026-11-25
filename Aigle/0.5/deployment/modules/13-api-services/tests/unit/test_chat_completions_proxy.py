"""P10: /chat/completions forwards `response_format` to Module 07 and keeps its structured errors."""
import os
import sys

os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

PATH = f"/api/{get_settings().api_version}/chat/completions"
FORMAT = {"type": "json_schema", "json_schema": {"name": "e", "schema": {"type": "object"}}}
client = TestClient(app)


@pytest.fixture(autouse=True)
def _env():
    settings = get_settings()
    previous = (app.dependency_overrides.get(get_current_user), settings.gr_enabled)
    app.dependency_overrides[get_current_user] = lambda: {"sub": "u1"}
    settings.gr_enabled = False
    yield
    settings.gr_enabled = previous[1]
    if previous[0] is None:
        app.dependency_overrides.pop(get_current_user, None)
    else:
        app.dependency_overrides[get_current_user] = previous[0]


def _module07(monkeypatch, status, body):
    sent = []

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, **kw):
            sent.append(json)
            kwargs = {"json": body} if isinstance(body, dict) else {"text": body}
            return httpx.Response(status, request=httpx.Request("POST", url), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return sent


def test_response_format_is_forwarded(monkeypatch):
    sent = _module07(monkeypatch, 200, {"choices": [{"message": {"content": "{}"}}]})
    r = client.post(PATH, json={"model": "m", "messages": [{"role": "user", "content": "x"}],
                                "response_format": FORMAT})
    assert r.status_code == 200 and sent[0]["response_format"] == FORMAT


def test_structured_output_error_keeps_its_code(monkeypatch):
    error = {"error": {"message": "bad", "type": "server_error", "code": "structured_output_error",
                       "violations": ["$: not JSON"]}}
    _module07(monkeypatch, 502, {"detail": error})
    r = client.post(PATH, json={"model": "m", "messages": [], "response_format": FORMAT})
    assert r.status_code == 502 and r.json()["detail"] == error


def test_non_json_upstream_error_stays_text(monkeypatch):
    _module07(monkeypatch, 503, "upstream down")
    r = client.post(PATH, json={"model": "m", "messages": []})
    assert r.status_code == 503 and r.json()["detail"] == "upstream down"
