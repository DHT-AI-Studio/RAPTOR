"""/guard/check/{input,output,conversation} are deprecated (GRD-X1): same body
as before, plus X-Deprecated and a successor Link header — input/output point
to POST /guardrail/policy/decide, conversation to /api/v0.2/check/conversation."""
from __future__ import annotations

import pytest
from app.routers import check_guard
from app.services.guard_classifier import CombinedVerdict
from fastapi import FastAPI
from fastapi.testclient import TestClient

_SAFE = CombinedVerdict(safe=True, categories=[], category_names={}, raw="safe")
_BODY = {"safe": True, "categories": [], "category_names": {}, "raw": "safe", "conflict": None, "results": None}


@pytest.fixture
def client(monkeypatch):
    async def _on():
        return True

    async def _classify(content, role="user", *, policy=None):
        return _SAFE

    async def _classify_conversation(messages, *, policy=None):
        return _SAFE

    monkeypatch.setattr(check_guard, "is_enabled", _on)
    monkeypatch.setattr(check_guard.guard_classifier, "classify", _classify)
    monkeypatch.setattr(check_guard.guard_classifier, "classify_conversation", _classify_conversation)
    app = FastAPI()
    app.include_router(check_guard.router, prefix="/guard")
    return TestClient(app)


@pytest.mark.parametrize("path, body, successor", [
    ("/guard/check/input", {"content": "hi"}, "/guardrail/policy/decide"),
    ("/guard/check/output", {"content": "hi"}, "/guardrail/policy/decide"),
    ("/guard/check/conversation", {"messages": [{"role": "user", "content": "hi"}]}, "/api/v0.2/check/conversation"),
])
def test_deprecated_endpoints_carry_the_headers_and_an_unchanged_body(client, path, body, successor):
    resp = client.post(path, json=body)
    assert resp.status_code == 200
    assert resp.headers["X-Deprecated"] == "true"
    assert resp.headers["Link"] == f'<{successor}>; rel="successor-version"'
    assert resp.json() == _BODY


def test_headers_are_set_even_when_guardrail_is_disabled(client, monkeypatch):
    async def _off():
        return False

    monkeypatch.setattr(check_guard, "is_enabled", _off)
    resp = client.post("/guard/check/input", json={"content": "hi"})
    assert resp.headers["X-Deprecated"] == "true"


def test_openapi_marks_them_deprecated_but_not_raw():
    from app import main

    paths = main.app.openapi()["paths"]
    for path in ("/guard/check/input", "/guard/check/output", "/guard/check/conversation"):
        assert paths[path]["post"].get("deprecated") is True
    assert not paths["/guard/check/raw"]["post"].get("deprecated")
    assert not paths["/api/v0.2/check/input"]["post"].get("deprecated")
