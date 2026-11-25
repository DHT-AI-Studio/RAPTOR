"""/api/v0.2/check/* over HTTP after the move into policy_check_service
(GRD-X1): the response must be exactly what it was, and the audit row now
carries the caller's module/request_id."""
from __future__ import annotations

import pytest
from app.routers import check_v0_2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .conftest import safe_verdict, standard_policy, unsafe_verdict

_SAFE_BODY = {"safe": True, "categories": [], "category_names": {}, "raw": "safe", "conflict": None, "results": None}
_UNSAFE_BODY = {"safe": False, "categories": ["S1"], "category_names": {"S1": "name-S1"}, "raw": "unsafe",
                "conflict": None, "results": None}
_DISABLED_BODY = {"safe": True, "categories": [], "category_names": {}, "raw": "guardrail disabled",
                  "conflict": None, "results": None}


@pytest.fixture
def client(policy_env):
    app = FastAPI()
    app.include_router(check_v0_2.router, prefix="/api/v0.2")
    return TestClient(app)


def _conversation():
    return {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}


@pytest.mark.parametrize("path, body", [
    ("/api/v0.2/check/input", {"content": "hello"}),
    ("/api/v0.2/check/output", {"content": "hello"}),
    ("/api/v0.2/check/conversation", _conversation()),
])
def test_safe_response_is_unchanged(client, policy_env, path, body):
    policy_env.verdicts = [safe_verdict()]
    assert client.post(path, json=body).json() == _SAFE_BODY


@pytest.mark.parametrize("path, body", [
    ("/api/v0.2/check/input", {"content": "bad"}),
    ("/api/v0.2/check/output", {"content": "bad"}),
    ("/api/v0.2/check/conversation", _conversation()),
])
def test_unsafe_response_is_unchanged(client, policy_env, path, body):
    policy_env.policy = standard_policy()
    policy_env.verdicts = [unsafe_verdict("S1")]
    assert client.post(path, json=body).json() == _UNSAFE_BODY


@pytest.mark.parametrize("path, body", [
    ("/api/v0.2/check/input", {"content": "x"}),
    ("/api/v0.2/check/conversation", _conversation()),
])
def test_disabled_response_is_unchanged(client, policy_env, path, body):
    policy_env.enabled = False
    assert client.post(path, json=body).json() == _DISABLED_BODY


def test_debug_variant_includes_prompts(client, policy_env):
    policy_env.verdicts = [unsafe_verdict("S1")]
    body = client.post("/api/v0.2/debug/check/input", json={"content": "bad"}).json()
    assert body["safe"] is False
    assert body["results"][0]["prompt"] == "prompt"


def test_audit_now_carries_module_and_request_id(client, policy_env):
    policy_env.verdicts = [unsafe_verdict("S1")]
    client.post("/api/v0.2/check/input", json={"content": "bad", "module": "07", "request_id": "r-1"})
    assert len(policy_env.audits) == 1
    assert (policy_env.audits[0]["module"], policy_env.audits[0]["request_id"]) == ("07", "r-1")


def test_v0_2_is_not_marked_deprecated(client):
    resp = client.post("/api/v0.2/check/input", json={"content": "hello"})
    assert "X-Deprecated" not in resp.headers
    assert "Link" not in resp.headers
