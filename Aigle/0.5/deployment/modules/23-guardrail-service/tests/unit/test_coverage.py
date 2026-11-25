"""GET /guardrail/coverage — minimal GB-22 coverage matrix (GRD-X1)."""
from __future__ import annotations

from app.routers import coverage
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _entries() -> dict[str, dict]:
    app = FastAPI()
    app.include_router(coverage.router, prefix="/guardrail")
    resp = TestClient(app).get("/guardrail/coverage")
    assert resp.status_code == 200
    return {e["endpoint"]: e for e in resp.json()}


def test_decide_and_v0_2_are_policy_aware():
    entries = _entries()
    for path in ("/guardrail/policy/decide", "/api/v0.2/check/input",
                 "/api/v0.2/check/output", "/api/v0.2/check/conversation"):
        assert entries[path]["policy_aware"] is True
        assert entries[path]["deprecated"] is False


def test_guard_check_is_deprecated_with_decide_as_successor():
    entries = _entries()
    successors = {"input": "/guardrail/policy/decide", "output": "/guardrail/policy/decide",
                  "conversation": "/api/v0.2/check/conversation"}
    for kind, successor in successors.items():
        entry = entries[f"/guard/check/{kind}"]
        assert entry["deprecated"] is True
        assert entry["policy_aware"] is False
        assert entry["successor"] == successor
    assert entries["/guard/check/raw"]["deprecated"] is False


def test_every_listed_endpoint_exists_in_the_app():
    from app import main

    paths = main.app.openapi()["paths"]
    for entry in coverage.ENFORCEMENT_POINTS:
        assert entry.method.lower() in paths[entry.endpoint], entry.endpoint


def test_coverage_route_is_mounted():
    from app import main

    assert "get" in main.app.openapi()["paths"]["/guardrail/coverage"]
