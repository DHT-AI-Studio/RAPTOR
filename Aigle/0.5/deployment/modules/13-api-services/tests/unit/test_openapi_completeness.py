"""
Automated OpenAPI-schema completeness/regression test.

Guards against exactly the two failure modes this session found by hand:
  * a router mounted but every route inside it commented out (personal_db.py,
    pre-fix) — `include_router()` succeeds either way, so nothing trips on
    startup; only an assertion on the *paths* the schema actually contains
    catches it.
  * a proxy router that was simply never built at all (guardrail.py, pre-fix)
and one more a schema check catches for free: any path that leaked onto the
gateway under the wrong version prefix (`/api/v1/*`, an internal backend
convention — see benchmark_service_url — rather than the gateway's own
`/api/{api_version}/*`).

Run from the 13-api-services/ directory:
    python -m pytest test_openapi_completeness.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

_V = get_settings().api_version  # "0.5"

# One real, schema-visible path per router mounted in main.py — catches "the
# whole router is silently empty or missing" the way `include_router()`
# succeeding never would. media_sync.py is deliberately excluded: every one
# of its routes is declared `include_in_schema=False`, so it can never
# appear in the OpenAPI schema by design, not by bug.
REQUIRED_PATHS = {
    f"/api/{_V}/sso/login",
    f"/api/{_V}/search/hybrid",
    f"/api/{_V}/search/video_search",
    f"/api/{_V}/asset/fileupload_analysis",
    f"/api/{_V}/processing/cache/all",
    f"/api/{_V}/chat/completions",
    f"/api/{_V}/training/submit",
    f"/api/{_V}/benchmark/health",
    f"/api/{_V}/guardrail/policies",
    f"/api/{_V}/aiml/models/local",
    f"/api/{_V}/a2a/query",
    f"/api/{_V}/memory/store",
    f"/api/{_V}/search/people",
    f"/api/{_V}/identity/identities",
    f"/api/{_V}/identity/{{identity_id}}/mergeinto",
    f"/api/{_V}/mcp",
}


def _schema_paths() -> set:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    return set(resp.json()["paths"].keys())


def test_required_paths_present():
    """Every router mounted in main.py contributes at least one real,
    schema-visible route."""
    missing = REQUIRED_PATHS - _schema_paths()
    assert not missing, f"Missing from OpenAPI schema: {sorted(missing)}"


# test_personal_db_fully_active() removed: personal_db.py's entire router is
# now deliberately commented out (superseded by search.py/video_search.py
# calling module 25 directly under the old 0.3-style path names -- see that
# router's own "search" section comment), so there are no personal-db routes
# left to assert are active.


def test_guardrail_fully_active():
    """All 15 guardrail endpoints (13 unique paths — /policies and
    /policies/{policy_id} each carry two HTTP methods) present: module 23's
    "Guardrail Policies" and "Guardrail (system)" routes, nothing else."""
    expected = {
        f"/api/{_V}/guardrail/policies",
        f"/api/{_V}/guardrail/policies/active",
        f"/api/{_V}/guardrail/policies/template",
        f"/api/{_V}/guardrail/policies/upload",
        f"/api/{_V}/guardrail/policies/{{policy_id}}",
        f"/api/{_V}/guardrail/policies/{{policy_id}}/activate",
        f"/api/{_V}/guardrail/policies/{{policy_id}}/deactivate",
        f"/api/{_V}/guardrail/system/enable",
        f"/api/{_V}/guardrail/system/disable",
        f"/api/{_V}/guardrail/system/policy/enable",
        f"/api/{_V}/guardrail/system/policy/disable",
        f"/api/{_V}/guardrail/system/status",
        f"/api/{_V}/guardrail/{{policy_id}}",
    }
    missing = expected - _schema_paths()
    assert not missing, f"guardrail routes missing: {sorted(missing)}"


def test_no_public_v1_paths():
    """`/api/v1/*` is an internal backend convention (module 22's benchmark
    service — see benchmark_service_url in app/core/config.py), never a path
    the gateway itself should expose. Every gateway route is versioned as
    `/api/{api_version}/*` (legacy `/api/0.3/*` callers are rewritten to that
    by LegacyApiAliasMiddleware before routing ever sees them) — a bare
    `/api/v1/*` path on the schema would mean a router got mounted without
    the `f"/api/{_v}/..."` prefix convention every other router uses, or an
    internal backend path leaked through a proxy unstripped.
    """
    leaked = {p for p in _schema_paths() if p.startswith("/api/v1")}
    assert not leaked, f"Public /api/v1/* paths must not be exposed by the gateway: {sorted(leaked)}"
