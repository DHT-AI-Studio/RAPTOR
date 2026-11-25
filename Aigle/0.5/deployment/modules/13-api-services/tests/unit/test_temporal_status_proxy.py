"""
Unit tests for `/api/{v}/temporal/status/{version_id}` (temporal.py, P12): the gateway asks Module 25
for the caller's own source, passes the job state through, and keeps Module 25's 404.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_temporal_status_proxy.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app
from app.routers.search import get_personal_db_service
from app.services.personal_db_service import PersonalDBService
from fastapi.testclient import TestClient

FAKE_SUB = "jwt-verified-user-42"
STATUS = f"/api/{get_settings().api_version}/temporal/status"
JOB = {"job_id": "abc", "source_version_id": "881212da-8593-51fb-9fed-918395c859c5", "status": "complete",
       "segments_total": 18, "segments_done": 18, "facts": 55, "segments_with_problems": []}

client = TestClient(app)


@pytest.fixture
def module25(monkeypatch):
    """The real PersonalDBService over a fake Module 25 transport; records the requests."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, json={"detail": "No temporal job for version missing"})
        return httpx.Response(200, json=JOB)

    svc = PersonalDBService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings=get_settings())
    overrides = {get_current_user: lambda: {"sub": FAKE_SUB}, get_personal_db_service: lambda: svc}
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    yield seen
    for k, v in previous.items():
        if v is None:
            app.dependency_overrides.pop(k, None)
        else:
            app.dependency_overrides[k] = v


def test_status_is_passed_through_for_the_callers_own_database(module25):
    r = client.get(f"{STATUS}/{JOB['source_version_id']}")
    assert r.status_code == 200 and r.json() == JOB
    [req] = module25
    assert req.method == "GET"
    assert req.url.path == f"/personal/temporal/status/{JOB['source_version_id']}"
    assert req.headers["X-Branch-ID"] == FAKE_SUB          # scoped by the JWT, never by the caller


def test_no_job_stays_404(module25):
    r = client.get(f"{STATUS}/missing")
    assert r.status_code == 404 and "No temporal job" in r.json()["detail"]


@pytest.mark.parametrize("bad", ["a%2Fb", "..%2F..%2Fpersonal%2Fsearch", "x%3Fq%3D1", "a b"])
def test_version_id_cannot_reach_another_module25_route(module25, bad):
    r = client.get(f"{STATUS}/{bad}")
    assert r.status_code in (404, 422)
    assert module25 == []                                    # nothing forwarded


def test_capabilities_come_from_module_25_with_gateway_routes(monkeypatch):
    import httpx

    def handler(request):
        assert request.url.path == "/personal/temporal/capabilities"
        return httpx.Response(200, json={"contract_versions": [1, 2], "tkg": {"route": "/personal/search/tkg"},
                                         "operations": {"status": "/personal/temporal/status/{version_id}"}})

    svc = PersonalDBService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings=get_settings())
    overrides = {get_current_user: lambda: {"sub": FAKE_SUB}, get_personal_db_service: lambda: svc}
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    try:
        r = client.get(f"/api/{get_settings().api_version}/temporal/capabilities")
    finally:
        for k, v in previous.items():
            app.dependency_overrides.pop(k, None) if v is None else app.dependency_overrides.__setitem__(k, v)
    body = r.json()
    assert r.status_code == 200 and body["contract_versions"] == [1, 2]
    assert body["tkg"]["route"] == "/api/{version}/search/tkg"                 # the gateway's own paths
    assert body["operations"]["status"] == "/api/{version}/temporal/status/{version_id}"
    assert body["operations"]["reason"] == "/api/{version}/temporal/reason/{version_id}"
    # P17 / P21: the gateway's own routes, not Module 25's internal ones
    assert body["operations"]["verify"] == "/api/{version}/temporal/verify"
    assert body["operations"]["expiry"] == "/api/{version}/temporal/expiry/{version_id}"
    assert body["operations"]["parse"] == "/api/{version}/temporal/parse"          # TMP-7


def test_reason_and_recompute_are_proxied_for_the_callers_own_database():
    import httpx
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers["X-Branch-ID"]))
        if request.method == "POST":
            return httpx.Response(202, json={"source_version_id": "v1", "queued": True})
        return httpx.Response(200, json={"status": "consistent", "derived_facts": [], "stale": False})

    svc = PersonalDBService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings=get_settings())
    overrides = {get_current_user: lambda: {"sub": FAKE_SUB}, get_personal_db_service: lambda: svc}
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    base = f"/api/{get_settings().api_version}/temporal/reason/v1"
    try:
        assert client.get(base).json()["status"] == "consistent"
        r = client.post(f"{base}/recompute")
    finally:
        for k, v in previous.items():
            if v is None:
                app.dependency_overrides.pop(k, None)
            else:
                app.dependency_overrides[k] = v
    assert (r.status_code, r.json()["queued"]) == (202, True)
    assert seen == [("GET", "/personal/temporal/reason/v1", FAKE_SUB),
                    ("POST", "/personal/temporal/reason/v1/recompute", FAKE_SUB)]


def _proxied(handler):
    svc = PersonalDBService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings=get_settings())
    overrides = {get_current_user: lambda: {"sub": FAKE_SUB}, get_personal_db_service: lambda: svc}
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    return previous


def _restore(previous):
    for k, v in previous.items():
        if v is None:
            app.dependency_overrides.pop(k, None)
        else:
            app.dependency_overrides[k] = v


VERIFY = f"/api/{get_settings().api_version}/temporal/verify"
CLAIMS = {"claims": [{"kind": "date", "fact": "tf-1", "date": "2005"}], "evidence": ["tf-1"], "complete": False}


def test_verify_is_proxied_for_the_callers_own_facts():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers["X-Branch-ID"], request.read()))
        return httpx.Response(200, json={"verdicts": [{"verified": True}], "all_verified": True, "facts": {}})

    previous = _proxied(handler)
    try:
        r = client.post(VERIFY, json={**CLAIMS, "timezone": "Asia/Taipei"})
    finally:
        _restore(previous)
    assert r.status_code == 200 and r.json()["all_verified"] is True
    [(method, path, branch, body)] = seen
    assert (method, path, branch) == ("POST", "/personal/temporal/verify", FAKE_SUB)
    import json
    assert json.loads(body) == {**CLAIMS, "timezone": "Asia/Taipei"}


def test_verify_passes_module_25s_refusal_through_and_checks_its_own_limits():
    def handler(request):
        return httpx.Response(400, json={"detail": {"error": "INVALID_TIMEZONE", "timezone": "Mars/Base"}})

    previous = _proxied(handler)
    try:
        r = client.post(VERIFY, json={**CLAIMS, "timezone": "Mars/Base"})
        too_many = client.post(VERIFY, json={"claims": [{"kind": "absence"}] * 51, "evidence": []})
    finally:
        _restore(previous)
    assert r.status_code == 400 and r.json()["detail"]["error"] == "INVALID_TIMEZONE"
    assert too_many.status_code == 422                       # refused here, Module 25 never asked


PARSE = f"/api/{get_settings().api_version}/temporal/parse"


def test_parse_is_proxied_to_module_25():
    """TMP-7: the question's time condition, read by Module 25's normaliser; its answer comes back as is."""
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers["X-Branch-ID"], request.read()))
        return httpx.Response(200, json={"status": "RESOLVED", "match": "inside",
                                         "window": {"start": "2012-01-01T00:00:00.000+08:00", "end": None}})

    previous = _proxied(handler)
    try:
        r = client.post(PARSE, json={"text": "2011年以後發生什麼", "timezone": "Asia/Taipei"})
        empty = client.post(PARSE, json={"text": ""})
    finally:
        _restore(previous)
    assert r.status_code == 200 and r.json()["window"]["start"].startswith("2012-01-01")
    [(method, path, branch, body)] = seen
    assert (method, path, branch) == ("POST", "/personal/temporal/parse", FAKE_SUB)
    import json
    assert json.loads(body) == {"text": "2011年以後發生什麼", "timezone": "Asia/Taipei"}
    assert empty.status_code == 422                          # refused here, Module 25 never asked


def test_expiry_is_proxied_and_an_upload_without_one_stays_404():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers["X-Branch-ID"]))
        if request.url.path.endswith("/upload-1"):
            return httpx.Response(404, json={"detail": "No expiry registered for version upload-1"})
        return httpx.Response(200, json={"expires_at": "2026-10-29T07:00:00Z", "state": "pending", "expired": False})

    base = f"/api/{get_settings().api_version}/temporal/expiry"
    previous = _proxied(handler)
    try:
        ok = client.get(f"{base}/inline-1")
        missing = client.get(f"{base}/upload-1")
        bad = client.get(f"{base}/a%2Fb")
    finally:
        _restore(previous)
    assert ok.status_code == 200 and ok.json()["state"] == "pending"
    assert missing.status_code == 404
    assert bad.status_code in (404, 422)
    assert seen == [("/personal/temporal/expiry/inline-1", FAKE_SUB), ("/personal/temporal/expiry/upload-1", FAKE_SUB)]
