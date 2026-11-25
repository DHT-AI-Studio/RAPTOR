"""
CM-7 / VIE01-410: GET /api/{v}/compliance/report -- compliance-auditor only;
the period and format go to Module 33 unchanged, and its JSON or CSV comes
back as it is, with the file name and section headers.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_compliance_report_proxy.py -v
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
from app.api.dependencies import get_current_user, get_http_client
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

REPORT = f"/api/{get_settings().api_version}/compliance/report"
PERIOD = {"from": "2026-11-01T00:00:00Z", "to": "2026-11-30T23:59:59Z"}
CSV_BODY = "section,occurred_at\r\naccess_log,2026-11-03T08:30:00+00:00\r\n"
_user: dict = {}


class _FakeClient:
    def __init__(self):
        self.calls: list = []
        self.response = httpx.Response(200, json={"totals": {"access_log": 1}},
                                       headers={"x-report-truncated-sections": "",
                                                "x-report-unavailable-sections": "moderation_flags"})

    async def request(self, method, url, headers=None, timeout=None, params=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}),
                           "params": dict(params or {}), "timeout": timeout})
        self.response.request = httpx.Request(method, url)
        return self.response


@pytest.fixture
def fake():
    fake = _FakeClient()

    async def _current():
        return _user

    saved = {dep: app.dependency_overrides.get(dep) for dep in (get_current_user, get_http_client)}
    app.dependency_overrides[get_current_user] = _current
    app.dependency_overrides[get_http_client] = lambda: fake
    _user.clear()
    _user.update({"sub": "auditor-1"})
    yield fake
    for dep, prev in saved.items():
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev


client = TestClient(app)
AUTH = {"Authorization": "Bearer x"}


def _auditor():
    _user["realm_access"] = {"roles": ["compliance-auditor"]}


@pytest.mark.parametrize("roles", [[], ["gdpr-admin"], ["moderation-admin"], ["gdpr-admin", "moderation-admin"]])
def test_only_compliance_auditors_get_the_report(fake, roles):
    _user["realm_access"] = {"roles": roles}
    r = client.get(REPORT, params=PERIOD, headers=AUTH)
    assert r.status_code == 403 and "compliance-auditor" in r.json()["detail"]
    assert fake.calls == []


def test_auditor_request_goes_to_module_33_with_the_internal_token(fake):
    _auditor()
    r = client.get(REPORT, params=PERIOD, headers=AUTH)
    assert r.status_code == 200 and r.json() == {"totals": {"access_log": 1}}
    [call] = fake.calls
    assert call["method"] == "GET" and call["url"].endswith("/reports/compliance")
    assert call["headers"] == {"X-Internal-Token": get_settings().gdpr_internal_token}
    assert call["timeout"] == get_settings().compliance_report_timeout
    assert r.headers["x-report-unavailable-sections"] == "moderation_flags"


def test_only_the_report_parameters_are_forwarded(fake):
    _auditor()
    client.get(REPORT, params={**PERIOD, "format": "csv", "max_rows": "500", "user_id": "bob"}, headers=AUTH)
    assert fake.calls[0]["params"] == {"from": "2026-11-01T00:00:00+00:00", "to": "2026-11-30T23:59:59+00:00",
                                       "format": "csv", "max_rows": 500}


def test_json_is_the_default_format(fake):
    _auditor()
    client.get(REPORT, params=PERIOD, headers=AUTH)
    assert fake.calls[0]["params"]["format"] == "json" and "max_rows" not in fake.calls[0]["params"]


def test_parameters_show_in_swagger():
    params = {p["name"]: p for p in app.openapi()["paths"][REPORT]["get"]["parameters"]}
    assert {"from", "to", "format", "max_rows"} <= set(params)
    assert params["from"]["required"] and params["to"]["required"]


@pytest.mark.parametrize("query", [{}, {"from": PERIOD["from"]}, {**PERIOD, "format": "xlsx"},
                                   {**PERIOD, "max_rows": "0"}, {"from": "yesterday", "to": PERIOD["to"]}])
def test_bad_parameters_are_rejected_before_module_33(fake, query):
    _auditor()
    assert client.get(REPORT, params=query, headers=AUTH).status_code == 422
    assert fake.calls == []


def test_csv_download_passes_through(fake):
    _auditor()
    fake.response = httpx.Response(
        200, content=CSV_BODY.encode(),
        headers={"content-type": "text/csv; charset=utf-8",
                 "content-disposition": 'attachment; filename="compliance-report_2026-11-01_2026-11-30.csv"',
                 "x-report-truncated-sections": "access_log", "x-internal-detail": "not for clients"})
    r = client.get(REPORT, params={**PERIOD, "format": "csv"}, headers=AUTH)
    assert r.status_code == 200 and r.text == CSV_BODY
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"].endswith('compliance-report_2026-11-01_2026-11-30.csv"')
    assert r.headers["x-report-truncated-sections"] == "access_log"
    assert "x-internal-detail" not in r.headers


@pytest.mark.parametrize("code", [422, 503])
def test_module_33_errors_pass_through(fake, code):
    _auditor()
    fake.response = httpx.Response(code, json={"detail": "nope"})
    assert client.get(REPORT, params=PERIOD, headers=AUTH).status_code == code


def test_unreachable_module_33_is_a_502(fake):
    _auditor()

    async def down(*args, **kwargs):
        raise httpx.ConnectError("refused")

    fake.request = down
    assert client.get(REPORT, params=PERIOD, headers=AUTH).status_code == 502
