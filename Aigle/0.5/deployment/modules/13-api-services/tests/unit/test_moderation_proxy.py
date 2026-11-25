"""
CM-5 / VIE01-408: the gateway's admin review API for the automated flagging
system -- moderation-admin only, proxies to Module 23's /moderation/flags/*.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_moderation_proxy.py -v
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
from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

MODERATION = f"/api/{get_settings().api_version}/moderation"
FLAG_ID = "11111111-1111-1111-1111-111111111111"
_RECORD = {
    "id": FLAG_ID, "branch_id": "b1", "version_id": "v1", "asset_path": "a1", "moment_id": None, "chunk_id": None,
    "category": "nsfw", "score": 0.9, "model": "shieldgemma-2", "policy_version": None,
    "status": "flagged", "event_id": None,
    "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
}
_user: dict = {}


class _FakeClient:
    def __init__(self):
        self.calls: list = []
        self._response = httpx.Response(200, json=_RECORD, request=httpx.Request("GET", "http://x"))

    async def request(self, method, url, headers=None, timeout=None, params=None, **kwargs):
        self.calls.append({"method": method, "url": url, "params": dict(params or {}), **kwargs})
        return self._response


class _FakeStorage:
    """Stands in for StorageService -- records (key, version_id, user_id, branch_id)
    per call. Defaults to a successful presign; tests flip `raise_error` to check the
    fail-open path."""
    def __init__(self):
        self.calls: list[tuple[str, str, str, str]] = []
        self.url: str | None = "https://signed.example/preview"
        self.raise_error = False

    async def get_presign_url(self, key, version_id, user):
        if self.raise_error:
            raise RuntimeError("presign boom")
        self.calls.append((key, version_id, user.user_id, user.branch_id))
        return self.url


@pytest.fixture
def fake():
    fake = _FakeClient()
    fake.storage = _FakeStorage()

    async def _current():
        return _user

    saved = {dep: app.dependency_overrides.get(dep)
             for dep in (get_current_user, get_http_client, get_storage_service)}
    app.dependency_overrides[get_current_user] = _current
    app.dependency_overrides[get_http_client] = lambda: fake
    app.dependency_overrides[get_storage_service] = lambda: fake.storage
    _user.clear()
    _user.update({"sub": "jwt-verified-user-42"})
    yield fake
    for dep, prev in saved.items():
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev


client = TestClient(app)
AUTH = {"Authorization": "Bearer x"}


@pytest.mark.parametrize(("method", "path", "kwargs"), [
    ("GET", "/flags", {}),
    ("GET", f"/flags/{FLAG_ID}", {}),
    ("PATCH", f"/flags/{FLAG_ID}", {"json": {"status": "cleared"}}),
])
def test_routes_are_moderation_admin_only(fake, method, path, kwargs):
    fake._response = httpx.Response(
        200, json=_RECORD if path != "/flags" else {"items": [_RECORD], "total": 1, "page": 1, "page_size": 50},
        request=httpx.Request(method, "http://x"),
    )
    r = client.request(method, f"{MODERATION}{path}", headers=AUTH, **kwargs)
    assert r.status_code == 403 and fake.calls == []

    _user["realm_access"] = {"roles": ["moderation-admin"]}
    r = client.request(method, f"{MODERATION}{path}", headers=AUTH, **kwargs)
    assert r.status_code == 200
    [call] = fake.calls
    assert call["method"] == method and call["url"].endswith(path)


def test_list_flags_only_forwards_provided_filters(fake):
    fake._response = httpx.Response(200, json={"items": [], "total": 0, "page": 1, "page_size": 50},
                                     request=httpx.Request("GET", "http://x"))
    _user["realm_access"] = {"roles": ["moderation-admin"]}
    r = client.get(f"{MODERATION}/flags", params={"status": "flagged", "branch_id": "b1"}, headers=AUTH)
    assert r.status_code == 200
    assert fake.calls[0]["params"] == {"page": 1, "page_size": 50, "status": "flagged", "branch_id": "b1"}


def test_update_flag_status_forwards_body(fake):
    _user["realm_access"] = {"roles": ["moderation-admin"]}
    client.patch(f"{MODERATION}/flags/{FLAG_ID}", json={"status": "cleared", "note": "false positive"}, headers=AUTH)
    assert fake.calls[0]["json"] == {"status": "cleared", "note": "false positive"}


def test_module_23_error_passes_through(fake):
    _user["realm_access"] = {"roles": ["moderation-admin"]}

    async def not_found(method, url, headers=None, timeout=None, params=None, **kwargs):
        return httpx.Response(404, json={"detail": "flag not found"}, request=httpx.Request(method, url))

    fake.request = not_found
    r = client.get(f"{MODERATION}/flags/{FLAG_ID}", headers=AUTH)
    assert r.status_code == 404


def test_flag_creation_is_not_exposed_through_the_gateway(fake):
    _user["realm_access"] = {"roles": ["moderation-admin"]}
    r = client.post(f"{MODERATION}/flags", json={"branch_id": "b1", "version_id": "v1", "asset_path": "a1",
                                                  "category": "nsfw", "model": "shieldgemma-2"}, headers=AUTH)
    assert r.status_code == 405
    assert fake.calls == []


# ---------------------------------------------------------------- asset_url enrichment


def test_an_asset_type_flag_gets_a_presigned_asset_url(fake):
    _user["realm_access"] = {"roles": ["moderation-admin"]}
    r = client.get(f"{MODERATION}/flags/{FLAG_ID}", headers=AUTH)

    assert r.status_code == 200 and r.json()["asset_url"] == "https://signed.example/preview"
    [(key, version_id, user_id, branch_id)] = fake.storage.calls
    assert (key, version_id) == ("a1", "v1")
    assert user_id == "jwt-verified-user-42"
    assert branch_id == "b1"   # the flag's own branch_id -- whose data this is, not the admin's


def test_a_moment_type_flag_still_resolves_an_asset_url(fake):
    """asset_path/version_id are always present, even on a moment/chunk flag -- an
    admin reviewing a flagged video moment still needs to see the actual video, not
    just a one-way moment_id hash."""
    fake._response = httpx.Response(
        200, json={**_RECORD, "moment_id": "m1", "asset_path": "assets/video.mp4"},
        request=httpx.Request("GET", "http://x"),
    )
    _user["realm_access"] = {"roles": ["moderation-admin"]}
    r = client.get(f"{MODERATION}/flags/{FLAG_ID}", headers=AUTH)

    assert r.status_code == 200 and r.json()["asset_url"] == "https://signed.example/preview"
    [(key, version_id, *_rest)] = fake.storage.calls
    assert (key, version_id) == ("assets/video.mp4", "v1")


def test_a_presign_failure_does_not_break_the_response(fake):
    fake.storage.raise_error = True
    _user["realm_access"] = {"roles": ["moderation-admin"]}
    r = client.get(f"{MODERATION}/flags/{FLAG_ID}", headers=AUTH)

    assert r.status_code == 200 and r.json()["asset_url"] is None


def test_list_flags_resolves_asset_url_for_every_item(fake):
    fake._response = httpx.Response(
        200, json={"items": [_RECORD, {**_RECORD, "id": "22222222-2222-2222-2222-222222222222"}],
                   "total": 2, "page": 1, "page_size": 50},
        request=httpx.Request("GET", "http://x"),
    )
    _user["realm_access"] = {"roles": ["moderation-admin"]}
    r = client.get(f"{MODERATION}/flags", headers=AUTH)

    assert r.status_code == 200
    assert all(item["asset_url"] == "https://signed.example/preview" for item in r.json()["items"])
    assert len(fake.storage.calls) == 2
