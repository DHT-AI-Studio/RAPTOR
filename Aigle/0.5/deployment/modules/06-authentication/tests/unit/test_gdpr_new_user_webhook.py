"""
GD-5 / VIE01-250: the registration webhook to Module 33.

Keycloak and Module 33 are both faked -- verifies that a successful
registration (self-service or admin-created) sends the new user's id to
POST /internal/gdpr/new-user, that a rolled-back registration never does,
and that a webhook failure never fails the registration itself.

Run from the 06-authentication/ directory:
    python -m pytest tests/unit/test_gdpr_new_user_webhook.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import requests
from app.core import gdpr
from app.core.config import settings
from app.routers import account, creat_user
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

USER_ID = "kc-uuid-1"


class _Resp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}
        self.text = ""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def webhook_calls(monkeypatch):
    calls = []

    def _post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return _Resp(204)

    monkeypatch.setattr(gdpr.requests, "post", _post)
    monkeypatch.setattr(settings, "GD_INTERNAL_TOKEN", "test-token")
    monkeypatch.setattr(settings, "SMTP_HOST", "")
    return calls


@pytest.fixture
def keycloak(monkeypatch):
    """Fake KeycloakUser/KeycloakMaster; `fail_role` makes role assignment
    fail so the route rolls the new user back."""
    state = {"fail_role": False, "deleted": []}
    ku = account.KeycloakUser
    monkeypatch.setattr(account.KeycloakMaster, "MasterToken", staticmethod(lambda: "master"))
    monkeypatch.setattr(creat_user.KeycloakMaster, "MasterToken", staticmethod(lambda: "master"))
    monkeypatch.setattr(ku, "user_create_api_basic", staticmethod(lambda **kw: _Resp(201)))
    monkeypatch.setattr(ku, "get_user_id_by_username", staticmethod(lambda **kw: USER_ID))
    monkeypatch.setattr(ku, "get_client_uuid", staticmethod(lambda **kw: "client-uuid"))

    def _role(**kw):
        if state["fail_role"]:
            raise RuntimeError("role lookup failed")
        return {"id": "r1", "name": "user_basic"}

    monkeypatch.setattr(ku, "get_client_role", staticmethod(_role))
    monkeypatch.setattr(ku, "assign_client_roles_to_user", staticmethod(lambda **kw: _Resp(204)))
    monkeypatch.setattr(ku, "user_delete_api", staticmethod(lambda **kw: state["deleted"].append(kw["user_id"])))
    return state


# Route functions are called directly rather than through a TestClient,
# which would need httpx -- not in this module's requirements.txt, and CI
# installs only that plus pytest.
def _register():
    return account.register(username="alice", email="a@x.io", firstName="", lastName="", realm_name="dhtsolution")


def test_self_registration_notifies_module_33(webhook_calls, keycloak):
    assert _register()["message"] == "Registration successful."
    assert webhook_calls == [{
        "url": f"{settings.PRIVACY_COMPLIANCE_URL}/internal/gdpr/new-user",
        "headers": {"X-Internal-Token": "test-token"},
        "json": {"user_id": USER_ID},
    }]


def test_rolled_back_registration_never_notifies(webhook_calls, keycloak):
    keycloak["fail_role"] = True
    with pytest.raises(HTTPException) as exc:
        _register()
    assert exc.value.status_code == 500
    assert keycloak["deleted"] == [USER_ID]
    assert webhook_calls == []


def test_admin_created_user_notifies_module_33(webhook_calls, keycloak):
    result = creat_user.create_user_api(
        user_data=creat_user.UserCreateRequest(username="bob", email="b@x.io"),
        realm_name="dhtsolution",
        token=HTTPAuthorizationCredentials(scheme="Bearer", credentials="admin"),
    )
    assert result["message"] == "User created successfully."
    assert webhook_calls[0]["json"] == {"user_id": USER_ID}


def test_webhook_failure_does_not_fail_registration(monkeypatch, keycloak):
    monkeypatch.setattr(settings, "GD_INTERNAL_TOKEN", "test-token")
    monkeypatch.setattr(settings, "SMTP_HOST", "")

    def _down(*a, **kw):
        raise requests.ConnectionError("module 33 down")

    monkeypatch.setattr(gdpr.requests, "post", _down)
    assert _register()["message"] == "Registration successful."


def test_notify_returns_false_on_error_status(monkeypatch):
    monkeypatch.setattr(settings, "GD_INTERNAL_TOKEN", "test-token")
    monkeypatch.setattr(gdpr.requests, "post", lambda *a, **kw: _Resp(500))
    assert gdpr.notify_new_user(USER_ID) is False


def test_notify_skipped_without_token(monkeypatch):
    monkeypatch.setattr(settings, "GD_INTERNAL_TOKEN", "")
    called = []
    monkeypatch.setattr(gdpr.requests, "post", lambda *a, **kw: called.append(1))
    assert gdpr.notify_new_user(USER_ID) is False
    assert called == []
