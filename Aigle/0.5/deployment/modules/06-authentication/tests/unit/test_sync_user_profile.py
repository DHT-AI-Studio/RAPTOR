"""
GD-5 / VIE01-250: sync_resources.py's unmanaged-attribute-policy sync.

Keeps Module 33's gdpr_consent_* writes from silently failing on a realm
whose users/profile doesn't allow admin writes to attributes it hasn't
declared -- see 06-authentication/README.md, "Unmanaged attributes". Lives
in its own REST sub-resource, not the realm-import JSON, so it needs this
sync step the same way the resource URIs below it do.

Run from the 06-authentication/ directory:
    python -m pytest tests/unit/test_sync_user_profile.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from app import sync_resources as sr


class _Resp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def profile_calls(monkeypatch):
    calls = {"get": 0, "put": []}

    def _get(url, headers=None, timeout=None):
        calls["get"] += 1
        return _Resp(200, calls["profile"])

    def _put(url, headers=None, json=None, timeout=None):
        calls["put"].append(json)
        return _Resp(200)

    monkeypatch.setattr(sr.requests, "get", _get)
    monkeypatch.setattr(sr.requests, "put", _put)
    return calls


def test_already_correct_is_a_noop(profile_calls):
    profile_calls["profile"] = {"unmanagedAttributePolicy": "ADMIN_EDIT", "attributes": []}
    changed = sr.sync_unmanaged_attribute_policy("tok", apply=True)
    assert changed is False
    assert profile_calls["put"] == []


def test_missing_policy_is_applied_without_losing_other_fields(profile_calls):
    profile_calls["profile"] = {"attributes": [{"name": "email"}], "groups": [{"name": "user-metadata"}]}
    changed = sr.sync_unmanaged_attribute_policy("tok", apply=True)
    assert changed is True
    [put] = profile_calls["put"]
    assert put["unmanagedAttributePolicy"] == "ADMIN_EDIT"
    assert put["attributes"] == [{"name": "email"}]          # rest of the profile preserved
    assert put["groups"] == [{"name": "user-metadata"}]


def test_wrong_policy_is_corrected(profile_calls):
    profile_calls["profile"] = {"unmanagedAttributePolicy": "ENABLED", "attributes": []}
    changed = sr.sync_unmanaged_attribute_policy("tok", apply=True)
    assert changed is True
    [put] = profile_calls["put"]
    assert put["unmanagedAttributePolicy"] == "ADMIN_EDIT"


def test_dry_run_reports_without_applying(profile_calls):
    profile_calls["profile"] = {"unmanagedAttributePolicy": "ENABLED"}
    changed = sr.sync_unmanaged_attribute_policy("tok", apply=False)
    assert changed is True
    assert profile_calls["put"] == []
