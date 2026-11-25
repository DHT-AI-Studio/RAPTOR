"""RA-1: /health and POST /audio/session/token."""
from __future__ import annotations

import json
import time

from .conftest import OTHER_PRIVATE_KEY, make_jwt


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "module": 34}


def test_issue_ticket(client, sync_redis):
    resp = client.post("/audio/session/token", headers={"Authorization": f"Bearer {make_jwt('alice')}"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"token", "expires_in"} and body["expires_in"] == 30
    key = f"rt:ws_ticket:{body['token']}"
    payload = json.loads(sync_redis.get(key))
    assert payload["user_id"] == "alice"
    assert abs(payload["issued_at"] - time.time()) < 5
    assert 0 < sync_redis.ttl(key) <= 30


def test_tickets_are_unique(client):
    headers = {"Authorization": f"Bearer {make_jwt()}"}
    tokens = {client.post("/audio/session/token", headers=headers).json()["token"] for _ in range(5)}
    assert len(tokens) == 5


def test_missing_or_malformed_authorization(client):
    for headers in ({}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer"}):
        resp = client.post("/audio/session/token", headers=headers)
        assert resp.status_code == 401
        assert resp.json()["error"] == "missing_token"


def test_invalid_expired_and_subless_tokens(client):
    cases = {
        "invalid_token": [make_jwt(key=OTHER_PRIVATE_KEY), "not-a-jwt", make_jwt(sub=None)],
        "token_expired": [make_jwt(exp_in=-10)],
    }
    for error, tokens in cases.items():
        for token in tokens:
            resp = client.post("/audio/session/token", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 401, token
            assert resp.json()["error"] == error
