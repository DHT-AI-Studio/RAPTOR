"""Integration tests for /guardrail/system/* against a REAL, RUNNING module 23
over HTTP (VIE01-153's AC: "enable/disable toggle, status endpoint" --
tests/unit/test_state.py already covers this logic with a FakeRedis; this
covers the real deployed process's own routes and whatever Redis/Postgres it
actually connects to."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_enable_disable_round_trip(client):
    r = await client.post("/guardrail/system/disable")
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert (await client.get("/guardrail/system/status")).json()["enabled"] is False

    r = await client.post("/guardrail/system/enable")
    assert r.status_code == 200 and r.json()["enabled"] is True
    assert (await client.get("/guardrail/system/status")).json()["enabled"] is True


async def test_policy_enable_disable_round_trip(client):
    r = await client.post("/guardrail/system/policy/disable")
    assert r.status_code == 200 and r.json()["policy_enabled"] is False
    assert (await client.get("/guardrail/system/status")).json()["policy_enabled"] is False

    r = await client.post("/guardrail/system/policy/enable")
    assert r.status_code == 200 and r.json()["policy_enabled"] is True
    assert (await client.get("/guardrail/system/status")).json()["policy_enabled"] is True


async def test_status_reports_the_real_active_policy(client):
    await client.post("/guardrail/system/enable")
    await client.post("/guardrail/system/policy/enable")

    active = await client.get("/guardrail/policies/active")
    status = (await client.get("/guardrail/system/status")).json()

    assert status["enabled"] is True and status["policy_enabled"] is True
    if active.status_code == 200:
        assert status["active_policy_name"] == active.json()["name"]
        assert status["active_policy_version"] == active.json()["version"]
    else:
        assert status["active_policy_name"] is None


async def test_status_reports_proxy_mounted_as_a_bool(client):
    """Not toggleable from a test (it's set once at process startup from
    whether the OpenAI-compatible proxy router actually mounted) -- just
    confirms the real deployed process reports it, not a stale/missing field."""
    status = (await client.get("/guardrail/system/status")).json()
    assert isinstance(status["proxy_mounted"], bool)
