"""Integration tests for /guardrail/policies/* against a REAL, RUNNING module 23
over HTTP (VIE01-153's AC: "CRUD round-trip, activate/deactivate, cannot delete
active policy" -- tests/unit/test_policy_store_update.py already covers this
logic with a monkeypatched pool; this covers the real deployed process: its
own routing, request validation, and whatever Postgres/Redis it actually
connects to, not ones this test constructs itself."""
from __future__ import annotations

import pytest

from .conftest import unique_name

pytestmark = pytest.mark.asyncio

RAW_CONTENT = "input_guardrail:\n  rules: []\n"


async def _create(client, name, version="v1"):
    r = await client.post("/guardrail/policies", params={"name": name, "version": version},
                           content=RAW_CONTENT, headers={"content-type": "text/plain"})
    assert r.status_code == 201, r.text
    return r.json()


async def test_create_then_get_round_trips_content(client, cleanup_policies):
    created = await _create(client, unique_name("crud"))
    cleanup_policies.append(created["id"])

    r = await client.get(f"/guardrail/policies/{created['id']}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["name"] == created["name"] and detail["version"] == "v1"
    assert detail["raw_content"] == RAW_CONTENT


async def test_list_policies_includes_the_created_policy(client, cleanup_policies):
    created = await _create(client, unique_name("list"))
    cleanup_policies.append(created["id"])

    r = await client.get("/guardrail/policies")
    assert r.status_code == 200
    assert created["id"] in {p["id"] for p in r.json()}


async def test_activate_then_deactivate_toggles_is_active(client, cleanup_policies):
    created = await _create(client, unique_name("toggle"))
    cleanup_policies.append(created["id"])

    r = await client.put(f"/guardrail/policies/{created['id']}/activate")
    assert r.status_code == 200 and r.json()["id"] == created["id"]
    assert (await client.get(f"/guardrail/policies/{created['id']}")).json()["is_active"] is True
    assert (await client.get("/guardrail/policies/active")).json()["id"] == created["id"]

    r = await client.put(f"/guardrail/policies/{created['id']}/deactivate")
    assert r.status_code == 200
    assert (await client.get(f"/guardrail/policies/{created['id']}")).json()["is_active"] is False


async def test_deactivating_a_policy_that_is_not_active_is_409(client, cleanup_policies):
    created = await _create(client, unique_name("notactive"))
    cleanup_policies.append(created["id"])

    r = await client.put(f"/guardrail/policies/{created['id']}/deactivate")
    assert r.status_code == 409


async def test_cannot_delete_the_active_policy(client, cleanup_policies):
    created = await _create(client, unique_name("delactive"))
    cleanup_policies.append(created["id"])
    await client.put(f"/guardrail/policies/{created['id']}/activate")

    r = await client.delete(f"/guardrail/policies/{created['id']}")
    assert r.status_code == 409
    assert "active" in r.json()["detail"].lower()
    # still there -- the delete must not have gone through
    assert (await client.get(f"/guardrail/policies/{created['id']}")).status_code == 200


async def test_can_delete_once_no_longer_active(client, cleanup_policies):
    created = await _create(client, unique_name("delok"))
    cleanup_policies.append(created["id"])
    await client.put(f"/guardrail/policies/{created['id']}/activate")
    await client.put(f"/guardrail/policies/{created['id']}/deactivate")

    r = await client.delete(f"/guardrail/policies/{created['id']}")
    assert r.status_code == 204
    assert (await client.get(f"/guardrail/policies/{created['id']}")).status_code == 404


async def test_cannot_delete_the_protected_default_policy(client):
    """Never registered with cleanup_policies -- if this guard is broken and
    the DELETE actually goes through, the real fallback policy every other
    guardrail check degrades to would be gone.

    Deactivates the default first: the active-policy check wins over the
    protected-name check when both are true (confirmed: deleting the
    default right after activating it hits the "active" 409, not
    "protected"), and deactivating the default specifically does not
    trigger the usual re-activate-the-default fallback. Always restores the
    invariant afterward, success or failure -- this is a real, possibly
    shared, deployment."""
    active = await client.get("/guardrail/policies/active")
    assert active.status_code == 200
    default = active.json()
    default_name = default["name"]
    if default_name != "default-standard-policy":
        pytest.skip(
            f"currently active policy is {default_name!r}, not the known built-in default -- "
            "won't deactivate+delete an unidentified live policy just to exercise this guard"
        )

    await client.put(f"/guardrail/policies/{default['id']}/deactivate")
    try:
        r = await client.delete(f"/guardrail/policies/{default['id']}")
        assert r.status_code == 409
        detail = r.json()["detail"].lower()
        assert "protected" in detail or "built-in" in detail or default_name.lower() in detail

        assert (await client.get(f"/guardrail/policies/{default['id']}")).status_code == 200
    finally:
        await client.put(f"/guardrail/policies/{default['id']}/activate")  # restore: some policy must stay active
