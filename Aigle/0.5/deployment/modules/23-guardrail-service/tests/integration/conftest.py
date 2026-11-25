"""Shared fixtures for the integration tests (VIE01-153's own follow-up).

These hit a REAL, RUNNING module 23 over HTTP -- not its Python functions
imported directly, and not its backing Postgres/Redis bypassing the app.
Importing the service layer only proves the logic is right against real
data stores; it says nothing about whether the actual deployed process is
up, wired correctly (routing, dependency injection, its own startup config),
or reachable at all. Point this at a running instance and run:

    GUARDRAIL_BASE_URL=http://localhost:8023 pytest tests/integration/

If it's unreachable the integration tests skip (so the pure-unit suite
still runs anywhere, same as every other module's tests/integration/). Ollama
(the actual guard models) is deliberately NOT covered here -- that needs a
real GPU-backed Ollama with the configured models pulled, out of scope for a
suite meant to run from a plain dev checkout.

None of these endpoints take auth (confirmed against app/routers/guardrail_policies.py
and app/routers/system.py -- no Depends on either), matching how they're meant to be
reached: from inside the platform network, not exposed to end users.
"""
from __future__ import annotations

import logging
import os
import uuid

import httpx
import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("GUARDRAIL_BASE_URL", "http://localhost:8023")


async def _reachable(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get("/guardrail/system/status", timeout=5.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


@pytest_asyncio.fixture
async def client():
    """A real httpx client against BASE_URL, or skip if it's not reachable."""
    async with httpx.AsyncClient(base_url=BASE_URL) as c:
        if not await _reachable(c):
            pytest.skip(f"module 23 not reachable at {BASE_URL} (set GUARDRAIL_BASE_URL)")
        yield c


def unique_name(label: str) -> str:
    """A policy `name` that can't collide with another test or a human's
    own policy on the same running instance."""
    return f"itest-{label}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def cleanup_policies(client):
    """Deletes every policy id this test created, via the real DELETE
    endpoint -- not direct SQL. A policy still active when cleanup runs
    (the test itself should have deactivated it, but might not have if it
    failed first) is deactivated here before the delete, so cleanup never
    silently leaves a stray active test policy behind."""
    created: list[str] = []
    yield created
    for policy_id in created:
        try:
            detail = await client.get(f"/guardrail/policies/{policy_id}")
            if detail.status_code == 200 and detail.json().get("is_active"):
                await client.put(f"/guardrail/policies/{policy_id}/deactivate")
            resp = await client.delete(f"/guardrail/policies/{policy_id}")
            if resp.status_code not in (204, 404, 409):
                logger.warning("cleanup: could not delete policy %s: %s %s", policy_id, resp.status_code, resp.text)
        except httpx.HTTPError as exc:
            logger.warning("cleanup: delete failed for policy %s: %s", policy_id, exc)


@pytest_asyncio.fixture(autouse=True)
async def _restore_real_state(client):
    """Safety net for a real, possibly-shared deployment: records what was
    enabled and which policy was active *before* any test in this run
    touches either (all via the real HTTP API), then restores both via that
    same API -- no matter which test failed, raised, or never got to its
    own cleanup code."""
    status = (await client.get("/guardrail/system/status")).json()
    active = await client.get("/guardrail/policies/active")
    original_active_id = active.json()["id"] if active.status_code == 200 else None

    yield

    await client.post("/guardrail/system/enable" if status["enabled"] else "/guardrail/system/disable")
    await client.post("/guardrail/system/policy/enable" if status["policy_enabled"] else "/guardrail/system/policy/disable")
    if original_active_id is not None:
        r = await client.put(f"/guardrail/policies/{original_active_id}/activate")
        if r.status_code != 200:
            logger.warning(
                "Could not restore original active policy id=%s after the integration run (status %s) -- "
                "it may have been deleted mid-test.", original_active_id, r.status_code,
            )
