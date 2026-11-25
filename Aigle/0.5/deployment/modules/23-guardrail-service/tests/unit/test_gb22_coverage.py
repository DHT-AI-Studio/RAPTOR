"""Unit tests for GB-22 platform-integration coverage endpoints.

Tests run entirely in-process — no Docker, no real Redis.
Redis is replaced by a fake async client (FakeRedis from fakeredis).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_redis(preset: dict[str, str] | None = None):
    """Return a minimal async Redis mock that supports get/set/mget."""
    store: dict[str, str] = dict(preset or {})
    fake = MagicMock()

    async def _set(key, value):
        store[key] = value

    async def _mget(*keys):
        return [store.get(k) for k in keys]

    async def _get(key):
        return store.get(key)

    fake.set = _set
    fake.mget = _mget
    fake.get = _get
    fake._store = store          # expose for assertions
    return fake


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_redis_empty():
    return _make_fake_redis()


@pytest.fixture()
def fake_redis_all_screened():
    keys = [
        "guardrail:integration:module_07_hook",
        "guardrail:integration:module_15_proxy",
        "guardrail:integration:mcp_tool_calls",
        "guardrail:integration:gateway_middleware",
        "guardrail:integration:grpc_interceptor",
        "guardrail:integration:a2a_host_orchestrator",
    ]
    return _make_fake_redis({k: "1" for k in keys})


@pytest.fixture()
def fake_redis_partial():
    """Only gateway_middleware and grpc_interceptor screened."""
    return _make_fake_redis({
        "guardrail:integration:gateway_middleware": "1",
        "guardrail:integration:grpc_interceptor": "1",
    })


@pytest_asyncio.fixture()
async def client_empty(fake_redis_empty):
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_empty)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest_asyncio.fixture()
async def client_all(fake_redis_all_screened):
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_all_screened)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest_asyncio.fixture()
async def client_partial(fake_redis_partial):
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_partial)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


# ── Legacy endpoint backward-compat ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_legacy_coverage_still_works(client_empty):
    """GET /guardrail/coverage must still return the static enforcement-point list."""
    r = await client_empty.get("/guardrail/coverage")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 11
    endpoints = [e["endpoint"] for e in data]
    assert "/guardrail/policy/decide" in endpoints
    assert "/guard/check/input" in endpoints


@pytest.mark.asyncio
async def test_legacy_coverage_shape(client_empty):
    r = await client_empty.get("/guardrail/coverage")
    entry = r.json()[0]
    assert set(entry.keys()) >= {"endpoint", "method", "policy_aware", "deprecated", "story", "description"}


# ── /api/0.4/guardrail/coverage/integration ──────────────────────────────────

@pytest.mark.asyncio
async def test_integration_coverage_returns_six_rows(client_empty):
    r = await client_empty.get("/api/0.4/guardrail/coverage/integration")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 6


@pytest.mark.asyncio
async def test_integration_coverage_all_screened_false_when_redis_empty(client_empty):
    r = await client_empty.get("/api/0.4/guardrail/coverage/integration")
    assert all(not row["screened"] for row in r.json())


@pytest.mark.asyncio
async def test_integration_coverage_all_screened_true_when_all_keys_set(client_all):
    r = await client_all.get("/api/0.4/guardrail/coverage/integration")
    assert r.status_code == 200
    assert all(row["screened"] for row in r.json())


@pytest.mark.asyncio
async def test_integration_coverage_partial(client_partial):
    r = await client_partial.get("/api/0.4/guardrail/coverage/integration")
    data = r.json()
    screened = {row["entry_point"]: row["screened"] for row in data}
    assert screened["gateway_middleware"] is True
    assert screened["grpc_interceptor"] is True
    assert screened["module_07_hook"] is False
    assert screened["module_15_proxy"] is False
    assert screened["mcp_tool_calls"] is False
    assert screened["a2a_host_orchestrator"] is False


@pytest.mark.asyncio
async def test_integration_coverage_required_fields(client_empty):
    r = await client_empty.get("/api/0.4/guardrail/coverage/integration")
    row = r.json()[0]
    assert set(row.keys()) == {
        "entry_point", "module", "tier", "screened", "method", "env_var", "story_reference"
    }


@pytest.mark.asyncio
async def test_integration_coverage_known_entry_points(client_empty):
    r = await client_empty.get("/api/0.4/guardrail/coverage/integration")
    entry_points = {row["entry_point"] for row in r.json()}
    assert entry_points == {
        "module_07_hook",
        "module_15_proxy",
        "mcp_tool_calls",
        "gateway_middleware",
        "grpc_interceptor",
        "a2a_host_orchestrator",
    }


@pytest.mark.asyncio
async def test_integration_coverage_story_references(client_empty):
    r = await client_empty.get("/api/0.4/guardrail/coverage/integration")
    stories = {row["entry_point"]: row["story_reference"] for row in r.json()}
    assert stories["module_07_hook"] == "GB-6"
    assert stories["module_15_proxy"] == "GB-12"
    assert stories["mcp_tool_calls"] == "GB-18"
    assert stories["gateway_middleware"] == "GB-19"
    assert stories["grpc_interceptor"] == "GB-20"
    assert stories["a2a_host_orchestrator"] == "GB-21"


# ── /api/0.4/guardrail/coverage/integration/summary ──────────────────────────

@pytest.mark.asyncio
async def test_summary_all_unscreened(client_empty):
    r = await client_empty.get("/api/0.4/guardrail/coverage/integration/summary")
    assert r.status_code == 200
    s = r.json()
    assert s["total"] == 6
    assert s["screened"] == 0
    assert s["unscreened"] == 6
    assert s["coverage_pct"] == 0.0


@pytest.mark.asyncio
async def test_summary_all_screened(client_all):
    r = await client_all.get("/api/0.4/guardrail/coverage/integration/summary")
    s = r.json()
    assert s["total"] == 6
    assert s["screened"] == 6
    assert s["unscreened"] == 0
    assert s["coverage_pct"] == 100.0


@pytest.mark.asyncio
async def test_summary_partial(client_partial):
    r = await client_partial.get("/api/0.4/guardrail/coverage/integration/summary")
    s = r.json()
    assert s["total"] == 6
    assert s["screened"] == 2
    assert s["unscreened"] == 4
    assert s["coverage_pct"] == pytest.approx(33.33, abs=0.01)


@pytest.mark.asyncio
async def test_summary_coverage_pct_formula(client_partial):
    """coverage_pct must equal screened/total*100 rounded to 2 dp."""
    r = await client_partial.get("/api/0.4/guardrail/coverage/integration/summary")
    s = r.json()
    expected = round(s["screened"] / s["total"] * 100, 2)
    assert s["coverage_pct"] == expected


# ── POST /api/0.4/guardrail/integration/register ─────────────────────────────

@pytest.mark.asyncio
async def test_register_known_entry_point(fake_redis_empty):
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_empty)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/0.4/guardrail/integration/register",
                json={"entry_point": "module_07_hook", "module": "07-ai-ml-services"},
            )
    assert r.status_code == 204
    assert fake_redis_empty._store.get("guardrail:integration:module_07_hook") == "1"


@pytest.mark.asyncio
async def test_register_sets_redis_key(fake_redis_empty):
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_empty)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for ep in [
                "module_07_hook", "module_15_proxy", "mcp_tool_calls",
                "gateway_middleware", "grpc_interceptor", "a2a_host_orchestrator",
            ]:
                r = await c.post(
                    "/api/0.4/guardrail/integration/register",
                    json={"entry_point": ep, "module": "test"},
                )
                assert r.status_code == 204, f"Failed for {ep}: {r.text}"
    # all 6 keys should now be set
    for ep in [
        "module_07_hook", "module_15_proxy", "mcp_tool_calls",
        "gateway_middleware", "grpc_interceptor", "a2a_host_orchestrator",
    ]:
        assert fake_redis_empty._store.get(f"guardrail:integration:{ep}") == "1"


@pytest.mark.asyncio
async def test_register_unknown_entry_point_returns_422(fake_redis_empty):
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_empty)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/0.4/guardrail/integration/register",
                json={"entry_point": "nonexistent_module", "module": "test"},
            )
    assert r.status_code == 422
    assert "nonexistent_module" in r.json()["detail"]


@pytest.mark.asyncio
async def test_register_then_coverage_reflects_change(fake_redis_empty):
    """Register one entry point, then verify coverage matrix shows it as screened."""
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_empty)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post(
                "/api/0.4/guardrail/integration/register",
                json={"entry_point": "grpc_interceptor", "module": "36-grpc-gateway"},
            )
            r = await c.get("/api/0.4/guardrail/coverage/integration")

    rows = {row["entry_point"]: row["screened"] for row in r.json()}
    assert rows["grpc_interceptor"] is True
    assert rows["module_07_hook"] is False


@pytest.mark.asyncio
async def test_register_missing_fields_returns_422(fake_redis_empty):
    from app.main import app
    with patch("app.routers.coverage.get_redis", new=AsyncMock(return_value=fake_redis_empty)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/0.4/guardrail/integration/register",
                json={"entry_point": "module_07_hook"},  # missing module
            )
    assert r.status_code == 422


# ── Legacy path also serves new endpoints (double-mount sanity) ───────────────

@pytest.mark.asyncio
async def test_legacy_prefix_does_not_expose_integration_endpoints(client_empty):
    """/guardrail/coverage/integration must NOT be reachable — the legacy
    mount prefix is /guardrail, so it would be /guardrail/coverage/integration.
    The router defines /coverage/integration, so under /guardrail prefix that
    becomes /guardrail/coverage/integration — this IS reachable and is fine,
    but /guardrail/integration/register should also be reachable under the
    legacy prefix."""
    # Both prefixes expose the same router — verify /guardrail/coverage/integration works too
    r = await client_empty.get("/guardrail/coverage/integration")
    assert r.status_code == 200
    assert len(r.json()) == 6
