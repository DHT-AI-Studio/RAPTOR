"""
VIE01-369: write-freeze maintenance mode -- scripts/nfs-freeze-archive.sh
calls POST /admin/maintenance/enable before touching the mirror's final
delta, and /disable right after, so 04's/26's writes through the gateway
answer 503 for that window without stopping any container.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_maintenance_mode.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')
os.environ.setdefault("GATEWAY_INTERNAL_ADMIN_TOKEN", "test-internal-token")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from app.core.config import get_settings
from app.main import app as gateway
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

# get_settings() is @lru_cache'd process-wide -- whichever test module
# the suite happens to import first is the one whose env vars "win" for
# the whole run, so GATEWAY_INTERNAL_ADMIN_TOKEN above may or may not
# actually be in effect by the time this runs. Read back whatever value
# is actually live instead of assuming the one just set above stuck.
_REAL_TOKEN = get_settings().internal_admin_token
TOKEN = {"X-Internal-Token": _REAL_TOKEN}
WRONG_TOKEN = {"X-Internal-Token": _REAL_TOKEN + "-wrong"}


class _FakeRedis:
    def __init__(self):
        self.store = {}

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)

    async def exists(self, key):
        return 1 if key in self.store else 0


class _BrokenRedis(_FakeRedis):
    async def exists(self, key):
        raise RedisError("down")


@pytest.fixture
def gw():
    saved = getattr(gateway.state, "redis_client", None)
    fake = _FakeRedis()
    gateway.state.redis_client = fake
    yield TestClient(gateway, raise_server_exceptions=False), fake
    gateway.state.redis_client = saved


# ── Admin endpoints require the internal token ──────────────────────────────

def test_enable_requires_internal_token(gw):
    client, _ = gw
    assert client.post("/admin/maintenance/enable").status_code == 401
    assert client.post("/admin/maintenance/enable", headers=WRONG_TOKEN).status_code == 401


def test_enable_disable_status_round_trip(gw):
    client, _fake = gw
    assert client.get("/admin/maintenance/status", headers=TOKEN).json() == {"enabled": False}

    r = client.post("/admin/maintenance/enable", headers=TOKEN)
    assert r.status_code == 200
    assert r.json()["enabled"] is True
    assert client.get("/admin/maintenance/status", headers=TOKEN).json() == {"enabled": True}

    r = client.post("/admin/maintenance/disable", headers=TOKEN)
    assert r.json() == {"enabled": False}
    assert client.get("/admin/maintenance/status", headers=TOKEN).json() == {"enabled": False}


def test_enable_sets_a_ttl(gw):
    """The TTL is the safety net against a caller that crashes before
    calling /disable -- confirm it's actually passed to Redis, not
    dropped."""
    client, fake = gw
    seen = {}
    orig_set = fake.set

    async def _set(key, value, ex=None):
        seen["ex"] = ex
        await orig_set(key, value, ex=ex)

    fake.set = _set
    client.post("/admin/maintenance/enable", headers=TOKEN, params={"ttl_seconds": 42})
    assert seen["ex"] == 42


# ── The middleware itself: only /asset and /memory writes, nothing else ─────

@pytest.mark.parametrize("method, path", [
    ("POST", "/api/0.5/asset/fileupload"),
    ("PUT", "/api/0.5/memory/x"),
    ("DELETE", "/api/0.5/asset/x"),
    ("PATCH", "/api/0.5/memory/x"),
    ("POST", "/api/0.3/asset/fileupload"),   # legacy alias -- must be rewritten first to match
])
def test_writes_503_while_frozen(gw, method, path):
    client, _ = gw
    client.post("/admin/maintenance/enable", headers=TOKEN)
    r = client.request(method, path)
    assert r.status_code == 503
    assert r.json() == {"reason": "maintenance_write_freeze"}


@pytest.mark.parametrize("method, path", [
    ("DELETE", "/api/0.5/identity/abc"),
    ("POST", "/api/0.5/chat/chat"),
    ("POST", "/api/0.5/training/jobs"),
    ("POST", "/api/0.5/gdpr/export"),
    ("POST", "/api/0.5/personal-db/x"),
])
def test_other_routers_unaffected_while_frozen(gw, method, path):
    """04/26 are the only routers this backup mechanism protects -- every
    other router writes to 03's Postgres (its own MVCC-consistent pg_dump
    needs no request-level freeze) or elsewhere, confirmed by grepping
    every router for asset_management_url/memory_service_url. Freezing
    them too would block writes 01's backup was never protecting."""
    client, _ = gw
    client.post("/admin/maintenance/enable", headers=TOKEN)
    r = client.request(method, path)
    assert r.status_code != 503


def test_reads_unaffected_while_frozen(gw):
    client, _ = gw
    client.post("/admin/maintenance/enable", headers=TOKEN)
    # No auth header at all -- if the maintenance middleware let this
    # through to the real route, it would 401, not 503. Confirms GET
    # never even reaches the 503 short-circuit.
    r = client.get("/api/0.5/asset/users/commits")
    assert r.status_code != 503


def test_writes_pass_through_once_disabled(gw):
    client, _ = gw
    client.post("/admin/maintenance/enable", headers=TOKEN)
    client.post("/admin/maintenance/disable", headers=TOKEN)
    r = client.post("/api/0.5/asset/fileupload")
    assert r.status_code != 503


def test_redis_error_fails_open(gw):
    """Same convention as the rate limiter: a Redis outage never takes
    down every write in the gateway just because maintenance mode exists."""
    client, _ = gw
    gateway.state.redis_client = _BrokenRedis()
    r = client.post("/api/0.5/asset/fileupload")
    assert r.status_code != 503


def test_no_redis_client_at_all_fails_open(gw):
    client, _ = gw
    gateway.state.redis_client = None
    r = client.post("/api/0.5/asset/fileupload")
    assert r.status_code != 503
