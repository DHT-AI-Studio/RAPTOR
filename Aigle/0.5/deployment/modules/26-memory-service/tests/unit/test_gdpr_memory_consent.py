"""
GD-5 / VIE01-250: DELETE /internal/gdpr/memory/user/{user_id} -- the
partial (memory-only) erasure Module 33 calls when a user withdraws memory
consent.

Run from the 26-memory-service/ directory:
    python -m pytest tests/unit/test_gdpr_memory_consent.py -v
"""
import os
import sys

os.environ.setdefault("MEM_REDIS_HOST", "localhost")
os.environ.setdefault("MEM_STORAGE_ROOT", "/tmp/gd5_test")
os.environ.setdefault("MEM_GDPR_INTERNAL_TOKEN", "test-internal-token")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

from core.config import settings
from core.dependencies import get_redis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers import gdpr as gdpr_router
from routers.management import get_management_service

# Same import-order fragility test_api.py documents for storage_root: this
# module's own os.environ.setdefault above only takes effect if it's the
# first to import core.config in the whole pytest run. Overwrite the
# singleton attribute directly so this module's token is authoritative
# regardless of import order.
settings.gdpr_internal_token = "test-internal-token"


class _FakeService:
    def __init__(self, found: bool):
        self.found = found
        self.deleted: list = []

    async def delete_all(self, user_id, redis):
        self.deleted.append((user_id, redis))
        return self.found


def _client(svc: _FakeService) -> TestClient:
    app = FastAPI()
    app.include_router(gdpr_router.router)
    app.dependency_overrides[get_management_service] = lambda: svc
    app.dependency_overrides[get_redis] = lambda: "redis"
    return TestClient(app)


def test_memory_erasure_requires_internal_token():
    svc = _FakeService(found=True)
    r = _client(svc).delete("/internal/gdpr/memory/user/u1")
    assert r.status_code == 401
    assert svc.deleted == []


def test_memory_erasure_deletes_that_users_memory():
    svc = _FakeService(found=True)
    r = _client(svc).delete(
        "/internal/gdpr/memory/user/u1", headers={"X-Internal-Token": settings.gdpr_internal_token},
    )
    assert r.status_code == 200
    assert r.json() == {"user_id": "u1", "deleted": True, "scope": "memory"}
    assert svc.deleted == [("u1", "redis")]


def test_memory_erasure_for_user_with_no_memory_is_not_an_error():
    r = _client(_FakeService(found=False)).delete(
        "/internal/gdpr/memory/user/ghost", headers={"X-Internal-Token": settings.gdpr_internal_token},
    )
    assert r.status_code == 200
    assert r.json()["deleted"] is False
