"""
MTR-1 / VIE01-214: upload must not trigger AI analysis.

Run from the 13-api-services/ directory:
    conda run -n CIE python -m pytest test_fileupload_decoupling.py -v

Same style as test_memory_proxy.py: dependency_overrides bypass real JWT and
the real asset-management HTTP call, and a hand-written capturing fake stands
in for the Kafka producer (no unittest.mock). Storage is faked at the
StorageService level rather than httpx, since asset.py calls svc.upload_file()
directly rather than issuing its own HTTP request.
"""
import os
import sys
from typing import ClassVar

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic", "document": "doc-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import app.routers.asset as asset_module
import pytest
from app.api.dependencies import get_current_user, get_storage_service
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

FAKE_SUB = "jwt-verified-user-42"
ASSET_PREFIX = f"/api/{get_settings().api_version}/asset"


async def _override_current_user():
    return {"sub": FAKE_SUB}


client = TestClient(app)


class _FakeStorageService:
    """Stands in for StorageService — returns a canned upload result instead
    of calling the real asset-management service over HTTP."""

    async def upload_file(self, primary_file, archive_date_or_ttl, destroy_date_or_ttl, user):
        return {
            "asset_path": "document/txt/mtr1_test",
            "version_id": "fake-version-1",
            "primary_filename": primary_file.filename,
            "status": "uploaded",
            "existence_info": {"exists": False},
        }

    async def add_associated_files(self, asset_path, associated_files, user, primary_version_id):
        return {"associated_filenames": []}


class _CapturingKafkaProducer:
    calls: ClassVar[list] = []

    async def send_and_wait(self, topic, message):
        _CapturingKafkaProducer.calls.append((topic, message))


@pytest.fixture(autouse=True)
def _overrides():
    """get_current_user/get_storage_service/get_kafka_producer are process-
    wide keys on the shared `app` singleton -- setting them as bare module-
    level statements (as this file originally did) leaks them into every
    other test file collected in the same pytest session. Scope them to
    this file's tests only, same pattern as test_register_processing.py /
    test_fileupload_decoupled.py."""
    prev_user = app.dependency_overrides.get(get_current_user)
    prev_storage = app.dependency_overrides.get(get_storage_service)
    prev_producer = app.dependency_overrides.get(asset_module.get_kafka_producer)

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_storage_service] = lambda: _FakeStorageService()
    _CapturingKafkaProducer.calls = []
    app.dependency_overrides[asset_module.get_kafka_producer] = lambda: _CapturingKafkaProducer()

    yield

    for dep, prev in (
        (get_current_user, prev_user),
        (get_storage_service, prev_storage),
        (asset_module.get_kafka_producer, prev_producer),
    ):
        if prev is None:
            app.dependency_overrides.pop(dep, None)
        else:
            app.dependency_overrides[dep] = prev


def _upload_file(field_bytes=b"mtr-1 decoupling test"):
    return {"primary_file": ("mtr1.txt", field_bytes, "text/plain")}


def test_fileupload_sends_zero_kafka_messages():
    r = client.post(
        f"{ASSET_PREFIX}/fileupload",
        headers={"Authorization": "Bearer fake"},
        files=_upload_file(),
    )
    assert r.status_code == 200
    assert r.json()["status"] == "uploaded"
    assert _CapturingKafkaProducer.calls == []


def test_fileupload_is_documented_in_openapi():
    schema = client.get("/openapi.json").json()
    assert f"{ASSET_PREFIX}/fileupload" in schema["paths"]
    assert "post" in schema["paths"][f"{ASSET_PREFIX}/fileupload"]


def test_fileupload_analysis_regression_still_sends_kafka():
    r = client.post(
        f"{ASSET_PREFIX}/fileupload_analysis",
        headers={"Authorization": "Bearer fake"},
        files=_upload_file(),
    )
    assert r.status_code == 200
    assert len(_CapturingKafkaProducer.calls) == 1
    _, message = _CapturingKafkaProducer.calls[0]
    assert message["payload"]["parameters"]["primary_filename"] == "mtr1.txt"
    assert "correlation_id" in r.json()["processing_result"]
