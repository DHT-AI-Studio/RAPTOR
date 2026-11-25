"""
VIE01-215: Register API (Module 04 + 13).

POST /processing/process-file is promoted off include_in_schema=False and
implements the four-branch guard by delegating the status lookup to Module
04's POST /register/{asset_path}/{version_id}: not_found -> 404,
conflict -> 409, "registered" (was unregistered) -> zero Kafka calls,
"needs_processing" (uploaded/failed) -> exactly one Kafka message, status
left unchanged in the response.

Follows test_fileupload_decoupled.py's convention: dependency_overrides
bypasses real JWT, httpx.AsyncClient and the Kafka producer are
faked/poisoned directly on app.state.

Run from the 13-api-services/ directory:
    python -m pytest test_register_processing.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app
from fastapi.testclient import TestClient

FAKE_SUB = "jwt-verified-user-1"


async def _override_current_user():
    return {"sub": FAKE_SUB}


@pytest.fixture(autouse=True)
def _override_auth():
    previous = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = _override_current_user
    yield
    if previous is None:
        app.dependency_overrides.pop(get_current_user, None)
    else:
        app.dependency_overrides[get_current_user] = previous


client = TestClient(app)

_V = get_settings().api_version

FAKE_METADATA = {
    "asset_path": "document/hello",
    "version_id": "v1-fake",
    "primary_filename": "hello.txt",
    "associated_filenames": [],
    "upload_date": "2026-09-16T00:00:00+08:00",
    "archive_date": None,
    "destroy_date": None,
    "status": "uploaded",
}


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://fake")
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=request,
                response=httpx.Response(self.status_code, json=self._json, request=request),
            )

    def json(self):
        return self._json


class _FakeRegisterHTTPClient:
    """Stands in for the httpx.AsyncClient that StorageService.register_asset
    posts through -- module-04's POST /register/{asset_path}/{version_id}.
    Response is controlled per-test via _register_response."""

    response: _FakeResponse = _FakeResponse({"outcome": "needs_processing", "metadata": FAKE_METADATA})

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        pass

    async def post(self, url, **kw):
        assert "/register/" in url, f"unexpected downstream URL: {url}"
        return type(self).response


class _PoisonKafkaProducer:
    async def _poison(self, *a, **kw):
        raise AssertionError("Kafka producer was called when it shouldn't have been")

    send = _poison
    send_and_wait = _poison


class _FakeWorkingKafkaProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, message):
        self.sent.append((topic, message))


@pytest.fixture(autouse=True)
def _fake_register_backend(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _FakeRegisterHTTPClient)
    monkeypatch.setattr(app.state, "http_client", _FakeRegisterHTTPClient(), raising=False)
    # Reset to the default per-test; individual tests override .response.
    _FakeRegisterHTTPClient.response = _FakeResponse({"outcome": "needs_processing", "metadata": FAKE_METADATA})


def _register(asset_path="document/hello", version_id="v1-fake"):
    return client.post(
        f"/api/{_V}/processing/process-file",
        json={"asset_path": asset_path, "version_id": version_id},
        headers={"Authorization": "Bearer fake"},
    )


def test_register_uploaded_asset_publishes_exactly_one_kafka_message(monkeypatch):
    """Integration test (VIE01-215): Upload result status='uploaded' ->
    Register -> exactly one Kafka message, and the response still reflects
    status='uploaded' (Register itself never writes "processing")."""
    fake_producer = _FakeWorkingKafkaProducer()
    monkeypatch.setattr(app.state, "kafka_producer", fake_producer, raising=False)
    _FakeRegisterHTTPClient.response = _FakeResponse(
        {"outcome": "needs_processing", "metadata": FAKE_METADATA}
    )

    resp = _register()

    assert resp.status_code == 200
    assert len(fake_producer.sent) == 1, "expected exactly one Kafka message"
    _topic, message = fake_producer.sent[0]
    assert message["payload"]["parameters"]["asset_path"] == FAKE_METADATA["asset_path"]
    assert message["payload"]["parameters"]["status"] == "uploaded"


def test_register_unregistered_asset_publishes_zero_kafka_and_returns_registered(monkeypatch):
    """"unregistered" fast path: Module 04 already flipped it to "active" and
    called Module 25 -- Register must not touch Kafka at all."""
    monkeypatch.setattr(app.state, "kafka_producer", _PoisonKafkaProducer(), raising=False)
    active_metadata = {**FAKE_METADATA, "status": "active"}
    _FakeRegisterHTTPClient.response = _FakeResponse(
        {"outcome": "registered", "metadata": active_metadata}
    )

    resp = _register()

    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "registered"
    assert body["metadata"]["status"] == "active"


def test_register_not_found_returns_404(monkeypatch):
    monkeypatch.setattr(app.state, "kafka_producer", _PoisonKafkaProducer(), raising=False)
    _FakeRegisterHTTPClient.response = _FakeResponse(
        {"detail": "Asset not found: document/missing/v1"}, status_code=404
    )

    resp = _register(asset_path="document/missing", version_id="v1")

    assert resp.status_code == 404


def test_register_processing_or_active_returns_409(monkeypatch):
    monkeypatch.setattr(app.state, "kafka_producer", _PoisonKafkaProducer(), raising=False)
    _FakeRegisterHTTPClient.response = _FakeResponse(
        {"detail": "Asset is already active"}, status_code=409
    )

    resp = _register()

    assert resp.status_code == 409


def test_process_file_is_documented_not_hidden():
    """VIE01-215: POST /processing/process-file must be promoted off
    include_in_schema=False, same as /fileupload was under VIE01-214."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    path_key = f"/api/{_V}/processing/process-file"
    assert path_key in schema["paths"], f"{path_key} missing from OpenAPI schema"
    op = schema["paths"][path_key]["post"]
    assert op.get("summary"), "process-file has no summary"
    assert op.get("description"), "process-file has no description"
