"""StorageService.add_associated_files() must forward to module 04's real
endpoint signature: POST /add-associated-files/{asset_path} with
primary_version_id as form data, not a second path segment. The old code
appended it to the URL (f"/add-associated-files/{asset_path}/{version_id}"),
which module 04's `{asset_path:path}` catch-all route silently absorbed into
asset_path itself, breaking every call through the gateway with a 404
("Asset not found for {asset_path}/{version_id}/{version_id}") -- confirmed
live against a real deployment, not just locally. primary_version_id is
deliberately optional on module 04's side (falls back to the latest active
version), which is exactly why it can't be a path segment like every other
asset endpoint's version_id.

Direct unit test of StorageService, not through TestClient/app -- the full
app's middleware stack has a pre-existing, unrelated failure in this shared
test env (same AttributeError on test_register_processing.py, which this
change never touched).

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_add_associated_files_url.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from app.core.config import Settings
from app.services.storage_service import StorageService, User


class _FakeUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str):
        self.filename = filename
        self.content_type = content_type
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json = json_body
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _CapturingHTTPClient:
    def __init__(self):
        self.calls = []

    async def post(self, url, **kw):
        self.calls.append((url, kw.get("data", {})))
        return _FakeResponse({"asset_path": "document/txt/foo", "version_id": "v-primary-123"})


@pytest.fixture
def service():
    client = _CapturingHTTPClient()
    settings = Settings(asset_management_url="http://raptor-asset-management:8000")
    return StorageService(client=client, settings=settings), client


@pytest.mark.asyncio
async def test_version_id_is_sent_as_form_data_not_a_second_path_segment(service):
    svc, client = service
    asset_path = "document/txt/foo"
    version_id = "v-primary-123"

    await svc.add_associated_files(
        asset_path=asset_path,
        associated_files=[_FakeUploadFile("extra.txt", b"extra content", "text/plain")],
        user=User(user_id="u1", branch_id="u1"),
        primary_version_id=version_id,
    )

    assert len(client.calls) == 1
    url, data = client.calls[0]
    assert url == "http://raptor-asset-management:8000/add-associated-files/document%2Ftxt%2Ffoo"
    assert version_id not in url
    assert data["primary_version_id"] == version_id


@pytest.mark.asyncio
async def test_no_primary_version_id_does_not_crash_on_quote_none(service):
    """primary_version_id is Optional at both the router and module-04's own
    endpoint -- the old code's quote(primary_version_id, safe='') would raise
    TypeError on quote(None, ...) before ever reaching module 04."""
    svc, client = service
    asset_path = "document/txt/foo"

    await svc.add_associated_files(
        asset_path=asset_path,
        associated_files=[_FakeUploadFile("extra.txt", b"extra content", "text/plain")],
        user=User(user_id="u1", branch_id="u1"),
    )

    url, data = client.calls[0]
    assert url == "http://raptor-asset-management:8000/add-associated-files/document%2Ftxt%2Ffoo"
    assert "primary_version_id" not in data
