"""CM-2: an image's moderation verdict in Module 04's asset metadata --
POST /update-moderation (called by Module 10's image orchestrator), the
moderation_flags field on metadata responses, and dedup clones keeping the
source version's verdict.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_moderation_flags.py -v
"""
import json

import pytest
from asset_management import endpoints
from asset_management.client import AssetManager
from asset_management.database import Database
from asset_management.models import (
    AssetMetadata,
    AssetMetadataResponse,
    model_to_response,
)
from conftest import _base_metadata_kwargs
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_upload import _FakeUploadDB, _FakeUploadObjectStore, _FakeUploadSearchSync

FLAGS = {"verdict": "unsafe", "model": "shieldgemma-2",
         "policies": {"dangerous": 0.01, "sexual": 0.97, "violence": 0.02},
         "latency_ms": 812.4, "checked_at": "2026-11-03T04:05:06.123456+00:00"}
HEADERS = {"X-User-ID": "u1", "X-Branch-ID": "b1"}


# ── POST /update-moderation ───────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    calls = []

    async def update_moderation(username, branch, asset_path, version_id, flags):
        calls.append((username, branch, asset_path, version_id, flags))
    monkeypatch.setattr(endpoints.manager, "update_moderation", update_moderation)
    # No `with`: the lifespan (Postgres, scheduler) isn't needed for this route.
    test_client = TestClient(endpoints.app)
    test_client.calls = calls
    return test_client


def test_update_moderation_stores_the_verdict(client):
    resp = client.post("/update-moderation/image/jpg/harbour/v7", json=FLAGS, headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json() == {"asset_path": "image/jpg/harbour", "version_id": "v7", "moderation_flags": FLAGS}
    assert client.calls == [("u1", "b1", "image/jpg/harbour", "v7", FLAGS)]


def test_update_moderation_keeps_model_specific_detail(client):
    body = {"verdict": "error", "model": "llama-guard-3-vision", "error": "HTTP 503: unavailable",
            "latency_ms": 20.0, "checked_at": "2026-11-03T04:05:06+00:00"}

    resp = client.post("/update-moderation/image/jpg/harbour/v7", json=body, headers=HEADERS)

    assert resp.status_code == 200
    assert client.calls[0][4]["error"] == "HTTP 503: unavailable"


@pytest.mark.parametrize("change", [
    {"verdict": "skipped"},
    {"verdict": None},
    {"model": None},
    {"latency_ms": "fast"},
    {"checked_at": "yesterday"},
])
def test_update_moderation_rejects_a_malformed_verdict(client, change):
    body = {**FLAGS, **change}
    body = {k: v for k, v in body.items() if v is not None}

    resp = client.post("/update-moderation/image/jpg/harbour/v7", json=body, headers=HEADERS)

    assert resp.status_code == 422
    assert client.calls == []


def test_update_moderation_needs_the_caller_headers(client):
    resp = client.post("/update-moderation/image/jpg/harbour/v7", json=FLAGS)

    assert resp.status_code == 422
    assert client.calls == []


# ── AssetManager.update_moderation() ─────────────────────────────────────────


class _FakeModerationDB:
    def __init__(self, found=True):
        self.found = found
        self.calls = []

    async def update_moderation(self, asset_path, version_id, branch, flags):
        self.calls.append((asset_path, version_id, branch, flags))
        return self.found


@pytest.mark.asyncio
async def test_manager_writes_the_flags():
    db = _FakeModerationDB()
    mgr = AssetManager(object_store=None, search_sync=None, db=db)

    await mgr.update_moderation("u1", "b1", "image/jpg/harbour", "v7", FLAGS)

    assert db.calls == [("image/jpg/harbour", "v7", "b1", FLAGS)]


@pytest.mark.asyncio
async def test_manager_404s_an_unknown_asset():
    mgr = AssetManager(object_store=None, search_sync=None, db=_FakeModerationDB(found=False))

    with pytest.raises(HTTPException) as exc:
        await mgr.update_moderation("u1", "b1", "image/jpg/nope", "v7", FLAGS)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_manager_refuses_path_traversal():
    db = _FakeModerationDB()
    mgr = AssetManager(object_store=None, search_sync=None, db=db)

    with pytest.raises(HTTPException) as exc:
        await mgr.update_moderation("u1", "b1", "image/../../etc", "v7", FLAGS)
    assert exc.value.status_code == 400
    assert db.calls == []


# ── reading it back ───────────────────────────────────────────────────────────


def test_metadata_responses_carry_the_flags():
    metadata = AssetMetadata(**_base_metadata_kwargs(moderation_flags=FLAGS))

    response = model_to_response(metadata, AssetMetadataResponse)

    assert response.moderation_flags == FLAGS
    assert json.loads(response.model_dump_json())["moderation_flags"] == FLAGS


def test_an_unmoderated_asset_has_no_flags():
    response = model_to_response(AssetMetadata(**_base_metadata_kwargs()), AssetMetadataResponse)

    assert response.moderation_flags is None


def _row(**overrides):
    from datetime import datetime, timezone
    row = {"asset_path": "image/jpg/harbour", "version_id": "v7", "primary_filename": "harbour.jpg",
           "associated_filenames": "[]", "upload_date": datetime.now(timezone.utc), "archive_date": None,
           "destroy_date": None, "updated_by": "u1", "branch": "b1", "status": "active",
           "checksum": "abc", "cloned_from": None}
    row.update(overrides)
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("stored, expected", [
    (json.dumps(FLAGS), FLAGS),      # asyncpg's JSONB without a codec
    (FLAGS, FLAGS),
    (None, None),
])
async def test_rows_decode_the_jsonb_column(stored, expected):
    metadata = await Database()._row_to_metadata(_row(moderation_flags=stored))

    assert metadata.moderation_flags == expected


@pytest.mark.asyncio
async def test_a_row_without_the_column_reads_as_unmoderated():
    metadata = await Database()._row_to_metadata(_row())

    assert metadata.moderation_flags is None


# ── dedup clone ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cloned_version_keeps_the_source_verdict():
    """Module 25's clone copies the Source's moderation_flags; the new version
    never reaches Module 10, so its Module 04 row takes the same verdict."""
    archived = AssetMetadata(**_base_metadata_kwargs(asset_path="image/jpg/other", version_id="v-old",
                                                     status="archived", moderation_flags=FLAGS))
    db = _FakeUploadDB(archived_match=archived)
    mgr = AssetManager(object_store=_FakeUploadObjectStore(version_id="v-new"),
                       search_sync=_FakeUploadSearchSync(), db=db)

    result = await mgr.upload_files("tester", "branch", (b"pixels", "harbour.jpg"))

    assert result.cloned_from == "v-old"
    assert result.moderation_flags == FLAGS
    assert db.saved[0].moderation_flags == FLAGS


@pytest.mark.asyncio
async def test_a_failed_clone_takes_no_verdict():
    archived = AssetMetadata(**_base_metadata_kwargs(asset_path="image/jpg/other", version_id="v-old",
                                                     status="archived", moderation_flags=FLAGS))
    db = _FakeUploadDB(archived_match=archived)
    mgr = AssetManager(object_store=_FakeUploadObjectStore(version_id="v-new"),
                       search_sync=_FakeUploadSearchSync(clone_raises=True), db=db)

    result = await mgr.upload_files("tester", "branch", (b"pixels", "harbour.jpg"))

    assert result.cloned_from is None
    assert result.moderation_flags is None


@pytest.mark.asyncio
async def test_a_new_upload_has_no_verdict_yet():
    db = _FakeUploadDB()
    mgr = AssetManager(object_store=_FakeUploadObjectStore(version_id="v-new"),
                       search_sync=_FakeUploadSearchSync(), db=db)

    result = await mgr.upload_files("tester", "branch", (b"pixels", "harbour.jpg"))

    assert result.moderation_flags is None
