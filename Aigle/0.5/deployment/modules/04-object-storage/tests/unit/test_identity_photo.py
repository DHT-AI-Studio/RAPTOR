"""AssetManager.upload_identity_photo() -- a person's seed photo (dev plan v1.4 §4.5).

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_identity_photo.py -v
"""
import base64

import pytest
from asset_management.client import AssetManager
from conftest import _FakeDB, _FakeSearchSync
from fastapi import HTTPException

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 64 + b"\xff\xd9"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


class _CapturingObjectStore:
    """Records batch_upload() calls; `unchanged` simulates LakeFS "commit: no changes"
    (identical bytes already on the branch), where the key is left out of "files"."""

    def __init__(self, unchanged: bool = False):
        self.calls = []
        self.unchanged = unchanged

    async def batch_upload(self, objects, branch, commit_message=None):
        self.calls.append({"objects": objects, "branch": branch, "message": commit_message})
        if self.unchanged:
            return {"files": []}
        return {"files": [{"key": o["key"], "checksum": "x", "version_id": "commit-1"} for o in objects]}


class _AuditDB(_FakeDB):
    def __init__(self):
        super().__init__(initial_status="active")
        self.logged = []

    async def log_access(self, *args, **kwargs):
        self.logged.append((args, kwargs))


def _manager(store, db=None):
    return AssetManager(object_store=store, search_sync=_FakeSearchSync(), db=db or _AuditDB())


@pytest.mark.asyncio
async def test_stores_the_photo_at_the_identity_key_in_one_commit():
    store, db = _CapturingObjectStore(), _AuditDB()
    res = await _manager(store, db).upload_identity_photo("u1", "u1_branch", "abc123", JPEG, "me.jpg")

    assert res == {"image_path": "identity/u1_branch/abc123.jpg", "image_version_id": "commit-1"}
    (call,) = store.calls
    assert call["branch"] == "u1_branch"
    assert [(o["key"], o["content_type"]) for o in call["objects"]] == [("identity/u1_branch/abc123.jpg", "image/jpeg")]
    assert call["objects"][0]["content"] == JPEG
    (args, _), = db.logged
    assert args[:6] == ("u1", "identity/u1_branch/abc123.jpg", "commit-1", "u1_branch", "Upload Identity Photo", True)


@pytest.mark.asyncio
async def test_extension_follows_the_content_not_the_filename():
    store = _CapturingObjectStore()
    res = await _manager(store).upload_identity_photo("u1", "b", "abc123", PNG, "photo.jpg")
    assert res["image_path"] == "identity/b/abc123.png"


@pytest.mark.asyncio
async def test_identical_reupload_falls_back_to_the_branch_ref():
    res = await _manager(_CapturingObjectStore(unchanged=True)).upload_identity_photo("u1", "b", "abc123", JPEG)
    assert res == {"image_path": "identity/b/abc123.jpg", "image_version_id": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_id,data,code", [
    ("../other/x", JPEG, 400),            # never a path the caller steers
    ("a/b", JPEG, 400),
    ("", JPEG, 400),
    ("abc123", b"", 400),
    ("abc123", b"%PDF-1.4 not a photo", 415),
    ("abc123", b"\xff\xd8\xff" + b"\x00" * (10 * 1024 * 1024), 413),
])
async def test_rejects_bad_requests_without_uploading(identity_id, data, code):
    store = _CapturingObjectStore()
    with pytest.raises(HTTPException) as exc:
        await _manager(store).upload_identity_photo("u1", "b", identity_id, data, "x")
    assert exc.value.status_code == code
    assert store.calls == []
