"""AssetManager.upload_files() -- the MTR-1 core upload path: fresh
uploads, dedup (Knowledge Hit / Clone Point), archived-path blocking, and
date validation.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_upload.py -v
"""
import pytest
from asset_management.client import AssetManager
from asset_management.models import AssetMetadata
from asset_management.utils import calculate_md5_bytes
from conftest import _base_metadata_kwargs
from fastapi import HTTPException


class _FakeUploadDB:
    """Configurable fake Database for upload_files() branch coverage."""

    def __init__(self, active_match=None, archived_match=None, path_record=None):
        self._active_match = active_match
        self._archived_match = archived_match
        self._path_record = path_record
        self.saved = []

    async def get_reusable_content(self, checksum, branch, status):
        return self._active_match if status == "active" else self._archived_match

    async def check_path_existence(self, asset_path, checksum, branch):
        return self._path_record

    async def save_metadata(self, metadata):
        self.saved.append(metadata)

    async def log_access(self, *args, **kwargs):
        pass


class _FakeUploadObjectStore:
    def __init__(self, version_id="v-new"):
        self.version_id = version_id

    async def upload_file(self, data, path, mime_type, branch):
        return {"key": path, "version_id": self.version_id, "checksum": calculate_md5_bytes(data)}

    async def delete_associated_files(self, asset_path, filename, branch):
        pass


class _FakeUploadSearchSync:
    def __init__(self, clone_raises=False):
        self.clone_raises = clone_raises
        self.clone_calls = []

    async def clone_point(self, **kwargs):
        self.clone_calls.append(kwargs)
        if self.clone_raises:
            raise RuntimeError("clone failed")


@pytest.mark.asyncio
async def test_upload_files_new_asset_status_is_uploaded():
    """MTR-1: a fresh upload with no dedup match writes status='uploaded', not 'active'."""
    db = _FakeUploadDB()
    mgr = AssetManager(object_store=_FakeUploadObjectStore(version_id="v1"), search_sync=_FakeUploadSearchSync(), db=db)
    result = await mgr.upload_files("tester", "branch", (b"hello", "hello.txt"))
    assert result.status == "uploaded"
    assert result.version_id == "v1"
    assert result.existence_info.exists is False
    assert db.saved and db.saved[0].status == "uploaded"


@pytest.mark.asyncio
async def test_upload_files_knowledge_hit_returns_existing_asset_unchanged():
    """A checksum match against an already-active asset short-circuits --
    the existing (active) record is returned as-is, no new 'uploaded' row.
    get_reusable_content() already filtered on status="active", so this is
    trusted directly -- no check_indexed() round-trip (removed: its
    Chunk-only definition of "indexed" false-negatived on summary-only
    content, e.g. audio with no transcribable speech, spuriously
    re-triggering analysis on every duplicate upload of it)."""
    existing = AssetMetadata(**_base_metadata_kwargs(asset_path="document/txt/existing", version_id="v-old", status="active"))
    db = _FakeUploadDB(active_match=existing)
    mgr = AssetManager(object_store=_FakeUploadObjectStore(), search_sync=_FakeUploadSearchSync(), db=db)
    result = await mgr.upload_files("tester", "branch", (b"hello", "hello.txt"))
    assert result.asset_path == "document/txt/existing"
    assert result.status == "active"
    assert result.existence_info.exists is True


@pytest.mark.asyncio
async def test_upload_files_blocked_when_path_archived_with_same_checksum():
    archived_record = AssetMetadata(**_base_metadata_kwargs(status="archived"))
    db = _FakeUploadDB(path_record=archived_record)
    mgr = AssetManager(object_store=_FakeUploadObjectStore(), search_sync=_FakeUploadSearchSync(), db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.upload_files("tester", "branch", (b"hello", "hello.txt"))
    assert exc_info.value.status_code == 400
    assert "archived" in exc_info.value.detail.lower()


@pytest.mark.parametrize("existing_status", ["uploaded", "processing", "failed", "unregistered"])
@pytest.mark.asyncio
async def test_upload_files_reuses_existing_record_for_non_active_non_archived_states(existing_status):
    """Re-uploading identical content to the same path while the existing
    record is uploaded/processing/failed/unregistered must reuse that
    record (existence_info.exists=True, status unchanged, no new version) --
    not fall through to a real LakeFS write, which always fails with
    "commit: no changes" (nothing changed) and used to surface as a
    misleading 409 ("...has no DB record. Delete the file from LakeFS and
    re-upload.") even though a DB record clearly exists."""
    existing = AssetMetadata(**_base_metadata_kwargs(version_id="v-existing", status=existing_status))
    db = _FakeUploadDB(path_record=existing)
    object_store = _FakeUploadObjectStore()
    mgr = AssetManager(object_store=object_store, search_sync=_FakeUploadSearchSync(), db=db)
    result = await mgr.upload_files("tester", "branch", (b"hello", "hello.txt"))

    assert result.version_id == "v-existing"
    assert result.status == existing_status
    assert result.existence_info.exists is True
    assert "process-file" in result.existence_info.message.lower()
    assert db.saved and db.saved[0].version_id == "v-existing"


@pytest.mark.asyncio
async def test_upload_files_clones_index_from_archived_content_at_different_path():
    """VIE01-214: status is "uploaded" even though a clone happened --
    cloned_from records which source version_id to trust register_asset()'s
    fast path with, instead of the old "unregistered" status abuse (that
    status is reserved for the real Unregister/Register toggle; a
    clone_point() dedup was never actually unregistered)."""
    archived_elsewhere = AssetMetadata(**_base_metadata_kwargs(asset_path="document/txt/other", version_id="v-old", status="archived"))
    db = _FakeUploadDB(archived_match=archived_elsewhere)
    sync = _FakeUploadSearchSync()
    mgr = AssetManager(object_store=_FakeUploadObjectStore(version_id="v-new"), search_sync=sync, db=db)
    result = await mgr.upload_files("tester", "branch", (b"hello", "hello.txt"))
    assert result.existence_info.exists is True
    assert "archived" in result.existence_info.message.lower()
    assert sync.clone_calls
    assert result.status == "uploaded"
    assert result.cloned_from == "v-old"


@pytest.mark.asyncio
async def test_upload_files_clone_failure_reports_exists_false():
    """cloned_from must stay None when clone_point() itself raises --
    register_asset() trusts this column without re-verifying Module 25, so
    a failed clone must never leave it set."""
    archived_elsewhere = AssetMetadata(**_base_metadata_kwargs(asset_path="document/txt/other", version_id="v-old", status="archived"))
    db = _FakeUploadDB(archived_match=archived_elsewhere)
    sync = _FakeUploadSearchSync(clone_raises=True)
    mgr = AssetManager(object_store=_FakeUploadObjectStore(version_id="v-new"), search_sync=sync, db=db)
    result = await mgr.upload_files("tester", "branch", (b"hello", "hello.txt"))
    assert result.existence_info.exists is False
    assert "failed to clone" in result.existence_info.message.lower()
    assert result.status == "uploaded"
    assert result.cloned_from is None


def test_cloned_from_excluded_from_serialization():
    """cloned_from is internal bookkeeping only -- /fileupload and /register
    both return AssetMetadata with no restricted response_model, so the
    only thing keeping it out of a public API response is this exclusion
    on the field itself."""
    metadata = AssetMetadata(**_base_metadata_kwargs(cloned_from="v-source"))
    assert "cloned_from" not in metadata.model_dump()
    assert "cloned_from" not in metadata.model_dump_json()
    assert metadata.cloned_from == "v-source"  # still a normal attribute internally


@pytest.mark.parametrize("archive,destroy,expected_detail_fragment", [
    (None, "1y", "cannot set a destroy date"),
    ("2000-01-01", None, "must be in the future"),
    ("2029-01-01", "2028-01-01", "destroy_date must be later"),
])
@pytest.mark.asyncio
async def test_upload_files_date_validation_errors(archive, destroy, expected_detail_fragment):
    db = _FakeUploadDB()
    mgr = AssetManager(object_store=_FakeUploadObjectStore(), search_sync=_FakeUploadSearchSync(), db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.upload_files("tester", "branch", (b"hello", "hello.txt"), archive_date_or_ttl=archive, destroy_date_or_ttl=destroy)
    assert exc_info.value.status_code == 400
    assert expected_detail_fragment in exc_info.value.detail.lower()
