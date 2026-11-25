"""AssetManager.add_associated_files() (MTR-1's status guard) and
_upload_assoc_batch() (VIE01-308's batch-commit path it's built on).

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_associated_files.py -v
"""
import pytest
from asset_management.client import AssetManager
from asset_management.models import AssetMetadata
from asset_management.utils import calculate_md5_bytes
from conftest import _base_metadata_kwargs, _FakeDB, _FakeSearchSync
from fastapi import HTTPException

# ── add_associated_files() (MTR-1) ────────────────────────────────────────

class _FakeObjectStore:
    async def upload_file(self, data, path, mime_type, branch):
        return {"key": path, "version_id": "fake-assoc-v1", "checksum": "irrelevant"}

    async def batch_upload(self, objects, branch, commit_message=None):
        return {
            "files": [{"key": o["key"], "checksum": "irrelevant", "version_id": "fake-batch-v1"} for o in objects],
        }


@pytest.mark.parametrize("status", ["uploaded", "processing", "active", "failed", "unregistered"])
@pytest.mark.asyncio
async def test_add_associated_files_allows_non_terminal_status(status):
    """MTR-1: add_associated_files() must accept 'uploaded' now, not just 'active'."""
    db = _FakeDB(initial_status=status)
    mgr = AssetManager(object_store=_FakeObjectStore(), search_sync=_FakeSearchSync(), db=db)
    result = await mgr.add_associated_files(
        "tester", "branch", "document/txt/foo",
        associated_files=[(b"data", "extra.txt")],
        primary_version_id="v1",
    )
    assert dict(result.associated_filenames)["extra.txt"] == "fake-batch-v1"


@pytest.mark.parametrize("status", ["archived", "destroyed"])
@pytest.mark.asyncio
async def test_add_associated_files_blocks_terminal_status(status):
    db = _FakeDB(initial_status=status)
    mgr = AssetManager(object_store=_FakeObjectStore(), search_sync=_FakeSearchSync(), db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.add_associated_files(
            "tester", "branch", "document/txt/foo",
            associated_files=[(b"data", "extra.txt")],
            primary_version_id="v1",
        )
    assert exc_info.value.status_code == 400


# ── _upload_assoc_batch() ────────────────────────────────────────────────────
# object_store.py's batch_upload() omits a key from "files" entirely when its
# group turned out byte-identical to what's already on the branch (instead of
# raising) -- omit_keys simulates that; raise_exception simulates a genuine
# object_store failure (network, LakeFS down, ...).

class _FakeBatchObjectStore:
    def __init__(self, raise_exception=None, omit_keys=None):
        self.raise_exception = raise_exception
        self.omit_keys = omit_keys or set()

    async def batch_upload(self, objects, branch, commit_message=None):
        if self.raise_exception is not None:
            raise self.raise_exception
        return {
            "files": [
                {"key": o["key"], "checksum": calculate_md5_bytes(o["content"]), "version_id": "batch-v1"}
                for o in objects if o["key"] not in self.omit_keys
            ]
        }


class _FakeAssocDB:
    def __init__(self, latest_asset=None):
        self._latest = latest_asset

    async def get_latest_asset(self, asset_path, branch):
        return self._latest


@pytest.mark.asyncio
async def test_upload_assoc_batch_success():
    mgr = AssetManager(object_store=_FakeBatchObjectStore(), search_sync="dummy", db=_FakeAssocDB())
    result = await mgr._upload_assoc_batch([(b"data", "extra.txt")], "document/txt/foo", "branch")
    assert result == [("extra.txt", "batch-v1")]


@pytest.mark.asyncio
async def test_upload_assoc_batch_omitted_key_reuses_existing_version():
    """A key omitted from batch_upload()'s "files" (its group turned out
    byte-identical to what's already on the branch) falls back to whatever
    version is already on record for that filename, instead of failing the
    whole associated-files upload."""
    existing = AssetMetadata(**_base_metadata_kwargs(associated_filenames=[("extra.txt", "old-v1")]))
    mgr = AssetManager(
        object_store=_FakeBatchObjectStore(omit_keys={"document/txt/foo/extra.txt"}),
        search_sync="dummy",
        db=_FakeAssocDB(latest_asset=existing),
    )
    result = await mgr._upload_assoc_batch([(b"data", "extra.txt")], "document/txt/foo", "branch")
    assert result == [("extra.txt", "old-v1")]


@pytest.mark.asyncio
async def test_upload_assoc_batch_object_store_failure_returns_empty_list():
    mgr = AssetManager(
        object_store=_FakeBatchObjectStore(raise_exception=Exception("LakeFS unreachable")),
        search_sync="dummy",
        db=_FakeAssocDB(),
    )
    result = await mgr._upload_assoc_batch([(b"data", "extra.txt")], "document/txt/foo", "branch")
    assert result == []


@pytest.mark.asyncio
async def test_upload_assoc_batch_omitted_key_with_no_prior_record_drops_entry():
    """The bug this whole feature fixes: a filename with no prior recorded
    version (never uploaded before) must never come back as (filename, None)
    -- it has to be dropped entirely, or it would silently write a null
    version_id to Postgres."""
    existing = AssetMetadata(**_base_metadata_kwargs(associated_filenames=[]))
    mgr = AssetManager(
        object_store=_FakeBatchObjectStore(omit_keys={"document/txt/foo/brand_new.txt"}),
        search_sync="dummy",
        db=_FakeAssocDB(latest_asset=existing),
    )
    result = await mgr._upload_assoc_batch([(b"data", "brand_new.txt")], "document/txt/foo", "branch")
    assert result == []


@pytest.mark.asyncio
async def test_upload_assoc_batch_empty_input_returns_empty_list():
    mgr = AssetManager(object_store=_FakeBatchObjectStore(), search_sync="dummy", db=_FakeAssocDB())
    result = await mgr._upload_assoc_batch([], "document/txt/foo", "branch")
    assert result == []
