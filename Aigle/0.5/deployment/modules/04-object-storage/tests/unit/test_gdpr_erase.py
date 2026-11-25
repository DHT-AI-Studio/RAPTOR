"""GD-1 / VIE01-246: gdpr_erase_user() purges every asset belonging to a
branch (= user_id). This now routes storage deletion through
bulk_delete_files_v2() (one batched call per asset, primary + associated
files together) instead of one delete_file_v2() per file -- same fix as
delfile's bulk_delete_files_v2()/batch_resolve_physical_addresses() batching.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_gdpr_erase.py -v
"""
import pytest
from asset_management.client import AssetManager
from asset_management.models import AssetMetadata
from conftest import _base_metadata_kwargs


class _FakeGdprDB:
    def __init__(self, assets):
        self.assets = assets
        self.deleted = []
        self.log_calls = []

    async def get_assets_by_branch(self, branch):
        return self.assets

    async def get_head_version(self, asset_path, branch):
        for a in self.assets:
            if a.asset_path == asset_path:
                return a.version_id
        return None

    async def delete_metadata(self, asset_path, version_id, branch):
        self.deleted.append((asset_path, version_id, branch))

    async def log_access(self, *args, **kwargs):
        self.log_calls.append(args)


class _FakeGdprObjectStore:
    def __init__(self, raise_on_delete=False):
        self.raise_on_delete = raise_on_delete
        self.bulk_delete_calls = []

    async def bulk_delete_files_v2(self, files, is_head=False, branch=None, asset_path=None):
        self.bulk_delete_calls.append({
            "files": files, "is_head": is_head, "branch": branch, "asset_path": asset_path,
        })
        if self.raise_on_delete:
            raise RuntimeError("storage delete failed")


class _FakeGdprSearchSync:
    def __init__(self):
        self.deleted = []

    async def delete_metadata(self, asset_path, version_id, branch):
        self.deleted.append((asset_path, version_id, branch))


def _asset(asset_path, version_id, associated_filenames=None):
    return AssetMetadata(**_base_metadata_kwargs(
        asset_path=asset_path, version_id=version_id,
        associated_filenames=associated_filenames or [],
    ))


@pytest.mark.asyncio
async def test_erases_primary_and_associated_files_in_one_bulk_call_per_asset():
    """Primary + every associated file of one asset go through a single
    bulk_delete_files_v2() call, not one delete per file."""
    asset = _asset(
        "document/txt/foo", "v1",
        associated_filenames=[("extra1.txt", "v-assoc-1"), ("extra2.txt", "v-assoc-2")],
    )
    db = _FakeGdprDB([asset])
    obj = _FakeGdprObjectStore()
    mgr = AssetManager(object_store=obj, search_sync=_FakeGdprSearchSync(), db=db)

    deleted_count = await mgr.gdpr_erase_user("branch")

    assert deleted_count == 1
    assert len(obj.bulk_delete_calls) == 1
    call = obj.bulk_delete_calls[0]
    assert set(call["files"]) == {
        ("document/txt/foo/foo.txt", "v1"),
        ("document/txt/foo/extra1.txt", "v-assoc-1"),
        ("document/txt/foo/extra2.txt", "v-assoc-2"),
    }
    assert call["asset_path"] == "document/txt/foo"
    assert call["is_head"] is True
    assert db.deleted == [("document/txt/foo", "v1", "branch")]


@pytest.mark.asyncio
async def test_one_bulk_call_per_asset_not_per_user():
    """Multiple assets for the same user each get their own bulk call --
    batch_resolve_physical_addresses() groups by a single asset_path prefix,
    so assets can't be merged into one call across different asset_paths."""
    assets = [_asset("document/txt/foo", "v1"), _asset("video/mp4/bar", "v2")]
    db = _FakeGdprDB(assets)
    obj = _FakeGdprObjectStore()
    mgr = AssetManager(object_store=obj, search_sync=_FakeGdprSearchSync(), db=db)

    deleted_count = await mgr.gdpr_erase_user("branch")

    assert deleted_count == 2
    assert len(obj.bulk_delete_calls) == 2
    assert {c["asset_path"] for c in obj.bulk_delete_calls} == {"document/txt/foo", "video/mp4/bar"}


@pytest.mark.asyncio
async def test_non_head_asset_version_passes_is_head_false():
    asset = _asset("document/txt/foo", "v-old")

    class _StaleHeadDB(_FakeGdprDB):
        async def get_head_version(self, asset_path, branch):
            return "v-new"  # a newer version is actually head

    db = _StaleHeadDB([asset])
    obj = _FakeGdprObjectStore()
    mgr = AssetManager(object_store=obj, search_sync=_FakeGdprSearchSync(), db=db)

    await mgr.gdpr_erase_user("branch")

    assert obj.bulk_delete_calls[0]["is_head"] is False


@pytest.mark.asyncio
async def test_survives_storage_delete_failure_and_still_erases_metadata():
    """A bulk_delete_files_v2() failure for one asset is caught and logged,
    not re-raised -- erasure of that asset's DB/search metadata (and any
    later assets) still completes, same best-effort contract as before."""
    asset = _asset("document/txt/foo", "v1")
    db = _FakeGdprDB([asset])
    obj = _FakeGdprObjectStore(raise_on_delete=True)
    search_sync = _FakeGdprSearchSync()
    mgr = AssetManager(object_store=obj, search_sync=search_sync, db=db)

    deleted_count = await mgr.gdpr_erase_user("branch")

    assert deleted_count == 1
    assert db.deleted == [("document/txt/foo", "v1", "branch")]
    assert search_sync.deleted == [("document/txt/foo", "v1", "branch")]


@pytest.mark.asyncio
async def test_no_assets_erases_nothing():
    db = _FakeGdprDB([])
    mgr = AssetManager(object_store=_FakeGdprObjectStore(), search_sync=_FakeGdprSearchSync(), db=db)
    assert await mgr.gdpr_erase_user("branch") == 0
