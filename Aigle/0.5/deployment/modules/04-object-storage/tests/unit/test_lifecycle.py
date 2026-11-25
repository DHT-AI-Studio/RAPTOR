"""AssetManager's post-upload lifecycle: update_status() (MTR-3),
destroy_v2(), the auto_archive()/auto_destroy() scheduler batch methods,
and update_expiration().

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_lifecycle.py -v
"""
import pytest
from asset_management.client import AssetManager
from asset_management.models import AssetMetadata
from conftest import _base_metadata_kwargs, _FakeDB, _FakeSearchSync
from fastapi import HTTPException

# ── update_status() (MTR-3) ────────────────────────────────────────────────

@pytest.mark.parametrize("status", ["active", "failed"])
@pytest.mark.asyncio
async def test_update_status_writes_db_and_never_calls_personal_db_set_status(status):
    """Module 25's Chunk/Source records default to status="active" at
    creation (the indexer's own success path) -- there's never a stored
    non-active record for update_status() to flip via _personal_db_set_status,
    for either outcome."""
    db = _FakeDB(initial_status="processing")
    sync = _FakeSearchSync(indexed=True)
    mgr = AssetManager(object_store="dummy", search_sync=sync, db=db)
    written = await mgr.update_status("tester", "branch", "document/txt/foo", "v1", status)
    assert db.metadata.status == status
    assert written == status
    assert sync.personal_db_calls == []


@pytest.mark.asyncio
async def test_update_status_active_trusts_orchestrator_without_checking_indexed():
    """The check_indexed() double-check (VIE01-255) is disabled: it raced
    against Module 25's async PA-8 consumer and downgraded genuinely-
    successful runs to "failed" before the consumer had caught up. Until
    that's fixed to check after the write is confirmed rather than right
    after the Kafka publish, update_status() trusts the orchestrator's
    claim directly -- even when Module 25 would (prematurely) say
    not-indexed."""
    db = _FakeDB(initial_status="processing")

    class _ExplodingCheckSearchSync(_FakeSearchSync):
        async def check_indexed(self, asset_path, version_id, branch, status=None):
            raise AssertionError("check_indexed() is disabled and must not be called")

    mgr = AssetManager(object_store="dummy", search_sync=_ExplodingCheckSearchSync(indexed=False), db=db)
    written = await mgr.update_status("tester", "branch", "document/txt/foo", "v1", "active")
    assert db.metadata.status == "active"
    assert written == "active"


@pytest.mark.asyncio
async def test_update_status_failed_does_not_check_indexed():
    """A "failed" write never needed the Module 25 round trip either."""
    db = _FakeDB(initial_status="processing")

    class _ExplodingCheckSearchSync(_FakeSearchSync):
        async def check_indexed(self, asset_path, version_id, branch, status=None):
            raise AssertionError("check_indexed() should not be called for status=failed")

    mgr = AssetManager(object_store="dummy", search_sync=_ExplodingCheckSearchSync(), db=db)
    await mgr.update_status("tester", "branch", "document/txt/foo", "v1", "failed")
    assert db.metadata.status == "failed"


@pytest.mark.asyncio
async def test_update_status_not_found_raises_404():
    class _EmptyDB(_FakeDB):
        async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
            return None

    mgr = AssetManager(object_store="dummy", search_sync=_FakeSearchSync(), db=_EmptyDB(initial_status="processing"))
    with pytest.raises(HTTPException) as exc_info:
        await mgr.update_status("tester", "branch", "document/txt/foo", "v1", "active")
    assert exc_info.value.status_code == 404


# ── destroy_v2() ──────────────────────────────────────────────────────────

class _FakeDestroyDB:
    def __init__(self, initial_status: str, associated_filenames=None):
        self.metadata = AssetMetadata(**_base_metadata_kwargs(
            status=initial_status, associated_filenames=associated_filenames or [],
        ))
        self.deleted = []

    async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
        return self.metadata

    async def get_head_version(self, asset_path, branch):
        return self.metadata.version_id

    async def delete_metadata(self, asset_path, version_id, branch):
        self.deleted.append((asset_path, version_id, branch))

    async def log_access(self, *args, **kwargs):
        pass


class _FakeNotFoundDB(_FakeDestroyDB):
    async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
        return None


class _FakeDestroyObjectStore:
    def __init__(self, raise_on_delete=False):
        self.raise_on_delete = raise_on_delete
        self.deleted_paths = []
        self.bulk_delete_calls = []

    async def bulk_delete_files_v2(self, files, is_head=False, branch=None, asset_path=None):
        self.bulk_delete_calls.append(files)
        self.deleted_paths.extend(key for key, _ in files)
        if self.raise_on_delete:
            raise RuntimeError("storage delete failed")


class _FakeDestroySearchSync:
    async def delete_metadata(self, asset_path, version_id, branch):
        pass


@pytest.mark.asyncio
async def test_destroy_v2_sets_status_destroyed_and_deletes_row():
    """destroy_v2 doesn't persist status='destroyed' -- it DELETEs the row and
    only sets .status on the in-memory object for the API response (see MTR-1
    commit message: this is why 'destroyed' had to be added to the Literal)."""
    db = _FakeDestroyDB(initial_status="archived")
    mgr = AssetManager(object_store=_FakeDestroyObjectStore(), search_sync=_FakeDestroySearchSync(), db=db)
    result = await mgr.destroy_v2("tester", "branch", "document/txt/foo", "v1")
    assert result.status == "destroyed"
    assert db.deleted == [("document/txt/foo", "v1", "branch")]


@pytest.mark.parametrize("status", ["uploaded", "processing", "active", "failed", "unregistered"])
@pytest.mark.asyncio
async def test_destroy_v2_blocks_non_archived_status(status):
    db = _FakeDestroyDB(initial_status=status)
    mgr = AssetManager(object_store=_FakeDestroyObjectStore(), search_sync=_FakeDestroySearchSync(), db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.destroy_v2("tester", "branch", "document/txt/foo", "v1")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_destroy_v2_not_found_raises_404():
    db = _FakeNotFoundDB(initial_status="archived")
    mgr = AssetManager(object_store=_FakeDestroyObjectStore(), search_sync=_FakeDestroySearchSync(), db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.destroy_v2("tester", "branch", "document/txt/foo", "v1")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_destroy_v2_survives_storage_delete_failure():
    """Primary-file delete failing at the storage layer is caught and logged,
    not re-raised -- destroy still completes."""
    db = _FakeDestroyDB(initial_status="archived")
    obj = _FakeDestroyObjectStore(raise_on_delete=True)
    mgr = AssetManager(object_store=obj, search_sync=_FakeDestroySearchSync(), db=db)
    result = await mgr.destroy_v2("tester", "branch", "document/txt/foo", "v1")
    assert result.status == "destroyed"
    assert obj.deleted_paths


@pytest.mark.asyncio
async def test_destroy_v2_bulk_deletes_primary_and_associated_files_together():
    """Primary file + every associated file go through bulk_delete_files_v2()
    in a single call, not one delete per file -- the whole point of routing
    destroy through the same batching object_store.py's transact/bulk-S3
    helpers already provide."""
    db = _FakeDestroyDB(
        initial_status="archived",
        associated_filenames=[("extra1.txt", "v-assoc-1"), ("extra2.txt", "v-assoc-2")],
    )
    obj = _FakeDestroyObjectStore()
    mgr = AssetManager(object_store=obj, search_sync=_FakeDestroySearchSync(), db=db)
    await mgr.destroy_v2("tester", "branch", "document/txt/foo", "v1")
    assert len(obj.bulk_delete_calls) == 1
    assert set(obj.deleted_paths) == {
        "document/txt/foo/foo.txt",
        "document/txt/foo/extra1.txt",
        "document/txt/foo/extra2.txt",
    }


# ── auto_archive() / auto_destroy() (MTR-1: get_assets_to_archive() fix) ──

class _FakeSchedulerDB:
    """Tracks multiple assets keyed by (asset_path, version_id) -- both
    scheduler methods process a batch, unlike the single-asset fakes above."""

    def __init__(self, assets: list):
        self._by_key = {(a.asset_path, a.version_id): a for a in assets}
        self.to_archive = list(assets)
        self.to_destroy = list(assets)

    async def get_assets_to_archive(self, current_date):
        return self.to_archive

    async def get_assets_to_destroy(self, current_date):
        return self.to_destroy

    async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
        return self._by_key.get((asset_path, version_id))

    async def get_head_version(self, asset_path, branch):
        for (p, v) in self._by_key:
            if p == asset_path:
                return v
        return None

    async def update_status(self, asset_path, version_id, status, branch):
        asset = self._by_key.get((asset_path, version_id))
        if asset:
            asset.status = status

    async def delete_metadata(self, asset_path, version_id, branch):
        self._by_key.pop((asset_path, version_id), None)

    async def log_access(self, *args, **kwargs):
        pass


class _FakeSchedulerObjectStore:
    def __init__(self):
        self.bulk_delete_calls = []

    async def bulk_delete_files_v2(self, files, is_head=False, branch=None, asset_path=None):
        self.bulk_delete_calls.append(files)


class _FakeSchedulerSearchSync:
    async def archive_metadata(self, *args, **kwargs):
        pass

    async def delete_metadata(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_auto_archive_archives_every_candidate():
    assets = [
        AssetMetadata(**_base_metadata_kwargs(asset_path=f"document/txt/asset{i}", version_id=f"v{i}", status="uploaded"))
        for i in range(3)
    ]
    db = _FakeSchedulerDB(assets)
    mgr = AssetManager(object_store=_FakeSchedulerObjectStore(), search_sync=_FakeSchedulerSearchSync(), db=db)
    archived = await mgr.auto_archive()
    assert {a.asset_path for a in archived} == {f"document/txt/asset{i}" for i in range(3)}
    assert all(a.status == "archived" for a in archived)


@pytest.mark.asyncio
async def test_auto_archive_one_failure_does_not_abort_the_batch():
    """Simulates a race: get_assets_to_archive() returned an asset that got
    archived by something else before this per-asset archive() call ran --
    archive()'s own guard rejects it, but the batch must still return the
    other, successfully-archived asset rather than losing everything."""
    good = AssetMetadata(**_base_metadata_kwargs(asset_path="document/txt/good", version_id="v-good", status="uploaded"))
    already_archived = AssetMetadata(**_base_metadata_kwargs(asset_path="document/txt/already", version_id="v-already", status="archived"))
    db = _FakeSchedulerDB([good, already_archived])
    mgr = AssetManager(object_store=_FakeSchedulerObjectStore(), search_sync=_FakeSchedulerSearchSync(), db=db)
    archived = await mgr.auto_archive()
    assert [a.asset_path for a in archived] == ["document/txt/good"]


@pytest.mark.asyncio
async def test_auto_destroy_destroys_every_candidate():
    assets = [
        AssetMetadata(**_base_metadata_kwargs(asset_path=f"document/txt/gone{i}", version_id=f"v{i}", status="archived"))
        for i in range(2)
    ]
    db = _FakeSchedulerDB(assets)
    obj = _FakeSchedulerObjectStore()
    mgr = AssetManager(object_store=obj, search_sync=_FakeSchedulerSearchSync(), db=db)
    destroyed = await mgr.auto_destroy()
    assert {a.asset_path for a in destroyed} == {"document/txt/gone0", "document/txt/gone1"}
    assert all(a.status == "destroyed" for a in destroyed)
    # auto_destroy() must route through destroy_v2() (bulk_delete_files_v2,
    # a real hard delete from SeaweedFS), not the old dead destroy() path
    # (delete_file(), LakeFS-only soft delete that never reclaimed physical
    # storage) -- this is the whole reason auto_destroy() was switched over.
    assert len(obj.bulk_delete_calls) == 2


@pytest.mark.asyncio
async def test_auto_destroy_one_failure_does_not_abort_the_batch():
    good = AssetMetadata(**_base_metadata_kwargs(asset_path="document/txt/good", version_id="v-good", status="archived"))
    not_yet_archived = AssetMetadata(**_base_metadata_kwargs(asset_path="document/txt/notready", version_id="v-nr", status="uploaded"))
    db = _FakeSchedulerDB([good, not_yet_archived])
    mgr = AssetManager(object_store=_FakeSchedulerObjectStore(), search_sync=_FakeSchedulerSearchSync(), db=db)
    destroyed = await mgr.auto_destroy()
    assert [a.asset_path for a in destroyed] == ["document/txt/good"]


# ── update_expiration() ────────────────────────────────────────────────────

class _FakeExpirationDB:
    def __init__(self, initial_status: str):
        self.metadata = AssetMetadata(**_base_metadata_kwargs(status=initial_status))
        self.update_calls = []

    async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
        return self.metadata

    async def update_asset_expiration(self, **kwargs):
        self.update_calls.append(kwargs)
        return True

    async def log_access(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_update_expiration_blocks_archived_asset():
    """Archived assets can't be reactivated by sneaking in new dates."""
    db = _FakeExpirationDB(initial_status="archived")
    mgr = AssetManager(object_store="dummy", search_sync="dummy", db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.update_expiration("tester", "branch", "document/txt/foo", "v1", "30d", "1y")
    assert exc_info.value.status_code == 400
    assert "cannot be reactivated" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_update_expiration_succeeds_and_sets_new_dates():
    db = _FakeExpirationDB(initial_status="uploaded")
    mgr = AssetManager(object_store="dummy", search_sync="dummy", db=db)
    result = await mgr.update_expiration("tester", "branch", "document/txt/foo", "v1", "30d", "1y")
    assert result.archive_date is not None
    assert result.destroy_date is not None
    assert result.destroy_date > result.archive_date
    # the DB write itself must carry the same dates the caller gets back, not just "something"
    assert len(db.update_calls) == 1
    assert db.update_calls[0]["new_archive_date"] == result.archive_date
    assert db.update_calls[0]["new_destroy_date"] == result.destroy_date


@pytest.mark.asyncio
async def test_update_expiration_rejects_destroy_date_without_archive_date():
    db = _FakeExpirationDB(initial_status="uploaded")
    mgr = AssetManager(object_store="dummy", search_sync="dummy", db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.update_expiration("tester", "branch", "document/txt/foo", "v1", None, "30d")
    assert exc_info.value.status_code == 400
    assert "permanent active" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_update_expiration_rejects_no_change_on_both():
    db = _FakeExpirationDB(initial_status="uploaded")
    mgr = AssetManager(object_store="dummy", search_sync="dummy", db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.update_expiration("tester", "branch", "document/txt/foo", "v1", "no_change", "no_change")
    assert exc_info.value.status_code == 400
    assert "no changes specified" in exc_info.value.detail.lower()
