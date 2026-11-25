"""AssetManager's own I/O-free helpers (path/filename sanitization),
AssetStatus Literal validation, and archive()'s/unregister_asset()'s/
register_asset()'s status-guard behavior.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_asset_manager_core.py -v
"""
import pytest
from asset_management.client import AssetManager
from asset_management.models import AssetMetadata
from conftest import _base_metadata_kwargs, _FakeDB, _FakeSearchSync
from fastapi import HTTPException
from pydantic import ValidationError

# ── AssetManager._sanitize_path / _sanitize_filename ─────────────────────
# Dummy non-None args bypass the real ObjectStore/Database/SearchStore
# connections in __init__ — these two methods touch no I/O.

_mgr = AssetManager(object_store="dummy", search_sync="dummy", db="dummy")


@pytest.mark.parametrize("path,expected", [
    ("a/b/c", "a/b/c"),
    ("/a/b/", "a/b"),
    ("a//b", "a/b"),
    ("a\\b", "a/b"),
])
def test_sanitize_path_normalizes(path, expected):
    assert _mgr._sanitize_path(path) == expected


def test_sanitize_path_empty_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        _mgr._sanitize_path("")
    assert exc_info.value.status_code == 400


def test_sanitize_path_traversal_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        _mgr._sanitize_path("a/../../etc/passwd")
    assert exc_info.value.status_code == 400
    assert "traversal" in exc_info.value.detail.lower()


@pytest.mark.parametrize("filename,expected", [
    ("my file.txt", "my_file.txt"),
    ("../../etc/passwd", "passwd"),
    ("report(final).pdf", "report_final_.pdf"),
])
def test_sanitize_filename_normalizes(filename, expected):
    assert _mgr._sanitize_filename(filename) == expected


def test_sanitize_filename_empty_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        _mgr._sanitize_filename("")
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("filename", [".", "..", "%2e%2e"])
def test_sanitize_filename_dot_variants_raise_400(filename):
    with pytest.raises(HTTPException) as exc_info:
        _mgr._sanitize_filename(filename)
    assert exc_info.value.status_code == 400


# ── AssetStatus Literal validation (MTR-1) ───────────────────────────────

@pytest.mark.parametrize("status", ["uploaded", "processing", "active", "failed", "unregistered", "archived", "destroyed"])
def test_asset_status_accepts_valid_values(status):
    metadata = AssetMetadata(**_base_metadata_kwargs(status=status))
    assert metadata.status == status


def test_asset_status_rejects_invalid_value():
    with pytest.raises(ValidationError):
        AssetMetadata(**_base_metadata_kwargs(status="not_a_real_status"))


# ── AssetManager guard-clause behavior (MTR-1 regression coverage) ───────

@pytest.mark.parametrize("status", ["uploaded", "processing", "active", "failed", "unregistered"])
@pytest.mark.asyncio
async def test_archive_allows_non_terminal_status(status):
    """MTR-1 / MTR-6 (VIE01-216): archive() must succeed from every
    pre-archive state -- all 5 non-terminal statuses, not just 'uploaded'
    and 'active'."""
    db = _FakeDB(initial_status=status)
    mgr = AssetManager(object_store="dummy", search_sync=_FakeSearchSync(), db=db)
    result = await mgr.archive("tester", "branch", "document/txt/foo", "v1")
    assert result.status == "archived"


@pytest.mark.parametrize("status", ["archived", "destroyed"])
@pytest.mark.asyncio
async def test_archive_blocks_terminal_status(status):
    """MTR-6: 'destroyed' stays blocked alongside 'archived' -- destroy()
    physically deletes the underlying files, so there is nothing left to
    archive. Not one of MTR-6's 5 pre-archive states."""
    db = _FakeDB(initial_status=status)
    mgr = AssetManager(object_store="dummy", search_sync=_FakeSearchSync(), db=db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.archive("tester", "branch", "document/txt/foo", "v1")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_register_returns_conflict_on_archived_asset():
    """MTR-6 (VIE01-216): Register must return a conflict outcome (mapped to
    409 at the HTTP layer, see endpoints.py's /register) on an archived
    asset, same as it already does for processing/active/destroyed."""
    db = _FakeDB(initial_status="archived")
    mgr = AssetManager(object_store="dummy", search_sync=_FakeSearchSync(), db=db)
    result = await mgr.register_asset(branch="branch", asset_path="document/txt/foo", version_id="v1")
    assert result == {"outcome": "conflict", "status": "archived"}


# ── unregister_asset() / register_asset() fast path (MTR-5 / VIE01-305) ────

@pytest.mark.asyncio
async def test_unregister_active_flips_status_and_calls_only_set_status():
    """The ticket's own negative requirement: Unregister must never touch
    Module 25's index-delete endpoint, only the status-set one. _FakeSearchSync
    has no delete_metadata/_personal_db_delete method at all -- if
    unregister_asset ever called one, this test would fail with an
    AttributeError, not silently pass."""
    db = _FakeDB(initial_status="active")
    sync = _FakeSearchSync()
    mgr = AssetManager(object_store="dummy", search_sync=sync, db=db)

    result = await mgr.unregister_asset("tester", "branch", "document/txt/foo", "v1")

    assert result.status == "unregistered"
    assert db.metadata.status == "unregistered"
    assert sync.personal_db_calls == [("document/txt/foo", "v1", "unregistered", "branch")]


@pytest.mark.parametrize("status", ["uploaded", "processing", "failed", "unregistered", "archived", "destroyed"])
@pytest.mark.asyncio
async def test_unregister_blocks_non_active_status(status):
    """Callable only when status='active'; 409 otherwise -- including from
    'unregistered' itself (not idempotent)."""
    db = _FakeDB(initial_status=status)
    sync = _FakeSearchSync()
    mgr = AssetManager(object_store="dummy", search_sync=sync, db=db)

    with pytest.raises(HTTPException) as exc_info:
        await mgr.unregister_asset("tester", "branch", "document/txt/foo", "v1")
    assert exc_info.value.status_code == 409
    # Rejected before any Module 25 call was attempted.
    assert sync.personal_db_calls == []


@pytest.mark.asyncio
async def test_unregister_not_found_raises_404():
    class _EmptyDB(_FakeDB):
        async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
            return None

    mgr = AssetManager(object_store="dummy", search_sync=_FakeSearchSync(), db=_EmptyDB(initial_status="active"))
    with pytest.raises(HTTPException) as exc_info:
        await mgr.unregister_asset("tester", "branch", "document/txt/foo", "v1")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_unregister_then_register_round_trip_touches_only_set_status():
    """Full round trip: active -> Unregister -> unregistered -> Register
    fast path -> active. Across both steps, Module 25 only ever sees two
    status-set calls (unregistered, then active) -- never a delete, never
    a Kafka dispatch (register_asset's "registered" outcome is returned
    before any Kafka code path is reached -- see client.py's own docstring
    and 13-api-services/app/routers/processing.py's early return on this
    exact outcome)."""
    db = _FakeDB(initial_status="active")
    sync = _FakeSearchSync()
    mgr = AssetManager(object_store="dummy", search_sync=sync, db=db)

    unregistered = await mgr.unregister_asset("tester", "branch", "document/txt/foo", "v1")
    assert unregistered.status == "unregistered"

    register_result = await mgr.register_asset(branch="branch", asset_path="document/txt/foo", version_id="v1")
    assert register_result["outcome"] == "registered"
    assert register_result["metadata"].status == "active"
    assert db.metadata.status == "active"

    assert sync.personal_db_calls == [
        ("document/txt/foo", "v1", "unregistered", "branch"),
        ("document/txt/foo", "v1", "active", "branch"),
    ]


@pytest.mark.asyncio
async def test_unregister_rejected_on_archived():
    """Ticket's explicit negative-test subtask: Unregister rejected on
    archived -- a real, not-synthetic status to reject, distinct from the
    parametrized sweep above."""
    db = _FakeDB(initial_status="archived")
    sync = _FakeSearchSync()
    mgr = AssetManager(object_store="dummy", search_sync=sync, db=db)

    with pytest.raises(HTTPException) as exc_info:
        await mgr.unregister_asset("tester", "branch", "document/txt/foo", "v1")
    assert exc_info.value.status_code == 409
    assert sync.personal_db_calls == []
