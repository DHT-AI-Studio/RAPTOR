"""
MTR-8 / VIE01-307: Scan API -- unit coverage for AssetManager.scan_assets().

Pure unit tests, no real Postgres -- the fake DB just records what it's
called with. The real SQL lookup (known duplicates, an asset mid-
"processing", unknowns) is covered live in test_scan_mtr8.py, which needs
real rows to mean anything.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_scan.py -v
"""
import pytest
from asset_management.client import AssetManager
from fastapi import HTTPException


class _FakeScanDB:
    def __init__(self):
        self.calls = []

    async def scan_assets(self, files, branch, concurrency=10):
        self.calls.append((files, branch, concurrency))
        return [{"path": p, "checksum": c, "status": "not_found"} for p, c in files]


def _mgr(db=None):
    return AssetManager(object_store="dummy", search_sync="dummy", db=db or _FakeScanDB())


# ── empty batch -> empty result, not an error ───────────────────────────────

@pytest.mark.asyncio
async def test_empty_batch_returns_empty_result():
    db = _FakeScanDB()
    mgr = _mgr(db)
    result = await mgr.scan_assets(files=[], branch="branch")
    assert result == []


# ── malformed checksum -> 400, whole batch rejected ─────────────────────────

@pytest.mark.parametrize("bad_checksum", [
    "",
    "not-hex",
    "deadbeef",  # too short to be MD5 (8 chars, not 32)
    "sha256:",  # prefix with nothing after it
    "md5:zz" + "0" * 30,  # right length, non-hex chars
])
@pytest.mark.asyncio
async def test_malformed_checksum_raises_400(bad_checksum):
    db = _FakeScanDB()
    mgr = _mgr(db)
    with pytest.raises(HTTPException) as exc_info:
        await mgr.scan_assets(files=[("doc.txt", bad_checksum)], branch="branch")
    assert exc_info.value.status_code == 400
    assert db.calls == [], "must validate before ever touching the DB"


@pytest.mark.asyncio
async def test_one_malformed_checksum_rejects_the_whole_batch():
    """A batch with 2 valid checksums and 1 bad one is entirely rejected --
    not a 200 with a per-file error mixed into the 6-state vocabulary."""
    db = _FakeScanDB()
    mgr = _mgr(db)
    valid = "d" * 32
    with pytest.raises(HTTPException) as exc_info:
        await mgr.scan_assets(
            files=[("a.txt", valid), ("b.txt", "garbage"), ("c.txt", valid)],
            branch="branch",
        )
    assert exc_info.value.status_code == 400
    assert db.calls == []


# ── checksum normalization: optional algo: prefix stripped, lowercased ─────

@pytest.mark.asyncio
async def test_sha256_prefix_is_stripped_and_lowercased_before_db_call():
    db = _FakeScanDB()
    mgr = _mgr(db)
    mixed_case_hex = "AABBCCDDEEFF00112233445566778899"[:32]
    await mgr.scan_assets(files=[("a.mp4", f"sha256:{mixed_case_hex}")], branch="branch")

    assert len(db.calls) == 1
    files_passed, branch_passed, concurrency_passed = db.calls[0]
    assert files_passed == [("a.mp4", mixed_case_hex.lower())]
    assert branch_passed == "branch"
    assert concurrency_passed == 10


@pytest.mark.asyncio
async def test_plain_hex_without_prefix_also_accepted():
    db = _FakeScanDB()
    mgr = _mgr(db)
    hex_checksum = "0" * 32
    await mgr.scan_assets(files=[("a.mp4", hex_checksum)], branch="branch")
    assert db.calls[0][0] == [("a.mp4", hex_checksum)]


# ── concurrency cap is forwarded (configurable, default 10) ────────────────

@pytest.mark.asyncio
async def test_custom_concurrency_is_forwarded_to_db():
    db = _FakeScanDB()
    mgr = _mgr(db)
    await mgr.scan_assets(files=[("a.mp4", "0" * 32)], branch="branch", concurrency=3)
    assert db.calls[0][2] == 3


# ── result pass-through ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_results_from_db_are_returned_unchanged():
    class _DBWithMatches:
        async def scan_assets(self, files, branch, concurrency=10):
            return [
                {"path": "a.mp4", "checksum": files[0][1], "status": "active",
                 "existing_asset_path": "video/mp4/a", "existing_version_id": "v1"},
                {"path": "b.mp4", "checksum": files[1][1], "status": "not_found"},
            ]

    mgr = _mgr(_DBWithMatches())
    result = await mgr.scan_assets(
        files=[("a.mp4", "a" * 32), ("b.mp4", "b" * 32)], branch="branch",
    )
    assert result == [
        {"path": "a.mp4", "checksum": "a" * 32, "status": "active",
         "existing_asset_path": "video/mp4/a", "existing_version_id": "v1"},
        {"path": "b.mp4", "checksum": "b" * 32, "status": "not_found"},
    ]
