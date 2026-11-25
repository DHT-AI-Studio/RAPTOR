"""GD-2 / VIE01-247: Module 04's export file index
(GET /internal/gdpr/user/{user_id}/files) -- metadata plus a download link
per stored file, never the bytes. Mocked Postgres, LakeFS and S3."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from asset_management import object_store as object_store_module
from asset_management.client import AssetManager
from asset_management.config import settings
from asset_management.models import AssetMetadata
from asset_management.object_store import ObjectStore

UPLOADED = datetime(2026, 10, 1, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))


def _asset(asset_path, primary, version, assoc=(), status="active"):
    return AssetMetadata(
        asset_path=asset_path, version_id=version, primary_filename=primary,
        associated_filenames=list(assoc), upload_date=UPLOADED, user="u1", branch="u1",
        status=status, checksum="0" * 32,
    )


class _Db:
    def __init__(self, assets):
        self.assets = assets
        self.branches = []

    async def get_assets_by_branch(self, branch):
        self.branches.append(branch)
        return self.assets


class _ObjectStore:
    def __init__(self, missing=()):
        self.calls = []
        self.missing = set(missing)

    async def gdpr_file_link(self, key, version_id, expires_in):
        self.calls.append((key, version_id, expires_in))
        if key in self.missing:
            raise RuntimeError("object not found")
        return {"size": 1234, "type": "application/pdf", "download_url": f"https://s3.example/{key}?v={version_id}"}


def _manager(assets, missing=()):
    manager = AssetManager.__new__(AssetManager)
    manager.db = _Db(assets)
    manager.object_store = _ObjectStore(missing)
    return manager


@pytest.mark.asyncio
async def test_lists_primary_and_associated_files_with_links():
    manager = _manager([
        _asset("document/pdf/report", "report.pdf", "v1", assoc=[("appendix.pdf", "v2"), ("notes.txt", "")]),
        _asset("video/mp4/clip", "clip.mp4", "v3", status="archived"),
    ])
    files = await manager.gdpr_list_files("u1", expires_in=259200)

    assert manager.db.branches == ["u1"]
    assert [(f["name"], f["version_id"], f["role"]) for f in files] == [
        ("report.pdf", "v1", "primary"),
        ("appendix.pdf", "v2", "associated"),
        ("notes.txt", "v1", "associated"),  # no own version -> the asset's
        ("clip.mp4", "v3", "primary"),
    ]
    report = files[0]
    assert report == {
        "name": "report.pdf", "asset_path": "document/pdf/report", "version_id": "v1", "role": "primary",
        "status": "active", "uploaded_at": UPLOADED, "size": 1234, "type": "application/pdf",
        "download_url": "https://s3.example/document/pdf/report/report.pdf?v=v1",
    }
    assert files[3]["status"] == "archived"  # every status is the user's data
    assert {expires for _, _, expires in manager.object_store.calls} == {259200}


@pytest.mark.asyncio
async def test_file_missing_from_storage_stays_listed_with_error():
    manager = _manager([_asset("document/pdf/report", "report.pdf", "v1")],
                       missing={"document/pdf/report/report.pdf"})
    [entry] = await manager.gdpr_list_files("u1", expires_in=3600)
    assert entry["name"] == "report.pdf"
    assert (entry["size"], entry["type"], entry["download_url"]) == (None, None, None)
    assert entry["error"] == "file not found in object storage"


@pytest.mark.asyncio
async def test_no_assets():
    assert await _manager([]).gdpr_list_files("u1", expires_in=3600) == []


# ── ObjectStore.gdpr_file_link: own TTL, physical address, filename ─────────

@pytest.mark.asyncio
async def test_file_link_presigns_physical_address_for_the_given_ttl(monkeypatch):
    monkeypatch.setattr(settings, "s3_bucket", "lakefs")
    # An underscore-containing host is exactly the real case this needs to
    # handle (boto3's own client construction refuses one -- see
    # _presign_path_style_get's docstring) -- so the fixture value isn't the
    # tidy "files.example.com" one might otherwise pick.
    monkeypatch.setattr(settings, "s3_public_url", "http://raptor_open_api.example:8333")
    monkeypatch.setattr(settings, "aws_access_key", "AKIATEST")
    monkeypatch.setattr(settings, "aws_secret_key", "secret")
    stat = SimpleNamespace(physical_address="s3://lakefs/data/gabc123/obj456", size_bytes=2048, content_type="video/mp4")
    refs = []

    def fake_repository(repository_id, client):
        return SimpleNamespace(ref=lambda version_id: refs.append(version_id) or SimpleNamespace(
            object=lambda path: refs.append(path) or SimpleNamespace(stat=lambda: stat)))

    monkeypatch.setattr(object_store_module.lakefs, "repository", fake_repository)
    store = ObjectStore.__new__(ObjectStore)
    store.repository, store.client = "repo", object()

    link = await store.gdpr_file_link("video/mp4/clip/影片 1.mp4", "v3", expires_in=259200)

    assert refs == ["v3", "video/mp4/clip/影片 1.mp4"]
    # Signed (SigV4) for the public endpoint itself, directly -- not signed
    # internally and host-rewritten (which would break a SigV4 signature),
    # and never through a real client against it either (this fixture's
    # underscore-containing host is exactly what boto3's own client
    # construction would refuse -- see _presign_path_style_get's docstring).
    url = link.pop("download_url")
    assert link == {"size": 2048, "type": "video/mp4"}
    assert url.startswith("http://raptor_open_api.example:8333/lakefs/data/gabc123/obj456?")
    assert "response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27%25E5%25BD%25B1%25E7%2589%2587%25201.mp4" in url
    assert "X-Amz-Expires=259200" in url
    assert "X-Amz-Signature=" in url
