"""ObjectStore.batch_resolve_physical_addresses() and bulk_delete_files_v2()'s use of it
(delfile follow-up): destroy-lifecycle hard delete used to resolve every file's
physical_address with its own blocking (no asyncio.to_thread) LakeFS stat() call -- for an
asset with thousands of associated files, this ran sequentially on the event loop, with no
batching at all, unlike the S3 delete step right after it (already chunked). Groups by
version_id and resolves a whole commit's worth of files with one objects/ls call instead,
same pattern as batch_generate_presigned_urls(); whatever that batch call doesn't resolve
still falls back to a per-file stat(), now off the event loop via asyncio.to_thread.

No live LakeFS/S3: fakes follow the established patterns in test_batch_presign_by_key.py
(_lakefs_session) and test_object_store_transact.py (_FakeSession/_FakeS3Client, lakefs.repository()).
"""
from __future__ import annotations

import pytest
from asset_management.object_store import ObjectStore


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _FakeLakefsSession:
    def __init__(self, listings):
        self.listings = listings
        self.calls: list[tuple[str, str]] = []
        self.urls: list[str] = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        version_id = url.split("/refs/")[1].split("/objects/ls")[0]
        prefix = url.split("prefix=")[1].split("&")[0] if "prefix=" in url else ""
        prefix = prefix.replace("%2F", "/")
        self.calls.append((version_id, prefix))
        results = [o for o in self.listings.get(version_id, []) if o["path"].startswith(prefix)]
        return _FakeResponse({"results": results, "pagination": {"has_more": False}})


def _store(listings):
    s = ObjectStore()
    s._lakefs_session = _FakeLakefsSession(listings)
    return s


@pytest.mark.asyncio
async def test_files_from_the_same_commit_cost_one_call():
    store = _store({
        "v1": [
            {"path": "document/txt/foo/foo.txt", "physical_address": "s3://unused/a"},
            {"path": "document/txt/foo/extra1.txt", "physical_address": "s3://unused/b"},
            {"path": "document/txt/foo/extra2.txt", "physical_address": "s3://unused/c"},
        ],
    })
    items = [("document/txt/foo/foo.txt", "v1"), ("document/txt/foo/extra1.txt", "v1"),
             ("document/txt/foo/extra2.txt", "v1")]
    resolved = await store.batch_resolve_physical_addresses("document/txt/foo", items)
    assert len(store._lakefs_session.calls) == 1
    assert resolved == {
        ("document/txt/foo/foo.txt", "v1"): "a",
        ("document/txt/foo/extra1.txt", "v1"): "b",
        ("document/txt/foo/extra2.txt", "v1"): "c",
    }


@pytest.mark.asyncio
async def test_different_commits_group_separately():
    store = _store({
        "v1": [{"path": "document/txt/foo/foo.txt", "physical_address": "s3://unused/a"}],
        "v-assoc": [{"path": "document/txt/foo/extra1.txt", "physical_address": "s3://unused/b"}],
    })
    items = [("document/txt/foo/foo.txt", "v1"), ("document/txt/foo/extra1.txt", "v-assoc")]
    resolved = await store.batch_resolve_physical_addresses("document/txt/foo", items)
    assert len(store._lakefs_session.calls) == 2
    assert set(resolved) == set(items)


@pytest.mark.asyncio
async def test_a_file_the_listing_does_not_return_is_simply_absent():
    store = _store({"v1": [{"path": "document/txt/foo/foo.txt", "physical_address": "s3://unused/a"}]})
    resolved = await store.batch_resolve_physical_addresses(
        "document/txt/foo", [("document/txt/foo/foo.txt", "v1"), ("document/txt/foo/gone.txt", "v1")])
    assert set(resolved) == {("document/txt/foo/foo.txt", "v1")}


@pytest.mark.asyncio
async def test_empty_input_makes_no_calls():
    store = _store({})
    assert await store.batch_resolve_physical_addresses("document/txt/foo", []) == {}
    assert store._lakefs_session.calls == []


@pytest.mark.asyncio
async def test_never_asks_lakefs_to_presign_the_listing():
    """Regression test: presign=true makes LakeFS overwrite physical_address with a
    presigned URL instead of the raw s3://bucket/key value this function needs to
    hard-delete the real object. Confirmed live against a real deployment: with
    presign=true, the "deleted" object was still downloadable byte-for-byte after a
    "successful" (0 errors) delete -- S3's DeleteObjects doesn't error on a key that
    was never real to begin with."""
    store = _store({
        "v1": [{"path": "document/txt/foo/foo.txt", "physical_address": "s3://unused/a"}],
    })
    await store.batch_resolve_physical_addresses("document/txt/foo", [("document/txt/foo/foo.txt", "v1")])
    assert store._lakefs_session.urls
    for url in store._lakefs_session.urls:
        assert "presign" not in url


@pytest.mark.asyncio
async def test_many_commits_resolve_concurrently_not_one_at_a_time():
    """Sequential resolution (one LakeFS call, wait, next call, wait...) was the original
    bug delfile shipped with: fine when files land in one or two commits, but a destroy
    whose associated files were attached across many small batches pays every commit's
    full network round trip serially. Verifies concurrency directly: a commit whose fake
    response takes longer must not block another commit's response from completing first."""
    order: list[str] = []

    class _SlowThenFastSession(_FakeLakefsSession):
        def get(self, url, timeout=None):
            version_id = url.split("/refs/")[1].split("/objects/ls")[0]
            if version_id == "slow":
                import time
                time.sleep(0.2)
            order.append(version_id)
            return super().get(url, timeout=timeout)

    store = ObjectStore()
    store._lakefs_session = _SlowThenFastSession({
        "slow": [{"path": "document/txt/foo/a.txt", "physical_address": "s3://unused/a"}],
        "fast": [{"path": "document/txt/foo/b.txt", "physical_address": "s3://unused/b"}],
    })
    await store.batch_resolve_physical_addresses(
        "document/txt/foo", [("document/txt/foo/a.txt", "slow"), ("document/txt/foo/b.txt", "fast")])
    # If these ran sequentially in insertion order, "slow" would always finish first
    # (it's resolved before "fast" is even started). Running concurrently, the fast
    # one's call completes first despite being grouped second.
    assert order[0] == "fast"


@pytest.mark.asyncio
async def test_concurrency_is_bounded_by_the_semaphore():
    in_flight = 0
    max_in_flight = 0

    class _TrackingSession(_FakeLakefsSession):
        def get(self, url, timeout=None):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            import time
            time.sleep(0.05)
            in_flight -= 1
            return super().get(url, timeout=timeout)

    listings = {f"v{i}": [{"path": f"document/txt/foo/{i}.txt", "physical_address": f"s3://unused/{i}"}] for i in range(20)}
    store = ObjectStore()
    store._lakefs_session = _TrackingSession(listings)
    items = [(f"document/txt/foo/{i}.txt", f"v{i}") for i in range(20)]

    await store.batch_resolve_physical_addresses("document/txt/foo", items, concurrency=3)
    assert max_in_flight <= 3


# ── bulk_delete_files_v2(): batch first, per-file stat() fallback for the rest ──

class _FakeStat:
    def __init__(self, physical_address):
        self.physical_address = physical_address


class _FakeObject:
    def __init__(self, path, physical_address):
        self._physical_address = physical_address
        self.path = path

    def stat(self):
        return _FakeStat(self._physical_address)


class _FakeRef:
    def __init__(self, objects_by_path):
        self._objects_by_path = objects_by_path

    def object(self, path):
        return _FakeObject(path, self._objects_by_path[path])


class _FakeRepo:
    def __init__(self, objects_by_path):
        self._objects_by_path = objects_by_path
        self.stat_calls: list[str] = []

    def ref(self, version_id):
        return _FakeRefTracking(self._objects_by_path, self.stat_calls)


class _FakeRefTracking(_FakeRef):
    def __init__(self, objects_by_path, stat_calls):
        super().__init__(objects_by_path)
        self._stat_calls = stat_calls

    def object(self, path):
        self._stat_calls.append(path)
        return super().object(path)


class _FakeS3Client:
    def __init__(self):
        self.delete_calls: list[list[str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def delete_objects(self, Bucket, Delete):
        keys = [o["Key"] for o in Delete["Objects"]]
        self.delete_calls.append(keys)
        return {"Deleted": [{"Key": k} for k in keys], "Errors": []}


class _FakeSession:
    def __init__(self, client):
        self._client = client

    def client(self, service_name, **kwargs):
        return self._client


@pytest.mark.asyncio
async def test_bulk_delete_resolves_via_batch_and_skips_the_stat_fallback_for_those(monkeypatch):
    import asset_management.object_store as object_store_module

    s = ObjectStore()
    s._lakefs_session = _FakeLakefsSession({
        "v1": [
            {"path": "document/txt/foo/foo.txt", "physical_address": "s3://unused/a"},
            {"path": "document/txt/foo/extra1.txt", "physical_address": "s3://unused/b"},
        ],
    })
    fake_repo = _FakeRepo({})  # the stat() fallback must never be reached
    monkeypatch.setattr(object_store_module.lakefs, "repository", lambda **kw: fake_repo)
    fake_s3 = _FakeS3Client()
    s.session = _FakeSession(fake_s3)

    result = await s.bulk_delete_files_v2(
        [("document/txt/foo/foo.txt", "v1"), ("document/txt/foo/extra1.txt", "v1")],
        asset_path="document/txt/foo",
    )
    assert result == {"deleted": 2, "errors": 0}
    assert fake_repo.stat_calls == []
    assert sorted(fake_s3.delete_calls[0]) == ["a", "b"]


@pytest.mark.asyncio
async def test_bulk_delete_falls_back_to_stat_for_whatever_the_batch_missed(monkeypatch):
    import asset_management.object_store as object_store_module

    s = ObjectStore()
    # the batch listing only has "foo.txt" -- "extra1.txt" (e.g. a different, unlisted
    # commit) must fall back to a per-file stat()
    s._lakefs_session = _FakeLakefsSession({
        "v1": [{"path": "document/txt/foo/foo.txt", "physical_address": "s3://unused/a"}],
    })
    fake_repo = _FakeRepo({"document/txt/foo/extra1.txt": "s3://unused/b"})
    monkeypatch.setattr(object_store_module.lakefs, "repository", lambda **kw: fake_repo)
    fake_s3 = _FakeS3Client()
    s.session = _FakeSession(fake_s3)

    result = await s.bulk_delete_files_v2(
        [("document/txt/foo/foo.txt", "v1"), ("document/txt/foo/extra1.txt", "v1")],
        asset_path="document/txt/foo",
    )
    assert result == {"deleted": 2, "errors": 0}
    assert fake_repo.stat_calls == ["document/txt/foo/extra1.txt"]
    assert sorted(fake_s3.delete_calls[0]) == ["a", "b"]


@pytest.mark.asyncio
async def test_no_asset_path_skips_the_batch_call_and_uses_stat_for_everything(monkeypatch):
    """Back-compat for any caller that doesn't have one shared asset_path to pass."""
    import asset_management.object_store as object_store_module

    s = ObjectStore()
    s._lakefs_session = _FakeLakefsSession({})  # must never be called
    fake_repo = _FakeRepo({"a/b.txt": "s3://unused/a"})
    monkeypatch.setattr(object_store_module.lakefs, "repository", lambda **kw: fake_repo)
    fake_s3 = _FakeS3Client()
    s.session = _FakeSession(fake_s3)

    result = await s.bulk_delete_files_v2([("a/b.txt", "v1")])
    assert result == {"deleted": 1, "errors": 0}
    assert s._lakefs_session.calls == []
    assert fake_repo.stat_calls == ["a/b.txt"]
