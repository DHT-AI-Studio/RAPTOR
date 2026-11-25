"""ObjectStore.batch_generate_presigned_urls_by_key() (MTR-24 follow-up): groups by each key's own
parent directory + version_id instead of one caller-supplied asset_path, so callers with keys from
several different assets (or a seed photo under its own identity/ prefix) still get the one-call-
per-directory speedup batch_generate_presigned_urls() already gives a single asset's own files.

No live LakeFS: _lakefs_session is swapped for a fake that serves canned per-commit listings and
records every (version_id, prefix) it was asked to list, same pattern as
test_object_store_transact.py's _FakeSession.
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
    """listings: {version_id: [{"path", "physical_address", "content_type"}, ...]} -- the full
    contents of that commit, as objects/ls?presign=true would return them. Each call is filtered
    by the prefix actually requested and recorded for assertions."""

    def __init__(self, listings):
        self.listings = listings
        self.calls: list[tuple[str, str]] = []

    def get(self, url, timeout=None):
        version_id = url.split("/refs/")[1].split("/objects/ls")[0]
        prefix = url.split("prefix=")[1].split("&")[0] if "prefix=" in url else ""
        prefix = prefix.replace("%2F", "/")
        self.calls.append((version_id, prefix))
        results = [o for o in self.listings.get(version_id, []) if o["path"].startswith(prefix)]
        return _FakeResponse({"results": results, "pagination": {"has_more": False}})


def _store(listings):
    s = ObjectStore()
    s._lakefs_session = _FakeLakefsSession(listings)
    s._redis = None
    return s


@pytest.mark.asyncio
async def test_keys_from_the_same_directory_and_commit_cost_one_call():
    store = _store({
        "v1": [
            {"path": "u/videos/vid1/face_t1.jpg", "physical_address": "https://internal/a?sig=1", "content_type": "image/jpeg"},
            {"path": "u/videos/vid1/face_t2.jpg", "physical_address": "https://internal/b?sig=2", "content_type": "image/jpeg"},
            {"path": "u/videos/vid1/face_t3.jpg", "physical_address": "https://internal/c?sig=3", "content_type": "image/jpeg"},
        ],
    })
    items = [("u/videos/vid1/face_t1.jpg", "v1"), ("u/videos/vid1/face_t2.jpg", "v1"),
             ("u/videos/vid1/face_t3.jpg", "v1")]
    resolved = await store.batch_generate_presigned_urls_by_key(items)
    assert len(store._lakefs_session.calls) == 1                       # one directory, one commit -> one call
    assert set(resolved) == set(items)
    assert resolved[("u/videos/vid1/face_t2.jpg", "v1")]["url"].endswith("/b?sig=2")


@pytest.mark.asyncio
async def test_different_directories_or_commits_group_separately():
    """A people-search response mixing face images from two different videos, plus a seed photo
    under identity/ -- each gets its own group, never silently merged or missed."""
    store = _store({
        "v1": [{"path": "u/videos/vidA/face_t1.jpg", "physical_address": "https://internal/a?sig=1", "content_type": None}],
        "v2": [{"path": "u/videos/vidB/face_t1.jpg", "physical_address": "https://internal/b?sig=2", "content_type": None}],
        "v3": [{"path": "identity/branch1/person1.jpg", "physical_address": "https://internal/c?sig=3", "content_type": None}],
    })
    items = [("u/videos/vidA/face_t1.jpg", "v1"), ("u/videos/vidB/face_t1.jpg", "v2"),
             ("identity/branch1/person1.jpg", "v3")]
    resolved = await store.batch_generate_presigned_urls_by_key(items)
    assert len(store._lakefs_session.calls) == 3
    assert {prefix for _, prefix in store._lakefs_session.calls} == {
        "u/videos/vidA/", "u/videos/vidB/", "identity/branch1/"}
    assert set(resolved) == set(items)


@pytest.mark.asyncio
async def test_a_key_the_listing_does_not_return_is_simply_absent():
    """Best-effort, same contract as batch_generate_presigned_urls(): a key list_objects didn't
    return (wrong commit, deleted, ...) is missing from the result, never raises."""
    store = _store({"v1": [{"path": "u/videos/vid1/face_t1.jpg", "physical_address": "https://internal/a",
                           "content_type": None}]})
    resolved = await store.batch_generate_presigned_urls_by_key(
        [("u/videos/vid1/face_t1.jpg", "v1"), ("u/videos/vid1/face_missing.jpg", "v1")])
    assert set(resolved) == {("u/videos/vid1/face_t1.jpg", "v1")}


@pytest.mark.asyncio
async def test_empty_input_makes_no_calls():
    store = _store({})
    assert await store.batch_generate_presigned_urls_by_key([]) == {}
    assert store._lakefs_session.calls == []


@pytest.mark.asyncio
async def test_a_root_level_key_with_no_directory_lists_with_no_prefix():
    store = _store({"v1": [{"path": "root.jpg", "physical_address": "https://internal/r", "content_type": None}]})
    resolved = await store.batch_generate_presigned_urls_by_key([("root.jpg", "v1")])
    assert store._lakefs_session.calls == [("v1", "")]
    assert resolved[("root.jpg", "v1")]["url"].endswith("/r")
