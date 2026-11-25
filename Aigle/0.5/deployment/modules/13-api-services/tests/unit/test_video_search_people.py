"""Unit tests for video_search.py's People integration.

People runs alongside the 4 RRF retrievers but is not folded into their
pool (see video_search.py's module docstring) -- these tests cover the
pieces specific to that: _people_coro's consent gating,
_people_hits_by_video's reshaping, and _attach_scoped_people's merge/
fallback logic. No FastAPI app/TestClient needed, since none of these
depend on request/response plumbing.
"""
import app.routers.video_search as vs
import pytest
from app.routers.video_search import (
    PersonHit,
    _attach_scoped_people,
    _people_coro,
    _people_hits_by_video,
)
from fastapi import HTTPException

# ---------------------------------------------------------------- _people_hits_by_video


def test_groups_segments_by_video_and_builds_person_hits():
    people_res = {"results": [
        {"version_id": "v1", "identity_id": "i1", "identity_name": "Alice", "trace_id": "t1",
         "first_sec": 10.0, "last_sec": 12.0, "confidence": 0.9,
         "keyframe": {"image_url": "http://x/alice.jpg"},
         "asset_path": "video/mp4/a", "filename": "a.mp4", "asset_url": "http://x/a.mp4"},
    ]}
    hits, meta = _people_hits_by_video(people_res)
    assert list(hits.keys()) == ["v1"]
    [hit] = hits["v1"]
    assert isinstance(hit, PersonHit)
    assert hit.identity_name == "Alice"
    assert hit.start_time == 10.0 and hit.end_time == 12.0
    assert hit.keyframe_url == "http://x/alice.jpg"
    assert meta["v1"] == {"asset_path": "video/mp4/a", "filename": "a.mp4", "asset_url": "http://x/a.mp4"}


def test_two_segments_in_the_same_video_both_land_under_it():
    people_res = {"results": [
        {"version_id": "v1", "identity_id": "i1", "identity_name": "Alice", "trace_id": "t1",
         "first_sec": 10.0, "last_sec": 12.0, "confidence": 0.9},
        {"version_id": "v1", "identity_id": "i1", "identity_name": "Alice", "trace_id": "t2",
         "first_sec": 40.0, "last_sec": 42.0, "confidence": 0.8},
    ]}
    hits, _ = _people_hits_by_video(people_res)
    assert len(hits["v1"]) == 2
    assert {h.trace_id for h in hits["v1"]} == {"t1", "t2"}


def test_a_segment_missing_version_id_is_skipped():
    people_res = {"results": [
        {"identity_id": "i1", "identity_name": "Alice", "trace_id": "t1",
         "first_sec": 10.0, "last_sec": 12.0},
    ]}
    hits, meta = _people_hits_by_video(people_res)
    assert hits == {}
    assert meta == {}


def test_missing_keyframe_leaves_keyframe_url_none():
    people_res = {"results": [
        {"version_id": "v1", "identity_id": "i1", "identity_name": "Alice", "trace_id": "t1",
         "first_sec": 10.0, "last_sec": 12.0},
    ]}
    hits, _ = _people_hits_by_video(people_res)
    assert hits["v1"][0].keyframe_url is None


# ---------------------------------------------------------------- _people_coro


class _FakeSvc:
    def __init__(self, result=None):
        self.result = result if result is not None else {"results": []}
        self.calls: list = []

    async def people_search(self, user_id, body):
        self.calls.append((user_id, body))
        return self.result


class _FakeConsent:
    def __init__(self, granted: bool, unavailable: bool = False):
        self.granted = granted
        self.unavailable = unavailable
        self.calls: list = []

    async def require(self, user_id, purpose):
        self.calls.append((user_id, purpose))
        if self.unavailable:
            raise HTTPException(status_code=503, detail={"error": "CONSENT_CHECK_UNAVAILABLE"})
        if not self.granted:
            raise HTTPException(status_code=403, detail={"error": "PERMISSION_DENIED"})


@pytest.mark.asyncio
async def test_people_coro_calls_people_search_when_consent_is_granted():
    svc = _FakeSvc(result={"results": [{"version_id": "v1"}]})
    consent = _FakeConsent(granted=True)
    out = await _people_coro(svc, consent, "u1", "alice", 50)
    assert out == {"results": [{"version_id": "v1"}]}
    assert consent.calls == [("u1", "face_detection")]
    assert svc.calls == [("u1", {"query": "alice", "limit": 50})]


@pytest.mark.asyncio
async def test_people_coro_returns_empty_results_without_consent_instead_of_raising():
    svc = _FakeSvc()
    consent = _FakeConsent(granted=False)
    out = await _people_coro(svc, consent, "u1", "alice", 50)
    assert out == {"results": []}
    assert svc.calls == []   # never even calls module 25 once consent is denied


@pytest.mark.asyncio
async def test_people_coro_propagates_a_non_403_consent_failure():
    """503 CONSENT_CHECK_UNAVAILABLE (or any other error) is a real failure, not a
    routine "not granted" signal -- must propagate so asyncio.gather's
    return_exceptions=True / the isinstance(Exception) check in video_search sees it,
    same as a failed bm25/vector/graphrag/tkg call."""
    svc = _FakeSvc()
    consent = _FakeConsent(granted=False, unavailable=True)
    with pytest.raises(HTTPException) as exc_info:
        await _people_coro(svc, consent, "u1", "alice", 50)
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------- _attach_scoped_people


class _FakeScopedSvc:
    def __init__(self, result=None, exc: Exception | None = None):
        self.result = result if result is not None else {"results": []}
        self.exc = exc
        self.calls: list = []

    async def people_in_videos(self, user_id, body):
        self.calls.append((user_id, body))
        if self.exc:
            raise self.exc
        return self.result


@pytest.fixture(autouse=True)
def _no_op_image_urls(monkeypatch):
    """_attach_scoped_people calls add_image_urls on the scoped response --
    stub it out, same as the real function does when there's nothing to
    resolve (no keyframe paths in these fixtures)."""
    async def fake(data, storage, user_id):
        return None
    monkeypatch.setattr(vs, "add_image_urls", fake)


@pytest.mark.asyncio
async def test_no_identities_matched_skips_the_scoped_call():
    svc = _FakeScopedSvc()
    people_by_video = {"v1": [PersonHit(identity_id="i1", identity_name="Alice", trace_id="t1",
                                        start_time=0.0, end_time=1.0)]}
    out = await _attach_scoped_people(svc, None, "u1", {"identities": []}, dict(people_by_video), {"v1"})
    assert out == people_by_video
    assert svc.calls == []


@pytest.mark.asyncio
async def test_a_failed_parallel_people_search_skips_the_scoped_call():
    """people_res is the Exception asyncio.gather(return_exceptions=True) gives back
    for a failed call -- treated as "no identities", same as an empty result."""
    svc = _FakeScopedSvc()
    out = await _attach_scoped_people(svc, None, "u1", RuntimeError("boom"), {}, {"v1"})
    assert out == {}
    assert svc.calls == []


@pytest.mark.asyncio
async def test_no_candidate_videos_skips_the_scoped_call():
    svc = _FakeScopedSvc()
    people_res = {"identities": [{"identity_id": "i1", "name": "Alice"}]}
    out = await _attach_scoped_people(svc, None, "u1", people_res, {}, set())
    assert out == {}
    assert svc.calls == []


@pytest.mark.asyncio
async def test_scoped_results_overwrite_the_truncated_unscoped_ones():
    """The real bug: the unscoped call's own limit already dropped v2's people
    entirely; the scoped call, restricted to {v1, v2}, finds both."""
    svc = _FakeScopedSvc(result={"results": [
        {"version_id": "v1", "identity_id": "i1", "identity_name": "Alice", "trace_id": "t1",
         "first_sec": 0.0, "last_sec": 1.0},
        {"version_id": "v2", "identity_id": "i1", "identity_name": "Alice", "trace_id": "t2",
         "first_sec": 5.0, "last_sec": 6.0},
    ]})
    people_res = {"identities": [{"identity_id": "i1", "name": "Alice"}]}
    people_by_video = {"v1": [PersonHit(identity_id="i1", identity_name="Alice", trace_id="old",
                                        start_time=99.0, end_time=100.0)]}
    out = await _attach_scoped_people(svc, None, "u1", people_res, people_by_video, {"v1", "v2"})
    [(called_user, called_body)] = svc.calls
    assert called_user == "u1" and called_body["identity_ids"] == ["i1"]
    assert set(called_body["version_ids"]) == {"v1", "v2"}  # video_ids is a set -- order isn't meaningful
    assert {t.trace_id for t in out["v1"]} == {"t1"}       # scoped result replaces v1's old (truncated) entry
    assert {t.trace_id for t in out["v2"]} == {"t2"}       # v2 now has people at all, unlike before the fix


@pytest.mark.asyncio
async def test_a_failed_scoped_call_falls_back_to_the_unscoped_results():
    svc = _FakeScopedSvc(exc=RuntimeError("module 25 unreachable"))
    people_res = {"identities": [{"identity_id": "i1", "name": "Alice"}]}
    people_by_video = {"v1": [PersonHit(identity_id="i1", identity_name="Alice", trace_id="t1",
                                        start_time=0.0, end_time=1.0)]}
    out = await _attach_scoped_people(svc, None, "u1", people_res, dict(people_by_video), {"v1"})
    assert out == people_by_video  # unchanged, not dropped to empty just because the scoped call failed
