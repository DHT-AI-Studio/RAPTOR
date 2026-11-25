"""assoc_lock.py: keyframe upload (orchestrator) and representative upload (face service)
never write the same video's associated-file list in Module 04 at the same time.

Runs both copies of the module (one per service) against an in-memory Redis stand-in.
"""
# ruff: noqa: F811  (pytest fixtures imported by name are "redefined" as test params by design)
import asyncio
import os
from pathlib import Path

import api_client as orch_api_client  # orchestrator, via conftest's sys.path
import assoc_lock as orch_lock
import pytest
from aiohttp import web
from face_detection_helpers import face  # noqa: F401  (fixture)

_SERVICES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "worker", "services")


class FakeRedis:
    """SET NX EX and the compare-and-delete script, with expiry on the event loop clock."""

    def __init__(self, down=False):
        self.data = {}
        self.down = down

    def _live(self, key):
        item = self.data.get(key)
        if item and item[1] <= asyncio.get_running_loop().time():
            del self.data[key]
            return None
        return item

    async def set(self, key, value, nx=False, ex=None):
        if self.down:
            raise ConnectionError("Connection refused")
        if nx and self._live(key):
            return None
        self.data[key] = (value, asyncio.get_running_loop().time() + ex)
        return True

    async def eval(self, script, numkeys, key, token):
        item = self._live(key)
        if item and item[0] == token:
            del self.data[key]
            return 1
        return 0

    def holder(self, key):
        item = self._live(key)
        return item[0] if item else None


@pytest.fixture(params=["orchestrator", "face"])
def lock(request, face, monkeypatch):
    module = orch_lock if request.param == "orchestrator" else face.assoc_lock
    monkeypatch.setattr(module, "_POLL_SEC", 0.01)
    return module


def test_both_services_carry_the_same_lock():
    """The two copies must stay identical (bar their header line), or the services
    could lock under different keys and the lock would protect nothing."""
    texts = [Path(_SERVICES, svc, "assoc_lock.py").read_text().splitlines()[1:]
             for svc in ("video_orchestrator_service", "video_face_detection_service")]
    assert texts[0] == texts[1]


@pytest.mark.asyncio
async def test_writers_to_the_same_video_take_turns(lock):
    r, timeline = FakeRedis(), []

    async def writer(name):
        async with lock.associated_files_lock("b", "u/videos/clip", "v1", client=r) as held:
            timeline.append((name, "in", held))
            await asyncio.sleep(0.05)               # the add-associated-files round trip
            timeline.append((name, "out", held))

    await asyncio.gather(writer("keyframes"), writer("faces"))
    assert [(step, held) for _, step, held in timeline] == [("in", True), ("out", True), ("in", True), ("out", True)]
    assert timeline[0][0] == timeline[1][0]          # one finished before the other started
    assert r.data == {}                              # released


@pytest.mark.asyncio
async def test_different_videos_do_not_wait_for_each_other(lock):
    r = FakeRedis()
    async with lock.associated_files_lock("b", "u/videos/a", "v1", client=r, wait=0) as first:
        async with lock.associated_files_lock("b", "u/videos/b", "v1", client=r, wait=0) as second:
            assert first and second


@pytest.mark.asyncio
async def test_gives_up_waiting_and_uploads_anyway(lock):
    r = FakeRedis()
    await r.set(lock.lock_key("b", "p", "v1"), "someone-else", nx=True, ex=60)
    async with lock.associated_files_lock("b", "p", "v1", client=r, wait=0.05) as held:
        assert held is False
    assert r.holder(lock.lock_key("b", "p", "v1")) == "someone-else"   # theirs is left alone


@pytest.mark.asyncio
async def test_redis_down_never_blocks_the_upload(lock):
    entered = False
    async with lock.associated_files_lock("b", "p", "v1", client=FakeRedis(down=True)) as held:
        entered = True
        assert held is False
    assert entered


@pytest.mark.asyncio
async def test_an_expired_lock_is_not_released_from_under_its_new_owner(lock):
    r = FakeRedis()
    key = lock.lock_key("b", "p", "v1")
    async with lock.associated_files_lock("b", "p", "v1", client=r, ttl=0.05):
        await asyncio.sleep(0.1)                     # ours expired mid-upload...
        await r.set(key, "next-writer", nx=True, ex=60)   # ...and the next writer took it
    assert r.holder(key) == "next-writer"


@pytest.mark.asyncio
async def test_a_crashed_holder_frees_the_video_after_the_ttl(lock):
    r = FakeRedis()
    await r.set(lock.lock_key("b", "p", "v1"), "crashed", nx=True, ex=0.05)
    async with lock.associated_files_lock("b", "p", "v1", client=r, wait=1) as held:
        assert held


# ── both upload paths hold the lock for the whole Module 04 request ──────────

async def _module04(r, key):
    """add-associated-files stub that records whether the lock was held during the request."""
    seen = []

    async def add_associated_files(request):
        await request.read()
        seen.append(r.holder(key))
        return web.json_response({"associated_filenames": [["face_a.jpg", "c1"]]})

    app = web.Application()
    app.router.add_post("/add-associated-files/{asset_path:.*}", add_associated_files)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", seen


@pytest.mark.asyncio
async def test_representative_upload_holds_the_lock(face, monkeypatch):
    r = FakeRedis()
    key = face.assoc_lock.lock_key("b1", "u1/videos/clip", "v1")
    runner, url, seen = await _module04(r, key)
    monkeypatch.setattr(face.assoc_lock, "_client", r)
    monkeypatch.setattr(face.asset_client, "ASSET_MANAGEMENT_URL", url)
    try:
        await face.asset_client.upload_representatives("u1", "b1", "u1/videos/clip", "v1", [("face_a.jpg", b"\xff\xd8")])
    finally:
        await runner.cleanup()
    assert len(seen) == 1 and seen[0] is not None    # held while Module 04 handled it
    assert r.holder(key) is None                     # released afterwards


@pytest.mark.asyncio
async def test_keyframe_upload_holds_the_lock(tmp_path, monkeypatch):
    r = FakeRedis()
    key = orch_lock.lock_key("b1", "u1/videos/clip", "v1")
    runner, url, seen = await _module04(r, key)
    monkeypatch.setattr(orch_lock, "_client", r)
    monkeypatch.setattr(orch_api_client, "ASSET_MANAGEMENT_URL", url)
    frame = tmp_path / "moment_0001" / "keyframes" / "kf_0.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"\xff\xd8")
    try:
        await orch_api_client.SeaweedFSClient().register_keyframes("u1", "b1", "u1/videos/clip", "v1", [str(frame)])
    finally:
        await runner.cleanup()
    assert len(seen) == 1 and seen[0] is not None
    assert r.holder(key) is None
