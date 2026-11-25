"""Tracking, quality scoring, intra-video clustering and the full per-video
analysis of video_face_detection_service (dev plan v1.4 §4.3 steps 4–8).

Run from the 11-video-processing/ directory:
    FACE_MODEL_DIR=/path/to/models/face pytest tests/unit/ -v
"""
# ruff: noqa: F811  (pytest fixtures imported by name are "redefined" as test params by design)
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from aiohttp import web
from face_detection_helpers import (  # noqa: F401  (face / detector / face_image / second_face_image are fixtures)
    FakeDetector,
    FakeFace,
    detector,
    face,
    face_image,
    requires_ffprobe,
    second_face_image,
)

FPS = 8.0
GREY = np.full((240, 320, 3), 128, dtype=np.uint8)


def _frames(face, n, start=0):
    return [face.sampler.Frame(frame_no=i, timestamp_sec=i / FPS, image=GREY) for i in range(start, start + n)]


def _box(x, y=60, size=50):
    return (x, y, x + size, y + size)


def _track(face, detections_per_frame, shot_changes=()):
    """detections_per_frame[i] is the list of FakeFace on sampled frame i."""
    tracker = face.tracker.FaceTracker(aligner=FakeDetector().align)
    for frame, faces in zip(_frames(face, len(detections_per_frame)), detections_per_frame):
        tracker.update(frame, faces, shot_changed=frame.frame_no in shot_changes)
    return tracker.finish()


# ── tracking ──────────────────────────────────────────────────

def test_moving_face_becomes_one_trace(face):
    traces = _track(face, [[FakeFace(_box(20 + 6 * i))] for i in range(16)])
    assert len(traces) == 1
    assert len(traces[0].frames) == 16
    assert [f.frame_no for f in traces[0].frames] == list(range(16))


def test_two_simultaneous_faces_keep_separate_traces(face):
    # list order flips every frame; the tracker must follow position, not list order
    frames = []
    for i in range(16):
        a, b = FakeFace(_box(20 + 4 * i)), FakeFace(_box(250 - 4 * i, y=120))
        frames.append([a, b] if i % 2 else [b, a])
    traces = _track(face, frames)
    assert len(traces) == 2
    for t in traces:
        xs = [f.bbox[0] for f in t.frames]
        assert xs == sorted(xs) or xs == sorted(xs, reverse=True)   # each trace stays on one face
        assert len(t.frames) == 16


def test_shot_change_ends_every_trace(face):
    traces = _track(face, [[FakeFace(_box(40))] for _ in range(16)], shot_changes={8})
    assert [len(t.frames) for t in traces] == [8, 8]


def test_short_gap_is_bridged(face):
    # face missing for 4 samples (0.5s) -> within the 1.0s tolerance, same trace
    frames = [[FakeFace(_box(40 + 2 * i))] if not 6 <= i < 10 else [] for i in range(16)]
    traces = _track(face, frames)
    assert len(traces) == 1 and len(traces[0].frames) == 12


def test_long_gap_starts_a_new_trace(face):
    # missing for 12 samples (1.5s) -> beyond the tolerance
    frames = [[FakeFace(_box(40))] if not 6 <= i < 18 else [] for i in range(26)]
    traces = _track(face, frames)
    assert [len(t.frames) for t in traces] == [6, 8]


def test_low_confidence_detection_extends_but_never_starts_a_trace(face):
    # a head turn: confidence drops below the high threshold mid-trace
    frames = [[FakeFace(_box(40 + i), confidence=0.9 if i < 8 else 0.55)] for i in range(16)]
    traces = _track(face, frames)
    assert len(traces) == 1 and len(traces[0].frames) == 16
    # the same low-confidence face on its own never opens a trace
    assert _track(face, [[FakeFace(_box(40), confidence=0.55)] for _ in range(16)]) == []


def test_short_traces_are_dropped(face):
    assert _track(face, [[FakeFace(_box(40))] for _ in range(3)]) == []      # 3 samples < 0.5s
    assert len(_track(face, [[FakeFace(_box(40))] for _ in range(4)])) == 1  # 4 samples = 0.5s


def test_trace_keeps_best_frames_for_embedding_and_a_representative(face):
    # the face's size wobbles a little frame to frame (enough to change its quality, not to break the track)
    sizes = [60, 70, 62, 72, 64, 68, 61, 71, 63, 69, 65, 67]
    traces = _track(face, [[FakeFace(_box(40, size=s))] for s in sizes])
    t = traces[0]
    assert len(t.embed_crops) == 8
    assert t.representative.thumbnail_jpeg.startswith(b"\xff\xd8")
    # the representative is the highest-quality frame of the trace
    assert t.representative.frame.quality == max(f.quality for f in t.frames)


# ── quality ───────────────────────────────────────────────────

def test_frontalness_symmetric_vs_profile(face):
    frontal = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)
    turned = np.array([[30, 40], [70, 40], [56, 60], [42, 80], [67, 80]], dtype=np.float32)
    profile = np.array([[30, 40], [70, 40], [66, 60], [55, 80], [70, 80]], dtype=np.float32)
    beyond = np.array([[30, 40], [70, 40], [75, 60], [60, 80], [72, 80]], dtype=np.float32)
    assert face.quality.frontalness(frontal) == pytest.approx(1.0)
    assert 0.3 < face.quality.frontalness(turned) < 0.6
    assert face.quality.frontalness(profile) < 0.15
    assert face.quality.frontalness(beyond) == 0.0       # nose past the eye: full profile


def test_frontalness_ignores_head_tilt(face):
    frontal = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)
    angle = np.deg2rad(25)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    tilted = (frontal - 50) @ rot.T + 50
    assert face.quality.frontalness(tilted) == pytest.approx(1.0, abs=1e-5)


def test_sharpness_drops_when_blurred(face):
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (112, 112), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert face.quality.sharpness(sharp) > 0.9 > face.quality.sharpness(blurred)


def test_exposure_excludes_dark_and_blown_out_faces(face):
    assert face.quality.exposure(np.full((112, 112), 10, np.uint8)) == 0.0
    assert face.quality.exposure(np.full((112, 112), 250, np.uint8)) == 0.0
    assert face.quality.exposure(np.full((112, 112), 128, np.uint8)) == 1.0


def test_bigger_face_scores_higher(face):
    small = face.quality.assess(GREY, FakeFace(_box(40, size=30)))
    big = face.quality.assess(GREY, FakeFace(_box(40, size=112)))
    assert big.size == 1.0 and small.size < big.size


def test_thumbnail_is_a_bounded_jpeg(face):
    img = np.full((720, 1280, 3), 100, dtype=np.uint8)
    jpeg = face.quality.thumbnail_jpeg(img, np.array([400, 100, 800, 600], dtype=np.float32))
    thumb = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert max(thumb.shape[:2]) <= 256


# ── clustering ────────────────────────────────────────────────

def _unit(cos_to_x, other_axis=1):
    v = np.zeros(512, dtype=np.float32)
    v[0], v[other_axis] = cos_to_x, np.sqrt(1 - cos_to_x ** 2)
    return v


def _trace(start, end, embedding, frames=8, px=100, conf=0.9):
    rep = SimpleNamespace(min_side=px, quality=SimpleNamespace(score=conf, confidence=conf))
    return SimpleNamespace(start_sec=start, end_sec=end, embedding=embedding,
                           frames=[None] * frames, representative=rep)


def test_non_overlapping_traces_merge_at_045_with_no_time_limit(face):
    a, b = _trace(0, 1, _unit(1.0)), _trace(100, 101, _unit(0.5))
    assert len(face.clustering.cluster_traces([a, b])) == 1
    c = _trace(100, 101, _unit(0.4))
    assert len(face.clustering.cluster_traces([a, c])) == 2


def test_overlapping_traces_need_075_and_clear_faces(face):
    a = _trace(0, 2, _unit(1.0))
    assert len(face.clustering.cluster_traces([a, _trace(1, 3, _unit(0.8))])) == 1
    assert len(face.clustering.cluster_traces([a, _trace(1, 3, _unit(0.7))])) == 2
    assert len(face.clustering.cluster_traces([a, _trace(1, 3, _unit(0.8), px=50)])) == 2
    assert len(face.clustering.cluster_traces([a, _trace(1, 3, _unit(0.8), conf=0.55)])) == 2


def test_overlap_is_checked_at_group_level(face):
    # A and B never overlap and match strongly; C matches B but is on screen together with A.
    # Pairwise-only checks would chain A–B–C into one person; the group-level check must not.
    a = _trace(0, 1, _unit(1.0))
    b = _trace(5, 6, _unit(0.95))
    c = _trace(0.5, 1.5, _unit(0.95, other_axis=2), px=30)
    groups = face.clustering.cluster_traces([a, b, c])
    members = sorted(sorted(id(t) for t in g.traces) for g in groups)
    assert members == sorted([sorted([id(a), id(b)]), [id(c)]])


def test_group_embedding_is_frame_weighted(face):
    a = _trace(0, 1, _unit(1.0), frames=30)
    b = _trace(5, 6, _unit(0.6), frames=10)
    (group,) = face.clustering.cluster_traces([a, b])
    assert group.frame_count == 40
    assert float(group.embedding @ a.embedding) > float(group.embedding @ b.embedding)


def test_groups_are_ordered_by_first_appearance(face):
    late, early = _trace(50, 51, _unit(1.0)), _trace(0, 1, _unit(0.0))
    groups = face.clustering.cluster_traces([late, early])
    assert groups[0].traces == [early]


# ── full analysis ─────────────────────────────────────────────

def _make_three_shot_video(path, face_image=None, fps=25):
    """face shot (0–1.5s) → empty green shot (1.5–3s) → face again elsewhere (3–4.5s)."""
    w, h = 640, 360
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(fps * 4.5)):
        t = i / fps
        frame = np.full((h, w, 3), (40, 160, 40) if 1.5 <= t < 3.0 else 90, dtype=np.uint8)
        if face_image is not None and not 1.5 <= t < 3.0:
            x = 80 if t < 1.5 else 380
            fh, fw = face_image.shape[:2]
            frame[60:60 + fh, x:x + fw] = face_image
        writer.write(frame)
    writer.release()
    return path


@requires_ffprobe
def test_analysis_with_fake_detector(face, tmp_path):
    # FakeDetector "sees" a face on the grey shots only (green shot averages ~80 < 85 < grey 90)
    path = _make_three_shot_video(str(tmp_path / "v.mp4"))
    analysis = face.analysis.analyze_video(path, FakeDetector(face_below=200), "u1/videos/clip/v1")
    assert len(analysis.detection.shot_boundaries) == 2
    summary = analysis.summary()
    assert summary["traces"] >= 1 and summary["groups"] >= 1
    assert all(t.trace_id and t.embedding.shape == (512,) and t.embed_crops == [] for t in analysis.traces)
    # IDs are deterministic per video version, so a retried write to Module 25 stays idempotent
    again = face.analysis.analyze_video(path, FakeDetector(face_below=200), "u1/videos/clip/v1")
    assert [t.trace_id for t in again.traces] == [t.trace_id for t in analysis.traces]


@requires_ffprobe
def test_analysis_same_person_across_shots_is_one_group(face, detector, face_image, tmp_path):
    path = _make_three_shot_video(str(tmp_path / "face.mp4"), face_image)
    analysis = face.analysis.analyze_video(path, detector, "u1/videos/clip/v1")
    assert len(analysis.traces) == 2           # the cut ends the first trace; the face returns later
    assert len(analysis.groups) == 1           # but it is the same person
    assert analysis.groups[0].group_id
    t1, t2 = analysis.traces
    assert t1.end_sec < 1.5 <= 3.0 <= t2.start_sec  # noqa: PLR0133 -- chain reads as "ends before 1.5s, resumes after 3.0s"
    assert float(t1.embedding @ t2.embedding) > 0.8


def _make_two_person_video(path, a, b, fps=25, shot_sec=1.5):
    """Four shots, each on its own background so every change is a hard cut:
    A alone → B alone → A and B on screen together → A alone again."""
    w, h = 640, 360
    shots = [((90, 90, 90), [(a, 80)]),
             ((40, 160, 40), [(b, 380)]),
             ((160, 60, 40), [(a, 60), (b, 380)]),
             ((60, 40, 160), [(a, 380)])]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for bg, faces in shots:
        for i in range(int(fps * shot_sec)):
            frame = np.full((h, w, 3), bg, dtype=np.uint8)
            for img, x in faces:
                fh, fw = img.shape[:2]
                frame[60:60 + fh, x + i:x + i + fw] = img      # drifts slowly, like a real shot
            writer.write(frame)
    writer.release()
    return path


@requires_ffprobe
def test_analysis_two_people_end_up_in_two_groups(face, detector, face_image, second_face_image, tmp_path):
    """Two different people: exactly two groups. A appears in shots 1, 3, 4 and B in shots 2, 3;
    in shot 3 they are on screen together, so A's and B's traces overlap in time and may
    only merge above the stricter co-presence threshold -- they must not."""
    path = _make_two_person_video(str(tmp_path / "two.mp4"), face_image, second_face_image)
    analysis = face.analysis.analyze_video(path, detector, "u1/videos/two/v1")

    assert len(analysis.detection.shot_boundaries) == 3
    assert len(analysis.traces) == 5              # A×3 shots + B×2 shots; every cut ends a trace
    assert len(analysis.groups) == 2

    cuts = analysis.detection.shot_boundaries

    def shots_of(group):     # shot number = 1 + cuts at or before the trace's start
        return sorted(1 + sum(c <= t.start_sec for c in cuts) for t in group.traces)

    by_first_shot = {shots_of(g)[0]: shots_of(g) for g in analysis.groups}
    assert by_first_shot == {1: [1, 3, 4], 2: [2, 3]}    # A in shots 1, 3, 4; B in shots 2, 3


# ── representative upload (HTTP contract with Module 04) ─────

class _NoRedis:
    async def set(self, *a, **kw):
        raise ConnectionError("no Redis in unit tests")

@pytest.mark.asyncio
async def test_upload_representatives_posts_associated_files(face, monkeypatch):
    received = {}

    async def add_associated_files(request):
        received["path"] = request.match_info["asset_path"]
        received["headers"] = (request.headers.get("X-User-ID"), request.headers.get("X-Branch-ID"))
        received["files"] = []
        async for part in await request.multipart():
            if part.name == "primary_version_id":
                received["version"] = await part.text()
            else:
                received["files"].append((part.name, part.filename, await part.read()))
        # Module 04 returns every associated file on the asset, keyframes included
        return web.json_response({"associated_filenames": [["keyframe_0001.jpg", "c0"], ["face_a.jpg", "c1"]]})

    app = web.Application()
    app.router.add_post("/add-associated-files/{asset_path:.*}", add_associated_files)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(face.asset_client, "ASSET_MANAGEMENT_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(face.assoc_lock, "_client", _NoRedis())     # lock covered in test_assoc_lock.py
    try:
        stored = await face.asset_client.upload_representatives("u1", "b1", "u1/videos/clip", "v1", [("face_a.jpg", b"\xff\xd8jpeg")])
    finally:
        await runner.cleanup()

    assert stored == {"face_a.jpg": "c1"}          # only what this call uploaded
    assert received["path"] == "u1/videos/clip"
    assert received["headers"] == ("u1", "b1")
    assert received["version"] == "v1"
    assert received["files"] == [("associated_files", "face_a.jpg", b"\xff\xd8jpeg")]


# ── trace/group write (HTTP contract with Module 25) ─────────

@pytest.mark.asyncio
async def test_write_face_traces_inits_the_user_db_then_posts(face, monkeypatch):
    received = []

    async def init(request):
        received.append(("init", request.headers.get("X-User-ID"), request.headers.get("X-Branch-ID"), None))
        return web.json_response({"user_id": "u1", "database": "user_u1", "created": False})

    async def face_traces(request):
        received.append(("face-traces", request.headers.get("X-User-ID"), request.headers.get("X-Branch-ID"),
                         await request.json()))
        return web.json_response({"matches": []}, status=201)

    app = web.Application()
    app.router.add_post("/internal/db/init", init)
    app.router.add_post("/personal/index/face-traces", face_traces)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(face.personal_db_client, "PERSONAL_DB_SERVICE_URL", f"http://127.0.0.1:{port}")
    try:
        result = await face.personal_db_client.write_face_traces("u1", {"version_id": "v1", "traces": [], "groups": []})
    finally:
        await runner.cleanup()

    assert result == {"matches": []}
    # the database may not exist yet when faces finish before content indexing, so init comes first
    assert received == [("init", "u1", "u1", None),
                        ("face-traces", "u1", "u1", {"version_id": "v1", "traces": [], "groups": []})]


@pytest.mark.asyncio
async def test_write_face_traces_raises_on_rejected_write(face, monkeypatch):
    async def init(request):
        return web.json_response({"created": False})

    async def face_traces(request):
        return web.json_response({"detail": "group g1 references unknown traces"}, status=422)

    app = web.Application()
    app.router.add_post("/internal/db/init", init)
    app.router.add_post("/personal/index/face-traces", face_traces)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(face.personal_db_client, "PERSONAL_DB_SERVICE_URL", f"http://127.0.0.1:{port}")
    try:
        with pytest.raises(RuntimeError, match="422"):
            await face.personal_db_client.write_face_traces("u1", {"version_id": "v1", "traces": [], "groups": []})
    finally:
        await runner.cleanup()
