"""video_face_detection_service (dev plan v1.4 §4.3 / §4.5): probing, shot
boundaries, fixed 8fps sampling, detection, the orchestrator's Kafka branch and
the synchronous /embed-face endpoint.

Run from the 11-video-processing/ directory:
    FACE_MODEL_DIR=/path/to/models/face pytest tests/unit/ -v

Model-dependent tests are skipped without insightface + the ONNX files, and
ffprobe-dependent ones without ffprobe on PATH (see face_detection_helpers).
"""
# ruff: noqa: F811  (pytest fixtures imported by name are "redefined" as test params by design)
import os

import aiohttp
import cv2
import numpy as np
import pytest
import yarl
from face_detection_helpers import (  # noqa: F401  (face / detector / face_image / second_face_image are fixtures)
    FakeDetector,
    FakeProducer,
    detector,
    face,
    face_image,
    make_face_video,
    make_video,
    requires_ffprobe,
    second_face_image,
)
from fastapi.testclient import TestClient


def _jpeg(image):
    return cv2.imencode(".jpg", image)[1].tobytes()


# ── sampling ──────────────────────────────────────────────────

@pytest.mark.parametrize("source_fps", [30, 25])
def test_sampler_picks_nearest_frame_every_125ms(face, tmp_path, source_fps):
    path = make_video(str(tmp_path / f"v{source_fps}.mp4"), source_fps, 3.0)
    samples = list(face.sampler.sample_frames(face.sampler.decode_frames(path)))

    # targets 0, 0.125, ..., 2.875 (the last frame is before 3.0s)
    assert len(samples) == 24
    for k, s in enumerate(samples):
        # the nearest frame is within half a source-frame interval of its target;
        # the timestamp is the frame's own, not snapped to the 0.125s grid
        assert abs(s.timestamp_sec - k * 0.125) <= 0.5 / source_fps + 1e-6
        assert s.frame_no == round(s.timestamp_sec * source_fps)


def test_sampler_does_not_repeat_frames_for_low_fps_source(face, tmp_path):
    path = make_video(str(tmp_path / "v5.mp4"), 5, 3.0)
    samples = list(face.sampler.sample_frames(face.sampler.decode_frames(path)))
    frame_nos = [s.frame_no for s in samples]
    assert frame_nos == sorted(set(frame_nos))
    assert len(samples) == 15   # every frame of a 5fps source is nearest to some 8fps target


def test_sampler_handles_stream_not_starting_at_zero(face):
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = [face.sampler.Frame(i, 1.0 + i / 30, img) for i in range(30)]
    samples = list(face.sampler.sample_frames(frames))
    assert samples[0].frame_no == 0
    assert [s.frame_no for s in samples] == sorted({s.frame_no for s in samples})


# ── shot boundaries ───────────────────────────────────────────

@pytest.mark.parametrize("source_fps", [30, 25])
def test_shot_detector_finds_known_cut(face, tmp_path, source_fps):
    cut_frame = int(1.7 * source_fps)
    path = make_video(str(tmp_path / "cut.mp4"), source_fps, 4.0, cut_at_frame=cut_frame)
    cuts = face.shot_detector.detect_shots(path)
    assert len(cuts) == 1
    # compared frame by frame, so the cut lands on the source frame, well inside one 8fps sample
    assert abs(cuts[0] - cut_frame / source_fps) <= 0.125


def test_shot_detector_ignores_motion_within_shot(face, tmp_path):
    path = make_video(str(tmp_path / "nocut.mp4"), 30, 3.0)
    assert face.shot_detector.detect_shots(path) == []


# ── probe ─────────────────────────────────────────────────────

@requires_ffprobe
def test_probe_reads_resolution_fps_duration(face, tmp_path):
    path = make_video(str(tmp_path / "p.mp4"), 25, 2.0, size=(320, 240))
    info = face.probe.probe_video(path)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(25, abs=0.01)
    assert info.duration_sec == pytest.approx(2.0, abs=0.05)


# ── pipeline (single decode pass) ─────────────────────────────

@requires_ffprobe
def test_pipeline_single_pass_samples_detects_and_finds_cut(face, tmp_path):
    # 4s at 25fps with a cut at 2s. The first shot averages ~100 brightness, the second ~83;
    # FakeDetector "sees" a face only below 90, i.e. only in the second shot.
    path = make_video(str(tmp_path / "v.mp4"), 25, 4.0, cut_at_frame=50)
    seen = []
    detector = FakeDetector(face_below=90)
    result = face.pipeline.run_detection(
        path, detector,
        on_frame=lambda frame, faces, shot_changed: seen.append((frame.image.shape, len(faces), shot_changed, frame.timestamp_sec)))

    assert len(result.frames) == 32                              # 4s × 8fps
    assert detector.detect_calls == 32                           # detection on sampled frames only
    assert len(seen) == 32 and seen[0][0] == (240, 320, 3)       # the callback gets the full image
    # the shot change is flagged once, on the first sample at or after the cut
    flagged = [ts for _, _, changed, ts in seen if changed]
    assert len(flagged) == 1 and 2.0 <= flagged[0] < 2.0 + 0.125
    assert len(result.shot_boundaries) == 1
    assert abs(result.shot_boundaries[0] - 2.0) <= 0.125

    summary = result.summary()
    assert summary["sampled_frames"] == 32
    assert summary["shot_boundaries"] == 1
    assert summary["frames_with_faces"] == summary["face_detections"] == 16   # the 2s after the cut
    assert (summary["width"], summary["height"]) == (320, 240)


# ── fetching the source video ─────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_video_links_the_orchestrator_temp_file(face, tmp_path):
    src = tmp_path / "orchestrator_tmp.mp4"
    src.write_bytes(b"video-bytes")
    dest = tmp_path / "face" / "copy.mp4"
    await face.asset_client.fetch_video(str(dest), str(src), "a", "v", {}, {})
    assert dest.read_bytes() == b"video-bytes"
    assert os.stat(dest).st_ino == os.stat(src).st_ino   # hard link, no data copied
    src.unlink()                                          # the orchestrator's cleanup at pipeline end
    assert dest.read_bytes() == b"video-bytes"            # our copy survives it


# ── uploading representatives ─────────────────────────────────

def _representatives(n):
    return [(f"face_t{i}.jpg", b"jpeg") for i in range(n)]


@pytest.mark.asyncio
async def test_representatives_go_up_in_batches_under_module_04s_file_limit(face, monkeypatch):
    # Module 04 rejects a whole request carrying more than 1000 files
    batches = []

    async def fake_batch(user_id, branch_id, asset_path, version_id, files):
        batches.append(len(files))
        return {name: f"ver-{name}" for name, _ in files}

    monkeypatch.setattr(face.asset_client, "FACE_REPRESENTATIVE_BATCH_SIZE", 500)
    monkeypatch.setattr(face.asset_client, "_upload_batch", fake_batch)
    stored = await face.asset_client.upload_representatives("u", "b", "a", "v", _representatives(1203))
    assert batches == [500, 500, 203]
    assert len(stored) == 1203
    assert stored["face_t1202.jpg"] == "ver-face_t1202.jpg"


@pytest.mark.asyncio
async def test_representatives_already_committed_survive_a_later_batch_failing(face, monkeypatch):
    calls = []

    async def fake_batch(user_id, branch_id, asset_path, version_id, files):
        calls.append(len(files))
        if len(calls) == 2:
            raise RuntimeError("Representative upload returned 503")
        return {name: "ver" for name, _ in files}

    monkeypatch.setattr(face.asset_client, "FACE_REPRESENTATIVE_BATCH_SIZE", 2)
    monkeypatch.setattr(face.asset_client, "_upload_batch", fake_batch)
    stored = await face.asset_client.upload_representatives("u", "b", "a", "v", _representatives(5))
    assert calls == [2, 2]                                # stops at the failing batch
    assert set(stored) == {"face_t0.jpg", "face_t1.jpg"}


@pytest.mark.asyncio
async def test_representative_upload_raises_when_no_batch_succeeds(face, monkeypatch):
    async def fake_batch(user_id, branch_id, asset_path, version_id, files):
        raise RuntimeError("Representative upload returned 400")

    monkeypatch.setattr(face.asset_client, "_upload_batch", fake_batch)
    with pytest.raises(RuntimeError, match="400"):
        await face.asset_client.upload_representatives("u", "b", "a", "v", _representatives(3))


@pytest.mark.asyncio
async def test_no_representatives_means_no_request(face, monkeypatch):
    async def fake_batch(*args):
        raise AssertionError("no request expected")

    monkeypatch.setattr(face.asset_client, "_upload_batch", fake_batch)
    assert await face.asset_client.upload_representatives("u", "b", "a", "v", []) == {}


# ── Kafka handler ─────────────────────────────────────────────

def _face_request(video_file_path=None):
    return {
        "message_id": "m1",
        "correlation_id": "corr-1",
        "source_service": "video_orchestrator_service",
        "payload": {
            "request_id": "req-1",
            "user_id": "u1",
            "action": "face_detection",
            "parameters": {
                "asset_path": "u1/videos/clip",
                "version_id": "v1",
                "primary_filename": "clip.mp4",
                "video_file_path": video_file_path,
                "download_headers": {"X-User-ID": "u1", "X-Branch-ID": "b1"},
                "download_params": {},
            },
        },
    }


@pytest.fixture(autouse=True)
def consent(face, monkeypatch):
    """GD-11: Module 33's answer per check, in order -- None = granted. Granted unless a
    test says otherwise; `calls` records each user asked about."""
    state = {"answers": [], "calls": []}

    async def fake_denied(user_id):
        state["calls"].append(user_id)
        return state["answers"].pop(0) if state["answers"] else None

    monkeypatch.setattr(face.kafka_handler, "face_consent_denied", fake_denied)
    return state


@pytest.fixture
def face_temp_dir(face, tmp_path, monkeypatch):
    d = tmp_path / "face_tmp"
    monkeypatch.setattr(face.kafka_handler, "FACE_TEMP_DIR", str(d))
    return d


@pytest.fixture
def uploads(face, monkeypatch):
    calls = []

    async def fake_upload(user_id, branch_id, asset_path, version_id, files):
        calls.append({"user_id": user_id, "branch_id": branch_id, "asset_path": asset_path,
                      "version_id": version_id, "files": files})
        return {name: "commit-1" for name, _ in files}

    monkeypatch.setattr(face.kafka_handler, "upload_representatives", fake_upload)
    return calls


@pytest.fixture
def personal_db(face, monkeypatch):
    """Stands in for Module 25: records each write, reports one group of each outcome per call."""
    calls = []

    async def fake_write(user_id, body):
        calls.append({"user_id": user_id, "body": body})
        return {"matches": [{"identity_id": "i1", "created_identity": False},    # joined an existing identity
                            {"identity_id": "i2", "created_identity": True},     # matched an unassigned group
                            {"identity_id": None, "created_identity": False}]}   # no match

    monkeypatch.setattr(face.kafka_handler, "write_face_traces", fake_write)
    return calls


@requires_ffprobe
@pytest.mark.asyncio
async def test_kafka_handler_reports_summary_and_cleans_up(face, tmp_path, face_temp_dir, uploads, personal_db):
    # 2s clip; FakeDetector sees one (stationary) face on every frame -> one trace, one group
    src = make_video(str(tmp_path / "orchestrator_tmp.mp4"), 25, 2.0)
    producer = FakeProducer()

    await face.kafka_handler.FaceDetectionKafkaHandler(FakeDetector()).handle_request(
        _face_request(video_file_path=src), producer)

    assert len(producer.sent) == 1
    topic, msg = producer.sent[0]
    assert topic == "video-face-detection-results"
    assert msg["message_type"] == "RESPONSE"
    assert msg["correlation_id"] == "corr-1"
    assert msg["target_service"] == "video_orchestrator_service"
    results = msg["payload"]["results"]
    assert results["sampled_frames"] == 16
    assert (results["traces"], results["groups"], results["representatives_uploaded"]) == (1, 1, 1)
    # step 8: one representative JPEG per trace, uploaded as the video's associated files
    assert len(uploads) == 1
    call = uploads[0]
    assert (call["user_id"], call["branch_id"], call["asset_path"], call["version_id"]) == ("u1", "b1", "u1/videos/clip", "v1")
    (name, data), = call["files"]
    assert name.startswith("face_") and name.endswith(".jpg") and data.startswith(b"\xff\xd8")
    # step 9: the traces and groups go to Module 25, in the user database the video indexer also writes to
    (write,) = personal_db
    assert write["user_id"] == "u1"
    body = write["body"]
    assert (body["version_id"], body["asset_path"], body["filename"]) == ("v1", "u1/videos/clip", "clip.mp4")
    (trace,), (group,) = body["traces"], body["groups"]
    assert name == f"face_{trace['trace_id']}.jpg"
    assert trace["representative"]["image_path"] == f"u1/videos/clip/{name}"
    assert trace["representative"]["image_version_id"] == "commit-1"
    assert len(trace["embedding"]) == len(group["embedding"]) == 512
    assert len(trace["frames"]) == 16 and group["trace_ids"] == [trace["trace_id"]]
    assert (results["identities_linked"], results["identities_created"], results["groups_unassigned"]) == (1, 1, 1)
    assert os.path.exists(src)                     # the orchestrator's temp file is untouched
    assert list(face_temp_dir.iterdir()) == []     # our own copy is removed when done


@requires_ffprobe
@pytest.mark.asyncio
async def test_kafka_handler_representative_upload_failure_is_not_fatal(face, tmp_path, face_temp_dir, personal_db, monkeypatch):
    async def broken(*args):
        raise RuntimeError("Representative upload returned 500")

    monkeypatch.setattr(face.kafka_handler, "upload_representatives", broken)
    src = make_video(str(tmp_path / "orchestrator_tmp.mp4"), 25, 2.0)
    producer = FakeProducer()
    await face.kafka_handler.FaceDetectionKafkaHandler(FakeDetector()).handle_request(
        _face_request(video_file_path=src), producer)

    msg = producer.sent[0][1]
    assert msg["message_type"] == "RESPONSE"       # the analysis result still goes out
    assert msg["payload"]["results"]["traces"] == 1
    assert msg["payload"]["results"]["representatives_uploaded"] == 0
    assert "500" in msg["payload"]["results"]["representative_upload_error"]
    # the traces are still written, just without an image to point at
    rep = personal_db[0]["body"]["traces"][0]["representative"]
    assert rep["image_path"] is None and rep["image_version_id"] is None


@requires_ffprobe
@pytest.mark.asyncio
async def test_kafka_handler_reports_error_when_personal_db_write_fails(face, tmp_path, face_temp_dir, uploads, monkeypatch):
    async def broken(user_id, body):
        raise RuntimeError("Personal DB face-traces returned 500: boom")

    monkeypatch.setattr(face.kafka_handler, "write_face_traces", broken)
    src = make_video(str(tmp_path / "orchestrator_tmp.mp4"), 25, 2.0)
    producer = FakeProducer()
    await face.kafka_handler.FaceDetectionKafkaHandler(FakeDetector()).handle_request(
        _face_request(video_file_path=src), producer)

    msg = producer.sent[0][1]
    assert msg["message_type"] == "ERROR"          # the result is not stored anywhere, so the request failed
    assert "face-traces returned 500" in msg["payload"]["error"]["message"]
    assert list(face_temp_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_kafka_handler_reports_error_without_raising(face, face_temp_dir, monkeypatch):
    async def unreachable(**kwargs):
        raise RuntimeError("Asset API returned 404")

    monkeypatch.setattr(face.kafka_handler, "fetch_video", unreachable)
    producer = FakeProducer()
    await face.kafka_handler.FaceDetectionKafkaHandler(FakeDetector()).handle_request(_face_request(), producer)

    topic, msg = producer.sent[0]
    assert topic == "video-face-detection-results"
    assert msg["message_type"] == "ERROR"
    assert msg["payload"]["status"] == "error"
    assert "404" in msg["payload"]["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["version_id", "user_id"])
async def test_kafka_handler_rejects_request_without_ids(face, face_temp_dir, missing):
    request = _face_request()
    if missing == "user_id":
        del request["payload"]["user_id"]
    else:
        del request["payload"]["parameters"][missing]
    producer = FakeProducer()
    await face.kafka_handler.FaceDetectionKafkaHandler(FakeDetector()).handle_request(request, producer)
    assert producer.sent[0][1]["message_type"] == "ERROR"
    assert missing in producer.sent[0][1]["payload"]["error"]["message"]


# ── POST /embed-face ─────────────────────────────────────────

@pytest.fixture
def fake_client(face):
    face.sync_api.init(FakeDetector(face_below=200))
    return TestClient(face.sync_api.app)


def test_embed_face_returns_embedding(fake_client):
    dark = _jpeg(np.full((120, 120, 3), 60, dtype=np.uint8))
    r = fake_client.post("/embed-face", files={"file": ("f.jpg", dark, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["embedding"]) == 512
    assert body["face_count"] == 1
    assert len(body["bbox"]) == 4


def test_embed_face_400_when_no_face(fake_client):
    bright = _jpeg(np.full((120, 120, 3), 250, dtype=np.uint8))
    assert fake_client.post("/embed-face", files={"file": ("f.jpg", bright, "image/jpeg")}).status_code == 400


def test_embed_face_400_on_undecodable_file(fake_client):
    r = fake_client.post("/embed-face", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_embed_face_503_before_init(face):
    face.sync_api.init(None)
    client = TestClient(face.sync_api.app)
    r = client.post("/embed-face", files={"file": ("f.jpg", b"x", "image/jpeg")})
    assert r.status_code == 503


# ── real models (skipped without insightface + the ONNX files) ────

def test_detect_returns_bbox_and_landmark_shapes(detector, face_image):
    faces = detector.detect(face_image)
    assert len(faces) == 1
    f = faces[0]
    assert f.bbox.shape == (4,)
    assert f.landmarks.shape == (5, 2)
    assert f.confidence >= 0.5
    x1, y1, x2, y2 = f.bbox
    h, w = face_image.shape[:2]
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    assert np.all((f.landmarks[:, 0] >= x1) & (f.landmarks[:, 0] <= x2))
    assert np.all((f.landmarks[:, 1] >= y1) & (f.landmarks[:, 1] <= y2))


def test_detect_returns_nothing_on_frame_without_faces(detector):
    assert detector.detect(np.full((384, 640, 3), 128, dtype=np.uint8)) == []


def test_detect_drops_faces_below_min_size(detector, second_face_image):
    # this fixture, not face_image: RetinaFace MV2 only finds that one down to ~18px
    f = detector.detect(second_face_image)[0]
    small = cv2.resize(second_face_image, None, fx=15 / f.min_side, fy=15 / f.min_side)   # ~15px face
    canvas = np.full((384, 640, 3), 128, dtype=np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small
    min_face_px = detector.min_face_px
    try:
        detector.min_face_px = 0
        assert len(detector.detect(canvas)) == 1   # the model itself finds the small face
    finally:
        detector.min_face_px = min_face_px
    assert detector.detect(canvas) == []           # and the 20px floor drops it


def test_embed_is_512_dim_l2_normalized(detector, face_image):
    f = detector.detect(face_image)[0]
    emb = detector.embed([(face_image, f)])
    assert emb.shape == (1, 512)
    assert np.linalg.norm(emb[0]) == pytest.approx(1.0, abs=1e-4)
    assert detector.embed([]).shape == (0, 512)


def test_embed_same_face_is_similar_across_flip_and_scale(detector, face_image):
    f = detector.detect(face_image)[0]
    other = cv2.resize(cv2.flip(face_image, 1), None, fx=1.5, fy=1.5)
    emb = detector.embed([(face_image, f), (other, detector.detect(other)[0])])
    assert float(emb[0] @ emb[1]) > 0.8


def test_embed_face_endpoint_with_real_models(face, detector, face_image):
    face.sync_api.init(detector)
    r = TestClient(face.sync_api.app).post("/embed-face", files={"file": ("f.jpg", _jpeg(face_image), "image/jpeg")})
    assert r.status_code == 200
    assert len(r.json()["embedding"]) == 512


@requires_ffprobe
def test_pipeline_with_real_models(face, detector, face_image, tmp_path):
    path = make_face_video(str(tmp_path / "face.mp4"), face_image)
    result = face.pipeline.run_detection(path, detector)
    before = [f for f in result.frames if f.timestamp_sec < 2.0]
    after = [f for f in result.frames if f.timestamp_sec >= 2.0]
    assert len(result.frames) == 32
    assert all(len(f.faces) == 1 for f in before)
    assert all(len(f.faces) == 0 for f in after)
    assert len(result.shot_boundaries) == 1


# ── GD-11: face_detection consent ────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["no_consent", "consent_unavailable"])
async def test_kafka_handler_skips_without_consent_before_touching_the_video(face, face_temp_dir, uploads, personal_db,
                                                                            consent, monkeypatch, reason):
    async def must_not_fetch(**kwargs):
        raise AssertionError("the video must not be fetched without consent")

    monkeypatch.setattr(face.kafka_handler, "fetch_video", must_not_fetch)
    consent["answers"] = [reason]
    producer = FakeProducer()
    await face.kafka_handler.FaceDetectionKafkaHandler(FakeDetector()).handle_request(_face_request(), producer)

    (topic, msg), = producer.sent
    assert topic == "video-face-detection-results" and msg["message_type"] == "RESPONSE"
    assert msg["payload"]["results"] == {"skipped": reason}
    assert consent["calls"] == ["u1"]
    assert uploads == [] and personal_db == []


@pytest.mark.asyncio
async def test_kafka_handler_drops_the_analysis_when_consent_is_withdrawn_mid_job(face, face_temp_dir, uploads,
                                                                                personal_db, consent, monkeypatch):
    """Withdrawn while detecting: Module 33 has already purged this user's faces, so
    neither the representatives nor the traces may be written back."""
    analysed = []

    async def fake_fetch(dest_path, **kwargs):
        pass

    def fake_analyze(video_path, detector, label):
        analysed.append(label)
        return object()                              # never read: the second check stops first

    monkeypatch.setattr(face.kafka_handler, "fetch_video", fake_fetch)
    monkeypatch.setattr(face.kafka_handler, "analyze_video", fake_analyze)
    consent["answers"] = [None, "no_consent"]
    producer = FakeProducer()
    await face.kafka_handler.FaceDetectionKafkaHandler(FakeDetector()).handle_request(_face_request(), producer)

    assert analysed == ["u1/videos/clip/v1"]
    (_, msg), = producer.sent
    assert msg["message_type"] == "RESPONSE" and msg["payload"]["results"] == {"skipped": "no_consent"}
    assert consent["calls"] == ["u1", "u1"]
    assert uploads == [] and personal_db == []
    assert list(face_temp_dir.iterdir()) == []      # our copy is still cleaned up


class _Resp:
    def __init__(self, status, body):
        self.status, self.body = status, body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            url = yarl.URL("http://m33:8033/consent")
            raise aiohttp.ClientResponseError(aiohttp.RequestInfo(url, "GET", {}, url), (), status=self.status)

    async def json(self):
        return self.body


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body,expected", [
    (200, {"purposes": [{"id": "face_detection", "granted": True}]}, None),
    (200, {"purposes": [{"id": "face_detection", "granted": False}]}, "no_consent"),
    (200, {"purposes": [{"id": "memory", "granted": True}]}, "no_consent"),
    (404, {"detail": "User not found in Keycloak"}, "no_consent"),
    (503, {}, "consent_unavailable"),
    (200, {"unexpected": []}, "consent_unavailable"),
])
async def test_consent_client_reads_module_33(face, monkeypatch, status, body, expected):
    seen = {}

    class _Session:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url, headers=None):
            seen.update(url=url, headers=headers)
            return _Resp(status, body)

    monkeypatch.setattr(face.consent_client.aiohttp, "ClientSession", _Session)
    monkeypatch.setattr(face.consent_client, "PRIVACY_COMPLIANCE_URL", "http://m33:8033")
    monkeypatch.setattr(face.consent_client, "GD_INTERNAL_TOKEN", "tok")
    assert await face.consent_client.face_consent_denied("u1") == expected
    assert seen == {"url": "http://m33:8033/consent", "headers": {"X-User-ID": "u1", "X-Internal-Token": "tok"}}


@pytest.mark.asyncio
async def test_consent_client_fails_closed_when_module_33_is_down(face, monkeypatch):
    monkeypatch.setattr(face.consent_client, "PRIVACY_COMPLIANCE_URL", "http://127.0.0.1:1")
    assert await face.consent_client.face_consent_denied("u1") == "consent_unavailable"
