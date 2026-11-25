"""Fixtures and helpers for video_face_detection_service's unit tests.

Not a conftest.py: a second conftest under tests/unit/ would take over the
`conftest` module name that test_claim_and_status.py imports helpers from.
Test modules import the fixtures below by name instead.

conftest.py puts video_orchestrator_service on sys.path, and its autouse
fixture imports the orchestrator's redis_manager (and with it the
orchestrator's `config`). The face-detection service has its own modules
with the same names (config, kafka_handler, message_utils), so its modules
are loaded here in isolation and handed to tests through the `face` fixture
instead of being imported by name.

Tests that need the real models are skipped unless insightface is installed
and FACE_MODEL_DIR points at a directory holding retinaface_mv2.onnx and
glintr100.onnx (see the module README); tests that need ffprobe are skipped
when it is not on PATH.
"""
import importlib
import os
import shutil
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

_SERVICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "worker", "services", "video_face_detection_service",
)
_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
_MODULES = [
    "config", "probe", "sampler", "shot_detector", "model_files", "retinaface", "detector", "pipeline",
    "quality", "tracker", "clustering", "analysis",
    "assoc_lock", "asset_client", "personal_db_client", "consent_client", "message_utils", "kafka_handler", "sync_api",
]

requires_ffprobe = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")


@pytest.fixture(scope="session")
def face():
    """The service's modules, imported from its own directory with its own `config`."""
    saved = {name: sys.modules.pop(name) for name in _MODULES if name in sys.modules}
    sys.path.insert(0, _SERVICE_DIR)
    try:
        modules = {}
        for name in _MODULES:
            try:
                modules[name] = importlib.import_module(name)
            except ImportError:
                if name not in ("retinaface", "detector"):
                    raise
                modules[name] = None   # insightface / onnxruntime not installed (e.g. CI)
        return SimpleNamespace(**modules)
    finally:
        sys.path.remove(_SERVICE_DIR)
        for name in _MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.fixture(scope="session")
def detector(face):
    """Real FaceDetector; needs insightface plus the retinaface_mv2 / glintr100 ONNX files."""
    if face.detector is None:
        pytest.skip("insightface is not installed")
    if not (os.path.exists(face.config.FACE_DET_MODEL_PATH) and os.path.exists(face.config.FACE_REC_MODEL_PATH)):
        pytest.skip(f"face models not found ({face.config.FACE_DET_MODEL_PATH}); set FACE_MODEL_DIR")
    return face.detector.FaceDetector()


@pytest.fixture(scope="session")
def face_image():
    return cv2.imread(os.path.join(_FIXTURE_DIR, "face_astronaut.jpg"))


@pytest.fixture(scope="session")
def second_face_image():
    """A different person from face_image (see fixtures/README.md)."""
    return cv2.imread(os.path.join(_FIXTURE_DIR, "face_armstrong.jpg"))


def make_video(path, fps, duration_sec, cut_at_frame=None, size=(320, 240)):
    """Synthetic clip: a moving square (motion within the shot, not a cut);
    from cut_at_frame on, a completely different background (a hard cut)."""
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(round(fps * duration_sec)):
        second_shot = cut_at_frame is not None and i >= cut_at_frame
        frame = np.full((h, w, 3), (40, 160, 40) if second_shot else (200, 60, 30), dtype=np.uint8)
        x = (i * 3) % (w - 40)
        cv2.rectangle(frame, (x, 100), (x + 40, 140), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def make_face_video(path, face_image, fps=25, duration_sec=4.0, cut_at_sec=2.0):
    """First shot: the face on a grey background, drifting right; after cut_at_sec
    a green shot with no face."""
    w, h = 640, 360
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fh, fw = face_image.shape[:2]
    for i in range(int(fps * duration_sec)):
        if i / fps < cut_at_sec:
            frame = np.full((h, w, 3), 90, dtype=np.uint8)
            x = 100 + i
            frame[80:80 + fh, x:x + fw] = face_image
        else:
            frame = np.full((h, w, 3), (40, 160, 40), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class FakeFace:
    """Same shape as detector.FaceDetection; landmarks default to a frontal face inside the box."""

    def __init__(self, bbox=(10.0, 10.0, 60.0, 70.0), confidence=0.9, landmarks=None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.confidence = confidence
        if landmarks is None:
            x1, y1, x2, y2 = self.bbox
            w, h = x2 - x1, y2 - y1
            landmarks = [[x1 + 0.3 * w, y1 + 0.4 * h], [x1 + 0.7 * w, y1 + 0.4 * h], [x1 + 0.5 * w, y1 + 0.6 * h],
                         [x1 + 0.35 * w, y1 + 0.8 * h], [x1 + 0.65 * w, y1 + 0.8 * h]]
        self.landmarks = np.array(landmarks, dtype=np.float32)

    @property
    def min_side(self):
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


class FakeDetector:
    """Stands in for FaceDetector where the models aren't needed: one face on
    every frame whose mean brightness is below `face_below`, none otherwise."""

    def __init__(self, face_below=255.0):
        self.face_below = face_below
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        return [FakeFace()] if float(image.mean()) < self.face_below else []

    def align(self, image, face):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def embed_aligned(self, crops):
        """Every face gets the same embedding, so non-overlapping traces always cluster together."""
        return np.full((len(crops), 512), 1 / np.sqrt(512), dtype=np.float32)

    def embed_best_face(self, image):
        faces = self.detect(image)
        if not faces:
            return None
        embedding = np.full(512, 1 / np.sqrt(512), dtype=np.float32)
        return faces[0], embedding, len(faces)


class FakeProducer:
    def __init__(self):
        self.sent = []

    async def send(self, topic, value):
        self.sent.append((topic, value))
