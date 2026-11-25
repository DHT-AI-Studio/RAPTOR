"""model_files.ensure_models(): the face models are fetched into the cache volume on first start,
checked against SHA-256 on every start, and never left half-written. Uses file:// URLs, no network."""
# ruff: noqa: F811  (pytest fixtures imported by name are "redefined" as test params by design)
import hashlib
import os

import pytest
from face_detection_helpers import face  # noqa: F401  (fixture)


def _source(tmp_path, name, data):
    src = tmp_path / "src" / name
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(data)
    return src.as_uri(), hashlib.sha256(data).hexdigest()


def test_missing_file_is_downloaded_and_verified(face, tmp_path):
    url, sha = _source(tmp_path, "m.onnx", b"model bytes")
    dest = tmp_path / "cache" / "face" / "m.onnx"
    face.model_files.ensure_models([(str(dest), url, sha)])
    assert dest.read_bytes() == b"model bytes"
    assert [p.name for p in dest.parent.iterdir()] == ["m.onnx"]          # no partial file left behind


def test_present_and_matching_file_is_not_downloaded_again(face, tmp_path):
    url, sha = _source(tmp_path, "m.onnx", b"model bytes")
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"model bytes")
    os.remove(url[len("file://"):])                                          # a download attempt would fail
    face.model_files.ensure_models([(str(dest), url, sha)])
    assert dest.read_bytes() == b"model bytes"


def test_file_that_does_not_match_is_replaced(face, tmp_path):
    url, sha = _source(tmp_path, "m.onnx", b"model bytes")
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"truncated")
    face.model_files.ensure_models([(str(dest), url, sha)])
    assert dest.read_bytes() == b"model bytes"


def test_download_with_the_wrong_checksum_fails_and_keeps_nothing(face, tmp_path):
    url, _ = _source(tmp_path, "m.onnx", b"some other model")
    dest = tmp_path / "cache" / "m.onnx"
    with pytest.raises(RuntimeError, match="SHA-256"):
        face.model_files.ensure_models([(str(dest), url, "0" * 64)])
    assert not dest.exists() and os.listdir(dest.parent) == []


def test_default_settings_point_at_the_cache_volume_and_pin_both_files(face):
    cfg = face.config
    assert cfg.FACE_MODEL_DIR == "/hf_cache/face" or "FACE_MODEL_DIR" in os.environ
    assert {os.path.basename(p) for p, _, _ in cfg.FACE_MODEL_FILES} <= {"retinaface_mv2.onnx", "glintr100.onnx"}
    assert all(len(sha) == 64 for _, _, sha in cfg.FACE_MODEL_FILES)
    assert "af6d057c9b0ec4071d4c49c80e3539258798b609" in cfg.FACE_REC_MODEL_URL   # a fixed revision, not main
