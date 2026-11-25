"""P13: the video orchestrator reads when the recording was made from its container (capture_time.py) and
puts it on every moment; Module 25 validates it and dates the recording's deictic phrases from it.

The tag choice is tested on ffprobe's JSON; when ffprobe is installed, real files made with ffmpeg too.
"""
import json
import shutil
import subprocess
import types

import capture_time
import pytest


def _probe(monkeypatch, fmt=None, streams=(), returncode=0):
    out = json.dumps({"format": {"tags": fmt or {}}, "streams": [{"tags": t} for t in streams]}).encode()
    monkeypatch.setattr(capture_time.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=returncode, stdout=out, stderr=b""))


def test_quicktime_creationdate_wins_and_keeps_its_offset(monkeypatch):
    _probe(monkeypatch, {"creation_time": "2015-03-20T04:15:00.000000Z",
                         "com.apple.quicktime.creationdate": "2015-03-20T12:15:00+0800"})
    assert capture_time.read_capture_time("v.mov") == ("2015-03-20T12:15:00+08:00", "container_creation_time")


def test_container_creation_time_from_format_or_stream(monkeypatch):
    _probe(monkeypatch, {"creation_time": "2015-03-20T04:15:00.000000Z"})
    assert capture_time.read_capture_time("v.mp4") == ("2015-03-20T04:15:00.000000Z", "container_creation_time")
    _probe(monkeypatch, {}, streams=[{"creation_time": "2015-03-20T04:15:00Z"}])
    assert capture_time.read_capture_time("v.mp4") == ("2015-03-20T04:15:00Z", "container_creation_time")


def test_wav_broadcast_extension(monkeypatch):
    _probe(monkeypatch, {"origination_date": "2015-03-20", "origination_time": "12:15:00"})
    assert capture_time.read_capture_time("a.wav") == ("2015-03-20T12:15:00", "container_creation_time")


def test_wav_broadcast_extension_as_ffprobe_reports_it(monkeypatch):
    _probe(monkeypatch, {"date": "2015-03-20", "creation_time": "12:15:00", "time_reference": "0"})
    assert capture_time.read_capture_time("a.wav") == ("2015-03-20T12:15:00", "container_creation_time")


def test_a_time_without_a_full_date_is_not_used(monkeypatch):
    _probe(monkeypatch, {"date": "2015", "creation_time": "12:15:00"})
    assert capture_time.read_capture_time("a.wav") == (None, None)


def test_an_id3_release_date_is_not_a_recording_time(monkeypatch):
    _probe(monkeypatch, {"date": "2015", "title": "x"})
    assert capture_time.read_capture_time("a.mp3") == (None, None)


def test_failures_only_lose_the_date(monkeypatch):
    _probe(monkeypatch, returncode=1)
    assert capture_time.read_capture_time("bad.mp4") == (None, None)

    def missing(*a, **k):
        raise FileNotFoundError("ffprobe")
    monkeypatch.setattr(capture_time.subprocess, "run", missing)
    assert capture_time.read_capture_time("v.mp4") == (None, None)


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")


def _make(path, *args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(path)], check=True)


@needs_ffmpeg
def test_real_mp4(tmp_path):
    _make(tmp_path / "v.mp4", "-f", "lavfi", "-i", "color=c=black:s=16x16:d=1",
          "-metadata", "creation_time=2015-03-20T04:15:00Z")
    value, source = capture_time.read_capture_time(str(tmp_path / "v.mp4"))
    assert value.startswith("2015-03-20T04:15:00") and source == "container_creation_time"


@needs_ffmpeg
def test_real_mov_with_quicktime_creationdate(tmp_path):
    _make(tmp_path / "v.mov", "-f", "lavfi", "-i", "color=c=black:s=16x16:d=1", "-movflags", "use_metadata_tags",
          "-metadata", "com.apple.quicktime.creationdate=2015-03-20T12:15:00+0800")
    assert capture_time.read_capture_time(str(tmp_path / "v.mov")) == ("2015-03-20T12:15:00+08:00",
                                                                      "container_creation_time")


@needs_ffmpeg
def test_real_file_without_a_date(tmp_path):
    _make(tmp_path / "v.mp4", "-f", "lavfi", "-i", "color=c=black:s=16x16:d=1", "-map_metadata", "-1")
    assert capture_time.read_capture_time(str(tmp_path / "v.mp4")) == (None, None)
