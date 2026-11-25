"""services/media_probe.py: the duration ffprobe reports, and every way it must fail quietly (VIE01-160)."""
from __future__ import annotations

import asyncio
import functools
import http.server
import os
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
from services.media_probe import (
    _parse_duration,
    is_probeable,
    make_duration_probe,
    probe_duration_s,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe missing")


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    """A 3 s video, a 2 s audio file and a text file named like a video, served over plain HTTP."""
    directory = tmp_path_factory.mktemp("media")
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10", "-movflags", "+faststart",
             "-pix_fmt", "yuv420p", str(directory / "clip.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=2", str(directory / "tone.wav")], check=True)
    (directory / "not-really.mp4").write_text("this is text, not a video")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _script(tmp_path, body: str):
    path = tmp_path / "fake-ffprobe"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# ── what it reports ─────────────────────────────────────────────────────────

@needs_ffmpeg
@pytest.mark.asyncio
async def test_reports_the_duration_of_a_video(media_dir):
    assert await probe_duration_s(f"{media_dir}/clip.mp4", "clip.mp4") == pytest.approx(3.0, abs=0.2)


@needs_ffmpeg
@pytest.mark.asyncio
async def test_reports_the_duration_of_an_audio_file(media_dir):
    assert await probe_duration_s(f"{media_dir}/tone.wav", "tone.wav") == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize(("name", "expected"), [
    ("clip.mp4", True), ("CLIP.MOV", True), ("a/b/song.mp3", True), ("tone.wav", True),
    ("report.pdf", False), ("big.bin", False), ("noextension", False), ("", False),
])
def test_only_names_that_say_audio_or_video_are_probed(name, expected):
    assert is_probeable(name) is expected


@pytest.mark.parametrize(("output", "expected"), [
    (b"30.753333\n", 30.753333), (b"3\n", 3.0), (b"N/A\n", 0.0), (b"", 0.0), (b"-1\n", 0.0),
    (b"0\n", 0.0), (b"inf\n", 0.0), (b"nan\n", 0.0), (b"garbage", 0.0),
])
def test_only_a_positive_finite_number_counts_as_a_duration(output, expected):
    assert _parse_duration(output) == pytest.approx(expected)


# ── every failure is "unknown", never an error ──────────────────────────────

@pytest.mark.asyncio
async def test_a_file_that_is_not_media_is_not_even_probed(monkeypatch):
    async def must_not_run(*args, **kwargs):
        raise AssertionError("a subprocess was started for a file that is not audio or video")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_run)
    assert await probe_duration_s("http://127.0.0.1:1/report.pdf", "report.pdf") == 0.0


@pytest.mark.asyncio
async def test_a_missing_binary_is_unknown():
    assert await probe_duration_s("http://127.0.0.1:1/a.mp4", "a.mp4", ffprobe_binary="/nonexistent/ffprobe") == 0.0


@needs_ffmpeg
@pytest.mark.asyncio
async def test_an_unreachable_url_is_unknown():
    assert await probe_duration_s("http://127.0.0.1:1/a.mp4", "a.mp4") == 0.0


@needs_ffmpeg
@pytest.mark.asyncio
async def test_a_file_ffprobe_cannot_read_is_unknown(media_dir):
    assert await probe_duration_s(f"{media_dir}/not-really.mp4", "not-really.mp4") == 0.0


@pytest.mark.asyncio
async def test_a_probe_that_times_out_is_unknown_and_its_process_is_killed(tmp_path):
    pid_file = tmp_path / "pid"
    binary = _script(tmp_path, f"echo $$ > {pid_file}\nexec sleep 30\n")
    started = time.monotonic()
    assert await probe_duration_s("http://x/a.mp4", "a.mp4", ffprobe_binary=binary, timeout_s=0.3) == 0.0
    assert time.monotonic() - started < 5
    assert not _alive(int(pid_file.read_text()))


@pytest.mark.asyncio
async def test_a_cancelled_probe_kills_its_process(tmp_path):
    pid_file = tmp_path / "pid"
    binary = _script(tmp_path, f"echo $$ > {pid_file}\nexec sleep 30\n")
    task = asyncio.ensure_future(probe_duration_s("http://x/a.mp4", "a.mp4", ffprobe_binary=binary, timeout_s=30))
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    pid = int(pid_file.read_text())
    assert _alive(pid)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _alive(pid)


@pytest.mark.asyncio
async def test_ffprobe_is_given_a_protocol_whitelist_and_the_url_as_one_argument(tmp_path):
    args_file = tmp_path / "args"
    binary = _script(tmp_path, f"printf '%s\\n' \"$@\" > {args_file}\necho 1.5\n")
    url = "http://host/some path/a.mp4?X-Amz=1&y=2"
    assert await probe_duration_s(url, "a.mp4", ffprobe_binary=binary) == pytest.approx(1.5)
    args = args_file.read_text().splitlines()
    assert args[args.index("-protocol_whitelist") + 1] == "http,https,tcp,tls,crypto"
    assert args[-1] == url


# ── the switch ──────────────────────────────────────────────────────────────

def test_the_probe_is_off_when_the_setting_is_off():
    assert make_duration_probe(SimpleNamespace(probe_duration=False, ffprobe_binary="ffprobe", probe_duration_timeout_s=3.0)) is None


@pytest.mark.asyncio
async def test_the_probe_uses_the_configured_binary_and_timeout(tmp_path):
    binary = _script(tmp_path, "echo 7.25\n")
    probe = make_duration_probe(SimpleNamespace(probe_duration=True, ffprobe_binary=binary, probe_duration_timeout_s=3.0))
    assert await probe("http://x/a.mp4", "a.mp4") == pytest.approx(7.25)
