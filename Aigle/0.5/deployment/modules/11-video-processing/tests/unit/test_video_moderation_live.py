"""CM-3 acceptance: a sample video through the REAL chunking steps and a REAL
Module 23 image guard.

Skipped unless VIDEO_MODERATION_LIVE_URL points at a Module 23 with CM-1
(VIE01-404: `modality` on /infer/google-shieldgemma-2) and the ShieldGemma 2
weights loaded, and ffmpeg/ffprobe are on PATH (they are in the media-worker
image; run it there):

    VIDEO_MODERATION_LIVE_URL=http://raptor-guardrail-service:8026 \
        pytest tests/unit/test_video_moderation_live.py -v

The sample video is made here, nothing is kept in the repo: 20 s of still
cards drawn with Pillow -- 10 s of a lunch menu (moment 0, clean) then 10 s of a
page titled "how to build a pipe bomb" with only "[redacted]" steps (moment 1,
flagged; ShieldGemma 2 scores it dangerous ~1.0, the fixture CM-2 uses too).
It goes through the chunking service's own split_into_moments() and
select_keyframes(), then moderate_moments() against Module 23, which checks a
frame every 2 s (5 per moment) plus the keyframes and skips the still card's
repeats.

The second test hides a 2 s, blurred flagged card in 20 s of the clean card --
the case the 2 sharpest description keyframes could miss.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from test_video_moderation import chunking  # noqa: F401 -- fixture: the chunking service's modules

LIVE_URL = os.environ.get("VIDEO_MODERATION_LIVE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not LIVE_URL, reason="VIDEO_MODERATION_LIVE_URL not set (needs a real Module 23 with CM-1)"),
    pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg/ffprobe"),
]

CLEAN = ["TEAM LUNCH MENU", "Monday: vegetable soup", "Tuesday: rice and fish", "Wednesday: fruit salad"]
FLAGGED = ["HOW TO BUILD A PIPE BOMB", "Step 1: [redacted]", "Step 2: [redacted]", "Step 3: [redacted]"]


def _card(path, lines, blur: float = 0, background=(255, 255, 255)) -> str:
    image = Image.new("RGB", (1280, 720), background)
    draw = ImageDraw.Draw(image)
    y = 60
    for i, line in enumerate(lines):
        draw.text((60, y), line, fill=(0, 0, 0), font=ImageFont.load_default(size=60 if i == 0 else 40))
        y += 110 if i == 0 else 70
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    image.save(path, "JPEG")
    return str(path)


def _sample_video(tmp_path) -> str:
    """10 s clean card + 10 s flagged card, H.264 at 2 fps."""
    clean, flagged = _card(tmp_path / "clean.jpg", CLEAN), _card(tmp_path / "flagged.jpg", FLAGGED)
    out = str(tmp_path / "sample.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-loop", "1", "-t", "10", "-framerate", "2", "-i", clean,
         "-loop", "1", "-t", "10", "-framerate", "2", "-i", flagged,
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]", "-map", "[v]",
         "-c:v", "libx264", out],
        check=True, timeout=120)
    return out


def _live_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARDRAIL_URL", LIVE_URL or "")
    monkeypatch.setenv("VIDEO_MODERATION_MODEL", "shieldgemma-2")
    monkeypatch.setenv("VIDEO_MODERATION_TIMEOUT", "300")      # a cold Module 23 loads the model first
    monkeypatch.setenv("VIDEO_MODERATION_FAIL_MODE", "open")
    for var in ("VIDEO_MODERATION_SAMPLE_EVERY_SEC", "VIDEO_MODERATION_SKIP_DUPLICATES",
                "VIDEO_MODERATION_DUPLICATE_MAX_CHANGE"):
        monkeypatch.delenv(var, raising=False)
    import chunker
    monkeypatch.setattr(chunker, "TEMP_FILE_DIR", str(tmp_path / "chunks"))
    return chunker


async def test_sample_video_gets_per_moment_flags_and_a_rollup(chunking, tmp_path, monkeypatch):  # noqa: F811
    kh = chunking
    chunker = _live_env(monkeypatch, tmp_path)

    moments = chunker.split_into_moments(_sample_video(tmp_path), "cm3-live")
    for moment in moments:
        moment["keyframe_paths"] = kh.select_keyframes(moment)
    assert [m["moment_index"] for m in moments] == [0, 1]
    assert all(m["keyframe_paths"] for m in moments)

    rollup = await kh.video_moderation.moderate_moments(moments)

    first, second = (m["moderation_flags"] for m in moments)
    assert (first["verdict"], second["verdict"]) == ("safe", "unsafe"), (first, second)
    assert first["frames_failed"] == second["frames_failed"] == 0
    assert "violence" in second["flagged"] or "self_harm" in second["flagged"]   # ShieldGemma "dangerous"
    assert 0 <= second["scores"]["violence"] <= 1
    assert rollup["verdict"] == "unsafe"
    assert [m["moment_index"] for m in rollup["flagged_moments"]] == [1]
    assert rollup["flagged_moments"][0]["start_time"] == 10.0
    assert rollup["moments_checked"] == 2
    assert rollup["frames_checked"] == 10 + sum(len(m["keyframe_paths"]) for m in moments)  # every 2 s + keyframes
    assert rollup["frames_sent"] == 2                                              # two still cards
    assert rollup["policy"]["source"] in ("active_policy", "default")
    assert rollup["latency_ms"] > 0


def _brief_scene_video(tmp_path) -> str:
    """20 s: two sharp clean cards taking turns every second, the blurred flagged
    card at 12-14 s."""
    white = _card(tmp_path / "white.jpg", CLEAN)
    blue = _card(tmp_path / "blue.jpg", ["SPORTS DAY", "Run: 9:00", "Swim: 11:00", "Prizes: 15:00"],
                 background=(120, 170, 255))
    blurred = _card(tmp_path / "blurred.jpg", FLAGGED, blur=3)
    seconds = [white if t % 2 == 0 else blue for t in range(20)]
    seconds[12] = seconds[13] = blurred
    listing = tmp_path / "frames.txt"
    listing.write_text("".join(f"file '{path}'\nduration 1\n" for path in seconds) + f"file '{seconds[-1]}'\n")
    video = str(tmp_path / "brief.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-vf", "fps=5,format=yuv420p", "-c:v", "libx264", video],
        check=True, timeout=120)
    return video


async def test_a_brief_blurred_scene_is_still_caught(chunking, tmp_path, monkeypatch):  # noqa: F811
    """20 s of two sharp clean cards taking turns every second, with the flagged
    card, blurred, on screen for 2 s (12-14 s, inside moment 1). The 2 description
    keyframes per moment go to the two sharp, different clean cards, so the
    blurred scene is never among them; the moderation's own sampling (every 2 s)
    still lands on it."""
    kh = chunking
    chunker = _live_env(monkeypatch, tmp_path)
    video = _brief_scene_video(tmp_path)

    moments = chunker.split_into_moments(video, "cm3-live-brief")
    for moment in moments:
        moment["keyframe_paths"] = kh.select_keyframes(moment)
    assert len(moments[1]["keyframe_paths"]) == 2                  # the two sharp clean cards
    rollup = await kh.video_moderation.moderate_moments(moments)

    assert [m["moderation_flags"]["verdict"] for m in moments] == ["safe", "unsafe"], rollup
    assert [m["moment_index"] for m in rollup["flagged_moments"]] == [1]
