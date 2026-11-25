# services/video_chunking_service/video_moderation.py
"""CM-3 (VIE01-406): video keyframe moderation -- each 10 s moment's frames go to
Module 23's image guard as `video_frame`, the frame verdicts are aggregated per
moment, and the moments are rolled up to one verdict for the whole video.

A moment's frames are:

* frames sampled evenly, the first frame of every VIDEO_MODERATION_SAMPLE_EVERY_SEC
  window (at 0, 2, 4 ... s) -- so every scene at least that long is checked,
  blurry or not;
* scene cuts (VIE01-406 follow-up): the first frame of every shot that starts
  and ends between two samples, so a brief shot of its own (0.4 s, 1 s) is
  checked too, blurry or not -- the samples step over it and the keyframes only
  land on it when it is sharp. Found in the same ffmpeg pass as the samples
  (`select` with `scene`); a shot that already has a sample costs nothing extra;
* its 2 description keyframes -- the sharpest, most varied frames; and
* for a moment none of the above landed in (a last moment of a fraction of a
  second), its first frame, so it isn't reported as unchecked. When the video
  has no picture there at all -- its audio runs a few hundredths of a second
  past the last frame, and the moments are cut by the file's duration -- the
  moment is `no_picture`: nothing to check, not an error.

What none of these catch: something shown for less than the sample interval
inside one continuous shot (no cut before it). A frame that barely differs from
the last one sent (a still scene, a slide, a talking head that hasn't moved, a
keyframe next to a sample) reuses that frame's verdict instead of costing
another GPU call.

Runs in video_chunking_service right after the moments are cut, so the flags
travel with the moments: the orchestrator writes the video's rollup to Module 04
(POST /update-moderation, as CM-2 does for images) and the indexer stores each
moment's flags on its Module 25 Chunk and the rollup on the Source. Moderation
only records a verdict; it never stops a video from being indexed (unless the
fail mode is closed and Module 23 gives no verdict).

The categories, thresholds and decisions are Module 23's (CM-1, VIE01-404): with
`modality=video_frame` its answer carries a `moderation` block built from the
active policy (or its built-in default). Nothing here re-scores a frame.

Settings (environment):

* VIDEO_MODERATION_ENABLED -- false (default): no Module 23 call at all.
* GUARDRAIL_URL -- Module 23's base URL (same name modules 07, 10 and 13 read).
* VIDEO_MODERATION_MODEL -- "shieldgemma-2" (default) or "llama-guard-3-vision".
* VIDEO_MODERATION_SAMPLE_EVERY_SEC -- seconds between sampled frames (default 2,
  i.e. 5 per 10 s moment, plus its 2 keyframes; 1 catches every ~1 s scene for
  ~70 % more GPU time, 5 costs ~40 % less).
* VIDEO_MODERATION_SCENE_CUTS -- true (default): also check brief shots between two
  samples (above); false: samples and keyframes only.
* VIDEO_MODERATION_SCENE_THRESHOLD -- ffmpeg's scene-change score (0-1, default 0.3)
  above which a frame starts a new shot. Lower finds softer cuts and costs more.
* VIDEO_MODERATION_MAX_CUTS_PER_MOMENT -- at most this many brief-shot frames per
  moment (default 5), so fast-cut footage can't multiply the GPU time.
* VIDEO_MODERATION_SKIP_DUPLICATES -- true (default): a frame that barely differs
  from one already sent (for its moment, or the last one sent) reuses its verdict.
* VIDEO_MODERATION_DUPLICATE_MAX_CHANGE -- the share of pixels (0-1, default 0.005)
  that may change for a frame to still count as a duplicate. Pixels, not colour
  histograms, so a small new object in one corner still counts as a change.
* VIDEO_MODERATION_CONCURRENCY -- frame calls in flight at once (default 2; Module
  23 runs one inference per GPU at a time anyway).
* VIDEO_MODERATION_TIMEOUT -- seconds per frame call (default 180: a cold Module 23
  loads the model first).
* VIDEO_MODERATION_FAIL_MODE -- "open" (default): a frame Module 23 gives no verdict
  for is counted as failed and the video is still indexed; "closed": any such
  frame fails the video's job.
"""
from __future__ import annotations

import asyncio
import bisect
import logging
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_ENDPOINTS = {
    "shieldgemma-2": "/infer/google-shieldgemma-2",
    "llama-guard-3-vision": "/infer/Llama-Guard-3-11B-Vision",
}
_DECISION_RANK = {"allow": 0, "warn": 1, "review": 2, "block": 3}


class ModerationUnavailable(Exception):
    """Raised by moderate_moments() when the fail mode is closed and at least one
    frame got no verdict (or the settings are invalid)."""


def enabled() -> bool:
    return os.environ.get("VIDEO_MODERATION_ENABLED", "false").strip().lower() == "true"


def fail_closed() -> bool:
    return os.environ.get("VIDEO_MODERATION_FAIL_MODE", "open").strip().lower() == "closed"


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if low <= value <= high else default


def _skip_duplicates() -> bool:
    return os.environ.get("VIDEO_MODERATION_SKIP_DUPLICATES", "true").strip().lower() != "false"


def _scene_cuts() -> bool:
    return os.environ.get("VIDEO_MODERATION_SCENE_CUTS", "true").strip().lower() != "false"


def _settings() -> tuple[str, str, float]:
    """(model, url, timeout). ValueError on an unknown model."""
    model = os.environ.get("VIDEO_MODERATION_MODEL", "shieldgemma-2").strip()
    if model not in _ENDPOINTS:
        raise ValueError(f"VIDEO_MODERATION_MODEL must be one of {sorted(_ENDPOINTS)}, not {model!r}")
    url = os.environ.get("GUARDRAIL_URL", "http://raptor-guardrail-service:8026").rstrip("/") + _ENDPOINTS[model]
    return model, url, float(os.environ.get("VIDEO_MODERATION_TIMEOUT", "180"))


def parse_frame(body: Any) -> dict[str, Any]:
    """Module 23's `moderation` block for one frame; ValueError when the answer
    has none (a Module 23 without CM-1 ignores `modality`)."""
    if not isinstance(body, dict):
        raise ValueError("response is not a JSON object")
    moderation = body.get("moderation")
    if not isinstance(moderation, dict) or moderation.get("decision") not in _DECISION_RANK \
            or not isinstance(moderation.get("flagged"), list):
        raise ValueError("no moderation verdict in the response -- Module 23 needs CM-1 (VIE01-404)")
    return moderation


# ── frame sampling ──────────────────────────────────────────────────────────────

_PTS_TIME = re.compile(r"\bpts_time:(-?[0-9.]+)")
_SCENE_SCORE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def _select_filter(every: float, threshold: float | None) -> str:
    """ffmpeg filters picking the first frame of every `every`-second window (so
    frame times are exact: 0, 2, 4 ... s; the `fps` filter takes the middle of each
    window instead) plus, with a threshold, every frame whose scene-change score is
    above it -- the first frame of a new shot. `metadata=print` logs each picked
    frame's scene score (when it has one), `showinfo` its time."""
    expr = f"isnan(prev_t)+gt(floor(t/{every})\\,floor(prev_t/{every}))"
    if threshold is not None:
        expr += f"+gt(scene\\,{threshold})"
    return f"select={expr},metadata=print:key=lavfi.scene_score,showinfo"


def _parse_select_log(log: str) -> list[tuple[float, float]]:
    """(time, scene score) of each frame showinfo logged, in order. metadata=print
    logs a frame's score just before showinfo logs the frame; a frame without one
    (scene cuts off) has 0."""
    entries: list[tuple[float, float]] = []
    score = 0.0
    for line in log.splitlines():
        if "Parsed_metadata" in line and (m := _SCENE_SCORE.search(line)):
            score = float(m.group(1))
        elif "Parsed_showinfo" in line and (m := _PTS_TIME.search(line)):
            entries.append((float(m.group(1)), score))
            score = 0.0
    return entries


def _extract_frames(video_path: str, every: float, threshold: float | None,
                    out_dir: str) -> list[tuple[float, str, float]]:
    """(time, path, scene score) of every frame _select_filter picks, in one ffmpeg
    pass, GPU decode first and CPU if that fails, as frame_selector.extract_frames
    does. A timed-out or crashed ffmpeg, or frames whose times can't be read, count
    as a failed attempt -- the caller falls back to the description keyframes."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "f_%06d.jpg")
    vf = _select_filter(every, threshold)
    out = ["-vsync", "vfr", "-q:v", "2", pattern]
    gpu = ["ffmpeg", "-y", "-nostdin", "-nostats", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-noautorotate", "-i", video_path, "-vf", f"hwdownload,format=nv12,{vf}", *out]
    cpu = ["ffmpeg", "-y", "-nostdin", "-nostats", "-i", video_path, "-vf", vf, *out]
    for cmd in (gpu, cpu):
        log = ""
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, timeout=3600)
            log = result.stderr.decode(errors="replace")
            failure = None if result.returncode == 0 else log[-300:]
        except (subprocess.SubprocessError, OSError) as exc:
            failure = str(exc)
        files = sorted(f for f in os.listdir(out_dir) if f.startswith("f_") and f.endswith(".jpg"))
        times = _parse_select_log(log)
        if failure is None and files and len(times) != len(files):
            failure = f"{len(files)} frames but {len(times)} frame times in ffmpeg's log"
        if failure is None and files:
            return [(t, os.path.join(out_dir, f), score) for (t, score), f in zip(times, files)]
        logger.warning(f"[moderation] frame sampling ({'gpu' if '-hwaccel' in cmd else 'cpu'}) failed: {failure}")
        for f in files:
            os.remove(os.path.join(out_dir, f))
    return []


def _extract_at(video_path: str, t: float, out_path: str) -> tuple[str | None, bool]:
    """(path, no_picture) for one frame at `t` seconds -- the first frame of a moment
    nothing else landed in. no_picture is True when ffmpeg ran fine but the video
    has no picture there: the moment is past the end of the video stream, as in a
    file whose audio runs a few hundredths of a second longer than its video (the
    moments are cut by the file's duration, audio included)."""
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", video_path,
           "-frames:v", "1", "-q:v", "2", out_path]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"[moderation] no frame at {t:.3f} s: {exc}")
        return None, False
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, False
    if result.returncode != 0:
        logger.warning(f"[moderation] no frame at {t:.3f} s: {result.stderr.decode(errors='replace')[-300:]}")
    return None, result.returncode == 0


def split_frames(frames: list[tuple[float, str, float]], every: float, threshold: float | None,
                 video_end: float) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """(samples, brief-shot frames) of the frames _extract_frames picked. A sample is
    the first frame picked in its `every`-second window. A cut starts a shot that
    lasts until the next cut (or the end of the video); its frame is kept only when
    no sample falls inside that shot -- a shot with a sample is already checked."""
    samples: list[tuple[float, str]] = []
    cuts: list[tuple[float, str, bool]] = []          # (time, path, is also a sample)
    windows: set[int] = set()
    for t, path, score in frames:
        window = math.floor(t / every + 1e-6)
        is_sample = window not in windows
        if is_sample:
            windows.add(window)
            samples.append((t, path))
        if threshold is not None and score > threshold:
            cuts.append((t, path, is_sample))
    sample_times = [t for t, _ in samples]
    brief: list[tuple[float, str]] = []
    for i, (t, path, is_sample) in enumerate(cuts):
        shot_end = cuts[i + 1][0] if i + 1 < len(cuts) else video_end
        first_sample = bisect.bisect_left(sample_times, t)
        if not is_sample and not (first_sample < len(sample_times) and sample_times[first_sample] < shot_end):
            brief.append((t, path))
    return samples, brief


def sample_frames(moments: list[dict[str, Any]], every: float, threshold: float | None = None,
                  max_cuts: int = 5) -> tuple[list[list[str]], str | None, list[int], set[int]]:
    """The frames to check for each moment -- its samples and brief-shot frames in
    time order, then its description keyframes -- the directory to delete
    afterwards (None when there is no video to sample, e.g. moments built without
    one), how many brief-shot frames each moment got, and the moments with no
    picture at all (past the end of the video stream). A frame belongs to the
    moment whose [start_time, end_time) holds its time. A moment left with no frame
    at all gets its first frame."""
    keyframes = [list(m.get("keyframe_paths") or []) for m in moments]
    if not moments or not moments[0].get("video_path") or not moments[0].get("output_dir"):
        return keyframes, None, [0] * len(moments), set()
    video = moments[0]["video_path"]
    out_dir = os.path.join(os.path.dirname(moments[0]["output_dir"]), "moderation_frames")
    frames = _extract_frames(video, every, threshold, out_dir)
    samples, brief = split_frames(frames, every, threshold, float(moments[-1].get("end_time") or 0.0))

    def moment_of(t: float) -> int | None:
        return next((k for k, m in enumerate(moments) if m["start_time"] <= t < m["end_time"]), None)

    timed: list[list[tuple[float, str]]] = [[] for _ in moments]
    for t, path in samples:
        if (k := moment_of(t)) is not None:
            timed[k].append((t, path))
    cut_counts = [0] * len(moments)
    for t, path in brief:
        k = moment_of(t)
        if k is None:
            continue
        if cut_counts[k] >= max_cuts:
            logger.debug(f"[moderation] moment {k}: more than {max_cuts} brief shots, skipping the one at {t} s")
            continue
        timed[k].append((t, path))
        cut_counts[k] += 1
    lists = [[path for _, path in sorted(row)] + kf for row, kf in zip(timed, keyframes)]
    no_picture: set[int] = set()
    for k, m in enumerate(moments):
        if lists[k]:
            continue
        first, none_there = _extract_at(video, float(m["start_time"]), os.path.join(out_dir, f"start_{k:04d}.jpg"))
        if first:
            lists[k] = [first]
        elif none_there:
            no_picture.add(k)
    return lists, out_dir, cut_counts, no_picture


def changed_share(path_a: str, path_b: str) -> float | None:
    """Share of pixels (0-1) that differ noticeably between two frames, compared
    as 160x90 greyscale; None when either can't be read."""
    try:
        import cv2  # in the worker image (frame_selector uses it too)
    except ImportError:
        return None  # can't compare: the frame is simply sent

    a, b = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE), cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return None
    a = cv2.resize(a, (160, 90), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (160, 90), interpolation=cv2.INTER_AREA)
    return float((cv2.absdiff(a, b) > 25).mean())


def plan_calls(frame_lists: list[list[str]], skip_duplicates: bool,
               max_change: float) -> tuple[list[str], list[list[int]]]:
    """Which frames go to Module 23, and whose verdict each frame gets. Returns
    (frames to send, per moment: for each of its frames the index into the frames
    sent). A frame that barely differs from a frame already sent -- one sent for
    the same moment, or the last one sent before it -- points at that one instead
    of being sent again; an unreadable frame is always sent (and fails there,
    counted like any other failed frame)."""
    sent: list[str] = []
    plan: list[list[int]] = []
    for paths in frame_lists:
        row: list[int] = []
        for path in paths:
            match = None
            if skip_duplicates and sent:
                candidates = list(dict.fromkeys(row + [len(sent) - 1]))   # this moment's, then the last sent
                for i in candidates:
                    share = changed_share(sent[i], path)
                    if share is not None and share <= max_change:
                        match = i
                        break
            if match is None:
                sent.append(path)
                match = len(sent) - 1
            row.append(match)
        plan.append(row)
    return sent, plan


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


async def _moderate_frame(session: aiohttp.ClientSession, url: str, path: str) -> dict[str, Any]:
    image_bytes = await asyncio.to_thread(_read_bytes, path)
    form = aiohttp.FormData()
    form.add_field("image", image_bytes, filename=os.path.basename(path))
    form.add_field("modality", "video_frame")
    async with session.post(url, data=form) as resp:
        if resp.status != 200:
            raise ValueError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        return parse_frame(await resp.json(content_type=None))


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _no_picture_flags(model: str) -> dict[str, Any]:
    """A moment with no picture to check (past the end of the video stream): not an
    error, nothing flagged, and left out of moments_checked/moments_failed."""
    return {"model": model, "verdict": "no_picture", "decision": None, "flagged": [], "scores": {},
            "frames_checked": 0, "frames_failed": 0, "frames_reused": 0, "frames_scene_cut": 0}


def aggregate_moment(frames: list[dict[str, Any] | BaseException], model: str) -> dict[str, Any]:
    """One moment's flags from its frames' Module 23 verdicts (or the exception a
    frame failed with): the worst decision, every flagged category, and each
    category's highest score (ShieldGemma 2 only -- Llama Guard gives none)."""
    ok = [f for f in frames if isinstance(f, dict)]
    failed = [f for f in frames if not isinstance(f, dict)]
    flags: dict[str, Any] = {"model": model, "frames_checked": len(ok), "frames_failed": len(failed)}
    if failed:
        flags["error"] = _reason(failed[0])
    if not ok:
        flags.update(verdict="error", decision=None, flagged=[], scores={})
        if not frames:
            flags["error"] = "no frames"
        return flags

    flagged: list[str] = []
    scores: dict[str, float] = {}
    for frame in ok:
        for name in frame["flagged"]:
            if name not in flagged:
                flagged.append(name)
        for name, cat in (frame.get("categories") or {}).items():
            score = cat.get("score") if isinstance(cat, dict) else None
            if isinstance(score, (int, float)):
                scores[name] = max(scores.get(name, 0.0), float(score))
    flags.update(
        verdict="unsafe" if flagged else "safe",
        decision=max((f["decision"] for f in ok), key=_DECISION_RANK.__getitem__),
        flagged=flagged,
        scores=scores,
        not_covered=ok[0].get("not_covered", []),
    )
    return flags


def rollup(moments: list[dict[str, Any]], model: str, policy: dict[str, Any] | None,
           started: float) -> dict[str, Any]:
    """The video's moderation_flags (Module 04 + Module 25 Source) from its
    moments' flags. `unsafe` when any moment is; otherwise `error` when any moment
    could not be checked -- an incompletely checked video is never called safe --
    else `safe`. A `no_picture` moment (past the end of the video stream) is
    neither checked nor failed."""
    per_moment = [m.get("moderation_flags") or {} for m in moments]
    unsafe = [(m, f) for m, f in zip(moments, per_moment) if f.get("verdict") == "unsafe"]
    failed = [f for f in per_moment if f.get("verdict") == "error"]
    no_picture = sum(1 for f in per_moment if f.get("verdict") == "no_picture")
    flagged: list[str] = []
    for _, f in unsafe:
        flagged.extend(n for n in f["flagged"] if n not in flagged)
    decisions = [f["decision"] for f in per_moment if f.get("decision")]
    verdict = "unsafe" if unsafe else ("error" if failed else "safe")
    flags = {
        "verdict": verdict,
        "model": model,
        "modality": "video_frame",
        "decision": max(decisions, key=_DECISION_RANK.__getitem__) if decisions else None,
        "flagged": flagged,
        "flagged_moments": [
            {"moment_index": m.get("moment_index"), "start_time": m.get("start_time"),
             "end_time": m.get("end_time"), "decision": f["decision"], "flagged": f["flagged"]}
            for m, f in unsafe
        ],
        "moments_checked": len(per_moment) - len(failed) - no_picture,
        "moments_failed": len(failed),
        "moments_no_picture": no_picture,
        "frames_checked": sum(f.get("frames_checked", 0) for f in per_moment),
        "frames_failed": sum(f.get("frames_failed", 0) for f in per_moment),
        "frames_reused": sum(f.get("frames_reused", 0) for f in per_moment),
        "policy": policy,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if failed:
        flags["error"] = next((f.get("error") for f in failed if f.get("error")), "moment not checked")
    return flags


def _policy(frame: dict[str, Any]) -> dict[str, Any]:
    return {k: frame.get(k) for k in ("source", "policy_name", "policy_version", "moderation_version")}


async def moderate_moments(moments: list[dict[str, Any]]) -> dict[str, Any]:
    """Sample each moment's frames, moderate them, set each moment's
    `moderation_flags` and return the video's rollup. Raises ModerationUnavailable
    only in closed fail mode (or on invalid settings, whatever the mode)."""
    started = time.perf_counter()
    try:
        model, url, timeout = _settings()
    except ValueError as exc:
        raise ModerationUnavailable(str(exc)) from exc
    every = _float_env("VIDEO_MODERATION_SAMPLE_EVERY_SEC", 2.0, 0.1, 60.0)
    max_change = _float_env("VIDEO_MODERATION_DUPLICATE_MAX_CHANGE", 0.005, 0.0, 1.0)
    threshold = _float_env("VIDEO_MODERATION_SCENE_THRESHOLD", 0.3, 0.01, 1.0) if _scene_cuts() else None
    max_cuts = _int_env("VIDEO_MODERATION_MAX_CUTS_PER_MOMENT", 5)
    gate = asyncio.Semaphore(_int_env("VIDEO_MODERATION_CONCURRENCY", 2))

    frame_lists, tmp_dir, cut_counts, no_picture = await asyncio.to_thread(
        sample_frames, moments, every, threshold, max_cuts)
    try:
        sent, plan = await asyncio.to_thread(plan_calls, frame_lists, _skip_duplicates(), max_change)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async def one(path: str) -> dict[str, Any] | BaseException:
                async with gate:
                    try:
                        return await _moderate_frame(session, url, path)
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as exc:  # asyncio's: the worker runs Python 3.10
                        logger.warning(f"[moderation] {model} gave no verdict for {path}: {_reason(exc)}")
                        return exc

            results = await asyncio.gather(*(one(p) for p in sent))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    policy = None
    first_use: set[int] = set()
    for k, (moment, row, cuts) in enumerate(zip(moments, plan, cut_counts)):
        if k in no_picture:
            moment["moderation_flags"] = _no_picture_flags(model)
            continue
        frames = [results[i] for i in row]
        reused = 0
        for i in row:
            if i in first_use:
                reused += 1      # a duplicate: this verdict was already used by an earlier frame
            else:
                first_use.add(i)
        flags = aggregate_moment(frames, model)
        flags["frames_reused"] = reused
        flags["frames_scene_cut"] = cuts
        moment["moderation_flags"] = flags
        if policy is None:
            policy = next((_policy(f) for f in frames if isinstance(f, dict)), None)

    flags = rollup(moments, model, policy, started)
    flags["sample_every_sec"] = every
    flags["frames_sent"] = len(sent)
    flags["scene_threshold"] = threshold
    flags["frames_scene_cut"] = sum(cut_counts)
    unchecked = sum(1 for k, row in enumerate(plan) if not row and k not in no_picture)
    if fail_closed() and flags["frames_failed"] + unchecked > 0:
        raise ModerationUnavailable(f"{flags['frames_failed']} frame(s) got no verdict: {flags.get('error')}")
    logger.info(f"[moderation] {len(moments)} moments, {flags['frames_checked']} frames "
                f"({flags['frames_sent']} sent, every {every} s, {flags['frames_scene_cut']} brief shots): {flags['verdict']} "
                f"(flagged moments {len(flags['flagged_moments'])}, failed {flags['moments_failed']}, "
                f"{flags['latency_ms']} ms)")
    return flags
