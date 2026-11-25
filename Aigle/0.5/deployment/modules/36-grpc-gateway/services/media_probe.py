"""Media duration for DownloadAsset's MediaHeader.duration_sec (VIE01-160).

Nothing in Raptor records how long a video or audio file is, so the gateway asks ffprobe (it ships
with the ffmpeg already in the image) for the container's duration, reading the same presigned URL the
download streams from. It is a best effort and never gets in the way of a download:

- only files whose name says audio or video are probed, everything else is 0.0 straight away;
- the probe runs concurrently with opening the storage stream and is given `timeout_s`, so the header is
  delayed by at most that long, and usually by much less;
- a missing binary, an unreachable URL, a file ffprobe cannot read, a timeout, or a cancelled download
  all end the same way: 0.0 ("unknown"), and the subprocess is killed.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
from collections.abc import Awaitable, Callable

_logger = logging.getLogger("grpc_gateway.media_probe")

DurationProbe = Callable[[str, str], Awaitable[float]]
"""(presigned url, file name) -> duration in seconds, or 0.0 when unknown."""

_MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".3gp",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".amr",
})


def is_probeable(filename: str) -> bool:
    return os.path.splitext(filename or "")[1].lower() in _MEDIA_EXTENSIONS


def _parse_duration(output: bytes) -> float:
    try:
        value = float(output.decode("utf-8", "replace").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0  # "N/A", empty, or anything else that is not a number
    return value if math.isfinite(value) and value > 0 else 0.0


async def probe_duration_s(url: str, filename: str, *, ffprobe_binary: str = "ffprobe", timeout_s: float = 3.0) -> float:
    if not is_probeable(filename):
        return 0.0
    args = [
        ffprobe_binary, "-v", "error",
        # the URL is built by the gateway, but never let ffprobe follow anything but plain http(s)
        "-protocol_whitelist", "http,https,tcp,tls,crypto",
        "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        url,
    ]
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, stdin=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        return _parse_duration(stdout) if proc.returncode == 0 else 0.0
    except asyncio.TimeoutError:
        _logger.info("ffprobe did not answer within %.1fs for %s; duration_sec stays 0.", timeout_s, filename)
        return 0.0
    except (OSError, ValueError) as exc:  # no such binary, cannot exec
        _logger.warning("ffprobe could not be run (%s); duration_sec stays 0.", exc)
        return 0.0
    finally:
        # a timeout, a cancelled download or an error all leave no process behind
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()


def make_duration_probe(settings) -> DurationProbe | None:
    """The probe the server uses, or None when GRPC_GATEWAY_PROBE_DURATION is off."""
    if not settings.probe_duration:
        return None

    async def probe(url: str, filename: str) -> float:
        return await probe_duration_s(
            url, filename, ffprobe_binary=settings.ffprobe_binary, timeout_s=settings.probe_duration_timeout_s
        )

    return probe


class _ProbeHandle:
    def __init__(self, task: asyncio.Future | None):
        self._task = task

    async def duration_s(self) -> float:
        if self._task is None:
            return 0.0
        try:
            return await self._task
        except Exception:
            _logger.warning("duration probe failed; duration_sec stays 0.", exc_info=True)
            return 0.0  # a probe must never fail a download


@contextlib.asynccontextmanager
async def background_probe(probe: DurationProbe | None, url: str, filename: str):
    """Runs the probe as a task alongside the storage request and cancels it when the download ends
    or is cancelled, so a slow ffprobe can never outlive its stream."""
    task = asyncio.ensure_future(probe(url, filename)) if probe is not None else None
    try:
        yield _ProbeHandle(task)
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
