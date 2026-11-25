"""
MTR-7 / VIE01-306: GPU-only thumbnail extraction worker.

Deliberately separate from Module 04's main `app` service -- same pattern
module 11 (video-processing) uses for its own GPU-needing stages
(video-chunking, video-ocr-frame, video-frame-description): isolate the
one thing that actually needs NVDEC/CUDA into its own container, so a
crash or driver hiccup here can never take down the core asset-management
API, and `docker-compose.yml`'s GPU reservation stays scoped to this one
service instead of Module 04's whole container.

Called synchronously over HTTP by asset_management/thumbnail_service.py
(internal-network only, not part of the public API) -- same style module
11's video-frame-description uses to call video-ocr-frame directly
(VIDEO_OCR_SYNC_URL), not a Kafka round trip. This feature is a plain
GET-and-get-bytes-back request, not an async job, so synchronous HTTP is
the right fit here even though most of this codebase's GPU work is
Kafka-driven.

All business logic (404/409/400 on the asset's own state, the 7-day cache,
presigned-URL generation) stays in thumbnail_service.py / Module 04's main
app -- this worker only knows "given a URL and a timestamp, probe its
duration and decode one frame." It owns no database, no cache, no asset
state.
"""
import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FFPROBE_TIMEOUT_SECONDS = 15
FFMPEG_TIMEOUT_SECONDS = 30
FFMPEG_CPU_FALLBACK_TIMEOUT_SECONDS = 20

app = FastAPI()


class ExtractRequest(BaseModel):
    url: str
    second: float


async def _probe_duration(url: str) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        url,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=FFPROBE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise HTTPException(status_code=502, detail="Timed out probing video duration")

    if proc.returncode != 0 or not stdout.strip():
        logger.error(f"ffprobe failed for thumbnail duration check: {stderr.decode(errors='replace')}")
        raise HTTPException(status_code=502, detail="Failed to probe video duration")
    try:
        return float(stdout.strip())
    except ValueError:
        raise HTTPException(status_code=502, detail="Could not parse video duration")


async def _run_ffmpeg_decode(url: str, second: float, hwaccel: bool, timeout: float) -> bytes:
    # No -hwaccel_output_format cuda: that keeps the decoded frame as an
    # opaque CUDA hwframe, which hwdownload can only transfer back verbatim
    # -- its own `format=` option asks it to BE that format, not convert to
    # it, so hwdownload rejects anything that isn't the source's actual
    # native format (nv12 only for plain 8-bit 4:2:0; a 10-bit source
    # decodes to p010le/p016le, triggering "Invalid output format nv12 for
    # hwframe download", confirmed live against a real 10-bit HEVC asset).
    # Without it, ffmpeg's hwaccel layer downloads automatically in
    # whatever software format the source actually is, and the plain
    # (non-hwframe) `format` filter converts *that* to yuv420p in software
    # -- unlike hwdownload, that one is a real converter, not a verbatim
    # transfer, so it isn't tied to a single source format. NVDEC decode
    # itself is unaffected either way (confirmed via -loglevel verbose).
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-ss", str(second), "-i", url,
        "-frames:v", "1",
        "-vf", "format=yuv420p",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2",
        "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"timed out ({'NVDEC' if hwaccel else 'CPU'})")

    if proc.returncode != 0 or not stdout:
        raise RuntimeError(
            f"{'NVDEC' if hwaccel else 'CPU'} decode failed: {stderr.decode(errors='replace')}"
        )
    return stdout


async def _extract_frame_nvdec(url: str, second: float) -> bytes:
    """NVDEC first; falls back to software decode on any failure (timeout,
    non-zero exit) -- the GPU can be genuinely out of memory under contention
    from other services sharing it (confirmed live, not a driver quirk), and
    thumbnail extraction has no business failing outright just because the
    GPU is busy with something else."""
    try:
        return await _run_ffmpeg_decode(url, second, hwaccel=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except RuntimeError as nvdec_error:
        logger.warning(f"ffmpeg NVDEC thumbnail extraction failed, retrying on CPU: {nvdec_error}")
        try:
            return await _run_ffmpeg_decode(
                url, second, hwaccel=False, timeout=FFMPEG_CPU_FALLBACK_TIMEOUT_SECONDS
            )
        except RuntimeError as cpu_error:
            logger.error(f"ffmpeg CPU fallback also failed: {cpu_error}")
            raise HTTPException(status_code=502, detail="Thumbnail extraction failed")


@app.post("/extract")
async def extract(req: ExtractRequest) -> Response:
    if req.second < 0:
        raise HTTPException(status_code=400, detail="second must be >= 0")

    duration = await _probe_duration(req.url)
    if req.second > duration:
        raise HTTPException(
            status_code=400,
            detail=f"second={req.second} is beyond video duration ({duration:.2f}s)",
        )

    jpeg_bytes = await _extract_frame_nvdec(req.url, req.second)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/health")
async def health():
    return {"status": "ok"}
