import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

USE_NVENC = False
FORCE_CPU = os.environ.get("TRANSCODE_GPU_ENABLE", "false").lower() != "true"
CACHE_DIR = Path("/cache")
CACHE_MAX_BYTES = 10 * 1024 ** 3  # 10 GB


async def _check_nvenc() -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=0.1:size=256x256:rate=1",
            "-c:v", "h264_nvenc", "-f", "null", "-",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global USE_NVENC
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if FORCE_CPU:
        USE_NVENC = False
        print("[transcode] NVENC: disabled (forced CPU)", flush=True)
    else:
        USE_NVENC = await _check_nvenc()
        print(f"[transcode] NVENC: {'enabled (GPU)' if USE_NVENC else 'disabled (CPU fallback)'}", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


def _cache_key(url: str) -> str:
    parsed = urlparse(url)
    stable = parsed.scheme + "://" + parsed.netloc + parsed.path
    return hashlib.sha256(stable.encode()).hexdigest()


def _evict_cache():
    files = sorted(CACHE_DIR.glob("*.mp4"), key=lambda f: f.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    while total > CACHE_MAX_BYTES and files:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


def _stream_args(source_url: str) -> list[str]:
    if USE_NVENC:
        return [
            "ffmpeg", "-y",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", "hevc_cuvid",
            "-i", source_url,
            "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28",
            "-c:a", "aac",
            "-movflags", "frag_keyframe+empty_moov",
            "-f", "mp4", "pipe:1",
        ]
    return [
        "ffmpeg", "-y", "-i", source_url,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac",
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4", "pipe:1",
    ]


@app.get("/transcode/")
async def transcode(url: str):
    key = _cache_key(url)
    cached = CACHE_DIR / f"{key}.mp4"

    # Cache hit — full file, supports range requests / seeking
    if cached.exists():
        print(f"[transcode] cache hit: {key[:12]}", flush=True)
        return FileResponse(str(cached), media_type="video/mp4")

    # Cache miss — stream immediately, save to cache in background
    print(f"[transcode] cache miss, streaming: {key[:12]}", flush=True)
    tmp = CACHE_DIR / f"{key}.tmp"

    async def generate():
        proc = await asyncio.create_subprocess_exec(
            *_stream_args(url),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        f = None
        ok = False
        try:
            f = open(tmp, "wb")
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                yield chunk
            await proc.wait()
            ok = proc.returncode == 0
        except Exception:
            raise
        finally:
            if f:
                f.close()
            if ok:
                tmp.rename(cached)
                _evict_cache()
            else:
                tmp.unlink(missing_ok=True)

    return StreamingResponse(generate(), media_type="video/mp4")
