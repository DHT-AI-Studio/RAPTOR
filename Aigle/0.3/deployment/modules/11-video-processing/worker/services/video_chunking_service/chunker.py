# services/video_chunking_service/chunker.py

import os
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from config import (
    MOMENT_DURATION_SEC, TEMP_FILE_DIR,
    NORMALIZE_HEIGHT, NORMALIZE_VIDEO_CODEC, NORMALIZE_CRF, NORMALIZE_PRESET,
    NORMALIZE_AUDIO_CODEC, NORMALIZE_AUDIO_BITRATE, NORMALIZE_AUDIO_SAMPLERATE,
    NORMALIZE_PIX_FMT,
)

logger = logging.getLogger(__name__)

_ROTATE_TRANSPOSE = {
    90:  "transpose=1,",
    180: "hflip,vflip,",
    270: "transpose=2,",
}

_CUDA_DECODE_CODECS = {"h264", "hevc", "vp9", "mpeg2video", "mpeg4", "mjpeg", "vc1"}


def _probe_codec(video_path: str) -> str:
    """Return the video codec name from stream metadata."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            return streams[0].get("codec_name", "")
    except (json.JSONDecodeError, KeyError):
        pass
    return ""


def _probe_rotation(video_path: str) -> int:
    """Return video rotation degrees (0/90/180/270) from stream metadata."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            return int(streams[0].get("tags", {}).get("rotate", 0))
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return 0


def normalize_video(video_path: str, request_id: str) -> str:
    """
    Normalize video to a standard format:
    - Resolution: up to NORMALIZE_HEIGHT (e.g. 720p), keep aspect ratio
    - Codec: H.264 (libx264), CRF 23, preset fast
    - Audio: AAC 128k, 44.1kHz
    - Pixel format: yuv420p
    Returns path to normalized file (in TEMP_FILE_DIR/<request_id>/).
    """
    out_dir = os.path.join(TEMP_FILE_DIR, request_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "normalized.mp4")

    codec = _probe_codec(video_path)
    rotation = _probe_rotation(video_path)
    rotate_filter = _ROTATE_TRANSPOSE.get(rotation, "")

    # Fallback 0: full GPU (scale_cuda). Skipped for codecs without CUDA decode support
    # (e.g. AV1) and for rotated videos (scale_cuda cannot consume CPU-side transpose output).
    if not rotate_filter and codec in _CUDA_DECODE_CODECS:
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", video_path,
            "-vf", f"scale_cuda=-2:{NORMALIZE_HEIGHT}",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", str(NORMALIZE_CRF),
            "-c:a", NORMALIZE_AUDIO_CODEC,
            "-b:a", NORMALIZE_AUDIO_BITRATE,
            "-ar", str(NORMALIZE_AUDIO_SAMPLERATE),
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0:
            logger.info(f"Video normalized to 720p H.264: {out_path}")
            return out_path
        logger.warning(f"Full-GPU normalization failed, trying CPU decode + nvenc: {result.stderr.decode(errors='replace')}")

    # Fallback 1: CPU decode + nvenc with explicit rotation correction
    cmd = [
        "ffmpeg", "-y",
        "-noautorotate",
        "-i", video_path,
        "-vf", f"{rotate_filter}scale=-2:{NORMALIZE_HEIGHT},format=yuv420p",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", str(NORMALIZE_CRF),
        "-c:a", NORMALIZE_AUDIO_CODEC,
        "-b:a", NORMALIZE_AUDIO_BITRATE,
        "-ar", str(NORMALIZE_AUDIO_SAMPLERATE),
        "-metadata:s:v:0", "rotate=0",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode == 0:
        logger.info(f"Video normalized to 720p H.264: {out_path}")
        return out_path

    # Fallback 2: full CPU (libx264)
    logger.warning(f"CPU decode + nvenc failed, falling back to libx264: {result.stderr.decode(errors='replace')}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"format=yuv420p,scale=-2:{NORMALIZE_HEIGHT}",
        "-c:v", NORMALIZE_VIDEO_CODEC,
        "-crf", str(NORMALIZE_CRF),
        "-preset", NORMALIZE_PRESET,
        "-pix_fmt", NORMALIZE_PIX_FMT,
        "-c:a", NORMALIZE_AUDIO_CODEC,
        "-b:a", NORMALIZE_AUDIO_BITRATE,
        "-ar", str(NORMALIZE_AUDIO_SAMPLERATE),
        "-metadata:s:v:0", "rotate=0",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg normalization failed: {result.stderr.decode(errors='replace')}"
        )
    logger.info(f"Video normalized to 720p H.264: {out_path}")
    return out_path


def probe_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def extract_audio_segment(video_path: str, start: float, end: float, output_dir: str) -> str:
    """Extract a mono 16kHz WAV audio segment for ASR."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"audio_{int(start)}_{int(end)}.wav")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", video_path,
        "-ar", "16000", "-ac", "1", "-vn",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return out_path


def split_into_moments(video_path: str, request_id: str) -> List[Dict[str, Any]]:
    """
    Split video into MOMENT_DURATION_SEC-second moments.
    WAV extraction runs in parallel via ThreadPoolExecutor.
    Returns list of moment metadata dicts (without keyframes — those are added by frame_selector).
    """
    duration = probe_duration(video_path)

    # Build moment metadata first
    moments = []
    start = 0.0
    moment_index = 0
    while start < duration:
        end = min(start + MOMENT_DURATION_SEC, duration)
        output_dir = os.path.join(TEMP_FILE_DIR, request_id, f"moment_{moment_index}")
        os.makedirs(output_dir, exist_ok=True)
        moments.append({
            "moment_index": moment_index,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "output_dir": output_dir,
            "audio_segment_path": None,
            "video_path": video_path,
        })
        moment_index += 1
        start = end

    # Extract WAV segments in parallel
    def _extract(moment):
        return extract_audio_segment(
            video_path,
            moment["start_time"],
            moment["end_time"],
            moment["output_dir"],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        audio_paths = list(pool.map(_extract, moments))

    for moment, audio_path in zip(moments, audio_paths):
        moment["audio_segment_path"] = audio_path

    logger.info(f"Split video into {len(moments)} moments ({MOMENT_DURATION_SEC}s each), WAV extracted in parallel")
    return moments
