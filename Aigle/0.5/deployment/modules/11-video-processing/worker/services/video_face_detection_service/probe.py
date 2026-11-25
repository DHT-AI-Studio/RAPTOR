# services/video_face_detection_service/probe.py
# 讀取影片基本資訊：解析度、原始 fps、長度

import json
import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_sec: float


def _parse_rate(rate: str) -> float:
    num, _, den = rate.partition("/")
    den_f = float(den) if den else 1.0
    return float(num) / den_f if den_f else 0.0


def probe_video(video_path: str) -> VideoInfo:
    """用 ffprobe 讀取第一條影像串流的資訊。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.decode(errors='replace')}")

    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream in {video_path}")
    stream = streams[0]

    # avg_frame_rate 在 VFR 影片上比 r_frame_rate 準；某些容器會回 0/0
    fps = _parse_rate(stream.get("avg_frame_rate", "0/0")) or _parse_rate(stream.get("r_frame_rate", "0/0"))
    duration = stream.get("duration") or data.get("format", {}).get("duration") or 0

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration_sec=float(duration),
    )
