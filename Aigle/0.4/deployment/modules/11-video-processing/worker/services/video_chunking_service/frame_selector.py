# services/video_chunking_service/frame_selector.py

import os
import logging
import subprocess
from typing import List, Dict, Any
import cv2
import numpy as np
from config import FRAMES_PER_MOMENT, FRAME_SAMPLE_FPS, BLUR_THRESHOLD, DIVERSITY_THRESHOLD

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, start: float, end: float, output_dir: str) -> List[str]:
    """Extract frames at FRAME_SAMPLE_FPS from a video segment."""
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-noautorotate",
        "-threads", "4",
        "-ss", str(start), "-to", str(end),
        "-i", video_path,
        "-vf", f"fps={FRAME_SAMPLE_FPS},hwdownload,format=nv12",
        "-q:v", "2",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    frames = sorted(f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg"))

    if result.returncode != 0 or not frames:
        if result.returncode == 0:
            logger.warning("GPU frame extraction produced no frames, falling back to CPU")
        else:
            logger.warning(f"GPU frame extraction failed, falling back to CPU: {result.stderr.decode(errors='replace')}")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
            "-vf", f"fps={FRAME_SAMPLE_FPS}",
            "-q:v", "2",
            pattern,
        ]
        cpu_result = subprocess.run(cmd, capture_output=True, timeout=60)
        if cpu_result.returncode != 0:
            logger.warning(f"CPU frame extraction failed: {cpu_result.stderr.decode(errors='replace')}")
        frames = sorted(f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg"))

    # Last resort: grab a single frame from the middle of the segment
    if not frames:
        mid = (start + end) / 2
        single = os.path.join(output_dir, "frame_0001.jpg")
        r = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(mid), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", single,
        ], capture_output=True, timeout=30)
        if r.returncode == 0 and os.path.exists(single):
            logger.warning(f"Using single mid-segment frame as last resort for {start}-{end}s")
            frames = [single]
        else:
            logger.warning(f"All frame extraction attempts failed for {start}-{end}s")
    return [os.path.join(output_dir, f) for f in frames]


def laplacian_variance(path: str) -> float:
    """Higher value = sharper frame."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def histogram_similarity(path_a: str, path_b: str) -> float:
    """Return similarity [0,1] between two frames using HSV histogram correlation."""
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None or img_b is None:
        return 1.0  # treat as identical if unreadable
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def select_keyframes(moment: Dict[str, Any]) -> List[str]:
    """
    Extract frames, score by sharpness, then greedily select diverse keyframes:
    each additional keyframe must be visually different enough from already selected ones.
    """
    output_dir = os.path.join(moment["output_dir"], "frames")
    frames = extract_frames(
        moment["video_path"],
        moment["start_time"],
        moment["end_time"],
        output_dir,
    )

    if not frames:
        logger.warning(f"No frames extracted for moment {moment['moment_index']}")
        return []

    all_scored = sorted([(laplacian_variance(f), f) for f in frames], key=lambda x: x[0], reverse=True)
    scored = [(score, path) for score, path in all_scored if score >= BLUR_THRESHOLD]
    if not scored:
        logger.warning(f"All frames below blur threshold for moment {moment['moment_index']}, using sharpest available")
        scored = all_scored

    selected = []
    for _, path in scored:
        if len(selected) >= FRAMES_PER_MOMENT:
            break
        too_similar = any(
            histogram_similarity(path, sel) > DIVERSITY_THRESHOLD
            for sel in selected
        )
        if not too_similar:
            selected.append(path)

    # Fallback: if not enough diverse frames, fill with sharpest remaining
    if len(selected) < FRAMES_PER_MOMENT:
        for _, path in scored:
            if path not in selected:
                selected.append(path)
            if len(selected) >= FRAMES_PER_MOMENT:
                break

    # Sort by filename to preserve temporal order
    selected.sort(key=lambda p: os.path.basename(p))

    # Delete unselected frames to free disk space
    selected_set = set(selected)
    for f in frames:
        if f not in selected_set:
            try:
                os.remove(f)
            except OSError:
                pass

    logger.debug(
        f"Moment {moment['moment_index']}: {len(frames)} extracted, "
        f"{len(selected)} selected, {len(frames) - len(selected)} discarded"
    )
    return selected
