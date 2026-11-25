# services/video_face_detection_service/pipeline.py
# 單支影片的偵測流程（v1.4 §4.3 處理流程 1–4）：
#   1. probe 影片資訊
#   2. 鏡頭切點偵測（逐幀）
#   3. 固定 8fps 取樣
#   4. 每張取樣幀做人臉偵測
# 2、3 共用同一次解碼：每個解碼幀先餵給切點偵測，再交給取樣器挑幀。
#
# 追蹤需要在取樣幀的影像還在記憶體時挑出每條軌跡的最佳幀，
# 所以 on_frame 回呼會拿到完整的 Frame（含影像）、這幀的偵測結果，
# 以及「上一張取樣幀之後是否經過鏡頭切點」；DetectionResult 本身只保留不含影像的偵測資料。
#
# detector 由呼叫端傳入（只需要 detect(image) 方法），這個模組不 import insightface。

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from probe import VideoInfo, probe_video
from sampler import Frame, decode_frames, sample_frames
from shot_detector import ShotBoundaryDetector

logger = logging.getLogger(__name__)


@dataclass
class FrameDetections:
    frame_no: int
    timestamp_sec: float
    faces: list[Any]        # detector.FaceDetection


@dataclass
class DetectionResult:
    info: VideoInfo
    shot_boundaries: list[float] = field(default_factory=list)
    frames: list[FrameDetections] = field(default_factory=list)   # 每張取樣幀一筆，含沒有臉的幀
    elapsed_sec: float = 0.0

    def summary(self) -> dict:
        """給 orchestrator 記錄用的摘要；不含逐幀資料，避免訊息過大。"""
        return {
            "width": self.info.width,
            "height": self.info.height,
            "fps": round(self.info.fps, 3),
            "duration_sec": round(self.info.duration_sec, 3),
            "sampled_frames": len(self.frames),
            "frames_with_faces": sum(1 for f in self.frames if f.faces),
            "face_detections": sum(len(f.faces) for f in self.frames),
            "shot_boundaries": len(self.shot_boundaries),
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


def run_detection(
    video_path: str,
    detector: Any,
    on_frame: Callable[[Frame, list[Any], bool], None] | None = None,
) -> DetectionResult:
    started = time.monotonic()
    result = DetectionResult(info=probe_video(video_path))
    shots = ShotBoundaryDetector()

    def decoded():
        for frame in decode_frames(video_path):
            shots.feed(frame.timestamp_sec, frame.image)
            yield frame

    prev_ts = None
    for frame in sample_frames(decoded()):
        faces = detector.detect(frame.image)
        result.frames.append(FrameDetections(frame.frame_no, frame.timestamp_sec, faces))
        if on_frame is not None:
            # 以時間戳判斷：取樣器可能回頭輸出前一個解碼幀，此時切點偵測器已多看了一幀，
            # 不能只看 cuts 的數量有沒有增加
            shot_changed = prev_ts is not None and any(prev_ts < c <= frame.timestamp_sec for c in shots.cuts)
            on_frame(frame, faces, shot_changed)
        prev_ts = frame.timestamp_sec

    result.shot_boundaries = shots.cuts
    result.elapsed_sec = time.monotonic() - started
    logger.info(f"Face detection finished for {video_path}: {result.summary()}")
    return result
