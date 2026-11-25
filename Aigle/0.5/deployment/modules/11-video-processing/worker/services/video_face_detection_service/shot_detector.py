# services/video_face_detection_service/shot_detector.py
# 鏡頭切點偵測：比較相鄰已解碼幀的 HSV 平均絕對差（與 PySceneDetect ContentDetector 相同的判準）。
# 逐幀比對而不是只比取樣幀，切點時間精度到原始幀，不受 8fps 取樣間隔限制。
# 切點會強制結束進行中的軌跡（v1.4 §4.3），避免把不同鏡頭的不同人接在一起。

import logging

import cv2
import numpy as np
from config import SHOT_ANALYSIS_WIDTH, SHOT_MIN_LENGTH_SEC, SHOT_THRESHOLD
from sampler import decode_frames

logger = logging.getLogger(__name__)


class ShotBoundaryDetector:
    """串流式偵測：逐幀呼叫 feed()，結束後讀 cuts。"""

    def __init__(self, threshold: float = SHOT_THRESHOLD, min_shot_sec: float = SHOT_MIN_LENGTH_SEC):
        self.threshold = threshold
        self.min_shot_sec = min_shot_sec
        self.cuts: list[float] = []
        self._prev_hsv: np.ndarray | None = None

    def feed(self, timestamp_sec: float, image: np.ndarray) -> bool:
        """回傳這一幀是否為新鏡頭的第一幀。"""
        h, w = image.shape[:2]
        small_h = max(1, round(h * SHOT_ANALYSIS_WIDTH / w))
        small = cv2.resize(image, (SHOT_ANALYSIS_WIDTH, small_h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)

        is_cut = False
        if self._prev_hsv is not None:
            score = float(np.mean(np.abs(hsv - self._prev_hsv)))
            last_cut = self.cuts[-1] if self.cuts else None
            if score >= self.threshold and (last_cut is None or timestamp_sec - last_cut >= self.min_shot_sec):
                self.cuts.append(timestamp_sec)
                is_cut = True
        self._prev_hsv = hsv
        return is_cut


def detect_shots(video_path: str, threshold: float = SHOT_THRESHOLD) -> list[float]:
    """掃過整支影片，回傳所有切點時間（秒）。切點時間 = 新鏡頭第一幀的時間戳。"""
    detector = ShotBoundaryDetector(threshold=threshold)
    for frame in decode_frames(video_path):
        detector.feed(frame.timestamp_sec, frame.image)
    logger.info(f"Detected {len(detector.cuts)} shot boundaries in {video_path}")
    return detector.cuts
