# services/video_face_detection_service/sampler.py
# 固定 8fps 取樣：每 125ms 取時間戳最接近的已解碼幀，與原始 fps 無關（v1.4 §4.3）。
#
# decode_frames() 逐幀解碼整支影片；sample_frames() 從解碼串流裡挑出取樣幀。
# 兩者分開，讓同一次解碼可以同時餵給 ShotBoundaryDetector（每幀）和人臉偵測（取樣幀）。

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np
from config import FACE_SAMPLE_FPS

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    frame_no: int          # 原始影片中的解碼幀序號（從 0 起算）
    timestamp_sec: float   # 該幀實際的播放時間，不是取樣目標時間
    image: np.ndarray      # BGR


def decode_frames(video_path: str) -> Iterator[Frame]:
    """依序解碼每一幀，附上實際時間戳。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_no = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # 部分容器讀不到 pts，POS_MSEC 會一直是 0；此時用名目 fps 換算
            if ts <= 0.0 and frame_no > 0 and nominal_fps > 0:
                ts = frame_no / nominal_fps
            yield Frame(frame_no=frame_no, timestamp_sec=ts, image=image)
            frame_no += 1
    finally:
        cap.release()


def sample_frames(frames: Iterable[Frame], fps: float = FACE_SAMPLE_FPS) -> Iterator[Frame]:
    """
    對每個取樣目標時間 k/fps，挑出時間戳最接近的解碼幀。
    原始 fps 低於取樣 fps 時，同一幀可能是多個目標的最近幀，只輸出一次。
    """
    interval = 1.0 / fps
    target_idx = 0
    prev: Frame | None = None
    last_emitted = -1

    def pick(target: float, before: Frame | None, after: Frame) -> Frame:
        if before is not None and target - before.timestamp_sec <= after.timestamp_sec - target:
            return before
        return after

    for cur in frames:
        while target_idx * interval <= cur.timestamp_sec:
            chosen = pick(target_idx * interval, prev, cur)
            if chosen.frame_no != last_emitted:
                last_emitted = chosen.frame_no
                yield chosen
            target_idx += 1
        prev = cur
