# services/video_face_detection_service/tracker.py
# 人臉追蹤（v1.4 §4.3）：把逐幀偵測串成連續軌跡（trace）。
#   - Kalman 等速模型預測每條軌跡下一幀的位置（狀態與 SORT 相同：中心、面積、長寬比及其速度）
#   - 兩階段 Hungarian（IoU）配對：高信心偵測先配；沒配到的軌跡再跟低信心偵測配，
#     撐過轉頭、動態模糊時信心下降的幀，避免軌跡斷掉。只有高信心偵測能開新軌跡
#   - 鏡頭切點強制結束所有進行中的軌跡，避免不同鏡頭的不同人被接在一起
#   - 中斷超過 FACE_TRACK_MAX_GAP_SEC 結束；短於 FACE_TRACK_MIN_LENGTH_SEC 的軌跡丟棄
#   - 所有時間容忍以秒設定，執行時依取樣 fps 換算成幀數
#
# 取樣幀的完整影像只在 update() 當下存在，所以每條軌跡在追蹤過程中就保留：
#   品質最好的前 N 張「對齊後的臉」（之後一起算 embedding），以及最佳一張的代表圖（JPEG）。

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from config import (
    FACE_EMBED_FRAMES_PER_TRACE,
    FACE_SAMPLE_FPS,
    FACE_TRACK_HIGH_CONF,
    FACE_TRACK_IOU_HIGH,
    FACE_TRACK_IOU_LOW,
    FACE_TRACK_MAX_GAP_SEC,
    FACE_TRACK_MIN_LENGTH_SEC,
)
from quality import Quality, assess, thumbnail_jpeg
from scipy.optimize import linear_sum_assignment


@dataclass
class TraceFrame:
    frame_no: int
    timestamp_sec: float
    bbox: list[float]          # x1, y1, x2, y2
    confidence: float
    landmarks: list[list[float]]
    quality: float


@dataclass
class Representative:
    frame: TraceFrame
    quality: Quality
    min_side: float
    thumbnail_jpeg: bytes


@dataclass
class Trace:
    frames: list[TraceFrame]
    representative: Representative
    embed_crops: list[np.ndarray] = field(repr=False)   # 對齊後的最佳 N 張，算完 embedding 即清空
    trace_id: str | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)
    representative_version_id: str | None = None   # 代表圖上傳 Module 04 後的版本，存在 {asset_path}/face_{trace_id}.jpg

    @property
    def start_sec(self) -> float:
        return self.frames[0].timestamp_sec

    @property
    def end_sec(self) -> float:
        return self.frames[-1].timestamp_sec


class _KalmanBox:
    """等速 Kalman：x = [cx, cy, area, aspect, vcx, vcy, varea]，量測 [cx, cy, area, aspect]。"""

    _F = np.eye(7)
    _F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0
    _H = np.eye(4, 7)
    _R = np.diag([1.0, 1.0, 10.0, 10.0])
    _Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

    def __init__(self, bbox: np.ndarray):
        self.x = np.zeros(7)
        self.x[:4] = self._to_z(bbox)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    @staticmethod
    def _to_z(bbox: np.ndarray) -> np.ndarray:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return np.array([bbox[0] + w / 2, bbox[1] + h / 2, w * h, w / max(h, 1e-6)])

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        return self.bbox()

    def update(self, bbox: np.ndarray) -> None:
        y = self._to_z(bbox) - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self._H) @ self.P

    def bbox(self) -> np.ndarray:
        area, aspect = max(self.x[2], 1e-6), max(self.x[3], 1e-6)
        w = math.sqrt(area * aspect)
        h = area / w
        return np.array([self.x[0] - w / 2, self.x[1] - h / 2, self.x[0] + w / 2, self.x[1] + h / 2])


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _match(track_boxes: np.ndarray, det_boxes: np.ndarray, min_iou: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian 一對一指派；IoU 低於門檻的配對視為未配對。回傳 (配對, 未配對軌跡, 未配對偵測)。"""
    iou = _iou_matrix(track_boxes, det_boxes)
    matches = []
    if iou.size:
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
    matched_t = {r for r, _ in matches}
    matched_d = {c for _, c in matches}
    return (matches,
            [i for i in range(len(track_boxes)) if i not in matched_t],
            [j for j in range(len(det_boxes)) if j not in matched_d])


class _ActiveTrack:
    def __init__(self, seq: int, embed_frames: int):
        self.seq = seq
        self.kf: _KalmanBox | None = None
        self.frames: list[TraceFrame] = []
        self.candidates: list[tuple[float, int, np.ndarray]] = []   # min-heap (score, frame_no, aligned crop)
        self.best: Representative | None = None
        self.embed_frames = embed_frames

    @property
    def last_ts(self) -> float:
        return self.frames[-1].timestamp_sec

    def add(self, frame, face, aligner: Callable) -> None:
        if self.kf is None:
            self.kf = _KalmanBox(face.bbox)
        else:
            self.kf.update(face.bbox)
        q = assess(frame.image, face)
        self.frames.append(TraceFrame(
            frame_no=frame.frame_no,
            timestamp_sec=frame.timestamp_sec,
            bbox=[round(float(v), 1) for v in face.bbox],
            confidence=round(float(face.confidence), 4),
            landmarks=[[round(float(x), 1), round(float(y), 1)] for x, y in face.landmarks],
            quality=round(q.score, 4),
        ))
        # 只有擠進前 N 名時才對齊（對齊需要這一幀的完整影像，之後就拿不到了）
        key = (q.score, -frame.frame_no)
        if len(self.candidates) < self.embed_frames or key > self.candidates[0][:2]:
            item = (q.score, -frame.frame_no, aligner(frame.image, face))
            if len(self.candidates) < self.embed_frames:
                heapq.heappush(self.candidates, item)
            else:
                heapq.heapreplace(self.candidates, item)
        # 代表幀：品質最高；全部為 0（例如都過暗）時退而取信心最高
        if self.best is None or (q.score, q.confidence) > (self.best.quality.score, self.best.quality.confidence):
            self.best = Representative(self.frames[-1], q, face.min_side, thumbnail_jpeg(frame.image, face.bbox))

    def to_trace(self) -> Trace:
        crops = [c for _, _, c in sorted(self.candidates, reverse=True)]
        return Trace(frames=self.frames, representative=self.best, embed_crops=crops)


class FaceTracker:
    """逐取樣幀呼叫 update()；結束後呼叫 finish() 取得軌跡（依開始時間排序）。"""

    def __init__(self, aligner: Callable, fps: float = FACE_SAMPLE_FPS,
                 high_conf: float = FACE_TRACK_HIGH_CONF, iou_high: float = FACE_TRACK_IOU_HIGH,
                 iou_low: float = FACE_TRACK_IOU_LOW, max_gap_sec: float = FACE_TRACK_MAX_GAP_SEC,
                 min_length_sec: float = FACE_TRACK_MIN_LENGTH_SEC,
                 embed_frames: int = FACE_EMBED_FRAMES_PER_TRACE):
        self.aligner = aligner
        self.high_conf, self.iou_high, self.iou_low = high_conf, iou_high, iou_low
        self.max_gap_sec = max_gap_sec
        self.min_frames = max(1, round(min_length_sec * fps))
        self.embed_frames = embed_frames
        self._active: list[_ActiveTrack] = []
        self._finished: list[_ActiveTrack] = []
        self._seq = 0

    def update(self, frame, faces: list, shot_changed: bool = False) -> None:
        if shot_changed:
            self._close(self._active)

        predicted = np.array([t.kf.predict() for t in self._active]).reshape(-1, 4)
        high = [f for f in faces if f.confidence >= self.high_conf]
        low = [f for f in faces if f.confidence < self.high_conf]

        # 第一階段：所有軌跡 × 高信心偵測
        m1, left_tracks, left_high = _match(predicted, np.array([f.bbox for f in high]).reshape(-1, 4), self.iou_high)
        # 第二階段：剩下的軌跡 × 低信心偵測
        m2, left_tracks2, _ = _match(predicted[left_tracks], np.array([f.bbox for f in low]).reshape(-1, 4), self.iou_low)

        for ti, di in m1:
            self._active[ti].add(frame, high[di], self.aligner)
        for ti, di in m2:
            self._active[left_tracks[ti]].add(frame, low[di], self.aligner)

        # 中斷太久的軌跡結束
        unmatched = [self._active[left_tracks[i]] for i in left_tracks2]
        self._close([t for t in unmatched if frame.timestamp_sec - t.last_ts > self.max_gap_sec])

        # 沒配到的高信心偵測開新軌跡（低信心偵測只用來延續既有軌跡）
        for di in left_high:
            track = _ActiveTrack(self._seq, self.embed_frames)
            self._seq += 1
            track.add(frame, high[di], self.aligner)
            self._active.append(track)

    def finish(self) -> list[Trace]:
        self._close(self._active)
        kept = [t for t in self._finished if len(t.frames) >= self.min_frames]
        kept.sort(key=lambda t: (t.frames[0].timestamp_sec, t.seq))
        return [t.to_trace() for t in kept]

    def _close(self, tracks: list[_ActiveTrack]) -> None:
        closing = {id(t) for t in tracks}
        self._finished.extend(tracks)
        self._active = [t for t in self._active if id(t) not in closing]
