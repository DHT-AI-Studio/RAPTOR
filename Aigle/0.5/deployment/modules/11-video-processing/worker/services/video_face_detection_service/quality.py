# services/video_face_detection_service/quality.py
# 人臉品質分數（v1.4 §4.3 代表幀選擇），五個因子相乘、各自落在 0–1：
#   信心 × 正臉程度 × 清晰度 × 臉框大小 × 曝光
# 用途有二：挑每條軌跡算 embedding 的最佳幾幀，以及挑代表幀（縮圖）。
# 清晰度與曝光在臉框裁切後縮到固定大小再算，與臉在畫面中的大小無關。

from dataclasses import dataclass

import cv2
import numpy as np
from config import (
    FACE_QUALITY_EXPOSURE_RANGE,
    FACE_QUALITY_SHARPNESS_SCALE,
    FACE_QUALITY_SIZE_REF_PX,
    FACE_THUMBNAIL_JPEG_QUALITY,
    FACE_THUMBNAIL_MARGIN,
    FACE_THUMBNAIL_MAX_PX,
)

_PATCH = 112


@dataclass
class Quality:
    confidence: float
    frontal: float
    sharpness: float
    size: float
    exposure: float

    @property
    def score(self) -> float:
        return self.confidence * self.frontal * self.sharpness * self.size * self.exposure


def frontalness(landmarks: np.ndarray) -> float:
    """
    鼻尖到左右眼距離的對稱度：正臉約 1，越側越接近 0。
    距離沿兩眼連線方向量：轉頭時鼻尖往一側眼睛靠，這個方向的差異最明顯；
    直線距離會被眼睛到鼻尖的垂直距離沖淡，明顯的側臉也有 0.5 分。沿兩眼連線量也不受頭部側傾影響。
    鼻尖跑到眼睛外側（接近全側臉）時為 0。
    """
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_axis = right_eye - left_eye
    eye_dist = float(np.linalg.norm(eye_axis))
    if eye_dist <= 0:
        return 0.0
    axis = eye_axis / eye_dist
    d_left = float(np.dot(nose - left_eye, axis))
    d_right = float(np.dot(right_eye - nose, axis))
    if d_left <= 0 or d_right <= 0:
        return 0.0
    return min(d_left, d_right) / max(d_left, d_right)


def face_patch(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """臉框內的灰階影像，縮放成固定大小。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (_PATCH, _PATCH), interpolation=cv2.INTER_AREA)


def sharpness(patch: np.ndarray) -> float:
    variance = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    return variance / (variance + FACE_QUALITY_SHARPNESS_SCALE)


def exposure(patch: np.ndarray) -> float:
    """過暗或過曝即使清晰也會扭曲特徵，直接排除。"""
    low, high = FACE_QUALITY_EXPOSURE_RANGE
    return 1.0 if low <= float(patch.mean()) <= high else 0.0


def assess(image: np.ndarray, face) -> Quality:
    patch = face_patch(image, face.bbox)
    return Quality(
        confidence=float(face.confidence),
        frontal=frontalness(face.landmarks),
        sharpness=sharpness(patch),
        size=min(1.0, face.min_side / FACE_QUALITY_SIZE_REF_PX),
        exposure=exposure(patch),
    )


def thumbnail_jpeg(image: np.ndarray, bbox: np.ndarray) -> bytes:
    """代表圖：臉框各邊向外擴張後裁切，長邊縮到上限，編成 JPEG。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * FACE_THUMBNAIL_MARGIN, (y2 - y1) * FACE_THUMBNAIL_MARGIN
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    crop = image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    longest = max(crop.shape[:2])
    if longest > FACE_THUMBNAIL_MAX_PX:
        scale = FACE_THUMBNAIL_MAX_PX / longest
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, FACE_THUMBNAIL_JPEG_QUALITY])[1].tobytes()
