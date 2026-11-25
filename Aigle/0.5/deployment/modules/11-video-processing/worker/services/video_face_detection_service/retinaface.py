# services/video_face_detection_service/retinaface.py
# RetinaFace 偵測模型的 ONNX 推論（前處理、anchor 解碼、NMS），只用 numpy + onnxruntime。
#
# 權重：yakhyo/retinaface-pytorch v0.0.1 的 retinaface_mv2.onnx（MobileNetV2 骨幹，MIT 授權），
# 前後處理依該 repo 的 onnx_inference.py：
#   輸入：BGR、減去 (104, 117, 123)、NCHW，高寬可變
#   輸出：loc (N, 4)、conf (N, 2，已做 softmax)、landmarks (N, 10)，相對於 anchor 的偏移量
#   anchor：stride 8 / 16 / 32，每格兩個，邊長 16,32 / 64,128 / 256,512；variance 0.1 / 0.2
#   landmarks 順序：左眼、右眼、鼻尖、左嘴角、右嘴角（影像上的左右），與 ArcFace 對齊模板一致

import math
from collections.abc import Sequence

import cv2
import numpy as np
import onnxruntime as ort

_BGR_MEAN = (104.0, 117.0, 123.0)
_STEPS = (8, 16, 32)
_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_VARIANCE = (0.1, 0.2)


def _anchors(height: int, width: int) -> np.ndarray:
    """(N, 4) 的 cx, cy, w, h，皆以輸入高寬正規化；順序為 stride → 列 → 行 → 邊長，與模型輸出一致。"""
    levels = []
    for step, sizes in zip(_STEPS, _MIN_SIZES, strict=True):
        rows, cols = math.ceil(height / step), math.ceil(width / step)
        cy, cx = np.meshgrid((np.arange(rows) + 0.5) * step / height, (np.arange(cols) + 0.5) * step / width,
                             indexing="ij")
        per_size = [np.stack([cx, cy, np.full_like(cx, s / width), np.full_like(cy, s / height)], axis=-1)
                    for s in sizes]
        levels.append(np.stack(per_size, axis=2).reshape(-1, 4))
    return np.concatenate(levels).astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        order = rest[inter / (areas[i] + areas[rest] - inter) <= threshold]
    return keep


class RetinaFace:
    def __init__(self, model_path: str, providers: Sequence[str], input_width: int,
                 threshold: float, nms_threshold: float):
        self.session = ort.InferenceSession(model_path, providers=list(providers))
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 3 or outputs[1].shape[-1] != 2 or outputs[2].shape[-1] != 10:
            raise RuntimeError(f"{model_path} is not a RetinaFace detection model "
                               f"(outputs: {[(o.name, o.shape) for o in outputs]})")
        self.input_name = inputs[0].name
        self.input_width = input_width
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self._anchor_cache: dict[tuple[int, int], np.ndarray] = {}

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        image 為 BGR。回傳 (boxes (N, 4) x1 y1 x2 y2, scores (N,), landmarks (N, 5, 2))，
        皆為原始影像像素座標，依 score 由高到低。
        """
        img_h, img_w = image.shape[:2]
        # 縮到固定寬度，高度依原比例取最接近的 32 倍數（最大 stride），不補邊
        width = self.input_width
        height = max(32, round(img_h * width / img_w / 32) * 32)
        resized = cv2.resize(image, (width, height)) if (width, height) != (img_w, img_h) else image
        blob = (resized.astype(np.float32) - _BGR_MEAN).transpose(2, 0, 1)[None].astype(np.float32)
        loc, conf, lmk = (out[0] for out in self.session.run(None, {self.input_name: blob}))

        if (height, width) not in self._anchor_cache:
            self._anchor_cache[(height, width)] = _anchors(height, width)
        scores = conf[:, 1]
        mask = scores > self.threshold
        priors, loc, lmk, scores = self._anchor_cache[(height, width)][mask], loc[mask], lmk[mask], scores[mask]
        if not len(scores):
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, 5, 2), np.float32))

        centers = priors[:, :2] + loc[:, :2] * _VARIANCE[0] * priors[:, 2:]
        sizes = priors[:, 2:] * np.exp(loc[:, 2:] * _VARIANCE[1])
        boxes = np.hstack([centers - sizes / 2, centers + sizes / 2]) * (img_w, img_h, img_w, img_h)
        landmarks = (priors[:, None, :2] + lmk.reshape(-1, 5, 2) * _VARIANCE[0] * priors[:, None, 2:]) * (img_w, img_h)

        keep = _nms(boxes, scores, self.nms_threshold)
        return boxes[keep].astype(np.float32), scores[keep].astype(np.float32), landmarks[keep].astype(np.float32)
