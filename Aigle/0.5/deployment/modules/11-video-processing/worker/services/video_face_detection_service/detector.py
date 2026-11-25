# services/video_face_detection_service/detector.py
# 人臉偵測 + 辨識（v1.4 §4.3），兩個可商用授權的 ONNX 模型（來源與授權見模組 README）：
#   偵測：RetinaFace MobileNetV2（retinaface_mv2.onnx，MIT），輸入寬 640，門檻 0.5，最小臉 20px
#         每張臉輸出 bbox、confidence、5 點 landmarks（左眼、右眼、鼻尖、左嘴角、右嘴角）
#   辨識：AuraFace（glintr100.onnx，Apache 2.0；ResNet100 ArcFace），依 landmarks 對齊成 112×112，
#         原圖與左右鏡像的輸出相加後正規化，輸出 512 維 L2 正規化向量
#
# InsightFace 自己的預訓練權重（buffalo_l 等）僅限非商業研究用途，這裡不使用；
# insightface 套件本身是 MIT，只用它的 ArcFace ONNX 包裝與對齊函式。
#
# 偵測與辨識分開（v1.4 §4.3）：embed() 不對每個偵測呼叫，
# 由追蹤挑出每條軌跡品質最好的幾張再算。

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from config import (
    FACE_DET_INPUT_WIDTH,
    FACE_DET_MODEL_PATH,
    FACE_DET_NMS_THRESHOLD,
    FACE_DET_THRESHOLD,
    FACE_EMBED_MIRROR,
    FACE_MIN_SIZE_PX,
    FACE_ORT_PROVIDERS,
    FACE_REC_MODEL_PATH,
)
from insightface.model_zoo import ArcFaceONNX, get_model
from insightface.utils import face_align
from retinaface import RetinaFace

logger = logging.getLogger(__name__)

# onnxruntime 只印 error 以上：glintr100 的輸出形狀以 batch=1 寫死在模型裡，embed() 一次送多張臉時
# 每次推論都會觸發 VerifyOutputSizes 警告（結果本身正確）。
# CUDA 是否真的退回 CPU，由下面 FaceDetector 自己的 provider 檢查記錄。
ort.set_default_logger_severity(3)

EMBEDDING_DIM = 512


@dataclass
class FaceDetection:
    bbox: np.ndarray        # (4,) x1, y1, x2, y2，原始影像像素座標
    confidence: float
    landmarks: np.ndarray   # (5, 2)，原始影像像素座標

    @property
    def min_side(self) -> float:
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


class FaceDetector:
    """載入一次後常駐；同一個實例同時供影片偵測與 /embed-face 使用。"""

    def __init__(self, det_model_path: str = FACE_DET_MODEL_PATH, rec_model_path: str = FACE_REC_MODEL_PATH,
                 input_width: int = FACE_DET_INPUT_WIDTH, threshold: float = FACE_DET_THRESHOLD,
                 min_face_px: int = FACE_MIN_SIZE_PX, providers: Sequence[str] = FACE_ORT_PROVIDERS,
                 mirror: bool = FACE_EMBED_MIRROR):
        providers = list(providers)
        self.det_model = RetinaFace(det_model_path, providers, input_width=input_width,
                                    threshold=threshold, nms_threshold=FACE_DET_NMS_THRESHOLD)
        # 以 .onnx 路徑載入走 insightface 依模型形狀自動路由；download 預設關閉，不會連網
        self.rec_model = get_model(rec_model_path, providers=providers)
        if not isinstance(self.rec_model, ArcFaceONNX):
            raise RuntimeError(f"{rec_model_path} is not a recognition model (routed to {type(self.rec_model).__name__})")

        self.rec_model.prepare(0 if "CUDAExecutionProvider" in providers else -1)
        self.min_face_px = min_face_px
        self.mirror = mirror

        for name, model in (("detection", self.det_model), ("recognition", self.rec_model)):
            active = model.session.get_providers()
            if providers[0] not in active:
                logger.warning(f"Face {name} model requested {providers[0]} but running on {active}")
            else:
                logger.info(f"Face {name} model running on {active}")

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """image 為 BGR。依 confidence 由高到低回傳。"""
        boxes, scores, kpss = self.det_model.detect(image)
        img_h, img_w = image.shape[:2]
        faces = []
        for bbox, score, kps in zip(boxes, scores, kpss, strict=True):
            bbox = bbox.copy()
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, img_w)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, img_h)
            face = FaceDetection(bbox=bbox, confidence=float(score), landmarks=kps)
            if face.min_side >= self.min_face_px:
                faces.append(face)
        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces

    def embed(self, faces: Sequence[tuple[np.ndarray, FaceDetection]]) -> np.ndarray:
        """
        faces 為 (該臉所在的 BGR 影像, 偵測結果) 的序列；回傳 (N, 512) L2 正規化向量。
        同一條軌跡的臉來自不同幀，所以每張臉各自帶自己的影像。
        """
        return self.embed_aligned([self.align(image, det) for image, det in faces])

    def align(self, image: np.ndarray, det: FaceDetection) -> np.ndarray:
        """依 5 點 landmarks 對齊成 ArcFace 的 112×112 輸入。追蹤時用來保存候選臉，之後再一起算 embedding。"""
        return face_align.norm_crop(image, landmark=det.landmarks, image_size=112)

    def embed_aligned(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """crops 為 align() 的輸出；回傳 (N, 512) L2 正規化向量。"""
        if not crops:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        # get_feat 回傳的是未正規化的原始輸出
        feats = self.rec_model.get_feat(list(crops)).astype(np.float32)
        if self.mirror:
            # 加上左右鏡像的輸出再正規化：單張臉的向量較穩定；分群與比對門檻是在開啟的情況下調的
            feats += self.rec_model.get_feat([np.ascontiguousarray(c[:, ::-1]) for c in crops]).astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)

    def embed_best_face(self, image: np.ndarray) -> tuple[FaceDetection, np.ndarray, int] | None:
        """
        單張照片用：偵測整張圖，取最主要的一張臉算 embedding。
        回傳 (該臉, (512,) 向量, 偵測到的臉數)；沒有臉回傳 None。
        「最主要」以 臉框面積 × confidence 判斷。
        """
        faces = self.detect(image)
        if not faces:
            return None
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) * f.confidence)
        return best, self.embed([(image, best)])[0], len(faces)
