# services/video_face_detection_service/config.py

import os

SERVICE_NAME = "video_face_detection_service"

# Kafka配置：orchestrator 在 chunking 完成後與影片／音訊分析平行送出（v1.4 §4.3）
KAFKA_GROUP_ID = 'video-face-detection-service-group'
KAFKA_TOPIC_REQUEST = "video-face-detection-requests"
KAFKA_TOPIC_RESPONSE = "video-face-detection-results"
# 長影片解碼 + 偵測可能超過預設 5 分鐘 poll 間隔
KAFKA_MAX_POLL_INTERVAL_MS = 3600000

# Asset Management API (Module 04)：orchestrator 的暫存檔不在時，從這裡下載原始影片
ASSET_MANAGEMENT_URL = os.environ.get("ASSET_MANAGEMENT_URL", "http://raptor-asset-management:8000")
# 代表圖每個請求最多帶幾張。Module 04 的 multipart 解析（Starlette 預設 max_files=1000）
# 超過 1000 個檔案會整個請求回 400，所以 trace 多的影片要分批送
FACE_REPRESENTATIVE_BATCH_SIZE = int(os.environ.get("FACE_REPRESENTATIVE_BATCH_SIZE", "500"))

# VIE01-369: retry budget for a representative-image batch upload -- a 5xx here
# (e.g. a --backup write-freeze window) is almost always transient, and this write
# happens after the expensive detection/clustering work is already done, with no
# Kafka redelivery to fall back on (this service's consumer auto-commits). 8
# attempts with a 90s-capped backoff sum to ~3.6 min before giving up.
WRITE_FREEZE_RETRY_COUNT = 8

# Personal DB Service (Module 25)：軌跡與分群結果寫到這裡
PERSONAL_DB_SERVICE_URL = os.environ.get("PERSONAL_DB_SERVICE_URL", "http://raptor-personal-db-service:8000")

# GD-11：人臉是 GDPR Art. 9 生物特徵，只處理有 face_detection 同意的使用者。
# 同意存在 Keycloak，只有 Module 33 讀得到，所以每支影片向 Module 33 的 GET /consent 查詢。
PRIVACY_COMPLIANCE_URL = os.environ.get("PRIVACY_COMPLIANCE_URL", "http://raptor-privacy-compliance-service:8033")
GD_INTERNAL_TOKEN = os.environ.get("GD_INTERNAL_TOKEN", "")

# 本服務自己的影片副本。orchestrator 的暫存檔在主流程結束時就會被刪掉，
# 而人臉偵測不在主流程的等待條件裡，可能比主流程晚結束
FACE_TEMP_DIR = os.environ.get("FACE_TEMP_DIR", "/tmp/media_processing/video_processing/face_detection")  # nosec B108 -- fixed base directory constant, matches the docker-compose volume layout, not a per-request predictable filename

# 模型配置：偵測 RetinaFace MobileNetV2（MIT）、辨識 AuraFace（Apache 2.0）。
# ONNX 檔案不打包進 image：啟動時檢查 FACE_MODEL_DIR（預設在共用的 /hf_cache volume 底下），
# 缺少或 SHA-256 不符就從下面的網址下載（model_files.py），之後重啟直接沿用。
# 網址可改指向內部鏡像；SHA-256 不變，檔案不同就不會啟動。
FACE_MODEL_DIR = os.environ.get("FACE_MODEL_DIR", "/hf_cache/face")
FACE_DET_MODEL_PATH = os.environ.get("FACE_DET_MODEL_PATH", os.path.join(FACE_MODEL_DIR, "retinaface_mv2.onnx"))
FACE_REC_MODEL_PATH = os.environ.get("FACE_REC_MODEL_PATH", os.path.join(FACE_MODEL_DIR, "glintr100.onnx"))
FACE_DET_MODEL_URL = os.environ.get(
    "FACE_DET_MODEL_URL",
    "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.onnx")
FACE_REC_MODEL_URL = os.environ.get(
    "FACE_REC_MODEL_URL",
    "https://huggingface.co/fal/AuraFace-v1/resolve/af6d057c9b0ec4071d4c49c80e3539258798b609/glintr100.onnx")
FACE_DET_MODEL_SHA256 = "3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757"
FACE_REC_MODEL_SHA256 = "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60"
# false：不下載，檔案必須已在 FACE_MODEL_DIR（無外網的環境先放好檔案）
FACE_MODEL_DOWNLOAD = os.environ.get("FACE_MODEL_DOWNLOAD", "true").lower() == "true"
# 啟動時要確保的檔案。以 FACE_DET_MODEL_PATH / FACE_REC_MODEL_PATH 指定自己的模型時，
# 那個檔案不在這裡，不驗證也不下載，避免被預設模型覆蓋
FACE_MODEL_FILES = [
    (path, url, sha256)
    for env, path, url, sha256 in (
        ("FACE_DET_MODEL_PATH", FACE_DET_MODEL_PATH, FACE_DET_MODEL_URL, FACE_DET_MODEL_SHA256),
        ("FACE_REC_MODEL_PATH", FACE_REC_MODEL_PATH, FACE_REC_MODEL_URL, FACE_REC_MODEL_SHA256),
    )
    if env not in os.environ
]
# 逗號分隔；CUDA 不可用時 onnxruntime 會退回下一個 provider
FACE_ORT_PROVIDERS = os.environ.get("FACE_ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",")

# 偵測參數（v1.4 §4.3）
FACE_DET_INPUT_WIDTH = 640         # 幀縮到此寬度再偵測，高度依原比例取 32 的倍數
FACE_DET_THRESHOLD = float(os.environ.get("FACE_DET_THRESHOLD", "0.5"))
FACE_DET_NMS_THRESHOLD = 0.4
FACE_MIN_SIZE_PX = int(os.environ.get("FACE_MIN_SIZE_PX", "20"))   # 臉框短邊，以原始解析度計

# 取樣：固定 8fps，每 125ms 取時間戳最接近的已解碼幀
FACE_SAMPLE_FPS = 8.0

# 鏡頭切點偵測：相鄰已解碼幀的 HSV 平均絕對差（0–255），超過即視為切點
SHOT_THRESHOLD = float(os.environ.get("FACE_SHOT_THRESHOLD", "27.0"))
SHOT_MIN_LENGTH_SEC = 0.5          # 兩個切點的最短間隔，壓掉閃光造成的連續誤判
SHOT_ANALYSIS_WIDTH = 160          # 比對前先縮小，只看整體色彩分布

# 追蹤（v1.4 §4.3）：Kalman 等速預測 + 兩階段 Hungarian（IoU）配對
# 1.4 沒有給數值的部分（高信心門檻、IoU 門檻、中斷容忍）為起始值，需以實際影片調整
FACE_TRACK_HIGH_CONF = float(os.environ.get("FACE_TRACK_HIGH_CONF", "0.6"))   # 第一階段配對 + 可開新軌跡
FACE_TRACK_IOU_HIGH = 0.2          # 第一階段（高信心偵測）最低 IoU
FACE_TRACK_IOU_LOW = 0.5           # 第二階段（低信心偵測）最低 IoU，較嚴，避免誤接
FACE_TRACK_MAX_GAP_SEC = float(os.environ.get("FACE_TRACK_MAX_GAP_SEC", "1.0"))   # 短暫遮擋／漏偵測的容忍時間
FACE_TRACK_MIN_LENGTH_SEC = 0.5    # 短於此長度的軌跡視為雜訊丟棄（8fps 約 4 幀）

# 每條軌跡只用品質最好的幾幀算 embedding（v1.4 §4.3）
FACE_EMBED_FRAMES_PER_TRACE = 8
# 每張臉的向量 = 原圖與左右鏡像的輸出相加後正規化。會讓相似度整體上移，
# 下面的分群門檻與 Module 25 的 face_match_threshold 都是在開啟的情況下調的，要改需一起重調
FACE_EMBED_MIRROR = True

# 品質分數 = 信心 × 正臉程度 × 清晰度 × 臉框大小 × 曝光（v1.4 §4.3 代表幀選擇）
FACE_QUALITY_SHARPNESS_SCALE = 100.0       # Laplacian 變異數換算成 0–1：v / (v + scale)
FACE_QUALITY_SIZE_REF_PX = 112             # 臉框短邊達此大小即視為滿分（ArcFace 輸入尺寸）
FACE_QUALITY_EXPOSURE_RANGE = (40, 215)    # 臉部平均亮度超出此範圍視為過暗／過曝，品質為 0

# 影片內分群（v1.4 §4.3）：有條件的階層式合併
# 門檻依辨識模型而定；以下是 AuraFace + 鏡像平均的值（buffalo_l 時為 0.45 / 0.75），換模型需重調
FACE_CLUSTER_SIM = 0.50                    # 時間不重疊：相似度門檻，不限時間間隔
FACE_CLUSTER_SIM_OVERLAP = 0.75            # 時間重疊（同框）：較高門檻
FACE_CLUSTER_OVERLAP_MIN_PX = 60           # 同框時兩邊的代表臉都需達此大小
FACE_CLUSTER_OVERLAP_MIN_CONF = 0.6        # 同框時兩邊的代表臉都需達此信心

# 代表圖（縮圖）：臉框向外擴張後裁切，長邊縮到此大小
FACE_THUMBNAIL_MARGIN = 0.4
FACE_THUMBNAIL_MAX_PX = 256
FACE_THUMBNAIL_JPEG_QUALITY = 90

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "video_face_detection_service.log"
