# services/video_ocr_frame_service/config.py

import os

SERVICE_NAME = "video_ocr_frame_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.57.156:19002,192.168.57.156:19003,192.168.57.156:19004'
KAFKA_GROUP_ID = 'video-ocr-frame-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "video-ocr-frame-requests"

# Topic配置 - produce to
KAFKA_TOPIC_RESPONSE = "video-ocr-frame-results"
KAFKA_TOPIC_DLQ = "video-ocr-frame-dlq"

# OCR 配置
OCR_OUTPUT_BASE_DIR = "/tmp/media_processing/video_processing/ocr_results"
OCR_FRAMES_BASE_DIR = "/tmp/media_processing/video_processing/ocr_frames"
OCR_BATCH_SIZE = 16
OCR_CONFIDENCE_THRESHOLD = 0.7
TEMP_FILE_CLEANUP_DELAY = 300  # 5分鐘後清理

# OCR 模型配置（可透過 .env 覆蓋）
OCR_DET_MODEL = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det")
OCR_REC_MODEL = os.environ.get("OCR_REC_MODEL", "PP-OCRv5_mobile_rec")
OCR_PRECISION  = os.environ.get("OCR_PRECISION", "fp16")

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "video_ocr_frame_service.log"
