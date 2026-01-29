# services/video_frame_description_service/config.py

import os

SERVICE_NAME = "video_frame_description_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_GROUP_ID = 'video-frame-description-service-group'

# Topic配置
KAFKA_TOPIC_REQUEST = "video-frame-description-requests"
KAFKA_TOPIC_RESPONSE = "video-frame-description-results"
KAFKA_TOPIC_DLQ = "video-frame-description-dlq"

# 檔案路徑配置
FRAME_DESCRIPTION_RESULTS_DIR = "/tmp/video_processing/frame_description_results"

# GPU配置
CUDA_VISIBLE_DEVICES = "2"  # 根據您的GPU配置調整
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# 模型配置
MODEL_PATH = 'OpenGVLab/InternVL3_5-4B'
MAX_MEMORY_PER_GPU = "36GiB"
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "do_sample": True
}

# 圖像處理配置
INPUT_SIZE = 448
MAX_NUM_PATCHES = 12

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "video_frame_description_service.log"

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3
