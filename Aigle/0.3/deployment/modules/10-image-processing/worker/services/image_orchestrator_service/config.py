# services/image_orchestrator_service/config.py

SERVICE_NAME = "image_orchestrator_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.57.156:19002,192.168.57.156:19003,192.168.57.156:19004'
KAFKA_GROUP_ID = 'image-orchestrator-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "image-processing-requests"

# Topic配置 - produce to
KAFKA_TOPIC_DESCRIPTION_REQUEST = "image-description-requests"
KAFKA_TOPIC_OCR_REQUEST = "image-ocr-requests"
KAFKA_TOPIC_INDEXER_REQUEST = "image-indexer-requests"
KAFKA_TOPIC_FINAL_RESULT = "image-processing-results"

# Topic配置 - listen to
KAFKA_TOPIC_DESCRIPTION_RESULT = "image-description-results"
KAFKA_TOPIC_OCR_RESULT = "image-ocr-results"
KAFKA_TOPIC_INDEXER_RESULT = "image-indexer-results"

KAFKA_TOPIC_DLQ = "image-processing-dlq"

# Asset Management API
import os
ASSET_MANAGEMENT_URL = os.environ.get("ASSET_MANAGEMENT_URL", "http://raptor-asset-management:8000")

# SeaweedFS配置
#SEAWEEDFS_BASE_URL = "http://192.168.57.156:8086"
SEAWEEDFS_TIMEOUT = 30
SEAWEEDFS_RETRY_COUNT = 3

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/media_processing/image_processing"
TEMP_FILE_CLEANUP_DELAY = 300  # 5分鐘後清理

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "image_orchestrator_service.log"


# Redis 配置
#REDIS_HOST = "192.168.57.156"
#REDIS_PORT = 6391
REDIS_DB = 0
REDIS_KEY_PREFIX = "image_orchestrator:"
REDIS_KEY_TTL = 86400