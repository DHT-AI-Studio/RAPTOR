# services/video_orchestrator_service/config.py

import os

SERVICE_NAME = "video_orchestrator_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.57.156:19002,192.168.57.156:19003,192.168.57.156:19004'
KAFKA_GROUP_ID = 'video-orchestrator-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "video-processing-requests"

# Topic配置 - produce to (並行處理)
KAFKA_TOPIC_VIDEO_ANALYSIS_REQUEST = "video-analysis-requests"
KAFKA_TOPIC_AUDIO_ANALYSIS_REQUEST = "audio-analysis-requests"

# Video moment chunking (inserted before analysis fan-out)
KAFKA_TOPIC_CHUNKING_REQUEST = "video-chunking-requests"
KAFKA_TOPIC_CHUNKING_RESULT  = "video-chunking-results"

# Topic配置 - produce to (後續處理)
KAFKA_TOPIC_SUMMARY_REQUEST = "video-summary-requests"
KAFKA_TOPIC_CONTEXTUALIZE_REQUEST = "video-contextualize-requests"
KAFKA_TOPIC_INDEXER_REQUEST = "video-indexer-requests"
KAFKA_TOPIC_FINAL_RESULT = "video-processing-results"

# Topic配置 - listen to
KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT = "video-analysis-results"
KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT = "audio-analysis-results"
KAFKA_TOPIC_SUMMARY_RESULT = "video-summary-results"
KAFKA_TOPIC_CONTEXTUALIZE_RESULT = "video-contextualize-results"
KAFKA_TOPIC_INDEXER_RESULT = "video-indexer-results"

KAFKA_TOPIC_DLQ = "video-processing-dlq"

# SeaweedFS配置
#SEAWEEDFS_BASE_URL = "http://192.168.57.156:8086"
SEAWEEDFS_TIMEOUT = 30
SEAWEEDFS_RETRY_COUNT = 3

# Asset Management API
ASSET_MANAGEMENT_URL = os.environ.get("ASSET_MANAGEMENT_URL", "http://raptor-asset-management:8000")

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/media_processing/video_processing"
TEMP_FILE_CLEANUP_DELAY = 300

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "video_orchestrator_service.log"

# Redis 配置
#REDIS_HOST = "192.168.57.156"
#REDIS_PORT = 6391
REDIS_DB = 0
REDIS_KEY_PREFIX = "video_orchestrator:"
REDIS_KEY_TTL = 86400

