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
# 人臉偵測（v1.4 §4.3）：與影片／音訊分析平行送出，但不在完成條件裡——
# 結果只寫進 Redis state，影片照原本的速度變成可搜尋
KAFKA_TOPIC_FACE_DETECTION_REQUEST = "video-face-detection-requests"
# 設為 false 可停止送出人臉偵測請求
FACE_DETECTION_ENABLED = os.environ.get("FACE_DETECTION_ENABLED", "true").lower() == "true"

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
KAFKA_TOPIC_FACE_DETECTION_RESULT = "video-face-detection-results"
KAFKA_TOPIC_SUMMARY_RESULT = "video-summary-results"
KAFKA_TOPIC_CONTEXTUALIZE_RESULT = "video-contextualize-results"
KAFKA_TOPIC_INDEXER_RESULT = "video-indexer-results"
# Published by Module 25 only once it has actually indexed the content (not by
# this module) -- shared across 09/10/11/12, each consuming via its own group.
KAFKA_TOPIC_INDEX_COMPLETED = "personal-index-completed"

KAFKA_TOPIC_DLQ = "video-processing-dlq"

# SeaweedFS配置
#SEAWEEDFS_BASE_URL = "http://192.168.57.156:8086"
SEAWEEDFS_TIMEOUT = 30
SEAWEEDFS_RETRY_COUNT = 3

# VIE01-369: retry budget for the final-status write (update_asset_status etc.) --
# separate from SEAWEEDFS_RETRY_COUNT because that one's tuned for an ordinary slow/
# flaky connection (download_file), not for riding out a --backup write-freeze window,
# which can run for as long as the coordinated pg_dump/ArcadeDB backup actually takes.
# 8 attempts with the 90s-capped backoff below sum to ~3.6 min before giving up.
WRITE_FREEZE_RETRY_COUNT = 8

# Asset Management API
ASSET_MANAGEMENT_URL = os.environ.get("ASSET_MANAGEMENT_URL", "http://raptor-asset-management:8000")

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/media_processing/video_processing"  # nosec B108 -- fixed base directory constant, matches the docker-compose volume layout, not a per-request predictable filename
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

