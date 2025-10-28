# services/video_orchestrator_service/config.py

SERVICE_NAME = "video_orchestrator_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_GROUP_ID = 'video-orchestrator-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "video-processing-requests"

# Topic配置 - produce to (並行處理)
KAFKA_TOPIC_VIDEO_ANALYSIS_REQUEST = "video-analysis-requests"
KAFKA_TOPIC_AUDIO_ANALYSIS_REQUEST = "audio-analysis-requests"

# Topic配置 - produce to (後續處理)
KAFKA_TOPIC_SUMMARY_REQUEST = "video-summary-requests"
KAFKA_TOPIC_SAVE_QDRANT_REQUEST = "video-results-save-qdrant-requests"
KAFKA_TOPIC_FINAL_RESULT = "video-processing-results"

# Topic配置 - listen to
KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT = "video-analysis-results"
KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT = "audio-analysis-results"
KAFKA_TOPIC_SUMMARY_RESULT = "video-summary-results"
KAFKA_TOPIC_SAVE_QDRANT_RESULT = "video-results-save-qdrant-results"

KAFKA_TOPIC_DLQ = "video-processing-dlq"

# SeaweedFS配置
#SEAWEEDFS_BASE_URL = "http://192.168.157.165:8086"
SEAWEEDFS_TIMEOUT = 30
SEAWEEDFS_RETRY_COUNT = 3

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/video_processing"
TEMP_FILE_CLEANUP_DELAY = 300

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "video_orchestrator_service.log"

# Redis 配置
#REDIS_HOST = "192.168.157.165"
#REDIS_PORT = 6391
REDIS_DB = 0
REDIS_KEY_PREFIX = "video_orchestrator:"
REDIS_KEY_TTL = 86400

