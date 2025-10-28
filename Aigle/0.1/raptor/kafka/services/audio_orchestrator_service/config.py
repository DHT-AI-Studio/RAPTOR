# services/audio_orchestrator_service/config.py

SERVICE_NAME = "audio_orchestrator_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_GROUP_ID = 'audio-orchestrator-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "audio-processing-requests"

# Topic配置 - produce to (並行處理)
KAFKA_TOPIC_CLASSIFIER_REQUEST = "audio-classifier-requests"
KAFKA_TOPIC_DIARIZATION_REQUEST = "audio-diarization-requests"
KAFKA_TOPIC_RECOGNIZER_REQUEST = "audio-recognizer-requests"

# Topic配置 - produce to (後續處理)
KAFKA_TOPIC_SUMMARY_REQUEST = "audio-summary-requests"
KAFKA_TOPIC_SAVE_QDRANT_REQUEST = "audio-results-save-qdrant-requests"
KAFKA_TOPIC_FINAL_RESULT = "audio-processing-results"

# Topic配置 - listen to
KAFKA_TOPIC_CLASSIFIER_RESULT = "audio-classifier-results"
KAFKA_TOPIC_DIARIZATION_RESULT = "audio-diarization-results"
KAFKA_TOPIC_RECOGNIZER_RESULT = "audio-recognizer-results"
KAFKA_TOPIC_SUMMARY_RESULT = "audio-summary-results"
KAFKA_TOPIC_SAVE_QDRANT_RESULT = "audio-results-save-qdrant-results"

KAFKA_TOPIC_DLQ = "audio-processing-dlq"

# SeaweedFS配置
#SEAWEEDFS_BASE_URL = "http://192.168.157.165:8086"
SEAWEEDFS_TIMEOUT = 30
SEAWEEDFS_RETRY_COUNT = 3

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/audio_processing"
MERGED_FILE_DIR = "/tmp/audio_processing/merged"
TEMP_FILE_CLEANUP_DELAY = 300  # 5分鐘後清理

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "audio_orchestrator_service.log"

# Redis 配置
#REDIS_HOST = "192.168.157.165"
#REDIS_PORT = 6391
REDIS_DB = 0
REDIS_KEY_PREFIX = "audio_orchestrator:"
REDIS_KEY_TTL = 86400

