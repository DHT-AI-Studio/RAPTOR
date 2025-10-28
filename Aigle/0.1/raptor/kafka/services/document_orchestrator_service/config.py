# services/document_orchestrator_service/config.py

SERVICE_NAME = "document_orchestrator_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_GROUP_ID = 'document-orchestrator-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "document-processing-requests"

# Topic配置 - produce to
KAFKA_TOPIC_ANALYSIS_REQUEST = "document-analysis-requests"
KAFKA_TOPIC_SUMMARY_REQUEST = "document-summary-requests"
KAFKA_TOPIC_SAVE_QDRANT_REQUEST = "document-results-save-qdrant-requests"
KAFKA_TOPIC_FINAL_RESULT = "document-processing-results"

# Topic配置 - listen to
KAFKA_TOPIC_ANALYSIS_RESULT = "document-analysis-results"
KAFKA_TOPIC_SUMMARY_RESULT = "document-summary-results"
KAFKA_TOPIC_SAVE_QDRANT_RESULT = "document-results-save-qdrant-results"

KAFKA_TOPIC_DLQ = "document-processing-dlq"

# SeaweedFS配置
#SEAWEEDFS_BASE_URL = "http://192.168.157.165:8086"
SEAWEEDFS_TIMEOUT = 30
SEAWEEDFS_RETRY_COUNT = 3

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/document_processing"
TEMP_FILE_CLEANUP_DELAY = 300  # 5分鐘後清理

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "document_orchestrator_service.log"

# Redis 配置
#REDIS_HOST = "192.168.157.165"
#REDIS_PORT = 6391
REDIS_DB = 0
REDIS_KEY_PREFIX = "document_orchestrator:"
REDIS_KEY_TTL = 86400