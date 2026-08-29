# services/video_analysis_service/config.py

SERVICE_NAME = "video_analysis_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.57.156:19002,192.168.57.156:19003,192.168.57.156:19004'
KAFKA_GROUP_ID = 'video-analysis-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "video-analysis-requests"

# Topic配置 - produce to
# Scene detection removed; keyframes come from video_chunking_service
KAFKA_TOPIC_OCR_FRAME_REQUEST = "video-ocr-frame-requests"
KAFKA_TOPIC_FRAME_DESCRIPTION_REQUEST = "video-frame-description-requests"
KAFKA_TOPIC_RESPONSE = "video-analysis-results"
KAFKA_TOPIC_DLQ = "video-analysis-dlq"

# Topic配置 - listen to
KAFKA_TOPIC_OCR_FRAME_RESULT = "video-ocr-frame-results"
KAFKA_TOPIC_FRAME_DESCRIPTION_RESULT = "video-frame-description-results"

# 檔案路徑配置
FRAME_EXTRACTION_BASE_DIR = "/tmp/media_processing/video_processing/frame_extraction"
MERGED_RESULTS_BASE_DIR = "/tmp/media_processing/video_processing/merged_results"
TEMP_FILE_CLEANUP_DELAY = 300  # 5分鐘後清理

# 幀提取配置
FRAME_EXTRACTION_FPS = 0.5  # 每秒提取幀數
FRAME_FORMAT = "jpg"
FRAME_QUALITY = 85

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "video_analysis_service.log"

# Redis 配置
#REDIS_HOST = "192.168.57.156"
#REDIS_PORT = 6391
REDIS_DB = 0
REDIS_KEY_PREFIX = "video_analysis:"
REDIS_KEY_TTL = 86400

