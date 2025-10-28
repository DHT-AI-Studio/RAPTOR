# services/audio_analysis_service/config.py

SERVICE_NAME = "audio_analysis_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_GROUP_ID = 'audio-analysis-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "audio-analysis-requests"

# Topic配置 - produce to
KAFKA_TOPIC_CLASSIFIER_REQUEST = "audio-classifier-requests"
KAFKA_TOPIC_RECOGNIZER_REQUEST = "audio-recognizer-requests"
KAFKA_TOPIC_DIARIZATION_REQUEST = "audio-diarization-requests"
KAFKA_TOPIC_RESPONSE = "audio-analysis-results"
KAFKA_TOPIC_DLQ = "audio-analysis-dlq"

# Topic配置 - listen to results
KAFKA_TOPIC_CLASSIFIER_RESULT = "audio-classifier-results"
KAFKA_TOPIC_RECOGNIZER_RESULT = "audio-recognizer-results"
KAFKA_TOPIC_DIARIZATION_RESULT = "audio-diarization-results"

# 檔案路徑配置
AUDIO_PROCESSING_BASE_DIR = "/tmp/video_processing"
AUDIO_MERGED_RESULTS_DIR = "/tmp/video_processing/audio_merged"
TEMP_FILE_CLEANUP_DELAY = 300  # 5分鐘後清理

# 音頻轉換配置
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "wav"

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "audio_analysis_service.log"

# Redis 配置
#REDIS_HOST = "192.168.157.165"
#REDIS_PORT = 6391
REDIS_DB = 0
REDIS_KEY_PREFIX = "audio_analysis:"
REDIS_KEY_TTL = 86400
