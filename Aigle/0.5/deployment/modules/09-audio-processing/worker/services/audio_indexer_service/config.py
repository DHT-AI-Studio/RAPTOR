import os

# services/audio_indexer_service/config.py


# Kafka 配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.57.156:19002,192.168.57.156:19003,192.168.57.156:19004'
KAFKA_TOPIC_SAVE_REQUEST = "audio-indexer-requests"
KAFKA_TOPIC_SAVE_RESULT = "audio-indexer-results"
KAFKA_GROUP_ID = "audio_indexer_service_group"

# 服務配置
SERVICE_NAME = "audio_indexer_service"
LOG_LEVEL = "INFO"

# Qdrant API 配置
#QDRANT_API_URL = "http://192.168.57.156:8815/insert_json"

# 重試配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# CM-4 / VIE01-403
M09_MODERATION_SCAN_ENABLED: bool = os.getenv("M09_MODERATION_SCAN_ENABLED", "true").lower() == "true"
M09_MODERATION_SCAN_TIMEOUT_MS: int = int(os.getenv("M09_MODERATION_SCAN_TIMEOUT_MS", "5000"))
GUARDRAIL_URL = os.getenv("GUARDRAIL_URL", "").strip()
