# services/document_save2qdrant_service/config.py

import os

# Kafka 配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_TOPIC_SAVE_REQUEST = "document-results-save-qdrant-requests"
KAFKA_TOPIC_SAVE_RESULT = "document-results-save-qdrant-results"
KAFKA_GROUP_ID = "document_save2qdrant_service_group"

# 服務配置
SERVICE_NAME = "document_save2qdrant_service"
LOG_LEVEL = "INFO"

# Qdrant API 配置
#QDRANT_API_URL = "http://192.168.157.165:8815/insert_json"

# 重試配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒
