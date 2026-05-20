# services/document_indexer_service/config.py

import os

# Kafka 配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.57.156:19002,192.168.57.156:19003,192.168.57.156:19004'
KAFKA_TOPIC_SAVE_REQUEST = "document-indexer-requests"
KAFKA_TOPIC_SAVE_RESULT = "document-indexer-results"
KAFKA_GROUP_ID = "document_indexer_service_group"

# 服務配置
SERVICE_NAME = "document_indexer_service"
LOG_LEVEL = "INFO"

# Qdrant API 配置
#QDRANT_API_URL = "http://192.168.57.156:8815/insert_json"

# Side-output topics for downstream consumers
KAFKA_TOPIC_GRAPH_REQUEST = "document-graph-requests"
KAFKA_TOPIC_INDEX_REQUEST = "opensearch-index-requests"

# 重試配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒
