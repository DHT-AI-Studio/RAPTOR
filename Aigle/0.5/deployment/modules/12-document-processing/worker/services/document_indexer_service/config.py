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

# Side-output topic for downstream consumers (module 17, retired -- see kafka_handler.py)
KAFKA_TOPIC_INDEX_REQUEST = "opensearch-index-requests"

# 重試配置
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# GD-6 / VIE01-251: PII scan of each chunk via Module 23 before personal
# indexing (pii_scan.py). GUARDRAIL_URL is the platform-wide shared name
# (root .env); the scan is on by default only when it's set, so existing
# deployments without it behave exactly as before. M12_PII_SCAN_ENABLED
# overrides either way.
GUARDRAIL_URL = os.getenv("GUARDRAIL_URL", "").strip()
# Empty counts as unset (compose passes ${VAR:-} through as "").
M12_PII_SCAN_ENABLED = (os.getenv("M12_PII_SCAN_ENABLED") or ("true" if GUARDRAIL_URL else "false")).strip().lower() == "true"
# Per-chunk Module 23 call timeout; a scan past it is skipped (fail-open).
M12_PII_SCAN_TIMEOUT_MS = int(os.getenv("M12_PII_SCAN_TIMEOUT_MS") or "1000")
# CM-4 / VIE01-403
M12_MODERATION_SCAN_ENABLED: bool = os.getenv("M12_MODERATION_SCAN_ENABLED", "true").lower() == "true"
M12_MODERATION_SCAN_TIMEOUT_MS: int = int(os.getenv("M12_MODERATION_SCAN_TIMEOUT_MS", "5000"))

