# services/video_contextualize_service/config.py

import os

SERVICE_NAME = "video_contextualize_service"

KAFKA_GROUP_ID = "video-contextualize-service-group"
KAFKA_TOPIC_REQUEST = "video-contextualize-requests"
KAFKA_TOPIC_RESULT  = "video-contextualize-results"
KAFKA_TOPIC_DLQ     = "video-contextualize-dlq"

# Inference (reuse same gateway as video_summary_service)
INFERENCE_URL         = os.environ.get("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
INFERENCE_MODEL       = os.environ.get("CONTEXTUALIZE_MODEL_NAME") or os.environ.get("INFERENCE_MODEL", "qwen3.5:9b")
INFERENCE_TIMEOUT     = 180
INFERENCE_TEMPERATURE = 0.2
INFERENCE_MAX_LENGTH  = 600  # JSON array for one batch
INFERENCE_THINK       = os.environ.get("INFERENCE_THINK", "false").lower() == "true"

# Batching / concurrency
BATCH_SIZE             = 8   # moments per LLM call
MAX_CONCURRENT_BATCHES = 3   # asyncio.gather fan-out cap

# Async task pool
MAX_WORKERS          = 4
MAX_CONCURRENT_TASKS = 6
TASK_TIMEOUT         = 240

TEMP_FILE_DIR = "/tmp/media_processing/video_processing/contextualize_results"

LOG_LEVEL = "INFO"
