# services/video_graph_service/config.py

KAFKA_TOPIC_REQUEST = "video-indexer-results"
KAFKA_TOPIC_RESULT  = "video-graph-ingest-results"
KAFKA_TOPIC_DLQ     = "video-graph-ingest-dlq"
KAFKA_GROUP_ID      = "video_graph_service_group"
SERVICE_NAME        = "video_graph_service"
LOG_LEVEL           = "INFO"
