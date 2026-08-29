# services/document_graph_service/config.py

SERVICE_NAME = "document_graph_service"

KAFKA_GROUP_ID = "document-graph-service-group"
KAFKA_TOPIC_GRAPH_REQUEST = "document-graph-requests"
KAFKA_TOPIC_GRAPH_RESULT = "document-graph-results"
KAFKA_TOPIC_DLQ = "document-graph-dlq"

# Neo4j
NEO4J_DB = "neo4j"

# Inference (entity extraction)
ENTITY_EXTRACTION_TASK = "text-generation"
ENTITY_EXTRACTION_ENGINE = "ollama"
ENTITY_EXTRACTION_MODEL = "qwenforsummary"

# Redis
REDIS_DB = 0
REDIS_KEY_PREFIX = "doc_graph:"
REDIS_KEY_TTL = 86400

LOG_LEVEL = "INFO"
LOG_FILE = "document_graph_service.log"
