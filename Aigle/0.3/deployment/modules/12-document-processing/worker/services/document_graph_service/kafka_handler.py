# services/document_graph_service/kafka_handler.py

import json
import asyncio
import logging
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from graph_builder import GraphBuilder
from neo4j_client import Neo4jClient
from config import (
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_GRAPH_REQUEST,
    KAFKA_TOPIC_GRAPH_RESULT,
    KAFKA_TOPIC_DLQ,
    SERVICE_NAME,
)
from dotenv import load_dotenv
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

logger = logging.getLogger(__name__)


class DocumentGraphKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.graph_builder = GraphBuilder()
        self.neo4j_client = None  # initialised in start_consumer

    async def start_consumer(self):
        self.neo4j_client = Neo4jClient()

        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_GRAPH_REQUEST,
            bootstrap_servers=self.bootstrap_servers,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        )

        await consumer.start()
        await producer.start()
        logger.info(f"DocumentGraphService listening on {KAFKA_TOPIC_GRAPH_REQUEST}")

        try:
            async for message in consumer:
                await self.process_message(message.value, producer)
        finally:
            await consumer.stop()
            await producer.stop()
            await self.neo4j_client.close()

    async def process_message(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        correlation_id = message.get("correlation_id", "unknown")
        try:
            payload = message.get("payload", {})
            parameters = payload.get("parameters", {})
            document_id = parameters.get("document_id", "")
            chunks = parameters.get("chunks", [])

            extracted = await self.graph_builder.extract_entities_and_relations(chunks)
            entities = extracted.get("entities", [])
            relations = extracted.get("relations", [])

            neo4j_result = await self.neo4j_client.upsert_entities_and_relations(
                entities, relations, document_id
            )
            await self.graph_builder.post_temporal_facts(relations, document_id)

            response = self._build_response(message, "success", {
                "document_id": document_id,
                "entities_written": neo4j_result["entities"],
                "relations_written": neo4j_result["relations"],
            })
            await producer.send_and_wait(KAFKA_TOPIC_GRAPH_RESULT, response)
            logger.info(f"Graph built for {document_id}: {neo4j_result}")

        except Exception as e:
            logger.error(f"[{correlation_id}] Graph build failed: {e}")
            response = self._build_response(message, "failed", {}, error=str(e))
            await producer.send_and_wait(KAFKA_TOPIC_DLQ, response)

    def _build_response(
        self, original: Dict[str, Any], status: str, results: Dict, error: str = ""
    ) -> Dict[str, Any]:
        import uuid
        from datetime import datetime, timezone
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": original.get("correlation_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": SERVICE_NAME,
            "target_service": original.get("source_service", ""),
            "message_type": "RESPONSE",
            "priority": original.get("priority", "MEDIUM"),
            "payload": {
                "request_id": original.get("payload", {}).get("request_id"),
                "status": status,
                "results": results,
                "error": error,
            },
            "retry_count": 0,
            "ttl": 3600,
        }
