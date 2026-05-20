# services/video_contextualize_service/kafka_handler.py

import asyncio
import json
import logging
from typing import Any, Dict

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from dotenv import load_dotenv
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

from config import (
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_DLQ,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_RESULT,
)
from contextualize import generate_contextual_prefixes
from message_utils import MessageBuilder

logger = logging.getLogger(__name__)


class VideoContextualizeKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.service_name = "video_contextualize_service"
        logger.info("VideoContextualizeKafkaHandler initialized")

    async def start_consumer(self):
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            bootstrap_servers=self.bootstrap_servers,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        )
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"),
        )

        await consumer.start()
        await producer.start()
        logger.info("Video Contextualize Kafka consumer started, waiting for messages...")

        try:
            async for message in consumer:
                asyncio.create_task(self.process_message(message.value, producer))
        finally:
            await consumer.stop()
            await producer.stop()

    async def process_message(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        message_id = message.get("message_id", "unknown")
        try:
            logger.info("Processing message: %s", message_id)

            if not self._validate_message(message):
                await self._send_error(producer, message, "Invalid message format", "INVALID_FORMAT")
                return

            if message["target_service"] != self.service_name:
                await self._send_error(
                    producer, message,
                    f"Wrong target service: {message['target_service']}",
                    "WRONG_TARGET",
                )
                return

            if message["payload"]["action"] == "video_contextualize":
                result = await self._handle_contextualize(message)
                response = MessageBuilder.create_contextualize_result_message(message, result)
                await self._send(producer, KAFKA_TOPIC_RESULT, response)
            else:
                await self._send_error(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION",
                )

        except Exception as exc:
            logger.error("Error processing message %s: %s", message_id, exc)
            await self._send_to_dlq(producer, message, str(exc))

    async def _handle_contextualize(self, message: Dict[str, Any]) -> Dict[str, Any]:
        parameters = message["payload"].get("parameters", {})
        video_id = parameters.get("video_id", "unknown")
        global_summary = parameters.get("global_summary", "")
        moments = parameters.get("moments", [])

        logger.info("Contextualizing video_id=%s with %d moments", video_id, len(moments))

        try:
            enriched_moments = await generate_contextual_prefixes(global_summary, moments)
        except Exception as exc:
            logger.error("Contextualization failed for video_id=%s: %s", video_id, exc)
            enriched_moments = [{**m, "contextual_prefix": ""} for m in moments]

        return {
            "request_id": message["payload"].get("request_id"),
            "action": "video_contextualize",
            "status": "success",
            "parameters": {
                "video_id": video_id,
                "moments": enriched_moments,
                # Passthrough: orchestrator forwards these to indexer unchanged
                "summary_result_path": parameters.get("summary_result_path"),
                "video_analysis_path": parameters.get("video_analysis_path"),
                "audio_analysis_path": parameters.get("audio_analysis_path"),
                "asset_path": parameters.get("asset_path"),
                "version_id": parameters.get("version_id"),
                "status": parameters.get("status"),
                "filename": parameters.get("filename"),
            },
        }

    def _validate_message(self, message: Dict[str, Any]) -> bool:
        if not all(k in message for k in ("message_id", "target_service", "payload")):
            logger.error("Missing required top-level fields: %s", message)
            return False
        payload = message.get("payload", {})
        if not all(k in payload for k in ("request_id", "action")):
            logger.error("Missing required payload fields: %s", payload)
            return False
        return True

    async def _send(self, producer: AIOKafkaProducer, topic: str, message: Dict[str, Any]):
        try:
            await producer.send_and_wait(topic, message)
            logger.info("Sent message %s to %s", message.get("message_id"), topic)
        except Exception as exc:
            logger.error("Failed to send to %s: %s", topic, exc)

    async def _send_error(self, producer, original_message, error_message, error_code):
        try:
            resp = MessageBuilder.create_error_response(original_message, error_message, error_code)
            await self._send(producer, KAFKA_TOPIC_RESULT, resp)
        except Exception as exc:
            logger.error("Failed to send error response: %s", exc)

    async def _send_to_dlq(self, producer, message, error):
        try:
            dlq_msg = MessageBuilder.create_dlq_message(message, error, message.get("retry_count", 0))
            await self._send(producer, KAFKA_TOPIC_DLQ, dlq_msg)
        except Exception as exc:
            logger.error("Failed to send to DLQ: %s", exc)
