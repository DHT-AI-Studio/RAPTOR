# services/audio_tts_service/kafka_handler.py

import json
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from tts_client import TTSClient
from config import (
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_TTS_REQUEST,
    KAFKA_TOPIC_TTS_RESULT,
    KAFKA_TOPIC_DLQ,
    SERVICE_NAME,
)
from dotenv import load_dotenv
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

logger = logging.getLogger(__name__)


class AudioTTSKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.tts_client = TTSClient()

    async def start_consumer(self):
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_TTS_REQUEST,
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
        logger.info(f"AudioTTSService listening on {KAFKA_TOPIC_TTS_REQUEST}")

        try:
            async for message in consumer:
                await self.process_message(message.value, producer)
        finally:
            await consumer.stop()
            await producer.stop()

    async def process_message(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        correlation_id = message.get("correlation_id", "unknown")
        try:
            payload = message.get("payload", {})
            parameters = payload.get("parameters", {})
            text = parameters.get("text", "")
            voice = parameters.get("voice", "default")
            speed = float(parameters.get("speed", 1.0))
            output_format = parameters.get("output_format", "wav")

            tts_result = await self.tts_client.synthesize(text, voice, speed, output_format)

            response = {
                "message_id": str(uuid.uuid4()),
                "correlation_id": message.get("correlation_id"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_service": SERVICE_NAME,
                "target_service": message.get("source_service", ""),
                "message_type": "RESPONSE",
                "priority": message.get("priority", "MEDIUM"),
                "payload": {
                    "request_id": payload.get("request_id"),
                    "status": "success",
                    "results": tts_result,
                },
                "retry_count": 0,
                "ttl": 3600,
            }
            await producer.send_and_wait(KAFKA_TOPIC_TTS_RESULT, response)
            logger.info(f"[{correlation_id}] TTS synthesis complete")

        except Exception as e:
            logger.error(f"[{correlation_id}] TTS failed: {e}")
            error_msg = {
                "message_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_service": SERVICE_NAME,
                "message_type": "ERROR",
                "priority": "HIGH",
                "payload": {"status": "failed", "error": str(e)},
                "retry_count": 0,
                "ttl": 3600,
            }
            await producer.send_and_wait(KAFKA_TOPIC_DLQ, error_msg)
