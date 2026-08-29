# services/video_chunking_service/kafka_handler.py

import json
import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from chunker import normalize_video, split_into_moments
from frame_selector import select_keyframes
from config import (
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_CHUNKING_REQUEST,
    KAFKA_TOPIC_CHUNKING_RESULT,
    KAFKA_TOPIC_DLQ,
    SERVICE_NAME,
)
from dotenv import load_dotenv
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


class VideoChunkingKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS

    async def start_consumer(self):
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_CHUNKING_REQUEST,
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
        logger.info(f"VideoChunkingService listening on {KAFKA_TOPIC_CHUNKING_REQUEST}")

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
            video_path = parameters.get("video_file_path") or parameters.get("temp_file_path") or parameters.get("file_path")
            request_id = payload.get("request_id", str(uuid.uuid4()))

            loop = asyncio.get_event_loop()

            # Normalize video to 720p H.264 before chunking
            logger.info(f"Normalizing video: {video_path}")
            normalized_path = await loop.run_in_executor(
                _executor, normalize_video, video_path, request_id
            )
            logger.info(f"Normalization complete: {normalized_path}")

            moments = await loop.run_in_executor(
                _executor, split_into_moments, normalized_path, request_id
            )

            # Extract keyframes for all moments in parallel
            keyframe_tasks = [
                loop.run_in_executor(_executor, select_keyframes, moment)
                for moment in moments
            ]
            keyframe_results = await asyncio.gather(*keyframe_tasks)
            for moment, keyframes in zip(moments, keyframe_results):
                moment["keyframe_paths"] = keyframes

            response = {
                "message_id": str(uuid.uuid4()),
                "correlation_id": message.get("correlation_id"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_service": SERVICE_NAME,
                "target_service": message.get("source_service", ""),
                "message_type": "RESPONSE",
                "priority": message.get("priority", "MEDIUM"),
                "payload": {
                    "request_id": request_id,
                    "status": "success",
                    "results": {
                        "moments": moments,
                        "moment_count": len(moments),
                        "original_parameters": parameters,
                    },
                },
                "retry_count": 0,
                "ttl": 3600,
            }
            await producer.send_and_wait(KAFKA_TOPIC_CHUNKING_RESULT, response)
            logger.info(f"[{correlation_id}] Chunked video into {len(moments)} moments")

            # Clean up normalized temp file (moments already record keyframe paths)
            try:
                if os.path.exists(normalized_path):
                    os.remove(normalized_path)
                    logger.info(f"[{correlation_id}] Cleaned up normalized video: {normalized_path}")
            except Exception as cleanup_err:
                logger.warning(f"[{correlation_id}] Failed to clean up normalized video: {cleanup_err}")

        except Exception as e:
            logger.error(f"[{correlation_id}] Chunking failed: {e}")
            error_msg = {
                "message_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_service": SERVICE_NAME,
                "message_type": "ERROR",
                "priority": "HIGH",
                "payload": {"status": "error", "error": str(e)},
                "retry_count": 0,
                "ttl": 3600,
            }
            await asyncio.gather(
                producer.send_and_wait(KAFKA_TOPIC_CHUNKING_RESULT, error_msg),
                producer.send_and_wait(KAFKA_TOPIC_DLQ, error_msg),
            )
