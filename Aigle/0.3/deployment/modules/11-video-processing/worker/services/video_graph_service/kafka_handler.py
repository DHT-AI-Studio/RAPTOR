# services/video_graph_service/kafka_handler.py
#
# Consumes video-indexer-results, forwards enriched records to
# module 20 (graph-service) POST /ingest/payload.

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import aiohttp
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from dotenv import load_dotenv

import config

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://raptor-graph-service:8843")

logger = logging.getLogger(__name__)


def _build_response(original: Dict, status: str, result_payload: Dict) -> Dict:
    return {
        "message_id":     str(uuid.uuid4()),
        "correlation_id": original.get("correlation_id"),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_service": config.SERVICE_NAME,
        "target_service": original.get("source_service"),
        "message_type":   "RESPONSE",
        "priority":       original.get("priority", "MEDIUM"),
        "payload":        result_payload,
        "retry_count":    0,
        "ttl":            3600,
    }


class VideoGraphKafkaHandler:

    def __init__(self):
        self.consumer = None
        self.producer = None

    async def start(self):
        self.consumer = AIOKafkaConsumer(
            config.KAFKA_TOPIC_REQUEST,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(
                v, ensure_ascii=False, default=str
            ).encode("utf-8"),
        )

        await self.consumer.start()
        await self.producer.start()
        logger.info(f"Listening on {config.KAFKA_TOPIC_REQUEST}")

        try:
            async for message in self.consumer:
                await self._process(message.value)
        finally:
            await self.stop()

    async def stop(self):
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()

    async def _process(self, message: Dict[str, Any]):
        message_id = message.get("message_id", "unknown")
        payload = message.get("payload", {})

        # 只處理 indexer 成功的訊息
        if payload.get("status") != "success":
            logger.debug(f"[{message_id}] skipping non-success message")
            return

        try:
            records: List[Dict] = payload.get("results", {}).get("records", [])
            if not records:
                logger.warning(f"[{message_id}] no records in payload, skipping")
                return

            version_id = payload.get("version_id", "unknown")
            branch_id = payload.get("user_id", "")
            logger.info(f"[{message_id}] forwarding {len(records)} records for {version_id}")

            result = await self._post_to_graph(records, branch_id=branch_id)

            await self._send(
                config.KAFKA_TOPIC_RESULT,
                _build_response(message, "success", {
                    "version_id": version_id,
                    "records_sent": len(records),
                    **result,
                }),
            )
            logger.info(f"[{message_id}] graph ingest done: {result}")

        except Exception as e:
            logger.error(f"[{message_id}] error: {e}", exc_info=True)
            await self._send(
                config.KAFKA_TOPIC_DLQ,
                _build_response(message, "failed", {"error": str(e)}),
            )

    async def _post_to_graph(self, records: List[Dict], branch_id: str = "") -> Dict:
        url = f"{GRAPH_SERVICE_URL}/ingest/payload"
        body = {"records": records, "options": {}, "branch_id": branch_id}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Graph service returned {resp.status}: {text}")
                return await resp.json()

    async def _send(self, topic: str, data: Dict):
        try:
            await self.producer.send_and_wait(topic, data)
        except Exception as e:
            logger.error(f"Failed to send to {topic}: {e}")
