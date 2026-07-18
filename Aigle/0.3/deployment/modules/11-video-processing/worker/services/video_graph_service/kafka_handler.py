# services/video_graph_service/kafka_handler.py
#
# Consumes video-indexer-results, forwards enriched records to
# module 20 (graph-service) POST /ingest/payload.

import asyncio
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
            records_path: str = payload.get("results", {}).get("records_path", "")
            if not records_path:
                logger.warning(f"[{message_id}] no records_path in payload, skipping")
                return
            if not os.path.exists(records_path):
                logger.warning(f"[{message_id}] records file not found: {records_path}, skipping")
                return

            with open(records_path, "r", encoding="utf-8") as f:
                records: List[Dict] = json.load(f)
            try:
                os.remove(records_path)
            except Exception as e:
                logger.warning(f"[{message_id}] failed to remove records file: {e}")

            if not records:
                logger.warning(f"[{message_id}] records file is empty, skipping")
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

        # Submit job — expect 202 + job_id
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 202:
                    text = await resp.text()
                    raise RuntimeError(f"Graph service returned {resp.status}: {text}")
                data = await resp.json()

        job_id = data["job_id"]
        status_url = f"{GRAPH_SERVICE_URL}/ingest/jobs/{job_id}"
        logger.info(f"Graph ingest job submitted: {job_id}")

        # Poll until done (max 2 hours)
        max_wait_sec = 7200
        poll_interval_sec = 15
        elapsed = 0

        while elapsed < max_wait_sec:
            await asyncio.sleep(poll_interval_sec)
            elapsed += poll_interval_sec

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 404:
                            raise RuntimeError(f"Job {job_id} not found — graph service may have restarted")
                        if resp.status != 200:
                            logger.warning(f"Job status check returned {resp.status}, retrying")
                            continue
                        job = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Job status check failed ({e}), retrying")
                continue

            if job["status"] == "done":
                logger.info(f"Graph ingest job {job_id} done in {elapsed}s")
                return job["result"]
            elif job["status"] == "failed":
                raise RuntimeError(f"Graph ingest job {job_id} failed: {job.get('error')}")
            # pending / running → keep polling

        raise RuntimeError(f"Graph ingest job {job_id} timed out after {max_wait_sec}s")

    async def _send(self, topic: str, data: Dict):
        try:
            await self.producer.send_and_wait(topic, data)
        except Exception as e:
            logger.error(f"Failed to send to {topic}: {e}")
