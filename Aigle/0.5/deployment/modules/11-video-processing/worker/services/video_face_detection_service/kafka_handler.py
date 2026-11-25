# services/video_face_detection_service/kafka_handler.py
# 接收 orchestrator 的人臉偵測請求（與影片／音訊分析平行，v1.4 §4.3）。
# 結果寫入 Module 25，摘要回報給 orchestrator 記錄，不在主流程的完成條件裡。

import asyncio
import json
import logging
import os
from typing import Any

import aiohttp
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from analysis import analyze_video
from asset_client import fetch_video, upload_representatives
from config import (
    FACE_TEMP_DIR,
    KAFKA_GROUP_ID,
    KAFKA_MAX_POLL_INTERVAL_MS,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_RESPONSE,
    SERVICE_NAME,
)
from consent_client import face_consent_denied
from message_utils import MessageBuilder
from personal_db_client import build_face_traces_request, write_face_traces

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

logger = logging.getLogger(__name__)


class FaceDetectionKafkaHandler:
    def __init__(self, detector):
        self.detector = detector
        os.makedirs(FACE_TEMP_DIR, exist_ok=True)

    async def start_consumer(self):
        """啟動 Kafka 消費者"""
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            enable_auto_commit=True,
            auto_offset_reset='latest',
            max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL_MS,
            session_timeout_ms=60000,
            heartbeat_interval_ms=20000,
        )
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )

        await consumer.start()
        await producer.start()

        try:
            logger.info(f"{SERVICE_NAME} started, listening for messages...")
            async for message in consumer:
                try:
                    await self.handle_request(message.value, producer)
                except Exception as e:  # noqa: BLE001 -- last-resort guard so one malformed message doesn't kill the consumer loop
                    logger.error(f"Error handling message: {e}")
        finally:
            await consumer.stop()
            await producer.stop()

    async def handle_request(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理一支影片：取得原始檔 → 偵測、追蹤、分群 → 上傳代表圖 → 寫入 Module 25 → 回報摘要。
        偵測／分群或寫入 Module 25 失敗回 ERROR；代表圖上傳失敗不影響結果，只記在摘要裡。

        GD-11：使用者沒有 face_detection 同意（或查不到）就不偵測，回報 {"skipped": 原因}。
        偵測完、上傳代表圖和寫入 Module 25 之前再查一次：偵測期間撤回同意的話，
        Module 33 已經刪過這個使用者的人臉資料，這時寫入會把刪掉的又放回去。"""
        video_path = None
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            request_id = payload["request_id"]
            user_id = payload.get("user_id")
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            if not asset_path or not version_id or not user_id:
                raise ValueError("asset_path, version_id and user_id are required")
            if reason := await face_consent_denied(user_id):
                await self._report_skipped(message, producer, reason)
                return

            ext = os.path.splitext(parameters.get("primary_filename") or "")[1] or ".mp4"
            video_path = os.path.join(FACE_TEMP_DIR, f"{request_id}{ext}")
            await fetch_video(
                dest_path=video_path,
                source_path=parameters.get("video_file_path"),
                asset_path=asset_path,
                version_id=version_id,
                download_headers=parameters.get("download_headers") or {},
                download_params=parameters.get("download_params") or {},
            )

            logger.info(f"Starting face analysis for {asset_path}/{version_id} (request {request_id})")
            # 解碼 + 推論是同步的 CPU/GPU 工作；丟到 thread，讓 /embed-face 在同一個 event loop 上照常回應
            analysis = await asyncio.to_thread(analyze_video, video_path, self.detector, f"{asset_path}/{version_id}")
            if reason := await face_consent_denied(user_id):
                await self._report_skipped(message, producer, reason)
                return
            summary = analysis.summary()
            summary.update(await self._upload_representatives(parameters, payload, analysis))
            summary.update(await self._write_personal_db(user_id, parameters, analysis))

            await producer.send(
                KAFKA_TOPIC_RESPONSE,
                MessageBuilder.create_processing_response(message, results=summary),
            )
            logger.info(f"Face detection result sent for {asset_path}/{version_id}")

        except Exception as e:  # noqa: BLE001 -- spans download, cv2/torch inference, and Module 25/04 calls; already logged + error response sent
            logger.error(f"Face detection failed for {message.get('correlation_id')}: {e}")
            await producer.send(
                KAFKA_TOPIC_RESPONSE,
                MessageBuilder.create_error_response(message, str(e), "FACE_DETECTION_FAILED"),
            )
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError as e:
                    logger.warning(f"Failed to remove {video_path}: {e}")

    async def _report_skipped(self, message: dict[str, Any], producer: AIOKafkaProducer, reason: str):
        """GD-11：沒有處理任何人臉，也沒有寫入任何東西。"""
        user_id = message["payload"].get("user_id")
        logger.info(f"Face detection skipped for {message.get('correlation_id')} (user {user_id}): {reason}")
        await producer.send(
            KAFKA_TOPIC_RESPONSE,
            MessageBuilder.create_processing_response(message, results={"skipped": reason}),
        )

    async def _upload_representatives(self, parameters: dict[str, Any], payload: dict[str, Any], analysis) -> dict[str, Any]:
        """每條軌跡的代表圖上傳到 Module 04（v1.4 §4.3 第 8 步），檔名 face_{trace_id}.jpg。"""
        headers = parameters.get("download_headers") or {}
        user_id = headers.get("X-User-ID") or payload.get("user_id")
        branch_id = headers.get("X-Branch-ID") or parameters.get("branch_id") or payload.get("branch_id")
        files = [(f"face_{t.trace_id}.jpg", t.representative.thumbnail_jpeg) for t in analysis.traces]
        if not files:
            return {"representatives_uploaded": 0}
        if not (user_id and branch_id):
            logger.warning("Missing user/branch for representative upload; skipped")
            return {"representatives_uploaded": 0, "representative_upload_error": "missing user_id/branch_id"}
        try:
            stored = await upload_representatives(
                user_id, branch_id, parameters["asset_path"], parameters["version_id"], files)
            for trace in analysis.traces:
                trace.representative_version_id = stored.get(f"face_{trace.trace_id}.jpg")
            return {"representatives_uploaded": len(stored)}
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            logger.warning(f"Representative upload failed (non-fatal): {e}")
            return {"representatives_uploaded": 0, "representative_upload_error": str(e)}

    async def _write_personal_db(self, user_id: str, parameters: dict[str, Any], analysis) -> dict[str, Any]:
        """寫入 Module 25（v1.4 §4.5）：軌跡、分群，並由 Module 25 做跨影片比對。回傳比對結果的計數。"""
        body = build_face_traces_request(analysis, parameters["version_id"], parameters["asset_path"],
                                         parameters.get("primary_filename"))
        result = await write_face_traces(user_id, body)
        matches = result.get("matches", [])
        return {
            "identities_linked": sum(bool(m["identity_id"]) and not m["created_identity"] for m in matches),
            "identities_created": sum(m["created_identity"] for m in matches),
            "groups_unassigned": sum(m["identity_id"] is None for m in matches),
        }
