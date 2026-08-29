# services/video_analysis_service/kafka_handler.py
# Scene detection removed.
# Keyframes are provided by video_chunking_service via orchestrator.
# Flow: receive moments → build synthetic scene dir → fan-out OCR + LVLM in parallel.

import asyncio
import json
import logging
import os
import shutil
import aiofiles
from typing import Dict, Any, List
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from result_merger import ResultMerger
from message_utils import MessageBuilder
from redis_manager import RedisStateManager
from datetime import datetime
from config import (
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_OCR_FRAME_REQUEST,
    KAFKA_TOPIC_FRAME_DESCRIPTION_REQUEST,
    KAFKA_TOPIC_RESPONSE,
    KAFKA_TOPIC_OCR_FRAME_RESULT,
    KAFKA_TOPIC_FRAME_DESCRIPTION_RESULT,
    STATE_TIMEOUT,
    MAX_RETRY_COUNT,
    SERVICE_NAME,
    FRAME_EXTRACTION_BASE_DIR,
)
from dotenv import load_dotenv

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(parent_dir, ".env")
load_dotenv(dotenv_path)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

logger = logging.getLogger(__name__)


class VideoAnalysisKafkaHandler:
    def __init__(self):
        self.result_merger = ResultMerger()
        self.redis_manager = RedisStateManager()

    async def start_consumer(self):
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            KAFKA_TOPIC_OCR_FRAME_RESULT,
            KAFKA_TOPIC_FRAME_DESCRIPTION_RESULT,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            enable_auto_commit=True,
            auto_offset_reset="latest",
        )

        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        )

        await consumer.start()
        await producer.start()

        try:
            logger.info(f"{SERVICE_NAME} started, listening for messages...")
            async for message in consumer:
                try:
                    await self.handle_message(message, producer)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        finally:
            await consumer.stop()
            await producer.stop()

    async def handle_message(self, message, producer: AIOKafkaProducer):
        topic = message.topic
        data = message.value

        try:
            if topic == KAFKA_TOPIC_REQUEST:
                await self.handle_video_analysis_request(data, producer)
            elif topic == KAFKA_TOPIC_OCR_FRAME_RESULT:
                await self.handle_ocr_frame_result(data, producer)
            elif topic == KAFKA_TOPIC_FRAME_DESCRIPTION_RESULT:
                await self.handle_frame_description_result(data, producer)
            else:
                logger.warning(f"Unknown topic: {topic}")
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            await self.send_error_response(data, producer, str(e), "MESSAGE_PROCESSING_ERROR")

    # ──────────────────────────────────────────────────────────────────────────
    # Request handler
    # ──────────────────────────────────────────────────────────────────────────

    async def handle_video_analysis_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """
        Process video analysis request.
        Uses moments (with keyframe_paths) from video_chunking_service instead of scene detection.
        Flow: receive moments → build synthetic scene dir → fan-out OCR + LVLM in parallel.
        """
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            request_id = payload["request_id"]
            primary_filename = parameters.get("primary_filename")
            moments: List[Dict[str, Any]] = parameters.get("moments", [])

            if not primary_filename:
                raise ValueError("Primary filename is required")

            if not moments:
                raise ValueError(
                    "No moments provided — video_chunking_service must complete before video_analysis_service"
                )

            logger.info(f"Building scene structure from {len(moments)} moments for: {request_id}")
            if moments:
                logger.info(f"First moment keys: {list(moments[0].keys())}")

            scene_dir, scene_json_path, total_keyframes = await self._build_scene_from_moments(
                moments, request_id
            )

            correlation_id = message["correlation_id"]
            state = {
                "original_message": message,
                "step": "parallel_analysis",
                "scene_dir": scene_dir,
                "scene_json_path": scene_json_path,
                "moments": moments,
                "total_keyframes": total_keyframes,
                "ocr_frame_result": None,
                "frame_description_result": None,
            }

            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")

            await self._send_parallel_frame_analysis_requests(message, producer, state)
            logger.info(f"Parallel OCR + LVLM requests sent for: {request_id}")

        except Exception as e:
            logger.error(f"Video analysis request failed: {e}")
            await self.send_error_response(message, producer, str(e), "VIDEO_ANALYSIS_FAILED")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _build_scene_from_moments(
        self, moments: List[Dict[str, Any]], request_id: str
    ):
        """
        Copy pre-selected keyframes into a unified scene directory named scene_frame_NNNN.jpg,
        and generate a scene_detection.json compatible with downstream OCR/LVLM services.
        """
        scene_dir = os.path.join(FRAME_EXTRACTION_BASE_DIR, request_id, "scenes")
        os.makedirs(scene_dir, exist_ok=True)

        scene_data = []
        frame_counter = 0

        for moment in moments:
            keyframes = moment.get("keyframe_paths", [])
            for kf_path in keyframes:
                dest = os.path.join(scene_dir, f"scene_frame_{frame_counter:04d}.jpg")
                if os.path.exists(kf_path):
                    shutil.copy2(kf_path, dest)
                else:
                    logger.warning(f"Keyframe not found: {kf_path}")

                scene_data.append({
                    "frame_index": frame_counter,
                    "timestamp": round(moment["start_time"], 2),
                    "combined_diff": 0.0,
                    "moment_index": moment["moment_index"],
                })
                frame_counter += 1

        scene_json_path = os.path.join(scene_dir, "scene_detection.json")
        async with aiofiles.open(scene_json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(scene_data, indent=2, ensure_ascii=False))

        logger.info(f"Built synthetic scene dir with {frame_counter} keyframes at: {scene_dir}")
        return scene_dir, scene_json_path, frame_counter

    async def _send_parallel_frame_analysis_requests(
        self,
        original_message: Dict[str, Any],
        producer: AIOKafkaProducer,
        state: Dict[str, Any],
    ):
        """Send OCR and Frame Description requests in parallel using chunking keyframes."""
        payload = original_message["payload"]
        parameters = payload["parameters"]
        scene_dir = state["scene_dir"]
        scene_json_path = state["scene_json_path"]
        total_keyframes = state["total_keyframes"]
        video_file_path = parameters.get("video_file_path") or parameters.get("file_path")

        ocr_frame_request = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="video_ocr_frame_service",
            action="ocr_frame",
            parameters={
                "primary_filename": parameters.get("primary_filename"),
                "frame_directory": scene_dir,
                "total_frames": total_keyframes,
                "scene_json_file_path": scene_json_path,
                "scene_output_directory": scene_dir,
            },
            file_path=scene_dir,
        )

        frame_description_request = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="video_frame_description_service",
            action="frame_description",
            parameters={
                "primary_filename": parameters.get("primary_filename"),
                "frame_directory": scene_dir,
                "total_frames": total_keyframes,
                "scene_json_file_path": scene_json_path,
                "scene_output_directory": scene_dir,
                "video_path": video_file_path,
                "moments": state["moments"],
            },
            file_path=scene_dir,
        )

        await asyncio.gather(
            producer.send(KAFKA_TOPIC_OCR_FRAME_REQUEST, ocr_frame_request),
            producer.send(KAFKA_TOPIC_FRAME_DESCRIPTION_REQUEST, frame_description_request),
        )
        logger.info(f"OCR and Frame Description requests sent for: {payload['request_id']}")

    # ──────────────────────────────────────────────────────────────────────────
    # Result handlers
    # ──────────────────────────────────────────────────────────────────────────

    async def handle_ocr_frame_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        correlation_id = message["correlation_id"]

        state = self.redis_manager.get_state(correlation_id)
        if state is None:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return

        if not self.redis_manager.update_state(correlation_id, {"ocr_frame_result": message["payload"]}):
            logger.error(f"Failed to update OCR result in Redis for: {correlation_id}")
            return

        state = self.redis_manager.get_state(correlation_id)
        logger.info(f"OCR frame result received for: {correlation_id}")
        await self.check_analysis_completion(producer, state, correlation_id)

    async def handle_frame_description_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        correlation_id = message["correlation_id"]

        state = self.redis_manager.get_state(correlation_id)
        if state is None:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return

        if not self.redis_manager.update_state(correlation_id, {"frame_description_result": message["payload"]}):
            logger.error(f"Failed to update frame description result in Redis for: {correlation_id}")
            return

        state = self.redis_manager.get_state(correlation_id)
        logger.info(f"Frame description result received for: {correlation_id}")
        await self.check_analysis_completion(producer, state, correlation_id)

    async def check_analysis_completion(
        self, producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str
    ):
        """Wait for both OCR + LVLM results then merge and emit response."""
        if state["ocr_frame_result"] is None or state["frame_description_result"] is None:
            return

        try:
            original_message = state["original_message"]
            payload = original_message["payload"]
            request_id = payload["request_id"]
            primary_filename = payload["parameters"].get("primary_filename")

            has_errors = False
            error_messages = []
            for service_name, result in [
                ("ocr_frame", state["ocr_frame_result"]),
                ("frame_description", state["frame_description_result"]),
            ]:
                if isinstance(result, dict) and result.get("status") == "error":
                    has_errors = True
                    error_messages.append(f"{service_name}: {result.get('error', 'Unknown error')}")

            if has_errors:
                await self.send_error_response(
                    original_message, producer,
                    f"Video analysis failed: {'; '.join(error_messages)}",
                    "ANALYSIS_FAILED",
                )
                return

            ocr_json_path = state["ocr_frame_result"]["file_path"]
            frame_description_json_path = state["frame_description_result"]["file_path"]

            merged_file_path = await self.result_merger.merge_all_video_results(
                scene_json_path=state["scene_json_path"],
                ocr_json_path=ocr_json_path,
                frame_description_json_path=frame_description_json_path,
                request_id=request_id,
                primary_filename=primary_filename,
            )

            combined_results = {
                "keyframe_analysis": {
                    "output_directory": state["scene_dir"],
                    "total_frames_extracted": state["total_keyframes"],
                },
                "ocr_frame": {
                    "status": state["ocr_frame_result"]["parameters"]["status"],
                    "result_file_path": state["ocr_frame_result"]["file_path"],
                    "total_scenes": state["ocr_frame_result"]["parameters"].get("total_scenes", 0),
                    "ocr_detected_count": state["ocr_frame_result"]["parameters"].get("ocr_detected_count", 0),
                },
                "frame_description": {
                    "status": state["frame_description_result"]["parameters"]["status"],
                    "result_file_path": state["frame_description_result"]["file_path"],
                    "total_descriptions": state["frame_description_result"]["parameters"].get("total_descriptions", 0),
                    "processing_service": "video_frame_description_service",
                },
                "merged_analysis": {
                    "merged_file_path": merged_file_path,
                    "format": "List of {frame_index, timestamp, text, frame_description} objects",
                },
            }

            response = MessageBuilder.create_processing_response(
                original_message=state["original_message"],
                status="SUCCESS",
                results=combined_results,
                file_path=merged_file_path,
            )

            await producer.send(KAFKA_TOPIC_RESPONSE, response)
            logger.info(f"Video analysis completed successfully for: {correlation_id}")
            logger.info(f"Merged results saved to: {merged_file_path}")

            await self.cleanup_processing_state(correlation_id)

        except Exception as e:
            logger.error(f"Error in analysis completion: {e}")
            await self.send_error_response(
                state["original_message"], producer, str(e), "ANALYSIS_COMPLETION_FAILED"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    async def send_error_response(
        self,
        original_message: Dict[str, Any],
        producer: AIOKafkaProducer,
        error_message: str,
        error_code: str,
    ):
        try:
            error_response = MessageBuilder.create_error_response(
                original_message=original_message,
                error_message=error_message,
                error_code=error_code,
            )
            # Must go to KAFKA_TOPIC_RESPONSE, not KAFKA_TOPIC_DLQ — video_orchestrator_service
            # only listens on the former, so sending errors to the DLQ meant it never learned
            # the job had failed and the correlation_id looked stuck forever instead.
            await producer.send(KAFKA_TOPIC_RESPONSE, error_response)
            logger.error(f"Error response sent: {error_message}")

            correlation_id = original_message.get("correlation_id")
            if correlation_id and self.redis_manager.exists(correlation_id):
                await self.cleanup_processing_state(correlation_id)

        except Exception as e:
            logger.error(f"Failed to send error response: {e}")

    async def cleanup_processing_state(self, correlation_id: str, keep_redis: bool = True):
        state = self.redis_manager.get_state(correlation_id)
        if state:
            if not keep_redis:
                self.redis_manager.delete_state(correlation_id)
                logger.info(f"Processing state deleted from Redis for: {correlation_id}")
            else:
                self.redis_manager.update_state(correlation_id, {
                    "step": "completed",
                    "completed_at": datetime.now().isoformat(),
                })
                logger.info(f"Processing state kept in Redis (will expire) for: {correlation_id}")
