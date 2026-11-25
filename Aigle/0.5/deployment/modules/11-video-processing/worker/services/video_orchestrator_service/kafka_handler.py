# services/video_orchestrator_service/kafka_handler.py

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiohttp
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from api_client import ClaimOutcome, SeaweedFSClient
from capture_time import read_capture_time
from config import (
    ASSET_MANAGEMENT_URL,
    FACE_DETECTION_ENABLED,
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_AUDIO_ANALYSIS_REQUEST,
    KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT,
    KAFKA_TOPIC_CHUNKING_REQUEST,
    KAFKA_TOPIC_CHUNKING_RESULT,
    KAFKA_TOPIC_CONTEXTUALIZE_REQUEST,
    KAFKA_TOPIC_CONTEXTUALIZE_RESULT,
    KAFKA_TOPIC_DLQ,
    KAFKA_TOPIC_FACE_DETECTION_REQUEST,
    KAFKA_TOPIC_FACE_DETECTION_RESULT,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_INDEX_COMPLETED,
    KAFKA_TOPIC_INDEXER_REQUEST,
    KAFKA_TOPIC_INDEXER_RESULT,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SUMMARY_RESULT,
    KAFKA_TOPIC_VIDEO_ANALYSIS_REQUEST,
    KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT,
    SERVICE_NAME,
)
from dotenv import load_dotenv
from message_utils import MessageBuilder, create_final_result_message
from moderation_flag_reporter import report_moderation_flags
from redis_manager import RedisStateManager

# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
logger = logging.getLogger(__name__)

# CJK Unified Ideographs + Extension A + Compatibility Ideographs -- enough
# to cover ordinary Chinese/Japanese text; not exhaustive of every CJK block,
# but this only affects spacing cosmetics, never correctness.
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _join_word_tokens(tokens: list[str]) -> str:
    """Join WhisperX word-level tokens with a space, except between two
    adjacent CJK characters.

    WhisperX's word-level alignment has no natural word boundary for CJK, so
    it tokenizes one character at a time -- a blanket space join turns
    "美國媒體" into "美 國 媒 體". A blanket "" join (the original bug) fixed
    that but glued space-separated languages together into run-on text
    instead (e.g. "Youunderstandthat?"). Insert a space everywhere except
    directly between two CJK characters, so Latin words keep their spacing
    and CJK text reads naturally.
    """
    out = ""
    for tok in tokens:
        if not tok:
            continue
        if out and not (_is_cjk(out[-1]) and _is_cjk(tok[0])):
            out += " "
        out += tok
    return out


def _load_json_file_sync(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def _report_moment_flags(
    branch_id: str, version_id: str, asset_path: str, moments: list[dict[str, Any]], rollup: dict[str, Any],
) -> None:
    """CM-5 / VIE01-408: report each flagged category of each flagged moment to
    Module 23's review store, alongside (not instead of) the moderation_flags
    already written to Module 04/25 by the caller. One row per moment per
    category -- moment_id is this video's own (version_id) plus the moment's
    index, since moments have no id of their own yet at this stage of the
    pipeline. asset_path (this video's own) travels on every row too, not just
    moment_id -- an admin reviewing the flag needs the actual video, not just
    a one-way hash."""
    model = rollup.get("model", "")
    policy_version = (rollup.get("policy") or {}).get("policy_version")
    url = os.environ.get("GUARDRAIL_URL", "http://raptor-guardrail-service:8026")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        for moment in moments:
            moment_flags = moment.get("moderation_flags") or {}
            flagged = moment_flags.get("flagged") or []
            idx = moment.get("moment_index")
            if not flagged or idx is None:
                continue
            # Same derivation as Module 25's kafka_consumer._stable_chunk_id() --
            # this is the exact chunk_id this moment's Chunk will get there, so a
            # later consumer can correlate this flag with that vertex.
            moment_id = hashlib.sha256(
                "\x1f".join([version_id, "moment", str(idx)]).encode("utf-8")
            ).hexdigest()
            await report_moderation_flags(
                session, url, branch_id=branch_id, version_id=version_id, asset_path=asset_path,
                categories=flagged, model=model, scores=moment_flags.get("scores"),
                policy_version=policy_version, moment_id=moment_id,
            )


class VideoOrchestratorKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.group_id = KAFKA_GROUP_ID
        self.service_name = SERVICE_NAME
        self.seaweedfs_client = SeaweedFSClient()
        self.redis_manager = RedisStateManager()

    async def start_consumer(self):
        """啟動 Kafka Consumer"""
        # 建立 consumers
        request_consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            max_poll_interval_ms=3600000,  # 1 hour, handles large video downloads
            session_timeout_ms=60000,
            heartbeat_interval_ms=20000,
            enable_auto_commit=False,  # VIE01-255: manual commit drives claim-based ack/redeliver
        )
        
        result_consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT,
            KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT,
            KAFKA_TOPIC_FACE_DETECTION_RESULT,
            KAFKA_TOPIC_CHUNKING_RESULT,
            KAFKA_TOPIC_SUMMARY_RESULT,
            KAFKA_TOPIC_CONTEXTUALIZE_RESULT,
            KAFKA_TOPIC_INDEXER_RESULT,
            KAFKA_TOPIC_INDEX_COMPLETED,
            bootstrap_servers=self.bootstrap_servers,
            group_id=f"{self.group_id}-results",
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # 建立 producer
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        
        await request_consumer.start()
        await result_consumer.start()
        await producer.start()
        
        logger.info("Video Orchestrator Kafka consumers started...")
        
        try:
            # 並行處理兩個 consumer
            await asyncio.gather(
                self.process_requests(request_consumer, producer),
                self.process_results(result_consumer, producer)
            )
        finally:
            await request_consumer.stop()
            await result_consumer.stop()
            await producer.stop()
            self.redis_manager.close()
    
    def _claim_ids_from_message(self, message: dict[str, Any]) -> tuple[str, str, str, str, str] | None:
        """VIE01-255: pull (asset_path, version_id, correlation_id, user_id,
        branch_id) out of a request message, or None if it doesn't carry
        enough to claim/report status on."""
        payload = message.get("payload", {})
        parameters = payload.get("parameters", {})
        headers = payload.get("asset_managemant_download_header", {})
        asset_path = parameters.get("asset_path")
        version_id = parameters.get("version_id")
        correlation_id = message.get("correlation_id")
        user_id = headers.get("X-User-ID")
        branch_id = headers.get("X-Branch-ID")
        if not all([asset_path, version_id, correlation_id, user_id, branch_id]):
            return None
        return asset_path, version_id, correlation_id, user_id, branch_id

    async def process_requests(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理影片處理請求"""
        async for message in consumer:
            ids = self._claim_ids_from_message(message.value)
            if ids is None:
                await self.handle_video_request(message.value, producer)
                await consumer.commit()
                continue

            asset_path, version_id, correlation_id, user_id, branch_id = ids
            outcome = await self.seaweedfs_client.claim_processing(
                asset_path, version_id, correlation_id, user_id, branch_id
            )
            if outcome is ClaimOutcome.CONFLICT:
                logger.info(f"Claim 409 for {correlation_id}: discarding, no processing")
                await consumer.commit()
                continue
            if outcome is ClaimOutcome.TRANSIENT_ERROR:
                logger.warning(f"Claim failed (transient) for {correlation_id}: leaving uncommitted for redelivery")
                continue

            await self.handle_video_request(message.value, producer)
            await consumer.commit()
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_video_request(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理影片處理請求"""
        try:
            logger.info(f"Processing video request: {message.get('message_id')}")
            
            # 驗證消息格式
            if not self.validate_message(message):
                await self.send_error_response(
                    producer, message,
                    "Invalid message format",
                    "INVALID_FORMAT",
                    message.get("correlation_id")
                )
                return

            # 檢查 TTL
            if self.is_message_expired(message):
                await self.send_error_response(
                    producer, message,
                    "Message expired",
                    "MESSAGE_EXPIRED",
                    message.get("correlation_id")
                )
                return

            # 檢查目標服務
            if message["target_service"] != self.service_name:
                await self.send_error_response(
                    producer, message,
                    f"Wrong target service: {message['target_service']}",
                    "WRONG_TARGET",
                    message.get("correlation_id")
                )
                return

            # 處理影片處理請求
            if message["payload"]["action"] == "video_processing":
                await self.handle_video_processing(message, producer)
            else:
                await self.send_error_response(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION",
                    message.get("correlation_id")
                )
                
        except Exception as e:  # noqa: BLE001 -- dispatches to the full video-processing pipeline, broad guard to still emit an error response
            logger.error(f"Error processing video request: {e}")
            # Must go through send_error_response (KAFKA_TOPIC_FINAL_RESULT), not
            # send_to_dlq (KAFKA_TOPIC_DLQ) — nothing consumes the DLQ topic, so
            # a failure sent there vanishes silently instead of reaching whatever
            # is waiting for this job's result.
            await self.send_error_response(
                producer, message, str(e), "DISPATCH_ERROR", message.get("correlation_id")
            )

    async def handle_video_processing(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理影片處理流程"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            # 提取必要資訊
            download_headers = payload.get("asset_managemant_download_header", {})
            download_params = payload.get("asset_managemant_download_paramater", {})
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            primary_filename = parameters.get("primary_filename")
            download_url = f"{ASSET_MANAGEMENT_URL}/filedownload/{asset_path}/{version_id}"

            if not all([download_headers, asset_path, version_id, primary_filename]):
                await self.send_error_response(
                    producer, message,
                    "Missing required parameters",
                    "MISSING_PARAMETERS",
                    message.get("correlation_id")
                )
                return

            # 如果 Module 13 已存到 NFS，直接用本地路徑；否則退回 LakeFS 下載
            nfs_path = parameters.get("temp_file_path")
            if nfs_path and os.path.exists(nfs_path):
                temp_file_path = nfs_path
                logger.info(f"Using NFS file directly: {temp_file_path}")
            else:
                if nfs_path:
                    logger.warning(f"NFS path not found ({nfs_path}), falling back to download")
                logger.info(f"Downloading video file via asset management API: {download_url}")
                temp_file_path = await self.seaweedfs_client.download_file(
                    download_url=download_url,
                    download_headers=download_headers,
                    download_params=download_params,
                    filename=primary_filename
                )
            
            # 初始化處理狀態
            correlation_id = message.get("correlation_id")
            _dl_headers = payload.get("asset_managemant_download_header", {})
            state = {
                "original_message": message,
                "temp_file_path": temp_file_path,
                "video_analysis_result": None,
                "audio_analysis_result": None,
                "summary_result": None,
                "save_result": None,
                "step": "parallel_analysis",
                "branch_id": parameters.get("branch_id") or _dl_headers.get("X-Branch-ID") or payload.get("branch_id") or "",
                "created_at": asyncio.get_event_loop().time()
            }
            
            # 存入 Redis
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")

            # First send chunking request; analysis fan-out happens after chunking result arrives
            await self.send_chunking_request(message, producer, temp_file_path, primary_filename)
            state["step"] = "chunking"
            self.redis_manager.set_state(correlation_id, state)

        except Exception as e:  # noqa: BLE001 -- spans download (incl. its own self-raised Exception), Redis, and Kafka produce; already logged + error response sent
            logger.error(f"Error in video processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Video processing failed: {e!s}",
                "PROCESSING_FAILED",
                message.get("correlation_id")
            )

    async def send_chunking_request(
        self,
        message: dict[str, Any],
        producer: AIOKafkaProducer,
        temp_file_path: str,
        primary_filename: str,
    ) -> None:
        """Send video-chunking-requests before analysis fan-out."""
        chunking_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="video_chunking_service",
            action="video_chunking",
            parameters={
                "video_file_path":  temp_file_path,
                "primary_filename": primary_filename,
            },
            temp_file_path=temp_file_path,
        )
        await producer.send(KAFKA_TOPIC_CHUNKING_REQUEST, chunking_request)
        logger.info(f"Chunking request sent for: {message.get('correlation_id')}")

    async def handle_chunking_result(
        self,
        message: dict[str, Any],
        producer: AIOKafkaProducer,
        state: dict[str, Any],
        correlation_id: str,
    ) -> None:
        """After chunking completes, fan out per-moment analysis."""
        payload = message.get("payload", {})

        # Try both payload.results.moments and payload.moments for robustness
        moments = None
        results_obj = payload.get("results")
        if isinstance(results_obj, dict):
            moments = results_obj.get("moments")
        if not moments:
            moments = payload.get("moments")
        if not moments:
            moments = []
            logger.error("No moments found in chunking result payload — aborting analysis fan-out")
            await self.handle_processing_error(producer, state, "Chunking service returned empty moments", "CHUNKING_EMPTY")
            return

        state["chunking_moments"] = moments
        # CM-3: the video's keyframe-moderation rollup (None when moderation is off);
        # each moment above carries its own moderation_flags.
        moderation_flags = results_obj.get("moderation_flags") if isinstance(results_obj, dict) else None
        state["moderation_flags"] = moderation_flags if isinstance(moderation_flags, dict) else None
        state["step"] = "parallel_analysis"
        self.redis_manager.set_state(correlation_id, state)

        original  = state["original_message"]
        temp_path = state["temp_file_path"]
        primary   = original["payload"]["parameters"].get("primary_filename", "")
        await self.send_parallel_analysis_requests(original, producer, temp_path, primary, moments)
        logger.info(f"Analysis fan-out dispatched after chunking for: {correlation_id} ({len(moments)} moments)")

        if state["moderation_flags"] is not None:
            ids = self._claim_ids_from_message(original)
            if ids:
                asset_path, version_id, _cid, user_id, branch_id = ids
                await self.seaweedfs_client.update_moderation(
                    user_id, branch_id, asset_path, version_id, state["moderation_flags"])
                try:
                    await _report_moment_flags(branch_id, version_id, asset_path, moments, state["moderation_flags"])
                except Exception as exc:  # noqa: BLE001 -- CM-5 reporting must never affect CM-3's own moderation result
                    logger.warning(f"[moderation-flag] report failed for {asset_path}/{version_id}: {exc}")

    async def send_parallel_analysis_requests(self, message: dict[str, Any], producer: AIOKafkaProducer, temp_file_path: str, primary_filename: str, moments: list | None = None):
        """同時發送影片和音頻分析請求"""
        parameters = message["payload"]["parameters"]

        # 創建影片分析請求（傳入 moments 供 video_analysis_service 使用）
        video_analysis_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="video_analysis_service",
            action="video_analysis",
            parameters={
                "file_type": parameters.get("file_type", "video"),
                "primary_filename": primary_filename,
                "video_file_path": temp_file_path,
                "moments": moments or [],
            },
            temp_file_path=temp_file_path
        )
        
        # 創建音頻分析請求 (從影片中提取音頻)
        audio_analysis_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="audio_analysis_service",
            action="audio_analysis",
            parameters={
                "file_type": "video",  # 從影片提取音頻
                "primary_filename": primary_filename,
                "video_file_path": temp_file_path,  # 影片檔案路徑，用於提取音頻
                "moments": moments or [],  # 供 ASR 分段使用
            },
            temp_file_path=temp_file_path
        )
        
        # 並行發送兩個分析請求
        await asyncio.gather(
            producer.send(KAFKA_TOPIC_VIDEO_ANALYSIS_REQUEST, video_analysis_request),
            producer.send(KAFKA_TOPIC_AUDIO_ANALYSIS_REQUEST, audio_analysis_request)
        )
        
        correlation_id = message.get("correlation_id")
        logger.info(f"Parallel analysis requests sent for: {correlation_id}")

        if FACE_DETECTION_ENABLED:
            await self.send_face_detection_request(message, producer, temp_file_path, primary_filename)

    async def send_face_detection_request(self, message: dict[str, Any], producer: AIOKafkaProducer, temp_file_path: str, primary_filename: str):
        """人臉偵測：第三個平行請求（v1.4 §4.3），不在完成條件裡，送出失敗也不影響主流程。

        temp_file_path 在主流程結束時就會被刪掉，而人臉偵測可能更晚才結束，
        所以一併傳下載資訊，讓人臉服務自己保留一份原始檔（能 hard link 就 link，不行就從 Module 04 下載）。
        """
        payload = message["payload"]
        correlation_id = message.get("correlation_id")
        try:
            face_request = MessageBuilder.create_processing_request(
                original_message=message,
                target_service="video_face_detection_service",
                action="face_detection",
                parameters={
                    "primary_filename": primary_filename,
                    "video_file_path": temp_file_path,
                    "download_headers": payload.get("asset_managemant_download_header", {}),
                    "download_params": payload.get("asset_managemant_download_paramater", {}),
                },
                temp_file_path=temp_file_path
            )
            await producer.send(KAFKA_TOPIC_FACE_DETECTION_REQUEST, face_request)
            state = self.redis_manager.get_state(correlation_id) or {}
            if "face_detection" not in state:   # 結果已先回來時不要蓋掉
                self.redis_manager.update_state(correlation_id, {"face_detection": {"status": "processing"}})
            logger.info(f"Face detection request sent for: {correlation_id}")
        except Exception as e:  # noqa: BLE001 -- face detection is not in the completion condition; a send failure here must never fail the video (see module docstring + test_face_detection_branch.py)
            logger.warning(f"Face detection request failed (non-fatal) for {correlation_id}: {e}")

    async def handle_service_result(self, message: dict[str, Any], producer: AIOKafkaProducer, topic: str):
        """處理各服務的結果"""
        state = None
        correlation_id = message.get("correlation_id")
        try:
            payload = message["payload"]
            if not correlation_id:
                logger.warning("Missing correlation_id in result message")
                return

            state = self.redis_manager.get_state(correlation_id)
            if not state:
                logger.warning(f"Unknown correlation_id: {correlation_id}")
                return

            # 人臉偵測不在主流程裡：成功或失敗都只記錄，必須在下面的統一錯誤偵測之前攔下，
            # 否則人臉偵測失敗會走 handle_processing_error，把整支影片判為失敗、
            # 並把 Module 04 的資產狀態寫成 failed
            if topic == KAFKA_TOPIC_FACE_DETECTION_RESULT:
                await self.handle_face_detection_result(message, correlation_id)
                return

            # 檢查結果狀態 (also catches "failed", not just "error")
            message_type = message.get("message_type", "")
            service_status = payload.get("status", "")
            if message_type == "ERROR" or service_status in ("error", "failed"):
                # create_error_response() puts "error" at the top level of
                # payload; create_summary_result_message() (also used on this
                # same topic when summary_results["status"] is "error") nests
                # the whole results dict -- including its own "error" key --
                # one level deeper under "results". Fall back to that nested
                # shape before giving up to "Unknown error".
                err = payload.get("error") or payload.get("results", {}).get("error")
                if isinstance(err, dict):
                    error_message = err.get("message", "Unknown error")
                elif isinstance(err, str):
                    error_message = err
                else:
                    error_message = (
                        payload.get("metadata", {}).get("error_message")
                        or "Unknown error"
                    )
                logger.error(f"Service error for {correlation_id}: {error_message}")
                await self.handle_processing_error(producer, state, error_message, "SERVICE_ERROR")
                return
            
            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_CHUNKING_RESULT:
                await self.handle_chunking_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT:
                await self.handle_video_analysis_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT:
                await self.handle_audio_analysis_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_SUMMARY_RESULT:
                await self.handle_summary_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_CONTEXTUALIZE_RESULT:
                await self.handle_contextualize_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_INDEXER_RESULT:
                await self.handle_save_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_INDEX_COMPLETED:
                await self.handle_index_completed(message, producer, state, correlation_id)

        except Exception as e:  # noqa: BLE001 -- dispatches to many result handlers; must stay broad so failures here still get reported instead of stranding the job (see comment below)
            logger.error(f"Error handling service result: {e}")
            logger.error(f"Message: {json.dumps(message, indent=2)}")
            # An exception here used to leave the asset stuck in "processing"
            # forever -- no failure ever got reported to Module 04, unlike
            # the sibling branch above (a downstream service explicitly
            # reporting "error"/"failed") which does call
            # handle_processing_error(). Any bug in this dispatch logic
            # itself (not just this specific race) silently stranded the
            # job with no retry, no failure signal. state.get("original_message")
            # guards the one case handle_processing_error() itself can't
            # recover from -- it needs that field too.
            if state and state.get("original_message"):
                try:
                    await self.handle_processing_error(producer, state, f"Unexpected error: {e}", "UNEXPECTED_ERROR")
                except Exception as inner_e:  # noqa: BLE001 -- last-resort fallback if even reporting the processing error fails
                    logger.error(f"Also failed to report processing error: {inner_e}")
            else:
                logger.error(f"Cannot report failure for {correlation_id} -- state missing or has no original_message")

    async def handle_video_analysis_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理影片分析結果"""
        updates = {"video_analysis_result": message["payload"]}
        if not self.redis_manager.update_state(correlation_id, updates):
            raise Exception("Failed to update state in Redis")
        
        logger.info(f"Video analysis result received for: {state['original_message']['payload']['request_id']}")

        state = self.redis_manager.get_state(correlation_id)
        await self.check_analysis_completion(producer, state, correlation_id)

    async def handle_audio_analysis_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理音頻分析結果"""
        updates = {"audio_analysis_result": message["payload"]}
        if not self.redis_manager.update_state(correlation_id, updates):
            raise Exception("Failed to update state in Redis")
        
        logger.info(f"Audio analysis result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        state = self.redis_manager.get_state(correlation_id)
        await self.check_analysis_completion(producer, state, correlation_id)


    async def handle_face_detection_result(self, message: dict[str, Any], correlation_id: str):
        """只寫進 Redis state，不參與完成條件；主流程可能已經結束（state 已是 complete）。"""
        payload = message.get("payload", {})
        if message.get("message_type") == "ERROR" or payload.get("status") in ("error", "failed"):
            face_detection = {
                "status": "failed",
                "error": payload.get("error") or payload.get("metadata", {}).get("error_message"),
            }
            logger.warning(f"Face detection failed for {correlation_id} (non-fatal): {face_detection['error']}")
        else:
            results = payload.get("results", {})
            # GD-11：沒有 face_detection 同意時人臉服務不偵測，回 {"skipped": 原因}
            face_detection = {"status": "skipped" if results.get("skipped") else "completed", "results": results}
            logger.info(f"Face detection result received for {correlation_id}: {face_detection['results']}")

        if not self.redis_manager.update_state(correlation_id, {"face_detection": face_detection}):
            logger.warning(f"Failed to record face detection result for {correlation_id}")

    async def check_analysis_completion(self, producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """檢查分析是否完成，如果完成則發送摘要請求"""
        if (state["video_analysis_result"] is not None and 
            state["audio_analysis_result"] is not None and
            state["step"] == "parallel_analysis"):
            
            try:
                if not self.redis_manager.update_state(correlation_id, {"step": "summary"}):
                    raise Exception("Failed to update step in Redis")
                
                # 提取必要資訊
                original_message = state["original_message"]
                original_params = original_message["payload"]["parameters"]

                
                # 從分析結果中提取數據路徑
                video_result = state["video_analysis_result"]
                audio_result = state["audio_analysis_result"]
                
                video_merged_path = None
                audio_merged_path = None

                # 從視頻分析結果提取 merged_analysis 路徑
                video_results = video_result.get("metadata", {}).get("results", {})
                if "merged_analysis" in video_results:
                    video_merged_path = video_results["merged_analysis"].get("merged_file_path")
                
                # 從音頻分析結果提取 merged_analysis 路徑（無音軌影片允許為 None）
                audio_results = audio_result.get("metadata", {}).get("results", {}) or audio_result.get("results", {})
                no_audio = audio_results.get("no_audio", False)
                if "merged_analysis" in audio_results:
                    audio_merged_path = audio_results["merged_analysis"].get("merged_file_path")

                # video_merged_path 必須，audio_merged_path 無音軌時允許為 None
                if not video_merged_path or (not audio_merged_path and not no_audio):
                    logger.error(f"Missing merged analysis paths - Video: {video_merged_path}, Audio: {audio_merged_path}")
                    await self.handle_processing_error(
                        producer, state,
                        "Missing merged analysis file paths",
                        "MISSING_MERGED_PATHS"
                    )
                    return

                if no_audio:
                    logger.info(f"No audio track in video, proceeding with video-only summary for: {correlation_id}")

                # 創建影片摘要請求，只傳送 merged_analysis 路徑
                summary_request = MessageBuilder.create_processing_request(
                    original_message=original_message,
                    target_service="video_summary_service",
                    action="video_summary",
                    parameters={
                        "filename": original_params.get("primary_filename"),
                        "asset_path": original_params.get("asset_path"),
                        "version_id": original_params.get("version_id"),
                        "status": original_params.get("status"),
                        "video_file_path": state["temp_file_path"],
                        # 只傳送 merged_analysis 的檔案路徑
                        "video_analysis_results": {
                            "merged_analysis": {
                                "merged_file_path": video_merged_path
                            }
                        },
                        "audio_analysis_results": {
                            "merged_analysis": {
                                "merged_file_path": audio_merged_path
                            }
                        }
                    },
                    temp_file_path=state["temp_file_path"]
                )
                
                await producer.send(KAFKA_TOPIC_SUMMARY_REQUEST, summary_request)
                logger.info(f"Summary request sent with merged analysis paths - Video: {video_merged_path}, Audio: {audio_merged_path} for: {correlation_id}")
                
            except Exception as e:  # noqa: BLE001 -- spans message building and Kafka produce; already logged + processing error reported
                logger.error(f"Error in analysis completion processing: {e}")
                await self.handle_processing_error(producer, state, str(e), "ANALYSIS_COMPLETION_FAILED")

    async def handle_summary_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理摘要結果 — 送出 contextualize request (Contextual RAG)"""
        updates = {
            "summary_result": message["payload"],
            "step": "contextualize",
        }
        if not self.redis_manager.update_state(correlation_id, updates):
            raise Exception("Failed to update state in Redis")
        state = self.redis_manager.get_state(correlation_id)

        summary_result = state["summary_result"]
        results = summary_result.get("results", {})
        parameters = results.get("parameters", {})
        summary_result_path = parameters.get("summary_result_path")

        if not summary_result_path:
            logger.error(f"No summary_result_path found for: {state['original_message']['payload']['request_id']}")
            await self.handle_processing_error(producer, state, "Summary result path not found", "MISSING_SUMMARY_PATH")
            return

        logger.info(f"Summary result path: {summary_result_path}")

        # Pre-read summary text (indexer deletes the file after indexing)
        _summary_pre = ""
        try:
            _result_data_pre = await asyncio.to_thread(_load_json_file_sync, summary_result_path)
            for _item in _result_data_pre:
                _p = _item.get("payload", {})
                if "summary" in _p:
                    _summary_pre = _p["summary"]
                    break
        except (OSError, json.JSONDecodeError) as _read_err:
            logger.warning(f"Could not pre-read summary file: {_read_err}")

        # Extract video/audio merged analysis paths
        video_analysis_path = None
        audio_analysis_path = None
        try:
            video_result = state.get("video_analysis_result", {})
            video_results = video_result.get("metadata", {}).get("results", {})
            video_analysis_path = video_results.get("merged_analysis", {}).get("merged_file_path")
        except AttributeError as e:
            logger.debug(f"Could not extract video_analysis_path: {e}")
        try:
            audio_result = state.get("audio_analysis_result", {})
            audio_results = audio_result.get("metadata", {}).get("results", {})
            audio_analysis_path = audio_results.get("merged_analysis", {}).get("merged_file_path")
        except AttributeError as e:
            logger.debug(f"Could not extract audio_analysis_path: {e}")

        logger.info(f"[DEBUG] video_analysis_path={video_analysis_path}, audio_analysis_path={audio_analysis_path}")

        # Persist in Redis for handle_contextualize_result fallback
        self.redis_manager.update_state(correlation_id, {
            "summary": _summary_pre,
            "summary_result_path": summary_result_path,
            "video_analysis_path": video_analysis_path,
            "audio_analysis_path": audio_analysis_path,
        })

        # Read global summary text
        global_summary = self._read_global_summary(summary_result_path)

        original_message = state["original_message"]
        original_params = original_message["payload"]["parameters"]
        primary_filename = original_params.get("primary_filename", "")
        # Same field, same fallback as image_orchestrator_service's own moment
        # text entries (message_utils.py) -- the real upload time, not
        # "whenever this processing step happened to run".
        upload_time = original_message["payload"].get("metadata", {}).get(
            "upload_timestamp", datetime.now(timezone.utc).isoformat())

        # Build per-moment text for the contextualize service
        chunking_moments = state.get("chunking_moments", [])
        moments = self._assemble_moments(chunking_moments, video_analysis_path, audio_analysis_path,
                                          primary_filename, upload_time)
        # P13: when the video was recorded, as its container says -- Module 25 dates the
        # recording's "yesterday" / "last week" from it (after validating it).
        capture_time, capture_time_source = await asyncio.to_thread(read_capture_time, state["temp_file_path"])
        if capture_time:
            for moment in moments:
                moment["capture_time"] = capture_time
                moment["capture_time_source"] = capture_time_source

        contextualize_request = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="video_contextualize_service",
            action="video_contextualize",
            parameters={
                "video_id": primary_filename.rsplit(".", 1)[0] if "." in primary_filename else primary_filename,
                "global_summary": global_summary,
                "moments": moments,
                "filename": primary_filename,
                "asset_path": original_params.get("asset_path"),
                "version_id": original_params.get("version_id"),
                "status": original_params.get("status"),
                "summary_result_path": summary_result_path,
                "video_analysis_path": video_analysis_path,
                "audio_analysis_path": audio_analysis_path,
            },
            temp_file_path=state["temp_file_path"],
        )

        await producer.send(KAFKA_TOPIC_CONTEXTUALIZE_REQUEST, contextualize_request)
        logger.info(f"Contextualize request sent with {len(moments)} moments for: {correlation_id}")

    async def handle_contextualize_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """收到 contextualized moments 後，轉送至 video_indexer_service"""
        updates = {"step": "save", "contextualize_result": message["payload"]}
        self.redis_manager.update_state(correlation_id, updates)
        state = self.redis_manager.get_state(correlation_id)

        payload = message.get("payload", {})
        results = payload.get("results", {})
        result_params = results.get("parameters", {})

        contextualized_moments = result_params.get("moments", [])

        # Prefer passthrough fields from contextualize result; fall back to Redis state
        summary_result_path = result_params.get("summary_result_path") or state.get("summary_result_path")
        video_analysis_path  = result_params.get("video_analysis_path")  or state.get("video_analysis_path")
        audio_analysis_path  = result_params.get("audio_analysis_path")  or state.get("audio_analysis_path")

        logger.info(f"[DEBUG] handle_contextualize_result: video_analysis_path={video_analysis_path}")

        original_message = state["original_message"]
        original_params  = original_message["payload"]["parameters"]

        save_request = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="video_indexer_service",
            action="document_indexing",
            parameters={
                "summary_result_path":    summary_result_path,
                "filename":               original_params.get("primary_filename"),
                "asset_path":             original_params.get("asset_path"),
                "version_id":             original_params.get("version_id"),
                "file_type":              original_params.get("file_type", "video"),
                "chunking_moments":       state.get("chunking_moments", []),
                "video_analysis_path":    video_analysis_path,
                "audio_analysis_path":    audio_analysis_path,
                "contextualized_moments": contextualized_moments,
                "moderation_flags":       state.get("moderation_flags"),   # CM-3
            },
            temp_file_path=state["temp_file_path"],
        )

        await producer.send(KAFKA_TOPIC_INDEXER_REQUEST, save_request)
        logger.info(
            f"Indexer request sent with {len(contextualized_moments)} contextualized moments for: {correlation_id}"
        )

    def _read_global_summary(self, summary_result_path: str) -> str:
        """Read the global summary text written by video_summary_service."""
        try:
            with open(summary_result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if item.get("payload", {}).get("embedding_type") == "summary":
                    return item["payload"].get("summary", "")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read global summary from {summary_result_path}: {e}")
        return ""

    def _assemble_moments(
        self,
        chunking_moments: list[dict[str, Any]],
        video_analysis_path: str | None,
        audio_analysis_path: str | None,
        primary_filename: str = "",
        upload_time: str = "",
    ) -> list[dict[str, Any]]:
        """Build per-moment dicts with ocr_text / asr_text / lvlm_desc for the contextualize service.

        filename/source/speaker/upload_time are stamped here too (not just
        ocr/asr/lvlm) — personal_index_publisher.py forwards these moments to
        Module 25 unchanged, and its _map_entry() already reads
        ep.get("filename")/ep.get("source")/ep.get("speaker")/
        ep.get("upload_time"); without this they silently stay null for every
        video moment, unlike every other media type.

        ASR/speaker extraction mirrors video_indexer_service's own
        _extract_asr_for_moment() (the Qdrant/OpenSearch-facing sibling of
        this method) exactly, word-level timestamps first: a spoken sentence
        can straddle a moment boundary (moments are a fixed 10s grid —
        video_chunking_service.config.MOMENT_DURATION_SEC — but WhisperX
        segments on natural speech pauses, not that grid), so segment-center
        matching alone can drop the whole sentence, speaker included, on one
        side of the boundary. Word-level timestamps split it correctly;
        segments without word-level data (older/partial ASR output) fall back
        to the same center-point check this method already used. speaker
        resolution: single speaker → that speaker, several → comma-joined,
        none → "Unknown".
        """
        source = primary_filename.rsplit(".", 1)[-1].lower() if "." in primary_filename else "unknown"
        video_frames: list[dict] = []
        moment_desc_map: dict[int, str] = {}
        if video_analysis_path and os.path.exists(video_analysis_path):
            try:
                with open(video_analysis_path, "r", encoding="utf-8") as f:
                    va = json.load(f)
                video_frames = va.get("frames", [])
                for md in va.get("moment_descriptions", []):
                    moment_desc_map[md.get("moment_index")] = md.get("lvlm_description", "")
                logger.info(f"Loaded {len(moment_desc_map)} VLM descriptions for contextualization")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load video analysis for contextualization: {e}")

        audio_chunks: list[dict] = []
        if audio_analysis_path and os.path.exists(audio_analysis_path):
            try:
                with open(audio_analysis_path, "r", encoding="utf-8") as f:
                    audio_chunks = json.load(f)
                if not isinstance(audio_chunks, list):
                    audio_chunks = []
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load audio analysis for contextualization: {e}")

        moments = []
        for moment in chunking_moments:
            start = moment.get("start_time", 0.0)
            end   = moment.get("end_time", 0.0)
            idx   = moment.get("moment_index", 0)

            moment_frames = sorted(
                [f for f in video_frames
                 if isinstance(f, dict) and start <= f.get("timestamp", -1) < end and f.get("text")],
                key=lambda f: f.get("timestamp", 0),
            )
            ocr_text = ", ".join(f"frame{i+1}:{{{f['text']}}}" for i, f in enumerate(moment_frames))

            word_tokens = []     # word-level matches: already carry their own spacing
            segment_texts = []   # whole-segment fallback matches: need an explicit separator
            speakers_in_moment = []
            for seg in audio_chunks:
                if not isinstance(seg, dict):
                    continue
                seg_speaker = seg.get("speaker")
                words = seg.get("words", [])
                if words:
                    for w in words:
                        word_start = float(w.get("start", -1))
                        if start <= word_start < end:
                            word_tokens.append(w.get("word", ""))
                            if seg_speaker:
                                speakers_in_moment.append(seg_speaker)
                else:
                    seg_start = float(seg.get("start_time", 0))
                    seg_end   = float(seg.get("end_time", 0))
                    if start <= (seg_start + seg_end) / 2 < end and seg.get("text"):
                        segment_texts.append(seg["text"])
                        if seg_speaker:
                            speakers_in_moment.append(seg_speaker)
            # Word-level tokens need _join_word_tokens(), not a blanket
            # " ".join() or "".join() -- see its docstring: WhisperX carries
            # no leading space per word (confirmed empirically: an earlier
            # "".join() here produced unreadable glued-together English, e.g.
            # "Youunderstandthat?"), but it also tokenizes CJK one character
            # at a time, so a blanket " ".join() instead turned "美國媒體"
            # into "美 國 媒 體". Whole-segment fallback texts are already
            # complete sentences, not individual characters, so a plain
            # " ".join() between them is correct regardless of language.
            asr_text = " ".join(p for p in (_join_word_tokens(word_tokens), " ".join(segment_texts)) if p)
            distinct_speakers = list(dict.fromkeys(speakers_in_moment))
            speaker = (distinct_speakers[0] if len(distinct_speakers) == 1
                       else ", ".join(distinct_speakers) if distinct_speakers else "Unknown")

            moments.append({
                "moment_index": idx,
                "start_time":   start,
                "end_time":     end,
                "ocr_text":     ocr_text,
                "asr_text":     asr_text,
                "lvlm_desc":    moment_desc_map.get(idx, ""),
                "filename":     primary_filename,
                "source":       source,
                "speaker":      speaker,
                "upload_time":  upload_time,
            })

        logger.info(f"Assembled {len(moments)} moments for contextualization")
        return moments

            
    async def handle_save_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理保存結果 -- this module's own job is done here, but Module 25 has
        only just been asked to index, not confirmed to have finished -- step
        stays "indexing" and update_asset_status(active) waits for
        handle_index_completed() (VIE01-255 update_status() race fix)."""
        try:
            save_result = message["payload"]

            # summary was pre-read before indexer deleted the file
            summary = state.get("summary", "")

            # text_entries = actual embedding_type:text entries the indexer built and sent to hybrid search
            text_entries = save_result.get("results", {}).get("text_entries", [])

            # 創建最終結果消息
            final_result = create_final_result_message(
                original_message=state["original_message"],
                video_analysis_result=state.get("video_analysis_result"),
                audio_analysis_result=state.get("audio_analysis_result"),
                summary_result=state.get("summary_result"),
                save_result=save_result
            )

            # 上傳 keyframe 到 Asset Management（非阻斷，失敗不影響流程）
            original_payload = state["original_message"]["payload"]
            original_params = original_payload.get("parameters", {})
            user_id = original_payload.get("user_id", "")
            branch_id = state.get("branch_id", "")
            asset_path = original_params.get("asset_path", "")
            version_id = original_params.get("version_id", "")
            all_keyframes = [
                kf for m in state.get("chunking_moments", [])
                for kf in m.get("keyframe_paths", [])
            ]
            if user_id and branch_id and asset_path and version_id and all_keyframes:
                await self.seaweedfs_client.register_keyframes(
                    user_id, branch_id, asset_path, version_id, all_keyframes
                )

            # 發送最終結果
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, final_result)
            logger.info(f"Final result sent for: {correlation_id}")

            # 清理資源
            if state.get("temp_file_path"):
                self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])
                logger.info(f"Cleaned up temp file: {state['temp_file_path']}")

            indexing_state = {
                "step": "indexing",
                "summary": summary,
                "text": text_entries,
                "branch_id": state.get("branch_id", ""),
                # set_state() below replaces the whole stored value -- original_message
                # must survive the trim: handle_index_completed() needs it to flip
                # status later, and handle_processing_error() needs it on any
                # failure path from here on (including from handle_index_completed()).
                "original_message": state["original_message"],
            }
            self._keep_face_result(correlation_id, indexing_state)
            if not self.redis_manager.set_state(correlation_id, indexing_state):
                raise Exception("Failed to update state in Redis")

            logger.info(f"Video processing saved (awaiting Module 25 indexing) with {len(text_entries)} text entries for: {correlation_id}")

        except Exception as e:  # noqa: BLE001 -- spans message building, Redis (incl. its own self-raised Exception), and Kafka produce; already logged + processing error reported
            logger.error(f"Error in handle_save_result: {e}")
            await self.handle_processing_error(
                producer, state,
                f"Failed to process save result: {e!s}",
                "SAVE_RESULT_PROCESSING_ERROR"
            )

    def _keep_face_result(self, correlation_id: str, new_state: dict[str, Any]) -> None:
        """人臉偵測不在完成條件裡，結果可能在任何時候回來（v1.4 §4.3）。
        set_state() 是整份覆寫：寫入前重新讀一次 state，把已經回來的人臉結果帶進 new_state，避免被洗掉。"""
        latest = self.redis_manager.get_state(correlation_id) or {}
        if "face_detection" in latest:
            new_state["face_detection"] = latest["face_detection"]

    async def handle_index_completed(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """Module 25 has actually finished indexing this asset (personal-index-completed) --
        only now is it safe to mark the job complete and flip the asset active
        (VIE01-255 update_status() race fix: this used to fire right after
        handle_save_result() merely enqueued the index-request)."""
        try:
            original_payload = state["original_message"]["payload"]
            original_params = original_payload.get("parameters", {})
            user_id = original_payload.get("user_id", "")
            branch_id = state.get("branch_id", "")
            asset_path = original_params.get("asset_path", "")
            version_id = original_params.get("version_id", "")

            final_state = {
                "step": "complete",
                "summary": state.get("summary", ""),
                "text": state.get("text", []),
                "branch_id": branch_id,
            }
            self._keep_face_result(correlation_id, final_state)
            if not self.redis_manager.set_state(correlation_id, final_state):
                raise Exception("Failed to update state in Redis")

            if user_id and branch_id and asset_path and version_id:
                await self.seaweedfs_client.update_asset_status(user_id, branch_id, asset_path, version_id, "active")

            logger.info(f"Video processing completed for: {correlation_id}")

        except Exception as e:  # noqa: BLE001 -- spans Redis (incl. its own self-raised Exception) and the asset-status call; already logged + processing error reported
            logger.error(f"Error in handle_index_completed: {e}")
            await self.handle_processing_error(
                producer, state,
                f"Failed to process index completion: {e!s}",
                "INDEX_COMPLETED_ERROR"
            )

    async def handle_processing_error(self, producer: AIOKafkaProducer, state: dict[str, Any], error_message: str, error_code: str):
        """處理錯誤並清理資源"""
        correlation_id = state["original_message"].get("correlation_id")

        try:
            self.redis_manager.update_state(correlation_id, {
                "step": "error",
                "error_code": error_code,
                "error_message": error_message,
            })
        except Exception as redis_err:  # noqa: BLE001 -- best-effort error-state marking inside the error handler itself; already logged
            logger.error(f"Failed to mark Redis state as error for {correlation_id}: {redis_err}")

        error_response = MessageBuilder.create_error_response(
            original_message=state["original_message"],
            error_message=error_message,
            error_code=error_code
        )
        await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)

        ids = self._claim_ids_from_message(state["original_message"])
        if ids:
            asset_path, version_id, _cid, user_id, branch_id = ids
            await self.seaweedfs_client.update_asset_status(user_id, branch_id, asset_path, version_id, "failed")

        self.seaweedfs_client.cleanup_temp_file(state.get("temp_file_path"))

        logger.info(f"Processing error handled for: {correlation_id} (code={error_code})")

    def validate_message(self, message: dict[str, Any]) -> bool:
        """驗證消息格式"""
        required_fields = [
            "message_id", "correlation_id", "timestamp",
            "source_service", "target_service", "message_type",
            "priority", "payload", "retry_count", "ttl"
        ]

        if not all(field in message for field in required_fields):
            return False

        payload = message.get("payload", {})
        if payload.get("action") == "video_processing":
            required_payload_fields = ["request_id", "action", "parameters"]
            if not all(field in payload for field in required_payload_fields):
                return False

            parameters = payload.get("parameters", {})
            required_param_fields = ["asset_path", "version_id", "primary_filename"]
            if not all(field in parameters for field in required_param_fields):
                return False

        return True

    def is_message_expired(self, message: dict[str, Any]) -> bool:
        """檢查消息是否過期"""
        try:
            timestamp = datetime.fromisoformat(message["timestamp"].replace('Z', '+00:00'))
            ttl = message.get("ttl", 3600)
            now = datetime.now(timezone.utc)
            return (now - timestamp).total_seconds() > ttl
        except (KeyError, ValueError, AttributeError) as e:
            logger.debug(f"Error checking message expiration, treating as not expired: {e}")
            return False
    
    async def send_error_response(
        self,
        producer: AIOKafkaProducer,
        original_message: dict[str, Any],
        error_message: str,
        error_code: str,
        correlation_id: str | None = None
    ):
        """發送錯誤響應"""
        try:
            error_response = MessageBuilder.create_error_response(
                original_message=original_message,
                error_message=error_message,
                error_code=error_code
            )
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
            logger.warning(f"Error response sent: {error_response['message_id']}")

            if correlation_id:
                payload = original_message.get("payload", {})
                parameters = payload.get("parameters", {})
                branch_id = (
                    parameters.get("branch_id")
                    or payload.get("asset_managemant_download_header", {}).get("X-Branch-ID")
                    or ""
                )
                ids = self._claim_ids_from_message(original_message)
                if ids:
                    asset_path, version_id, _cid, user_id, req_branch_id = ids
                    await self.seaweedfs_client.update_asset_status(user_id, req_branch_id, asset_path, version_id, "failed")

                self.redis_manager.set_state(correlation_id, {
                    "step": "error",
                    "error_code": error_code,
                    "error_message": error_message,
                    "branch_id": branch_id,
                })
        except Exception as e:  # noqa: BLE001 -- last-resort fallback if even sending the error response fails
            logger.error(f"Failed to send error response: {e}")

    async def send_to_dlq(self, producer: AIOKafkaProducer, original_message: dict[str, Any], error: str):
        """發送消息到 DLQ"""
        try:
            dlq_message = MessageBuilder.create_dlq_message(
                original_message=original_message,
                error=error,
                final_retry_count=original_message.get("retry_count", 0)
            )
            await producer.send(KAFKA_TOPIC_DLQ, dlq_message)
            logger.error(f"Message sent to DLQ: {dlq_message['message_id']}")
        except Exception as e:  # noqa: BLE001 -- last-resort fallback if even sending to the DLQ fails
            logger.error(f"Failed to send message to DLQ: {e}")
