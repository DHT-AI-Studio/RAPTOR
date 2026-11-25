# services/image_orchestrator_service/kafka_handler.py

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import image_moderation
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from api_client import ClaimOutcome, SeaweedFSClient
from capture_time import read_capture_time
from config import (
    ASSET_MANAGEMENT_URL,
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_DESCRIPTION_REQUEST,
    KAFKA_TOPIC_DESCRIPTION_RESULT,
    KAFKA_TOPIC_DLQ,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_INDEX_COMPLETED,
    KAFKA_TOPIC_INDEXER_REQUEST,
    KAFKA_TOPIC_INDEXER_RESULT,
    KAFKA_TOPIC_OCR_REQUEST,
    KAFKA_TOPIC_OCR_RESULT,
    KAFKA_TOPIC_REQUEST,
    SERVICE_NAME,
)
from dotenv import load_dotenv
from message_utils import ImageResultMerger, MessageBuilder, create_final_result_message
from redis_manager import RedisStateManager

# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
logger = logging.getLogger(__name__)

class ImageOrchestratorKafkaHandler:
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
            enable_auto_commit=False,  # VIE01-255: manual commit drives claim-based ack/redeliver
        )
        
        result_consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_DESCRIPTION_RESULT,
            KAFKA_TOPIC_OCR_RESULT,
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
        
        logger.info("Image Orchestrator Kafka consumers started...")
        
        try:
            # 並行處理兩個 consumer 和清理任務
            await asyncio.gather(
                self.process_requests(request_consumer, producer),
                self.process_results(result_consumer, producer),
                # self.periodic_cleanup()  # 添加定期清理任務
            )
        finally:
            await request_consumer.stop()
            await result_consumer.stop()
            await producer.stop()
            self.redis_manager.close()
    
    # async def periodic_cleanup(self):
    #     """定期清理過期狀態"""
    #     while True:
    #         try:
    #             await asyncio.sleep(300)  # 每5分鐘執行一次
    #             await self.cleanup_expired_states()
    #         except Exception as e:
    #             logger.error(f"Error in periodic cleanup: {e}")
        
    
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
        """處理圖片處理請求"""
        async for message in consumer:
            ids = self._claim_ids_from_message(message.value)
            if ids is None:
                await self.handle_image_request(message.value, producer)
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

            await self.handle_image_request(message.value, producer)
            await consumer.commit()
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_image_request(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理圖片處理請求"""
        try:
            logger.info(f"Processing image request: {message.get('message_id')}")
            
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

            # 處理圖片處理請求
            if message["payload"]["action"] == "image_processing":
                await self.handle_image_processing(message, producer)
            else:
                await self.send_error_response(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION",
                    message.get("correlation_id")
                )
                
        except Exception as e:  # noqa: BLE001 -- dispatch spans message dict access, HTTP download, Kafka and Redis failures, already logged + reported via send_error_response
            logger.error(f"Error processing image request: {e}")
            # Must go through send_error_response (result topic), not
            # send_to_dlq (KAFKA_TOPIC_DLQ) — nothing consumes the DLQ topic,
            # so a failure sent there vanishes silently instead of reaching
            # whatever is waiting for this job's result.
            await self.send_error_response(
                producer, message, str(e), "DISPATCH_ERROR", message.get("correlation_id")
            )
    
    async def handle_image_processing(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理圖片處理流程"""
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
                logger.info(f"Downloading file via asset management API: {download_url}")
                temp_file_path = await self.seaweedfs_client.download_file(
                    download_url=download_url,
                    download_headers=download_headers,
                    download_params=download_params,
                    filename=primary_filename
                )
            

            correlation_id = message["correlation_id"]
            _dl_headers = payload.get("asset_managemant_download_header", {})
            moderate = image_moderation.enabled()
            state = {
                "original_message": message,
                "temp_file_path": temp_file_path,
                "description_result": None,
                "ocr_result": None,
                "save_result": None,
                "step": "parallel_processing",  # parallel_processing -> merge -> save -> complete
                "branch_id": parameters.get("branch_id") or _dl_headers.get("X-Branch-ID") or payload.get("branch_id") or "",
                "created_at": asyncio.get_event_loop().time(),
                # CM-2: merging also waits for moderation when it is enabled.
                "moderation_done": not moderate,
                "moderation_flags": None,
            }
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")

            # 並行發送描述和OCR請求
            await self.send_parallel_requests(message, producer, temp_file_path, parameters)

            # CM-2: moderated while description and OCR run in their own services.
            # Awaited, not a background task, so the request isn't committed until
            # the verdict is in: a worker that dies meanwhile gets the message
            # redelivered instead of leaving the job waiting for a lost verdict.
            if moderate:
                await self.run_moderation(producer, correlation_id, temp_file_path, primary_filename)

        except Exception as e:  # noqa: BLE001 -- spans dict access, SeaweedFS download, Redis state and parallel Kafka sends; already logged + reported
            logger.error(f"Error in image processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Image processing failed: {e!s}",
                "PROCESSING_FAILED",
                message.get("correlation_id")
            )
    
    async def send_parallel_requests(self, message: dict[str, Any], producer: AIOKafkaProducer, temp_file_path: str, parameters: dict[str, Any]):
        """並行發送圖片描述和OCR請求"""
        # 發送圖片描述請求
        description_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="image_description_service",
            action="image_description",
            parameters={
                "file_type": parameters.get("file_type", "image"),
                "primary_filename": parameters.get("primary_filename")
            },
            temp_file_path=temp_file_path
        )
        
        # 發送OCR請求
        ocr_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="image_ocr_service",
            action="image_ocr",
            parameters={
                "file_type": parameters.get("file_type", "image"),
                "primary_filename": parameters.get("primary_filename")
            },
            temp_file_path=temp_file_path
        )
        
        # 並行發送兩個請求
        await asyncio.gather(
            producer.send(KAFKA_TOPIC_DESCRIPTION_REQUEST, description_request),
            producer.send(KAFKA_TOPIC_OCR_REQUEST, ocr_request)
        )
        
        correlation_id = message["correlation_id"]
        logger.info(f"Parallel requests sent for correlation_id: {correlation_id}")
        logger.info(f"Description request ID: {description_request['payload']['request_id']}")
        logger.info(f"OCR request ID: {ocr_request['payload']['request_id']}")

    async def run_moderation(self, producer: AIOKafkaProducer, correlation_id: str, image_path: str, filename: str | None):
        """CM-2: moderate the image, keep the flags in the job's state for the merge
        (Module 25) and write them to Module 04. Never raises: the request was
        already handed to the description and OCR services."""
        try:
            try:
                flags = await image_moderation.moderate_image(image_path, filename)
            except image_moderation.ModerationUnavailable as exc:
                if image_moderation.fail_closed():
                    state = self.redis_manager.get_state(correlation_id)
                    if state and state.get("step") == "parallel_processing":
                        await self.handle_processing_error(
                            producer, state, f"Image moderation unavailable: {exc}", "MODERATION_UNAVAILABLE")
                    return
                flags = exc.flags

            state = self.redis_manager.get_state(correlation_id)
            if not state or state.get("step") != "parallel_processing":
                logger.info(f"[moderation] job {correlation_id} already left parallel processing -- verdict not used")
                return
            # No await between storing the flags and the merge check: whichever of
            # description, OCR and moderation finishes last is the one that merges.
            self.redis_manager.update_state(correlation_id, {"moderation_flags": flags, "moderation_done": True})
            await self.check_and_merge_results(producer, correlation_id)

            ids = self._claim_ids_from_message(state["original_message"])
            if ids:
                asset_path, version_id, _cid, user_id, branch_id = ids
                await self.seaweedfs_client.update_moderation(user_id, branch_id, asset_path, version_id, flags)
                try:
                    await image_moderation.report_flags(flags, asset_path, version_id, branch_id)
                except Exception as exc:  # noqa: BLE001 -- CM-5 reporting must never affect CM-2's own moderation result
                    logger.warning(f"[moderation-flag] report_flags failed for {asset_path}: {exc}")
        except Exception as e:
            logger.error(f"Error in image moderation for {correlation_id}: {e}", exc_info=True)
            state = self.redis_manager.get_state(correlation_id)
            if state and state.get("original_message") and state.get("step") != "error":
                await self.handle_processing_error(producer, state, f"Image moderation failed: {e!s}", "MODERATION_FAILED")

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

            # 檢查結果狀態 (also catches "failed", not just "error")
            message_type = message.get("message_type", "")
            service_status = payload.get("status", "")
            if message_type == "ERROR" or service_status in ("error", "failed"):
                error_message = payload.get('error', {}).get('message')
                logger.error(f"Service error for correlation_id {correlation_id}: {payload.get('error')}")
                await self.handle_processing_error(producer, state, f"Service processing failed: {error_message}", "SERVICE_ERROR")
                return
            
            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_DESCRIPTION_RESULT:
                await self.handle_description_result(message, producer, correlation_id)
            elif topic == KAFKA_TOPIC_OCR_RESULT:
                await self.handle_ocr_result(message, producer, correlation_id)
            elif topic == KAFKA_TOPIC_INDEXER_RESULT:
                await self.handle_save_result(message, producer, correlation_id)
            elif topic == KAFKA_TOPIC_INDEX_COMPLETED:
                await self.handle_index_completed(message, producer, state, correlation_id)

        except Exception as e:  # noqa: BLE001 -- delegate handlers span dict access, Redis, file IO and Kafka; already logged + reported below
            logger.error(f"Error handling service result: {e}")
            # An exception here used to leave the asset stuck in "processing"
            # forever -- no failure ever got reported to Module 04, unlike
            # the sibling branch above (a downstream service explicitly
            # reporting "error"/"failed") which does call
            # handle_processing_error(). state.get("original_message") guards
            # the one case handle_processing_error() itself can't recover
            # from -- it needs that field too.
            if state and state.get("original_message"):
                try:
                    await self.handle_processing_error(producer, state, f"Unexpected error: {e}", "UNEXPECTED_ERROR")
                except Exception as inner_e:  # noqa: BLE001 -- last-resort reporting path itself spans Redis/Kafka/HTTP, already logged
                    logger.error(f"Also failed to report processing error: {inner_e}")
            else:
                logger.error(f"Cannot report failure for {correlation_id} -- state missing or has no original_message")

    async def handle_description_result(self, message: dict[str, Any], producer: AIOKafkaProducer, correlation_id: str):
        """處理圖片描述結果"""
        self.redis_manager.update_state(correlation_id, {
            "description_result": message["payload"]
        })
        logger.info(f"Description result received for correlation_id: {correlation_id}")
        
        # 檢查是否兩個結果都已收到
        await self.check_and_merge_results(producer, correlation_id)
    
    async def handle_ocr_result(self, message: dict[str, Any], producer: AIOKafkaProducer, correlation_id: str):
        """處理OCR結果"""
        self.redis_manager.update_state(correlation_id, {
            "ocr_result": message["payload"]
        })
        logger.info(f"OCR result received for correlation_id: {correlation_id}")
        
        # 檢查是否兩個結果都已收到
        await self.check_and_merge_results(producer, correlation_id)
    
    @staticmethod
    def _load_json_file_sync(file_path: str) -> Any:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def check_and_merge_results(self, producer: AIOKafkaProducer, correlation_id: str):
        """檢查並合併結果"""
        state = self.redis_manager.get_state(correlation_id)
        if not state:
            logger.error(f"State not found for correlation_id: {correlation_id}")
            return    
        # CM-2: three results join here now (description, OCR, moderation), so only
        # a job still in parallel processing merges -- never a second time, never
        # one that already failed. A state written before CM-2 has no
        # moderation_done and doesn't wait for it.
        if (state["description_result"] and state["ocr_result"] and state.get("moderation_done", True)
                and state.get("step") == "parallel_processing"):
            self.redis_manager.update_state(correlation_id, {"step": "merge"})
            try:
                # 合併結果
                # P13: when the photo was taken, as its EXIF says -- Module 25 records the photo as an
                # observation event at that time and dates deictic phrases in its text from it.
                capture = await asyncio.to_thread(read_capture_time, state.get("temp_file_path") or "")
                merged_file_path = ImageResultMerger.merge_results(
                    description_result=state["description_result"],
                    ocr_result=state["ocr_result"],
                    original_message=state["original_message"],
                    capture=capture,
                    moderation_flags=state.get("moderation_flags"),
                )
                
                # 趁合併檔案還存在，萃取 summary/text 存入 Redis
                summary = None
                text = None
                try:
                    merged_data = await asyncio.to_thread(self._load_json_file_sync, merged_file_path)
                    for item in merged_data:
                        item_payload = item.get("payload", {})
                        if item_payload.get("embedding_type") == "summary":
                            summary = item_payload.get("summary")
                        elif item_payload.get("embedding_type") == "text":
                            text = item_payload.get("text")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as read_err:
                    logger.warning(f"Could not pre-read merged file: {read_err}")

                self.redis_manager.update_state(correlation_id, {
                    "step": "save",
                    "merged_file_path": merged_file_path,
                    "summary": summary,
                    "text": text
                })

                
                # 發送保存請求
                save_request = MessageBuilder.create_processing_request(
                    original_message=state["original_message"],
                    target_service="image_indexer_service",
                    action="document_indexing",
                    parameters={
                        "merged_result_path": merged_file_path,
                        "filename": state["original_message"]["payload"]["parameters"].get("primary_filename"),
                    },
                    temp_file_path=state["temp_file_path"]
                )
                
                await producer.send(KAFKA_TOPIC_INDEXER_REQUEST, save_request)
                logger.info(f"Save request sent for correlation_id: {correlation_id}")
                logger.info(f"Save request ID: {save_request['payload']['request_id']}")
                logger.info(f"Merged file path: {merged_file_path}")
                

                
            except Exception as e:  # noqa: BLE001 -- spans merge_results' re-raised ValueError (wrapping file/dict errors), dict building and Kafka send; already logged + reported
                logger.error(f"Error merging results: {e}")
                await self.handle_processing_error(producer, state, f"Result merging failed: {e!s}", "MERGE_FAILED")
    
    async def handle_save_result(self, message: dict[str, Any], producer: AIOKafkaProducer, correlation_id: str):
        """處理保存結果 -- this module's own job is done here, but Module 25 has
        only just been asked to index, not confirmed to have finished -- step
        stays "indexing" and update_asset_status(active) waits for
        handle_index_completed() (VIE01-255 update_status() race fix)."""
        try:
            # 先獲取當前狀態
            state = self.redis_manager.get_state(correlation_id)
            if not state:
                logger.error(f"State not found for correlation_id: {correlation_id}")
                return
             # 創建最終結果消息
            final_result = create_final_result_message(
                original_message=state["original_message"],
                description_result=state["description_result"],
                ocr_result=state["ocr_result"],
                save_result=message["payload"]
            )
            # 發送到 Kafka
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, final_result)
            logger.info(f"Final result sent for correlation_id: {correlation_id}")

            # summary/text 已在 check_and_merge_results 存入 Redis state
            summary = state.get("summary")
            text = state.get("text")

            indexing_state = {
                "step": "indexing",
                "summary": summary,
                "text": text,
                "branch_id": state.get("branch_id", ""),
                # set_state() below replaces the whole stored value -- original_message
                # must survive the trim: handle_index_completed() needs it to flip
                # status later, and handle_processing_error() needs it on any
                # failure path from here on (including from handle_index_completed()).
                "original_message": state["original_message"],
            }
            self.redis_manager.set_state(correlation_id, indexing_state)
            logger.info(f"Redis updated to indexing state for {correlation_id}")
            # 清理臨時檔案
            self.cleanup_request_state(correlation_id)

        except Exception as e:
            logger.error(f"Error in handle_save_result: {e}", exc_info=True)
            state = self.redis_manager.get_state(correlation_id)
            if state:
                await self.handle_processing_error(producer, state, f"Failed to process save result: {e!s}", "SAVE_RESULT_ERROR")

    async def handle_index_completed(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """Module 25 has actually finished indexing this asset (personal-index-completed) --
        only now is it safe to mark the job complete and flip the asset active
        (VIE01-255 update_status() race fix: this used to fire right after
        handle_save_result() merely enqueued the index-request)."""
        try:
            final_state = {
                "step": "complete",
                "summary": state.get("summary"),
                "text": state.get("text"),
                "branch_id": state.get("branch_id", "")
            }
            self.redis_manager.set_state(correlation_id, final_state)
            logger.info(f"Redis updated to final state for {correlation_id}")

            ids = self._claim_ids_from_message(state["original_message"])
            if ids:
                asset_path, version_id, _cid, user_id, branch_id = ids
                await self.seaweedfs_client.update_asset_status(user_id, branch_id, asset_path, version_id, "active")

        except Exception as e:
            logger.error(f"Error in handle_index_completed: {e}", exc_info=True)
            await self.handle_processing_error(producer, state, f"Failed to process index completion: {e!s}", "INDEX_COMPLETED_ERROR")


    
    def cleanup_request_state(self, correlation_id: str):
        """清理請求狀態和相關資源"""
        state = self.redis_manager.get_state(correlation_id)
            
        if state:
            # 清理臨時檔案
            if "temp_file_path" in state:
                self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])
            
            # self.redis_manager.delete_state(correlation_id)
            logger.info(f"Cleaned up processing state for correlation_id: {correlation_id}")

    async def handle_processing_error(self, producer: AIOKafkaProducer, state: dict[str, Any], error_message: str, error_code: str):
        """處理錯誤並清理資源"""
        correlation_id = state["original_message"].get("correlation_id")

        try:
            self.redis_manager.update_state(correlation_id, {
                "step": "error",
                "error_code": error_code,
                "error_message": error_message,
            })
        except Exception as redis_err:  # noqa: BLE001 -- update_state() already fail-safe internally; kept as defensive backstop
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
        if payload.get("action") == "image_processing":
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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
        except Exception as e:  # noqa: BLE001 -- spans dict building, Kafka send, HTTP status update and Redis; already logged
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
        except (KeyError, KafkaError) as e:
            logger.error(f"Failed to send message to DLQ: {e}")
    
    # async def cleanup_expired_states(self):
    #     """清理過期的處理狀態（可選的背景任務）"""
    #     current_time = asyncio.get_event_loop().time()
    #     expired_correlations = []
        
    #     for correlation_id, state in self.processing_states.items():
    #         if current_time - state["created_at"] > 1800:  # 30分鐘超時
    #             expired_correlations.append(correlation_id)
        
    #     for correlation_id in expired_correlations:
    #         logger.warning(f"Cleaning up expired processing state: {correlation_id}")
    #         self.cleanup_request_state(correlation_id)
