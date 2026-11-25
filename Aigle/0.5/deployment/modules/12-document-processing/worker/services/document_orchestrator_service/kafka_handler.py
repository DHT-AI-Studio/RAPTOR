# services/document_orchestrator_service/kafka_handler.py

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from api_client import ClaimOutcome, SeaweedFSClient
from config import (
    ASSET_MANAGEMENT_URL,
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_ANALYSIS_REQUEST,
    KAFKA_TOPIC_ANALYSIS_RESULT,
    KAFKA_TOPIC_DLQ,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_INDEX_COMPLETED,
    KAFKA_TOPIC_INDEXER_REQUEST,
    KAFKA_TOPIC_INDEXER_RESULT,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SUMMARY_RESULT,
    SERVICE_NAME,
)
from dotenv import load_dotenv
from message_utils import MessageBuilder, create_final_result_message
from redis_manager import RedisStateManager

# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
logger = logging.getLogger(__name__)


def _load_json_file_sync(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class DocumentOrchestratorKafkaHandler:
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
            KAFKA_TOPIC_ANALYSIS_RESULT,
            KAFKA_TOPIC_SUMMARY_RESULT,
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
        
        logger.info("Document Orchestrator Kafka consumers started...")
        
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
        """處理文件處理請求"""
        async for message in consumer:
            ids = self._claim_ids_from_message(message.value)
            if ids is None:
                await self.handle_document_request(message.value, producer)
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

            await self.handle_document_request(message.value, producer)
            await consumer.commit()
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_document_request(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理文件處理請求"""
        try:
            logger.info(f"Processing document request: {message.get('message_id')}")
            correlation_id = message.get("correlation_id")
            
            # 驗證消息格式
            if not self.validate_message(message):
                await self.send_error_response(
                    producer, message, 
                    "Invalid message format", 
                    "INVALID_FORMAT",
                    correlation_id 
                )
                return
            
            # 檢查 TTL
            if self.is_message_expired(message):
                await self.send_error_response(
                    producer, message,
                    "Message expired",
                    "MESSAGE_EXPIRED",
                    correlation_id
                )
                return
            
            # 檢查目標服務
            if message["target_service"] != self.service_name:
                await self.send_error_response(
                    producer, message,
                    f"Wrong target service: {message['target_service']}",
                    "WRONG_TARGET",
                    correlation_id
                )
                return
            
            # 處理文件處理請求
            if message["payload"]["action"] == "document_processing":
                await self.handle_document_processing(message, producer)
            else:
                await self.send_error_response(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION",
                    correlation_id
                )
                
        except Exception as e:  # noqa: BLE001 -- dispatch spans message dict access, HTTP download, Kafka and Redis failures, already logged + reported via send_error_response
            logger.error(f"Error processing document request: {e}")
            # Must go through send_error_response (result topic), not
            # send_to_dlq (KAFKA_TOPIC_DLQ) — nothing consumes the DLQ topic,
            # so a failure sent there vanishes silently instead of reaching
            # whatever is waiting for this job's result.
            await self.send_error_response(
                producer, message, str(e), "DISPATCH_ERROR", message.get("correlation_id")
            )
    
    async def handle_document_processing(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理文件處理流程"""
        try:
            correlation_id = message.get("correlation_id")
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
                    correlation_id
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


            # 初始化處理狀態並儲存到 Redis
            _dl_headers = payload.get("asset_managemant_download_header", {})
            state = {
                "original_message": message,
                "temp_file_path": temp_file_path,
                "analysis_result": None,
                "summary_result": None,
                "save_result": None,
                "step": "analysis",  # analysis -> summary -> save -> complete
                "branch_id": parameters.get("branch_id") or _dl_headers.get("X-Branch-ID") or payload.get("branch_id") or "",
                "created_at": asyncio.get_event_loop().time()
            }
            if not self.redis_manager.set_state(correlation_id, state):
                logger.error(f"Failed to save state to Redis for: {correlation_id}")
                await self.send_error_response(
                    producer, message,
                    "Failed to initialize processing state",
                    "STATE_INIT_FAILED",
                    correlation_id
                )
                return
            
            processing_mode = self._determine_processing_mode(message, primary_filename)
            
            # 發送分析請求
            analysis_request = MessageBuilder.create_processing_request(
                original_message=message,
                target_service="document_analysis_service",
                action="document_analysis",
                parameters={
                    "file_type": parameters.get("file_type", "document"),
                    "primary_filename": primary_filename,
                    "processing_mode": processing_mode
                },
                temp_file_path=temp_file_path
            )
            
            await producer.send(KAFKA_TOPIC_ANALYSIS_REQUEST, analysis_request)
            logger.info(f"Analysis request sent for: {correlation_id}")
            
        except Exception as e:  # noqa: BLE001 -- spans dict access, SeaweedFS download, Redis state, and Kafka send; already logged + reported
            logger.error(f"Error in document processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Document processing failed: {e!s}",
                "PROCESSING_FAILED",
                correlation_id
            )
    
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
                # create_error_response() puts "error" at the top level of
                # payload; create_*_result_message() (the summarize/analyze
                # success-shaped envelope, also used on this same topic when
                # summary_results["status"] is "error") nests the whole
                # results dict -- including its own "error" key -- one level
                # deeper under "results". Check both so a guardrail-block or
                # other in-process failure doesn't fall through to the
                # generic "Unknown service error" message.
                error_info = payload.get("error") or payload.get("results", {}).get("error") or {}
                logger.error(f"Service error for {correlation_id}: {error_info}")
                error_message = error_info.get('message', 'Unknown service error')
                await self.handle_processing_error(producer, state, f"Service processing failed: {error_message}", "SERVICE_ERROR")
                return

            
            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_ANALYSIS_RESULT:
                await self.handle_analysis_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_SUMMARY_RESULT:
                await self.handle_summary_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_INDEXER_RESULT:
                await self.handle_save_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_INDEX_COMPLETED:
                await self.handle_index_completed(message, producer, state, correlation_id)
                
        except Exception as e:  # noqa: BLE001 -- delegate handlers span dict access, Redis, Kafka and HTTP status updates; already logged + reported below
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

    async def handle_analysis_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理分析結果"""
        state["analysis_result"] = message["payload"]
        state["step"] = "summary"
        
        # 從分析結果中提取 analysis_result_path 並讀取完整的 chunks 數據
        analysis_results = message["payload"].get("results", {})
        parameters = analysis_results.get("parameters", {})
        analysis_result_path = parameters.get("analysis_result_path")
        
        # 從 analysis_result_path 讀取完整的 chunks 數據並保存到 Redis
        if analysis_result_path and os.path.exists(analysis_result_path):
            try:
                chunks_data = await asyncio.to_thread(_load_json_file_sync, analysis_result_path)

                # 將 chunks 數據保存到 Redis
                if not self.redis_manager.set_state(correlation_id, {"chunks": chunks_data}):
                    logger.error(f"Failed to save chunks to Redis for: {correlation_id}")
                else:
                    logger.info(f"Saved chunks data to Redis for: {correlation_id}, total chunks: {len(chunks_data)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from analysis_result_path for {correlation_id}: {e}")
            except (OSError, UnicodeDecodeError) as e:
                logger.error(f"Error reading analysis_result_path for {correlation_id}: {e}")
        else:
            logger.warning(f"Analysis result path not found or does not exist: {analysis_result_path}")
        
        # 更新 Redis 狀態
        if not self.redis_manager.update_state(correlation_id, state):
            logger.error(f"Failed to update state in Redis for: {correlation_id}")
            return
            
        filename = parameters.get("filename")
        
        if not analysis_result_path:
            logger.error(f"No analysis_result_path found in analysis result for: {correlation_id}")
            await self.handle_processing_error(producer, state, "Analysis result path not found", "MISSING_ANALYSIS_PATH")
            return
        
        # 發送摘要請求
        summary_request = MessageBuilder.create_processing_request(
            original_message=state["original_message"],
            target_service="document_summary_service",
            action="document_summary",
            parameters={
                "analysis_result": {
                    "results": {
                        "parameters": {
                            "analysis_result_path": analysis_result_path,
                            "filename": filename
                        }
                    }
                }
            },
            temp_file_path=state["temp_file_path"]
        )
        
        await producer.send(KAFKA_TOPIC_SUMMARY_REQUEST, summary_request)
        logger.info(f"Summary request sent for: {correlation_id}")
    
    async def handle_summary_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理摘要結果"""
        state["summary_result"] = message["payload"]
        state["step"] = "save"
        
        # 從摘要結果中提取 summary 並保存到 Redis
        summary_results = message["payload"].get("results", {})
        summary_params = summary_results.get("parameters", {})
        summary_result_path = summary_params.get("summary_result_path")
        
        # 嘗試從 summary_result_path 讀取 summary
        if summary_result_path and os.path.exists(summary_result_path):
            try:
                summary_data = await asyncio.to_thread(_load_json_file_sync, summary_result_path)

                # summary_data 是一個列表，提取第一個元素的 summary
                if isinstance(summary_data, list) and len(summary_data) > 0:
                    first_item = summary_data[0]
                    if "payload" in first_item and "summary" in first_item["payload"]:
                        summary_text = first_item["payload"]["summary"]
                        # 將 summary 保存到 Redis
                        if not self.redis_manager.set_state(correlation_id, {"summary": summary_text}):
                            logger.error(f"Failed to save summary to Redis for: {correlation_id}")
                        else:
                            logger.info(f"Saved summary to Redis for: {correlation_id}")
                    else:
                        logger.warning(f"No 'summary' field found in first item's payload for: {correlation_id}")
                else:
                    logger.warning(f"Summary data is not a list or is empty for: {correlation_id}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from summary_result_path for {correlation_id}: {e}")
            except (OSError, UnicodeDecodeError) as e:
                logger.error(f"Error reading summary_result_path for {correlation_id}: {e}")
        else:
            logger.warning(f"Summary result path not found or does not exist: {summary_result_path}")
        
        if not self.redis_manager.update_state(correlation_id, state):
            logger.error(f"Failed to update state in Redis for: {correlation_id}")
            return
        original_payload = state["original_message"]["payload"]
        original_params = original_payload["parameters"]
        # 發送保存請求
        save_request = MessageBuilder.create_processing_request(
            original_message=state["original_message"],
            target_service="document_indexer_service",
            action="document_indexing",
            parameters={
                "summary_result_path": summary_result_path,
                "filename": original_params.get("primary_filename"),
            },
            temp_file_path=state["temp_file_path"]
        )
        
        await producer.send(KAFKA_TOPIC_INDEXER_REQUEST, save_request)
        logger.info(f"Save request sent for: {correlation_id}")
    
    async def handle_save_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理保存結果 -- this module's own job (chunk/summarize/save) is done here,
        but Module 25 has only just been asked to index, not confirmed to have
        finished -- step stays "indexing" and update_asset_status(active) waits
        for handle_index_completed() (VIE01-255 update_status() race fix)."""
        try:
            save_result_payload = message["payload"]

            # 準備完整的 Redis 狀態 - 包含 step、chunks 和 summary
            indexing_state = {
                "step": "indexing",
                "branch_id": state.get("branch_id", ""),
                # set_state() below replaces the whole stored value -- original_message
                # must survive the trim: handle_index_completed() needs it to flip
                # status later, and handle_processing_error() needs it on any
                # failure path from here on (including from handle_index_completed()).
                "original_message": state["original_message"],
            }

            # 從 state 中獲取已保存的 chunks 和 summary
            if state.get("chunks"):
                indexing_state["chunks"] = state["chunks"]

            if state.get("summary"):
                indexing_state["summary"] = state["summary"]

            # 更新 Redis 狀態
            if not self.redis_manager.set_state(correlation_id, indexing_state):
                logger.error(f"Failed to update indexing state in Redis for: {correlation_id}")
            else:
                logger.info(f"Successfully updated indexing state in Redis for: {correlation_id}")

            # 創建最終結果 (保持不變，發送完整的 Kafka 消息)
            final_result = create_final_result_message(
                original_message=state["original_message"],
                analysis_result=state["analysis_result"],
                summary_result=state["summary_result"],
                save_result=save_result_payload
            )
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, final_result)
            logger.info(f"Final result sent for: {correlation_id}")

            # 清理臨時文件
            if "temp_file_path" in state:
                self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])

        except Exception as e:  # noqa: BLE001 -- spans Redis state, Kafka send and temp-file cleanup; already logged + reported
            logger.error(f"Error in handle_save_result for {correlation_id}: {e}")
            await self.handle_processing_error(producer, state, str(e), "SAVE_RESULT_ERROR")

    async def handle_index_completed(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """Module 25 has actually finished indexing this asset (personal-index-completed) --
        only now is it safe to mark the job complete and flip the asset active
        (VIE01-255 update_status() race fix: this used to fire right after
        handle_save_result() merely enqueued the index-request)."""
        try:
            final_state = {
                "step": "complete",
                "branch_id": state.get("branch_id", "")
            }
            if state.get("chunks"):
                final_state["chunks"] = state["chunks"]
            if state.get("summary"):
                final_state["summary"] = state["summary"]

            if not self.redis_manager.set_state(correlation_id, final_state):
                logger.error(f"Failed to update final state in Redis for: {correlation_id}")
            else:
                logger.info(f"Successfully updated final state in Redis for: {correlation_id}")

            ids = self._claim_ids_from_message(state["original_message"])
            if ids:
                asset_path, version_id, _cid, user_id, branch_id = ids
                await self.seaweedfs_client.update_asset_status(user_id, branch_id, asset_path, version_id, "active")

        except Exception as e:  # noqa: BLE001 -- spans Redis state and HTTP status update; already logged + reported
            logger.error(f"Error in handle_index_completed for {correlation_id}: {e}")
            await self.handle_processing_error(producer, state, str(e), "INDEX_COMPLETED_ERROR")

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
        if payload.get("action") == "document_processing":
            required_payload_fields = ["request_id", "action", "parameters"]
            if not all(field in payload for field in required_payload_fields):
                return False
            
            parameters = payload.get("parameters", {})
            required_param_fields = ["asset_path", "version_id", "primary_filename", "status"]
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
        
    def _determine_processing_mode(self, message: dict[str, Any], filename: str) -> str:
        """決定文件處理模式"""
        try:
            # 從原始消息的 metadata 中獲取用戶選擇的處理模式
            metadata = message["payload"].get("metadata", {})
            original_metadata = metadata.get("original_metadata", {})
            user_processing_mode = original_metadata.get("processing_mode")
            
            # 檢查檔案副檔名
            file_extension = filename.lower().split('.')[-1] if '.' in filename else ""
            
            # 如果是 PDF 檔案且用戶指定了處理模式
            if file_extension == "pdf" and user_processing_mode in ["default", "ocr"]:
                logger.info(f"PDF file detected, using user-specified mode: {user_processing_mode}")
                return user_processing_mode
            
            # 如果是 PDF 檔案但沒有指定處理模式，使用預設
            elif file_extension == "pdf":
                logger.info("PDF file detected, using default mode")
                return "default"
            
            # 其他檔案類型一律使用 default 模式
            else:
                logger.info(f"Non-PDF file ({file_extension}), using default mode")
                return "default"

        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"Error determining processing mode: {e}, falling back to default")
            return "default"
