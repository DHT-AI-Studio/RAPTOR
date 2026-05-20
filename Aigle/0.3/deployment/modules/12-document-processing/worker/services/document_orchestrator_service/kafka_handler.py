# services/document_orchestrator_service/kafka_handler.py

import json
import asyncio
import logging
import os
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from api_client import SeaweedFSClient
from message_utils import MessageBuilder, create_final_result_message
from redis_manager import RedisStateManager 
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_ANALYSIS_REQUEST,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_INDEXER_REQUEST,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_ANALYSIS_RESULT,
    KAFKA_TOPIC_SUMMARY_RESULT,
    KAFKA_TOPIC_INDEXER_RESULT,
    KAFKA_TOPIC_DLQ,
    SERVICE_NAME,
    ASSET_MANAGEMENT_URL
)
from dotenv import load_dotenv
import os
# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
logger = logging.getLogger(__name__)

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
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        result_consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_ANALYSIS_RESULT,
            KAFKA_TOPIC_SUMMARY_RESULT,
            KAFKA_TOPIC_INDEXER_RESULT,
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
    
    async def process_requests(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理文件處理請求"""
        async for message in consumer:
            await self.handle_document_request(message.value, producer)
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_document_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
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
                
        except Exception as e:
            logger.error(f"Error processing document request: {e}")
            await self.send_to_dlq(producer, message, str(e))
    
    async def handle_document_processing(self, message: Dict[str, Any], producer: AIOKafkaProducer):
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
            status = parameters.get("status")
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
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Document processing failed: {str(e)}",
                "PROCESSING_FAILED",
                correlation_id
            )
    
    async def handle_service_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, topic: str):
        """處理各服務的結果"""
        try:
            payload = message["payload"]
            correlation_id = message.get("correlation_id")
            if not correlation_id:
                logger.warning(f"Missing correlation_id in message")
                return
            
            state = self.redis_manager.get_state(correlation_id)
            if not state:
                logger.warning(f"State not found in Redis for: {correlation_id}")
                return
            
            # 檢查結果狀態
            if payload.get("status") == "error":
                logger.error(f"Service error for {correlation_id}: {payload.get('error')}")
                # 清理並發送錯誤
                self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])
                
                error_message = payload.get('error', {}).get('message', 'Unknown service error')
                
                await self.send_error_response(
                    producer,
                    state["original_message"],
                    f"Service processing failed: {error_message}",
                    "SERVICE_ERROR",
                    correlation_id 
                )
                return

            
            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_ANALYSIS_RESULT:
                await self.handle_analysis_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_SUMMARY_RESULT:
                await self.handle_summary_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_INDEXER_RESULT:
                await self.handle_save_result(message, producer, state, correlation_id)
                
        except Exception as e:
            logger.error(f"Error handling service result: {e}")
    
    async def handle_analysis_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
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
                with open(analysis_result_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                # 將 chunks 數據保存到 Redis
                if not self.redis_manager.set_state(correlation_id, {"chunks": chunks_data}):
                    logger.error(f"Failed to save chunks to Redis for: {correlation_id}")
                else:
                    logger.info(f"Saved chunks data to Redis for: {correlation_id}, total chunks: {len(chunks_data)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from analysis_result_path for {correlation_id}: {e}")
            except Exception as e:
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
            await self.send_error_response(
                producer,
                state["original_message"],
                "Analysis result path not found",
                "MISSING_ANALYSIS_PATH",
                correlation_id
            )
            # await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
            # self.redis_manager.delete_state(correlation_id)
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
    
    async def handle_summary_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
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
                with open(summary_result_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
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
            except Exception as e:
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
    
    async def handle_save_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理保存結果"""
        try:
            save_result_payload = message["payload"]
            
            # 準備完整的 Redis 狀態 - 包含 step、chunks 和 summary
            final_state = {
                "step": "complete",
                "branch_id": state.get("branch_id", "")
            }

            # 從 state 中獲取已保存的 chunks 和 summary
            if state.get("chunks"):
                final_state["chunks"] = state["chunks"]

            if state.get("summary"):
                final_state["summary"] = state["summary"]
            
            # 更新 Redis 狀態
            if not self.redis_manager.set_state(correlation_id, final_state):
                logger.error(f"Failed to update final state in Redis for: {correlation_id}")
            else:
                logger.info(f"Successfully updated final state in Redis for: {correlation_id}")
            
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
            
        except Exception as e:
            logger.error(f"Error in handle_save_result for {correlation_id}: {e}")
            try:
                error_state = {"step": "complete", "error": str(e)}
                self.redis_manager.set_state(correlation_id, error_state)
            except:
                pass
            # self.redis_manager.delete_state(correlation_id)
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
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
    
    def is_message_expired(self, message: Dict[str, Any]) -> bool:
        """檢查消息是否過期"""
        try:
            from datetime import datetime, timezone
            timestamp = datetime.fromisoformat(message["timestamp"].replace('Z', '+00:00'))
            ttl = message.get("ttl", 3600)
            now = datetime.now(timezone.utc)
            
            return (now - timestamp).total_seconds() > ttl
        except:
            return False
    
    async def send_error_response(
        self, 
        producer: AIOKafkaProducer, 
        original_message: Dict[str, Any], 
        error_message: str,
        error_code: str,
        correlation_id: str = None
    ):
        """發送錯誤響應"""
        error_response = MessageBuilder.create_error_response(
            original_message=original_message,
            error_message=error_message,
            error_code=error_code
        )
        await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
        logger.warning(f"Error response sent: {error_response['message_id']}")
        # 更新 Redis 狀態
        if correlation_id:
            error_state = {
                "step": f"error:{error_message}"
            }
            if self.redis_manager.set_state(correlation_id, error_state):
                logger.info(f"Error state saved to Redis for: {correlation_id}")
            else:
                logger.error(f"Failed to save error state to Redis for: {correlation_id}")
    
    async def send_to_dlq(self, producer: AIOKafkaProducer, original_message: Dict[str, Any], error: str):
        """發送到 DLQ"""
        dlq_message = MessageBuilder.create_dlq_message(
            original_message=original_message,
            error=error,
            final_retry_count=original_message.get("retry_count", 0)
        )
        await producer.send(KAFKA_TOPIC_DLQ, dlq_message)
        logger.error(f"Message sent to DLQ: {dlq_message['message_id']}")
        
    def _determine_processing_mode(self, message: Dict[str, Any], filename: str) -> str:
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
                
        except Exception as e:
            logger.warning(f"Error determining processing mode: {e}, falling back to default")
            return "default"
