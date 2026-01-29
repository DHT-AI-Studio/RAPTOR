# services/document_summary_service/kafka_handler.py

import json
import uuid
import asyncio
import logging
import concurrent.futures
import os
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from document_summary import DocumentSummarizer
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SUMMARY_RESULT,
    KAFKA_TOPIC_DLQ,
    KAFKA_CONSUMER_CONFIG,
    ASYNC_PROCESSING_CONFIG,
    SUMMARY_RESULTS_DIR,
    #OLLAMA_URL,
    #OLLAMA_MODEL
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
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
logger = logging.getLogger(__name__)

class DocumentSummaryKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.request_topic = KAFKA_TOPIC_SUMMARY_REQUEST
        self.result_topic = KAFKA_TOPIC_SUMMARY_RESULT
        self.dlq_topic = KAFKA_TOPIC_DLQ
        self.summarizer = DocumentSummarizer(
            ollama_url=OLLAMA_URL,
            model_name=OLLAMA_MODEL
        )
        self.service_name = "document_summary_service"

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=ASYNC_PROCESSING_CONFIG["max_workers"],
            thread_name_prefix="DocumentSummary"
        )

        self.active_tasks = {}
        self.task_semaphore = asyncio.Semaphore(ASYNC_PROCESSING_CONFIG["max_concurrent_tasks"])
        
        logger.info(f"Initialized with {ASYNC_PROCESSING_CONFIG['max_workers']} workers, "
                   f"max concurrent tasks: {ASYNC_PROCESSING_CONFIG['max_concurrent_tasks']}")
        
    async def start_consumer(self):
        """啟動 Kafka Consumer"""
        consumer = AIOKafkaConsumer(
            self.request_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="document-summary-service-group",
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        
        await consumer.start()
        await producer.start()
        
        logger.info("Document Summary Kafka consumer started, waiting for messages...")
        logger.info(f"Consumer config: session_timeout={KAFKA_CONSUMER_CONFIG['session_timeout_ms']}ms, "
                   f"heartbeat_interval={KAFKA_CONSUMER_CONFIG['heartbeat_interval_ms']}ms")
        
        try:
            async for message in consumer:
                asyncio.create_task(self.process_message_async(message.value, producer))
        finally:
            await consumer.stop()
            await producer.stop()
            self.thread_pool.shutdown(wait=True)

    async def process_message_async(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """異步處理消息"""
        message_id = message.get('message_id', 'unknown')
        
        # 使用信號量控制並發數
        async with self.task_semaphore:
            try:
                logger.info(f"[ASYNC] Processing message: {message_id}")
                
                # 檢查消息格式
                if not self.validate_message(message):
                    await self.send_error_response(
                        producer, message, 
                        "Invalid message format", 
                        "INVALID_FORMAT"
                    )
                    return
                
                # 檢查 TTL
                if self.is_message_expired(message):
                    await self.send_error_response(
                        producer, message,
                        "Message expired",
                        "MESSAGE_EXPIRED"
                    )
                    return
                
                # 檢查目標服務
                if message["target_service"] != self.service_name:
                    await self.send_error_response(
                        producer, message,
                        f"Wrong target service: {message['target_service']}",
                        "WRONG_TARGET"
                    )
                    return
                
                # 處理文件摘要請求
                if message["payload"]["action"] == "document_summary":
                    # 記錄任務開始
                    self.active_tasks[message_id] = {
                        "start_time": asyncio.get_event_loop().time(),
                        "request_id": message["payload"].get("request_id")
                    }
                    
                    try:
                        summary_results = await self.handle_document_summary_async(message)
                        
                        response_message = MessageBuilder.create_summary_result_message(
                            original_message=message,
                            summary_results=summary_results
                        )
                        await self.send_response(producer, response_message)
                        
                        # 記錄處理時間
                        if ASYNC_PROCESSING_CONFIG["enable_task_monitoring"]:
                            processing_time = asyncio.get_event_loop().time() - self.active_tasks[message_id]["start_time"]
                            logger.info(f"[ASYNC] Message {message_id} processed in {processing_time:.2f}s")
                        
                    finally:
                        # 清理任務記錄
                        self.active_tasks.pop(message_id, None)
                        
                else:
                    await self.send_error_response(
                        producer, message,
                        f"Unknown action: {message['payload']['action']}",
                        "UNKNOWN_ACTION"
                    )
                    
            except Exception as e:
                logger.error(f"[ASYNC] Error processing message {message_id}: {e}")
                await self.send_to_dlq(producer, message, str(e))
                # 清理任務記錄
                self.active_tasks.pop(message_id, None)

    async def handle_document_summary_async(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """異步處理文件摘要請求"""
        try:
            loop = asyncio.get_event_loop()
            
            # 使用線程池執行同步版本的文件摘要
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool,
                    self.summarize_document_sync,
                    message["payload"]
                ),
                timeout=ASYNC_PROCESSING_CONFIG["task_timeout"]
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Document summary timeout for message: {message.get('message_id')}")
            return {
                "request_id": message["payload"].get("request_id"),
                "action": "document_summary",
                "status": "error",
                "error": {
                    "code": "PROCESSING_TIMEOUT",
                    "message": f"Processing timeout after {ASYNC_PROCESSING_CONFIG['task_timeout']} seconds",
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Document summary error: {e}")
            raise  # 重新拋出異常，讓上層處理 DLQ

    def summarize_document_sync(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步版本的文件摘要方法（用於線程池執行）
        
        Args:
            request_payload: 包含摘要請求的 payload
            
        Returns:
            摘要結果
        """
        try:
            # 提取請求參數
            request_id = request_payload["request_id"]
            parameters = request_payload.get("parameters", {})
            
            # 從 parameters.analysis_result.results.parameters 
            analysis_result = parameters.get("analysis_result", {})
            results = analysis_result.get("results", {})
            analysis_params = results.get("parameters", {})

            analysis_result_path = analysis_params.get("analysis_result_path")
            
            
            if not analysis_result_path:
                return {
                    "request_id": request_id,
                    "action": "document_summary",
                    "status": "error",
                    "error": {
                        "code": "MISSING_ANALYSIS_PATH",
                        "message": "Missing analysis_result_path in parameters",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 檢查分析結果檔案是否存在
            if not os.path.exists(analysis_result_path):
                return {
                    "request_id": request_id,
                    "action": "document_summary",
                    "status": "error",
                    "error": {
                        "code": "ANALYSIS_FILE_NOT_FOUND",
                        "message": f"Analysis result file not found: {analysis_result_path}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 提取其他參數
            filename = analysis_result.get("results", {}).get("parameters", {}).get("filename", "unknown")
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            status = parameters.get("status")
            
            logger.info(f"[SYNC] Processing summary for: {filename} from: {analysis_result_path}")
            logger.debug(f"[SYNC] Asset path: {asset_path}, Version ID: {version_id}")
            
            # 處理摘要
            start_time = datetime.now()
            summary_text = self.summarizer.summarize_from_json(analysis_result_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 檢查摘要結果
            if summary_text.startswith("處理文件時發生錯誤") or summary_text.startswith("摘要生成失敗"):
                return {
                    "request_id": request_id,
                    "action": "document_summary",
                    "status": "error",
                    "error": {
                        "code": "SUMMARY_GENERATION_FAILED",
                        "message": f"Summary generation failed: {summary_text}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 生成摘要結果檔案路徑
            summary_result_path = os.path.join(
                SUMMARY_RESULTS_DIR,
                f"summary_result_{request_id}_{filename}.json"  
            )
            try:
                # 讀取原始 JSON 文件
                with open(analysis_result_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                # 確保 original_data 是列表
                if not isinstance(original_data, list):
                    logger.error(f"Expected list format, got {type(original_data)}")
                    return {
                        "request_id": request_id,
                        "action": "document_summary",
                        "status": "error",
                        "error": {
                            "code": "INVALID_DATA_FORMAT",
                            "message": f"Expected list format, got {type(original_data).__name__}",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                
                new_summary_entry = {
                    "id": str(uuid.uuid4()),
                    "payload": {
                        "filename": filename,
                        "type": "documents",
                        "upload_time": datetime.now().isoformat(),
                        "embedding_type": "summary",
                        "asset_path": asset_path,
                        "version_id": version_id,
                        "status": status,
                        "summary": summary_text
                    }
                }
                modified_data = [new_summary_entry] + original_data
                
                # 保存為新的 JSON 文件
                with open(summary_result_path, 'w', encoding='utf-8') as f:
                    json.dump(modified_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"[SYNC] Added summary to {len(original_data)} items")
                
            except Exception as json_error:
                logger.error(f"Error creating JSON summary result: {json_error}")
                return {
                    "request_id": request_id,
                    "action": "document_summary",
                    "status": "error",
                    "error": {
                        "code": "JSON_CREATION_FAILED",
                        "message": f"Failed to create JSON summary result: {str(json_error)}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            logger.info(f"[SYNC] Summary result saved to: {summary_result_path}")
            
            # 計算檔案大小
            file_size_bytes = os.path.getsize(summary_result_path)
            summary_length = len(summary_text)
            
            # 返回成功結果
            return {
                "request_id": request_id,
                "action": "document_summary",
                "status": "success",
                "parameters": {
                    "summary_result_path": summary_result_path,
                    # "summary_text": summary_text,
                    "filename": filename,
                    "summary_length": summary_length,
                    "file_size_bytes": file_size_bytes,
                    "processing_time_seconds": processing_time,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "source_analysis_path": analysis_result_path,
                    "output_format": "json",
                    "items_processed": len(original_data) if 'original_data' in locals() else 0  
                }
                # "metadata": {
                #     "processed_at": datetime.now().isoformat(),
                #     "processing_method": "async_thread_pool",
                #     "model_used": OLLAMA_MODEL
                # }
            }
            
        except Exception as e:
            logger.error(f"[SYNC] Error generating summary: {e}")
            

            try:
                request_id = request_payload.get('request_id', 'unknown')
                filename = request_payload.get('parameters', {}).get('analysis_result', {}).get('results', {}).get('parameters', {}).get('filename', 'unknown')
                summary_result_path = os.path.join(
                    SUMMARY_RESULTS_DIR,
                    f"summary_result_{request_id}_{filename}.json"  
                )
                if os.path.exists(summary_result_path):
                    os.remove(summary_result_path)
                    logger.info(f"Cleaned up failed summary result file: {summary_result_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup summary result file: {cleanup_error}")
            
            return {
                "request_id": request_payload.get("request_id"),
                "action": "document_summary",
                "status": "error",
                "error": {
                    "code": "SUMMARY_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "error_type": type(e).__name__,
                    "processing_method": "async_thread_pool"
                }
            }

    def validate_message(self, message: Dict[str, Any]) -> bool:
        """驗證消息格式"""
        required_fields = [
            "message_id", "correlation_id", "timestamp", 
            "source_service", "target_service", "message_type", 
            "priority", "payload", "retry_count", "ttl"
        ]
        
        if not all(field in message for field in required_fields):
            logger.error(f"Missing required fields in message: {message}")
            return False
        
        payload = message.get("payload", {})
        if payload.get("action") == "document_summary":
            required_payload_fields = ["request_id", "action", "parameters"]
            if not all(field in payload for field in required_payload_fields):
                logger.error(f"Missing required payload fields: {payload}")
                return False
            
            parameters = payload.get("parameters", {})
            if "analysis_result" not in parameters:
                logger.error("Missing analysis_result in parameters")
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
    
    async def send_response(self, producer: AIOKafkaProducer, response_message: Dict[str, Any]):
        """發送響應消息"""
        await producer.send(self.result_topic, response_message)
        logger.info(f"Response sent: {response_message['message_id']}")
    
    async def send_error_response(
        self, 
        producer: AIOKafkaProducer, 
        original_message: Dict[str, Any], 
        error_message: str,
        error_code: str
    ):
        """發送錯誤響應"""
        error_response = MessageBuilder.create_error_response(
            original_message=original_message,
            error_message=error_message,
            error_code=error_code
        )
        await producer.send(self.result_topic, error_response)
        logger.warning(f"Error response sent: {error_response['message_id']}")
    
    async def send_to_dlq(self, producer: AIOKafkaProducer, original_message: Dict[str, Any], error: str):
        """發送到 DLQ"""
        dlq_message = MessageBuilder.create_dlq_message(
            original_message=original_message,
            error=error,
            final_retry_count=original_message.get("retry_count", 0)
        )
        await producer.send(self.dlq_topic, dlq_message)
        logger.error(f"Message sent to DLQ: {dlq_message['message_id']}")
    
    def get_active_tasks_info(self) -> Dict[str, Any]:
        """獲取當前活躍任務資訊（用於監控）"""
        current_time = asyncio.get_event_loop().time()
        tasks_info = {}
        
        for message_id, task_info in self.active_tasks.items():
            tasks_info[message_id] = {
                "request_id": task_info["request_id"],
                "running_time": current_time - task_info["start_time"]
            }
        
        return {
            "active_task_count": len(self.active_tasks),
            "max_concurrent_tasks": ASYNC_PROCESSING_CONFIG["max_concurrent_tasks"],
            "thread_pool_size": ASYNC_PROCESSING_CONFIG["max_workers"],
            "tasks": tasks_info
        }
