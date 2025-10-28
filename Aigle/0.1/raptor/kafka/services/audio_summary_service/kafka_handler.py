# services/audio_summary_service/kafka_handler.py

import json
import uuid
import asyncio
import logging
import concurrent.futures
import os
from typing import Dict, Any
from datetime import datetime
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from audio_summary import AudioSummary
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SUMMARY_RESULT,
    KAFKA_TOPIC_DLQ,
    SUMMARY_RESULTS_DIR,
    #OLLAMA_URL,
    #OLLAMA_MODEL,
    AUDIO_SUMMARY_CONFIG
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

class AudioSummaryKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.request_topic = KAFKA_TOPIC_SUMMARY_REQUEST
        self.result_topic = KAFKA_TOPIC_SUMMARY_RESULT
        self.dlq_topic = KAFKA_TOPIC_DLQ
        self.summarizer = AudioSummary(
            model_name=OLLAMA_MODEL,
            max_tokens_per_batch=AUDIO_SUMMARY_CONFIG["max_tokens_per_batch"],
            max_summary_length=AUDIO_SUMMARY_CONFIG["max_summary_length"],
            ollama_url=OLLAMA_URL
        )
        self.service_name = "audio_summary_service"
        
        # 簡化線程池配置
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="AudioSummary"
        )
        
        logger.info("Audio Summary Kafka Handler initialized")
        
    async def start_consumer(self):
        """啟動 Kafka Consumer"""
        consumer = AIOKafkaConsumer(
            self.request_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="audio-summary-service-group",
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        
        await consumer.start()
        await producer.start()
        
        logger.info("Audio Summary Kafka consumer started, waiting for messages...")
        
        try:
            async for message in consumer:
                asyncio.create_task(self.process_message(message.value, producer))
        finally:
            await consumer.stop()
            await producer.stop()
            self.thread_pool.shutdown(wait=True)

    async def process_message(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理消息"""
        message_id = message.get('message_id', 'unknown')
        
        try:
            logger.info(f"Processing message: {message_id}")
            
            # 基本驗證
            if not self.validate_message(message):
                await self.send_error_response(producer, message, "Invalid message format", "INVALID_FORMAT")
                return
            
            # 檢查目標服務
            if message["target_service"] != self.service_name:
                await self.send_error_response(producer, message, f"Wrong target service: {message['target_service']}", "WRONG_TARGET")
                return
            
            # 處理音訊摘要請求
            if message["payload"]["action"] == "audio_summary":
                summary_results = await self.handle_audio_summary(message)
                response_message = MessageBuilder.create_summary_result_message(
                    original_message=message,
                    summary_results=summary_results
                )
                await self.send_response(producer, response_message)
            else:
                await self.send_error_response(producer, message, f"Unknown action: {message['payload']['action']}", "UNKNOWN_ACTION")
                    
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            await self.send_to_dlq(producer, message, str(e))

    async def handle_audio_summary(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """處理音訊摘要請求"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self.summarize_audio_sync,
                message["payload"],
                message
            )
            return result
            
        except Exception as e:
            logger.error(f"Audio summary error: {e}")
            return {
                "request_id": message["payload"].get("request_id"),
                "action": "audio_summary",
                "status": "error",
                "error": {
                    "code": "AUDIO_SUMMARY_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }

    def summarize_audio_sync(self, request_payload: Dict[str, Any], kafka_message: Dict[str, Any] = None) -> Dict[str, Any]:
        """同步版本的音訊摘要方法"""
        try:
            request_id = request_payload["request_id"]
            file_path = request_payload.get("file_path")
            parameters = request_payload.get("parameters", {})
            
            if not file_path:
                return {
                    "request_id": request_id,
                    "action": "audio_summary",
                    "status": "error",
                    "error": {
                        "code": "MISSING_FILE_PATH",
                        "message": "Missing file_path in payload",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            if not os.path.exists(file_path):
                return {
                    "request_id": request_id,
                    "action": "audio_summary",
                    "status": "error",
                    "error": {
                        "code": "AUDIO_FILE_NOT_FOUND",
                        "message": f"Audio analysis file not found: {file_path}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            filename = parameters.get("filename", "unknown")
            logger.info(f"Processing audio summary for: {filename} from: {file_path}")
            
            # 處理摘要
            start_time = datetime.now()
            summary_text = self.summarizer.process_summary(file_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 檢查摘要結果
            if summary_text.startswith("錯誤:") or summary_text.startswith("摘要生成失敗"):
                return {
                    "request_id": request_id,
                    "action": "audio_summary",
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
                f"audio_summary_result_{request_id}_{filename}.json"  
            )
            
            try:
                # 讀取原始 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                # 確保 original_data 是列表
                if not isinstance(original_data, list):
                    logger.error(f"Expected list format, got {type(original_data)}")
                    return {
                        "request_id": request_id,
                        "action": "audio_summary",
                        "status": "error",
                        "error": {
                            "code": "INVALID_DATA_FORMAT",
                            "message": f"Expected list format, got {type(original_data).__name__}",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                # 從 Kafka 消息中提取需要的資訊
                summary_id = str(uuid.uuid4())
                summary_item = {
                    "id": summary_id,
                    "payload": {
                        "filename": parameters.get("filename", filename),  
                        "type": "audios",
                        "upload_time": datetime.now().isoformat(),
                        "embedding_type": "summary",
                        "asset_path": parameters.get("asset_path", ""), 
                        "version_id": parameters.get("version_id", ""),
                        "status": parameters.get("status", ""),
                        "summary": summary_text
                    }
                }
    
                new_data = [summary_item] + original_data
                
                # 保存為新的 JSON 文件
                with open(summary_result_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Added summary item to the beginning. Total items: {len(new_data)}")
                
            except Exception as json_error:
                logger.error(f"Error creating JSON summary result: {json_error}")
                return {
                    "request_id": request_id,
                    "action": "audio_summary",
                    "status": "error",
                    "error": {
                        "code": "JSON_CREATION_FAILED",
                        "message": f"Failed to create JSON summary result: {str(json_error)}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            logger.info(f"Audio summary result saved to: {summary_result_path}")
            
            # 計算檔案大小
            file_size_bytes = os.path.getsize(summary_result_path)
            summary_length = len(summary_text)
            
            # 返回成功結果
            return {
                "request_id": request_id,
                "action": "audio_summary",
                "status": "success",
                "parameters": {
                    "summary_result_path": summary_result_path,
                    "filename": filename,
                    "summary_length": summary_length,
                    "file_size_bytes": file_size_bytes,
                    "processing_time_seconds": processing_time,
                    "items_processed": len(new_data) 
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating audio summary: {e}")
            return {
                "request_id": request_payload.get("request_id"),
                "action": "audio_summary",
                "status": "error",
                "error": {
                    "code": "AUDIO_SUMMARY_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }

    def validate_message(self, message: Dict[str, Any]) -> bool:
        """驗證消息格式"""
        required_fields = ["message_id", "target_service", "payload"]
        
        if not all(field in message for field in required_fields):
            logger.error(f"Missing required fields in message: {message}")
            return False
        
        payload = message.get("payload", {})
        required_payload_fields = ["request_id", "action"]
        
        if not all(field in payload for field in required_payload_fields):
            logger.error(f"Missing required payload fields: {payload}")
            return False
        
        return True

    async def send_response(self, producer: AIOKafkaProducer, message: Dict[str, Any]):
        """發送響應消息"""
        try:
            await producer.send_and_wait(self.result_topic, message)
            logger.info(f"Sent response to {self.result_topic}: {message.get('message_id')}")
        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    async def send_error_response(self, producer: AIOKafkaProducer, original_message: Dict[str, Any], error_message: str, error_code: str):
        """發送錯誤響應"""
        try:
            error_response = MessageBuilder.create_error_response(
                original_message=original_message,
                error_message=error_message,
                error_code=error_code
            )
            await self.send_response(producer, error_response)
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")

    async def send_to_dlq(self, producer: AIOKafkaProducer, message: Dict[str, Any], error: str):
        """發送消息到 DLQ"""
        try:
            dlq_message = MessageBuilder.create_dlq_message(
                original_message=message,
                error=error,
                final_retry_count=message.get("retry_count", 0)
            )
            
            await producer.send_and_wait(self.dlq_topic, dlq_message)
            logger.error(f"Sent message to DLQ: {message.get('message_id')} - {error}")
        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {e}")
