# services/audio_classifier_service/kafka_handler.py

import json
import asyncio
import logging
import concurrent.futures
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from audio_classification import AudioClassificationClient
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_CLASSIFIER_REQUEST,
    KAFKA_TOPIC_CLASSIFIER_RESULT,
    KAFKA_TOPIC_DLQ,
    KAFKA_CONSUMER_CONFIG,
    ASYNC_PROCESSING_CONFIG
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

class AudioClassifierKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.request_topic = KAFKA_TOPIC_CLASSIFIER_REQUEST
        self.result_topic = KAFKA_TOPIC_CLASSIFIER_RESULT
        self.dlq_topic = KAFKA_TOPIC_DLQ
        self.classification_client = AudioClassificationClient()
        self.service_name = "audio_classifier_service"

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=ASYNC_PROCESSING_CONFIG["max_workers"],
            thread_name_prefix="AudioClassification"
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
            group_id="audio-classifier-service-group",
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            **KAFKA_CONSUMER_CONFIG
        )
        
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        
        await consumer.start()
        await producer.start()
        
        logger.info("Audio Classifier Kafka consumer started, waiting for messages...")
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
                
                # 處理音頻分類請求
                if message["payload"]["action"] == "audio_classification":
                    # 記錄任務開始
                    self.active_tasks[message_id] = {
                        "start_time": asyncio.get_event_loop().time(),
                        "request_id": message["payload"].get("request_id")
                    }
                    
                    try:
                        classification_results = await self.handle_audio_classification_async(message)
                        
                        response_message = MessageBuilder.create_classification_result_message(
                            original_message=message,
                            classification_results=classification_results
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
    
    async def handle_audio_classification_async(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """異步處理音頻分類請求"""
        try:
            loop = asyncio.get_event_loop()
            
            # 使用線程池執行同步版本的音頻分類
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool,
                    self.classification_client.classify_audio_sync,
                    message["payload"]
                ),
                timeout=ASYNC_PROCESSING_CONFIG["task_timeout"]
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Audio classification timeout for message: {message.get('message_id')}")
            return {
                "request_id": message["payload"].get("request_id"),
                "action": "audio_classification",
                "status": "error",
                "error": {
                    "code": "PROCESSING_TIMEOUT",
                    "message": f"Processing timeout after {ASYNC_PROCESSING_CONFIG['task_timeout']} seconds",
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
        except Exception as e:
            logger.error(f"Audio classification error: {e}")
            raise  # 重新拋出異常，讓上層處理 DLQ
    
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
        if payload.get("action") == "audio_classification":
            required_payload_fields = ["request_id", "action", "parameters"]
            if not all(field in payload for field in required_payload_fields):
                logger.error(f"Missing required payload fields: {payload}")
                return False
            
            parameters = payload.get("parameters", {})
            if "file_path" not in parameters:
                logger.error("Missing file_path in parameters")
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
