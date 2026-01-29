# services/image_processing_service/kafka_handler.py

import json
import asyncio
import logging
import os
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from image_processor import ImageProcessor
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_DESCRIPTION_REQUEST,
    KAFKA_TOPIC_OCR_REQUEST,
    KAFKA_TOPIC_DESCRIPTION_RESULT,
    KAFKA_TOPIC_OCR_RESULT,
    KAFKA_TOPIC_DLQ,
    SERVICE_NAME
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

class ImageProcessingKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.group_id = KAFKA_GROUP_ID
        self.service_name = SERVICE_NAME
        
        # 定義此服務可以處理的 target_service 列表
        self.accepted_target_services = [
            "image_processing_service",    
            "image_description_service",   
            "image_ocr_service"           
        ]
        
        self.processor = ImageProcessor()
        
    async def initialize(self):
        """初始化服務"""
        logger.info("Initializing Image Processing Service...")
        self.processor.initialize_model()
        logger.info("Image Processing Service initialized successfully")
    
    async def start_consumer(self):
        """啟動 Kafka Consumer - 監聽兩個 topic"""
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_DESCRIPTION_REQUEST,
            KAFKA_TOPIC_OCR_REQUEST,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        
        await consumer.start()
        await producer.start()
        
        logger.info("Image Processing Kafka consumer started, listening to multiple topics...")
        
        try:
            async for message in consumer:
                await self.handle_request(message.topic, message.value, producer)
        finally:
            await consumer.stop()
            await producer.stop()
    
    async def handle_request(self, topic: str, message: Dict[str, Any], producer: AIOKafkaProducer):
        """根據 topic 和 action 處理請求"""
        try:
            logger.info(f"Processing request from topic {topic}: {message.get('message_id')}")
            logger.debug(f"Full message content: {json.dumps(message, indent=2, ensure_ascii=False)}")
            
            # 驗證消息格式
            validation_result = self.validate_message(message)
            if not validation_result["valid"]:
                logger.error(f"Message validation failed: {validation_result['error']}")
                await self.send_error_response(
                    producer, message, topic,
                    f"Invalid message format: {validation_result['error']}", 
                    "INVALID_FORMAT"
                )
                return
            
            # 檢查 TTL
            if self.is_message_expired(message):
                logger.warning(f"Message expired: {message.get('message_id')}")
                await self.send_error_response(
                    producer, message, topic,
                    "Message expired",
                    "MESSAGE_EXPIRED"
                )
                return
            
            # 檢查目標服務 - 現在接受多個服務名稱
            target_service = message["target_service"]
            if target_service not in self.accepted_target_services:
                logger.warning(f"Wrong target service. Expected one of: {self.accepted_target_services}, Got: {target_service}")
                await self.send_error_response(
                    producer, message, topic,
                    f"Wrong target service: {target_service}",
                    "WRONG_TARGET"
                )
                return
            
            logger.info(f"Accepted target service: {target_service}")
            
            # 根據 action 處理請求
            action = message["payload"]["action"]
            logger.info(f"Processing action: {action}")
            
            if action == "image_description":
                await self.process_image_description(message, producer)
            elif action == "image_ocr":
                await self.process_image_ocr(message, producer)
            else:
                logger.error(f"Unknown action: {action}")
                await self.send_error_response(
                    producer, message, topic,
                    f"Unknown action: {action}",
                    "UNKNOWN_ACTION"
                )
                
        except Exception as e:
            logger.error(f"Error processing request from topic {topic}: {e}", exc_info=True)
            await self.send_to_dlq(producer, message, str(e))
    
    async def process_image_description(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理圖片描述"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            logger.info(f"Processing image description request: {payload['request_id']}")
            logger.debug(f"Parameters: {json.dumps(parameters, indent=2, ensure_ascii=False)}")
            
            # 獲取圖片路徑
            temp_file_path = parameters.get("temp_file_path")
            if not temp_file_path:
                logger.error("temp_file_path not found in parameters")
                await self.send_error_response(
                    producer, message, KAFKA_TOPIC_DESCRIPTION_REQUEST,
                    "temp_file_path not provided",
                    "MISSING_PARAMETER"
                )
                return
            
            if not os.path.exists(temp_file_path):
                logger.error(f"Image file not found: {temp_file_path}")
                await self.send_error_response(
                    producer, message, KAFKA_TOPIC_DESCRIPTION_REQUEST,
                    f"Image file not found: {temp_file_path}",
                    "FILE_NOT_FOUND"
                )
                return
            
            # 生成圖片描述
            logger.info(f"Generating description for: {temp_file_path}")
            description_result_path = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.processor.generate_description, 
                temp_file_path
            )
            
            logger.info(f"Description generated successfully, result saved to: {description_result_path}")
            
            # 發送成功響應
            success_response = MessageBuilder.create_description_success_response(
                original_message=message,
                result_path=description_result_path
            )
            
            await producer.send(KAFKA_TOPIC_DESCRIPTION_RESULT, success_response)
            logger.info(f"Description result sent for: {message['payload']['request_id']}")
            
        except Exception as e:
            logger.error(f"Error processing image description: {e}", exc_info=True)
            await self.send_error_response(
                producer, message, KAFKA_TOPIC_DESCRIPTION_REQUEST,
                f"Description generation failed: {str(e)}",
                "PROCESSING_FAILED"
            )
    
    async def process_image_ocr(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理圖片 OCR"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            logger.info(f"Processing image OCR request: {payload['request_id']}")
            logger.debug(f"Parameters: {json.dumps(parameters, indent=2, ensure_ascii=False)}")
            
            # 獲取圖片路徑
            temp_file_path = parameters.get("temp_file_path")
            if not temp_file_path:
                logger.error("temp_file_path not found in parameters")
                await self.send_error_response(
                    producer, message, KAFKA_TOPIC_OCR_REQUEST,
                    "temp_file_path not provided",
                    "MISSING_PARAMETER"
                )
                return
            
            if not os.path.exists(temp_file_path):
                logger.error(f"Image file not found: {temp_file_path}")
                await self.send_error_response(
                    producer, message, KAFKA_TOPIC_OCR_REQUEST,
                    f"Image file not found: {temp_file_path}",
                    "FILE_NOT_FOUND"
                )
                return
            
            # 提取圖片文字
            logger.info(f"Extracting text from: {temp_file_path}")
            ocr_result_path = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.processor.extract_text, 
                temp_file_path
            )
            
            logger.info(f"OCR completed successfully, result saved to: {ocr_result_path}")
            
            # 發送成功響應
            success_response = MessageBuilder.create_ocr_success_response(
                original_message=message,
                result_path=ocr_result_path
            )
            
            await producer.send(KAFKA_TOPIC_OCR_RESULT, success_response)
            logger.info(f"OCR result sent for: {message['payload']['request_id']}")
            
        except Exception as e:
            logger.error(f"Error processing image OCR: {e}", exc_info=True)
            await self.send_error_response(
                producer, message, KAFKA_TOPIC_OCR_REQUEST,
                f"OCR processing failed: {str(e)}",
                "PROCESSING_FAILED"
            )
    
    def validate_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """驗證消息格式 - 返回詳細的驗證結果"""
        try:
            required_fields = [
                "message_id", "correlation_id", "timestamp", 
                "source_service", "target_service", "message_type", 
                "priority", "payload", "retry_count", "ttl"
            ]
            
            # 檢查頂層必需字段
            missing_fields = [field for field in required_fields if field not in message]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}"
                }
            
            # 檢查 payload
            payload = message.get("payload", {})
            if not isinstance(payload, dict):
                return {
                    "valid": False,
                    "error": "payload must be a dictionary"
                }
            
            # 檢查 payload 必需字段
            required_payload_fields = ["request_id", "action", "parameters"]
            missing_payload_fields = [field for field in required_payload_fields if field not in payload]
            if missing_payload_fields:
                return {
                    "valid": False,
                    "error": f"Missing required payload fields: {missing_payload_fields}"
                }
            
            action = payload.get("action")
            if action not in ["image_description", "image_ocr"]:
                return {
                    "valid": False,
                    "error": f"Invalid action: {action}. Expected: image_description or image_ocr"
                }
            
            # 檢查 parameters
            parameters = payload.get("parameters", {})
            if not isinstance(parameters, dict):
                return {
                    "valid": False,
                    "error": "parameters must be a dictionary"
                }
            
            if "temp_file_path" not in parameters:
                return {
                    "valid": False,
                    "error": "temp_file_path not found in parameters"
                }
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    def is_message_expired(self, message: Dict[str, Any]) -> bool:
        """檢查消息是否過期"""
        try:
            from datetime import datetime, timezone
            timestamp = datetime.fromisoformat(message["timestamp"].replace('Z', '+00:00'))
            ttl = message.get("ttl", 3600)
            now = datetime.now(timezone.utc)
            
            age_seconds = (now - timestamp).total_seconds()
            is_expired = age_seconds > ttl
            
            if is_expired:
                logger.warning(f"Message expired: age={age_seconds}s, ttl={ttl}s")
            
            return is_expired
        except Exception as e:
            logger.error(f"Error checking message expiration: {e}")
            return False
    
    async def send_error_response(
        self, 
        producer: AIOKafkaProducer, 
        original_message: Dict[str, Any], 
        source_topic: str,
        error_message: str,
        error_code: str
    ):
        """發送錯誤響應"""
        try:
            action = original_message.get("payload", {}).get("action", "unknown")
            
            if action == "image_description":
                error_response = MessageBuilder.create_description_error_response(
                    original_message=original_message,
                    error_message=error_message,
                    error_code=error_code
                )
                result_topic = KAFKA_TOPIC_DESCRIPTION_RESULT
            elif action == "image_ocr":
                error_response = MessageBuilder.create_ocr_error_response(
                    original_message=original_message,
                    error_message=error_message,
                    error_code=error_code
                )
                result_topic = KAFKA_TOPIC_OCR_RESULT
            else:
                # 默認發送到 description result topic
                error_response = MessageBuilder.create_description_error_response(
                    original_message=original_message,
                    error_message=error_message,
                    error_code=error_code
                )
                result_topic = KAFKA_TOPIC_DESCRIPTION_RESULT
            
            logger.info(f"Sending error response to {result_topic}")
            logger.debug(f"Error response: {json.dumps(error_response, indent=2, ensure_ascii=False)}")
            
            await producer.send(result_topic, error_response)
            logger.warning(f"Error response sent to {result_topic}: {error_response['message_id']}")
            
        except Exception as e:
            logger.error(f"Failed to send error response: {e}", exc_info=True)
    
    async def send_to_dlq(self, producer: AIOKafkaProducer, original_message: Dict[str, Any], error: str):
        """發送到 DLQ"""
        try:
            dlq_message = MessageBuilder.create_dlq_message(
                original_message=original_message,
                error=error,
                final_retry_count=original_message.get("retry_count", 0)
            )
            await producer.send(KAFKA_TOPIC_DLQ, dlq_message)
            logger.error(f"Message sent to DLQ: {dlq_message['message_id']}")
        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {e}", exc_info=True)
