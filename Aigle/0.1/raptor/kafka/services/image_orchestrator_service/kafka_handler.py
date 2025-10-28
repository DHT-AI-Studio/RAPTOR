# services/image_orchestrator_service/kafka_handler.py

import json
import asyncio
import logging
import os
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from api_client import SeaweedFSClient
from message_utils import MessageBuilder, ImageResultMerger, create_final_result_message
from redis_manager import RedisStateManager
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_DESCRIPTION_REQUEST,
    KAFKA_TOPIC_OCR_REQUEST,
    KAFKA_TOPIC_SAVE_QDRANT_REQUEST,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_DESCRIPTION_RESULT,
    KAFKA_TOPIC_OCR_RESULT,
    KAFKA_TOPIC_SAVE_QDRANT_RESULT,
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
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        result_consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_DESCRIPTION_RESULT,
            KAFKA_TOPIC_OCR_RESULT,
            KAFKA_TOPIC_SAVE_QDRANT_RESULT,
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
        
    
    async def process_requests(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理圖片處理請求"""
        async for message in consumer:
            await self.handle_image_request(message.value, producer)
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_image_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理圖片處理請求"""
        try:
            logger.info(f"Processing image request: {message.get('message_id')}")
            
            # 驗證消息格式
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
            
            # 處理圖片處理請求
            if message["payload"]["action"] == "image_processing":
                await self.handle_image_processing(message, producer)
            else:
                await self.send_error_response(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION"
                )
                
        except Exception as e:
            logger.error(f"Error processing image request: {e}")
            await self.send_to_dlq(producer, message, str(e))
    
    async def handle_image_processing(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理圖片處理流程"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            # 提取必要資訊
            access_token = payload.get("access_token")
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            primary_filename = parameters.get("primary_filename")
            status = parameters.get("status")
            
            if not all([access_token, asset_path, version_id, primary_filename]):
                await self.send_error_response(
                    producer, message,
                    "Missing required parameters",
                    "MISSING_PARAMETERS"
                )
                return
            
            # 下載檔案
            logger.info(f"Downloading file: {asset_path}/{version_id}/{primary_filename}")
            temp_file_path = await self.seaweedfs_client.download_file(
                access_token=access_token,
                asset_path=asset_path,
                version_id=version_id,
                filename=primary_filename
            )
            

            correlation_id = message["correlation_id"]
            state = {
                "original_message": message,
                "temp_file_path": temp_file_path,
                "description_result": None,
                "ocr_result": None,
                "save_result": None,
                "step": "parallel_processing",  # parallel_processing -> merge -> save -> complete
                "created_at": asyncio.get_event_loop().time()
            }
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")
            
            # 並行發送描述和OCR請求
            await self.send_parallel_requests(message, producer, temp_file_path, parameters)
            
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Image processing failed: {str(e)}",
                "PROCESSING_FAILED"
            )
    
    async def send_parallel_requests(self, message: Dict[str, Any], producer: AIOKafkaProducer, temp_file_path: str, parameters: Dict[str, Any]):
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
    
    async def handle_service_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, topic: str):
        """處理各服務的結果"""
        try:
            # 使用 correlation_id 來找到對應的處理狀態
            correlation_id = message.get("correlation_id")
            
            if not correlation_id:
                logger.warning(f"Missing correlation_id in result message")
                return
            
            state = self.redis_manager.get_state(correlation_id)
            if not state:
                logger.warning(f"Unknown correlation_id: {correlation_id}")
                return
            
            payload = message["payload"]
            # 檢查結果狀態
            if payload.get("status") == "error":
                logger.error(f"Service error for correlation_id {correlation_id}: {payload.get('error')}")
                # 清理並發送錯誤
                self.cleanup_request_state(correlation_id)
                
                error_response = MessageBuilder.create_error_response(
                    original_message=state["original_message"],
                    error_message=f"Service processing failed: {payload.get('error', {}).get('message')}",
                    error_code="SERVICE_ERROR"
                )
                await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
                return
            
            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_DESCRIPTION_RESULT:
                await self.handle_description_result(message, producer, correlation_id)
            elif topic == KAFKA_TOPIC_OCR_RESULT:
                await self.handle_ocr_result(message, producer, correlation_id)
            elif topic == KAFKA_TOPIC_SAVE_QDRANT_RESULT:
                await self.handle_save_result(message, producer, correlation_id)
                
        except Exception as e:
            logger.error(f"Error handling service result: {e}")
    
    async def handle_description_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, correlation_id: str):
        """處理圖片描述結果"""
        self.redis_manager.update_state(correlation_id, {
            "description_result": message["payload"]
        })
        logger.info(f"Description result received for correlation_id: {correlation_id}")
        
        # 檢查是否兩個結果都已收到
        await self.check_and_merge_results(producer, correlation_id)
    
    async def handle_ocr_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, correlation_id: str):
        """處理OCR結果"""
        self.redis_manager.update_state(correlation_id, {
            "ocr_result": message["payload"]
        })
        logger.info(f"OCR result received for correlation_id: {correlation_id}")
        
        # 檢查是否兩個結果都已收到
        await self.check_and_merge_results(producer, correlation_id)
    
    async def check_and_merge_results(self, producer: AIOKafkaProducer, correlation_id: str):
        """檢查並合併結果"""
        state = self.redis_manager.get_state(correlation_id)
        if not state:
            logger.error(f"State not found for correlation_id: {correlation_id}")
            return    
        if state["description_result"] and state["ocr_result"]:      
            self.redis_manager.update_state(correlation_id, {"step": "merge"})
            try:
                # 合併結果
                merged_file_path = ImageResultMerger.merge_results(
                    description_result=state["description_result"],
                    ocr_result=state["ocr_result"],
                    original_message=state["original_message"]
                )
                
                self.redis_manager.update_state(correlation_id, {
                    "step": "save",
                    "merged_file_path": merged_file_path
                })

                
                # 發送保存請求
                save_request = MessageBuilder.create_processing_request(
                    original_message=state["original_message"],
                    target_service="image_save2qdrant_service",
                    action="save_to_qdrant",
                    parameters={
                        "merged_result_path": merged_file_path,
                        "filename": state["original_message"]["payload"]["parameters"].get("primary_filename"),
                    },
                    temp_file_path=state["temp_file_path"]
                )
                
                await producer.send(KAFKA_TOPIC_SAVE_QDRANT_REQUEST, save_request)
                logger.info(f"Save request sent for correlation_id: {correlation_id}")
                logger.info(f"Save request ID: {save_request['payload']['request_id']}")
                logger.info(f"Merged file path: {merged_file_path}")
                

                
            except Exception as e:
                logger.error(f"Error merging results: {e}")
                # 清理並發送錯誤
                self.cleanup_request_state(correlation_id)
                
                error_response = MessageBuilder.create_error_response(
                    original_message=state["original_message"],
                    error_message=f"Result merging failed: {str(e)}",
                    error_code="MERGE_FAILED"
                )
                await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
    
    async def handle_save_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, correlation_id: str):
        """處理保存結果"""
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
            
            summary = None
            text = None
            # 從 save_result 中提取
            save_result = message["payload"]
            if isinstance(save_result, dict) and "result_path" in save_result:
                result_path = save_result["result_path"]
                if isinstance(result_path, str) and os.path.exists(result_path):
                    try:
                        with open(result_path, 'r', encoding='utf-8') as f:
                            result_list = json.load(f)
                        
                        logger.info(f" Loaded {len(result_list)} items from {result_path}")
                        for item in result_list:
                            payload = item.get("payload", {})
                            
                            if payload.get("embedding_type") == "summary":
                                summary = payload.get("summary")
                                logger.info(f" Found summary: {summary[:100] if summary else None}...")
                            
                            if payload.get("embedding_type") == "text":
                                text = payload.get("text")
                                logger.info(f" Found text: {text}")
                        
                    except Exception as e:
                        logger.error(f" Failed to read/parse file {result_path}: {e}", exc_info=True)
                else:
                    logger.warning(f" result_path not found or invalid: {result_path}")
            else:
                logger.warning(f" result_path not in save_result")
                
            final_state = {
                "step": "complete",
                "summary": summary,
                "text": text
            }
            self.redis_manager.set_state(correlation_id, final_state)
            logger.info(f"Redis updated to final state for {correlation_id}")
            # 清理臨時檔案
            self.cleanup_request_state(correlation_id)
            
        except Exception as e:
            logger.error(f"Error in handle_save_result: {e}", exc_info=True)
            state = self.redis_manager.get_state(correlation_id)
            if state:
                error_response = MessageBuilder.create_error_response(
                    original_message=state["original_message"],
                    error_message=f"Failed to process save result: {str(e)}",
                    error_code="SAVE_RESULT_ERROR"
                )
                await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)


    
    def cleanup_request_state(self, correlation_id: str):
        """清理請求狀態和相關資源"""
        state = self.redis_manager.get_state(correlation_id)
            
        if state:
            # 清理臨時檔案
            if "temp_file_path" in state:
                self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])
            
            # self.redis_manager.delete_state(correlation_id)
            logger.info(f"Cleaned up processing state for correlation_id: {correlation_id}")

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
        if payload.get("action") == "image_processing":
            required_payload_fields = ["request_id", "action", "parameters"]
            if not all(field in payload for field in required_payload_fields):
                return False
            
            parameters = payload.get("parameters", {})
            required_param_fields = ["asset_path", "version_id", "primary_filename"]
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
        error_code: str
    ):
        """發送錯誤響應"""
        error_response = MessageBuilder.create_error_response(
            original_message=original_message,
            error_message=error_message,
            error_code=error_code
        )
        await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
        logger.warning(f"Error response sent: {error_response['message_id']}")
    
    async def send_to_dlq(self, producer: AIOKafkaProducer, original_message: Dict[str, Any], error: str):
        """發送到 DLQ"""
        dlq_message = MessageBuilder.create_dlq_message(
            original_message=original_message,
            error=error,
            final_retry_count=original_message.get("retry_count", 0)
        )
        await producer.send(KAFKA_TOPIC_DLQ, dlq_message)
        logger.error(f"Message sent to DLQ: {dlq_message['message_id']}")
    
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
