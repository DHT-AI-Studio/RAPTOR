# services/video_frame_description_service/kafka_handler.py

import asyncio
import json
import logging
import os
from typing import Dict, Any
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from frame_description_processor import FrameDescriptionProcessor
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_RESPONSE,
    KAFKA_TOPIC_DLQ,
    FRAME_DESCRIPTION_RESULTS_DIR,
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
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
logger = logging.getLogger(__name__)

class FrameDescriptionKafkaHandler:
    def __init__(self):
        self.processing_states = {}  # 追蹤處理狀態
        self.processor = FrameDescriptionProcessor()
        # 確保結果目錄存在
        os.makedirs(FRAME_DESCRIPTION_RESULTS_DIR, exist_ok=True)
    
    async def start_consumer(self):
        """啟動 Kafka 消費者"""
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            enable_auto_commit=True,
            auto_offset_reset='latest'
        )
        
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
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
        """處理接收到的消息"""
        topic = message.topic
        data = message.value
        
        try:
            if topic == KAFKA_TOPIC_REQUEST:
                await self.handle_frame_description_request(data, producer)
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            await self.send_error_response(data, producer, str(e), "MESSAGE_PROCESSING_ERROR")
    
    async def handle_frame_description_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理影片事件摘要請求"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            request_id = payload["request_id"]
            primary_filename = parameters.get("primary_filename")
            scene_json_file_path = parameters.get("scene_json_file_path")
            scene_output_directory = parameters.get("scene_output_directory")
            video_path = parameters.get("video_path")
            
            # 驗證必要參數
            if not scene_json_file_path or not os.path.exists(scene_json_file_path):
                raise ValueError(f"Scene JSON file not found: {scene_json_file_path}")
            
            if not scene_output_directory or not os.path.exists(scene_output_directory):
                raise ValueError(f"Scene output directory not found: {scene_output_directory}")
            
            if not primary_filename:
                raise ValueError("Primary filename is required")
            
            logger.info(f"Starting event summary processing for request: {request_id}")
            logger.info(f"Scene JSON file: {scene_json_file_path}")
            logger.info(f"Scene directory: {scene_output_directory}")
            
            # 執行事件摘要處理
            result_file_path = await self.processor.process_frame_descriptions(
                scene_json_path=scene_json_file_path,
                video_path=video_path,
                scene_output_directory=scene_output_directory,
                request_id=request_id,
                primary_filename=primary_filename
            )
            
            # 讀取處理結果以獲取統計資訊
            processing_stats = await self.get_processing_stats(result_file_path)
            
            # 發送成功響應
            response = MessageBuilder.create_processing_response(
                original_message=message,
                status="SUCCESS",
                file_path=result_file_path,
                processing_info={
                    "scene_images_used": processing_stats.get("scene_images_used", 0),
                    "images_processed": processing_stats.get("images_processed", 0),
                    "model_used": processing_stats.get("model", ""),
                    "processing_service": SERVICE_NAME,
                    "event_summary_generated": True,
                    "prompt_type": "event_summary"
                }
            )
            
            await producer.send(KAFKA_TOPIC_RESPONSE, response)
            
            logger.info(f"Event summary processing completed for request: {request_id}")
            logger.info(f"Results saved to: {result_file_path}")
            
        except Exception as e:
            logger.error(f"Event summary request failed: {e}")
            await self.send_error_response(message, producer, str(e), "EVENT_SUMMARY_PROCESSING_FAILED")

    
    async def get_processing_stats(self, result_file_path: str) -> Dict[str, Any]:
        """從結果檔案中獲取處理統計資訊"""
        try:
            import aiofiles
            async with aiofiles.open(result_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                result_data = json.loads(content)
            
            return {
                "scene_frames_used": result_data.get("scene_frames_used", 0),
                "frames_processed": result_data.get("processing_info", {}).get("frames_processed", 0),
                "model": result_data.get("processing_info", {}).get("model", ""),
                "input_size": result_data.get("processing_info", {}).get("input_size", 0),
                "max_patches": result_data.get("processing_info", {}).get("max_patches", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing stats from {result_file_path}: {e}")
            return {}
    
    async def send_error_response(
        self, 
        original_message: Dict[str, Any], 
        producer: AIOKafkaProducer,
        error_message: str,
        error_code: str
    ):
        """發送錯誤響應"""
        try:
            error_response = MessageBuilder.create_error_response(
                original_message=original_message,
                error_message=error_message,
                error_code=error_code
            )
            
            await producer.send(KAFKA_TOPIC_DLQ, error_response)
            logger.error(f"Error response sent: {error_message}")
                
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")
