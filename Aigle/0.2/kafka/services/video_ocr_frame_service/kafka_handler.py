# services/video_ocr_frame_service/kafka_handler.py

import asyncio
import json
import logging
import os
from typing import Dict, Any
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from ocr_frame import VideoOCRFrameProcessor
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_RESPONSE,
    KAFKA_TOPIC_DLQ,
    OCR_OUTPUT_BASE_DIR,
    OCR_FRAMES_BASE_DIR,
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

class OCRFrameKafkaHandler:
    def __init__(self):
        self.processing_states = {}  # 追蹤處理狀態
        # 確保結果目錄存在
        os.makedirs(OCR_OUTPUT_BASE_DIR, exist_ok=True)
        os.makedirs(OCR_FRAMES_BASE_DIR, exist_ok=True)
    
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
                await self.handle_ocr_frame_request(data, producer)
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            await self.send_error_response(data, producer, str(e), "MESSAGE_PROCESSING_ERROR")
    
    async def handle_ocr_frame_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理 OCR 幀請求"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            request_id = payload["request_id"]
            primary_filename = parameters.get("primary_filename")
            scene_output_directory = parameters.get("scene_output_directory")
            
            if not scene_output_directory or not os.path.exists(scene_output_directory):
                raise ValueError(f"Scene output directory not found: {scene_output_directory}")
            
            if not primary_filename:
                raise ValueError("Primary filename is required")
            
            logger.info(f"Starting OCR frame processing for request: {request_id}")
            logger.info(f"Scene directory: {scene_output_directory}")
            
            # 創建 OCR 輸出目錄
            ocr_frames_dir = os.path.join(
                OCR_FRAMES_BASE_DIR, 
                f"{request_id}_{primary_filename}_ocr_frames"
            )
            os.makedirs(ocr_frames_dir, exist_ok=True)
            
            # 執行 OCR 處理
            ocr_processor = VideoOCRFrameProcessor(output_dir=ocr_frames_dir)
            
            processing_result = await ocr_processor.process_scene_frames(
                request_id=request_id,
                primary_filename=primary_filename,
                scene_output_directory=scene_output_directory
            )
            
            ocr_results = processing_result.get("ocr_results", {})
            ocr_detected_count = len([r for r in ocr_results.values() if r.get("text", "")])
            
            # 保存結果到 JSON 檔案
            result_filename = f"ocr_frame_result_{request_id}_{primary_filename}.json"
            result_file_path = os.path.join(OCR_OUTPUT_BASE_DIR, result_filename)
            
            await self.save_results_to_json(processing_result, result_file_path)

            
            # 發送成功響應
            response = MessageBuilder.create_processing_response(
                original_message=message,
                status="SUCCESS",
                result_file_path=result_file_path,
                ocr_frames_dir=ocr_frames_dir,
                total_scenes=len(ocr_results),
                ocr_detected_count=ocr_detected_count
            )
            
            await producer.send(KAFKA_TOPIC_RESPONSE, response)
            
            logger.info(f"OCR frame processing completed for request: {request_id}")
            logger.info(f"Processed {len(ocr_results)} scenes")
            logger.info(f"OCR detected text in {ocr_detected_count} scenes")
            logger.info(f"Results saved to: {result_file_path}")
            logger.info(f"OCR frames saved to: {ocr_frames_dir}")
            
        except Exception as e:
            logger.error(f"OCR frame request failed: {e}")
            await self.send_error_response(message, producer, str(e), "OCR_FRAME_PROCESSING_FAILED")
    
    async def save_results_to_json(self, results: Dict[str, Any], file_path: str):
        """異步保存 OCR 結果到 JSON 檔案"""
        try:
            import aiofiles
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(results, indent=2, ensure_ascii=False))
            
            logger.info(f"OCR results saved to JSON file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results to JSON file {file_path}: {e}")
            raise
    
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
