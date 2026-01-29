# services/video_scene_detection_service/kafka_handler.py

import asyncio
import json
import logging
import os
import cv2
import numpy as np
import aiofiles
from typing import Dict, Any, List
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from scene_detection import SceneDetector
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_RESPONSE,
    KAFKA_TOPIC_DLQ,
    SCENE_OUTPUT_BASE_DIR,
    SCENE_RESULTS_BASE_DIR,
    SCENE_DETECTION_THRESHOLD,
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

class SceneDetectionKafkaHandler:
    def __init__(self):
        self.processing_states = {}  # 追蹤處理狀態
        # 確保結果目錄存在
        os.makedirs(SCENE_RESULTS_BASE_DIR, exist_ok=True)
    
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
                await self.handle_scene_detection_request(data, producer)
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            await self.send_error_response(data, producer, str(e), "MESSAGE_PROCESSING_ERROR")
    
    async def handle_scene_detection_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理場景檢測請求"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            request_id = payload["request_id"]
            frame_directory = parameters.get("frame_directory")
            primary_filename = parameters.get("primary_filename")
            total_frames = parameters.get("total_frames", 0)
            video_info = parameters.get("video_info", {})
            
            if not frame_directory or not os.path.exists(frame_directory):
                raise ValueError(f"Frame directory not found: {frame_directory}")
            
            if not primary_filename:
                raise ValueError("Primary filename is required")
            
            logger.info(f"Starting scene detection for request: {request_id}")
            logger.info(f"Frame directory: {frame_directory}, Total frames: {total_frames}")
            
            # 載入幀
            frames = await self.load_frames_from_directory(frame_directory)
            
            if not frames:
                raise ValueError(f"No frames found in directory: {frame_directory}")
            
            # 創建場景檢測器輸出目錄
            scene_output_dir = os.path.join(
                SCENE_OUTPUT_BASE_DIR, 
                f"{request_id}_{primary_filename}_scenes"
            )
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # 執行場景檢測
            scene_detector = SceneDetector(
                output_dir=scene_output_dir,
                diff_plot_path="scene_diff_plot.jpg"
            )
            
            # 從 video_info 獲取 extraction_fps
            extraction_fps = video_info.get("extraction_fps", 0.5)
            
            scenes = await scene_detector.detect(
                frames=frames,
                target_fps=extraction_fps,
                threshold=SCENE_DETECTION_THRESHOLD
            )
            
            # 保存結果到 JSON 檔案
            result_filename = f"scene_detection_result_{request_id}_{primary_filename}.json"
            result_file_path = os.path.join(SCENE_RESULTS_BASE_DIR, result_filename)
            
            await self.save_results_to_json(scenes, result_file_path)
            
            # 準備差異圖路徑
            diff_plot_path = os.path.join(scene_output_dir, "scene_diff_plot.jpg")
            
            # 發送成功響應（包含 JSON 檔案路徑和場景圖片目錄路徑）
            response = MessageBuilder.create_processing_response(
                original_message=message,
                status="SUCCESS",
                result_file_path=result_file_path,
                scene_output_dir=scene_output_dir,
                diff_plot_path=diff_plot_path,
                scenes_count=len(scenes)
            )
            
            await producer.send(KAFKA_TOPIC_RESPONSE, response)
            
            logger.info(f"Scene detection completed for request: {request_id}")
            logger.info(f"Detected {len(scenes)} scene changes")
            logger.info(f"Results saved to: {result_file_path}")
            logger.info(f"Scene images saved to: {scene_output_dir}")
            
        except Exception as e:
            logger.error(f"Scene detection request failed: {e}")
            await self.send_error_response(message, producer, str(e), "SCENE_DETECTION_FAILED")
    
    async def save_results_to_json(self, scenes: List[Dict[str, Any]], file_path: str):
        """異步保存場景變化結果到 JSON 檔案（只保存場景陣列）"""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(scenes, indent=2, ensure_ascii=False))
            
            logger.info(f"Scene detection results saved to JSON file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results to JSON file {file_path}: {e}")
            raise
    
    async def load_frames_from_directory(self, frame_directory: str) -> List[np.ndarray]:
        """從目錄載入所有幀"""
        try:
            # 獲取所有幀檔案
            frame_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                import glob
                frame_files.extend(glob.glob(os.path.join(frame_directory, ext)))
            
            # 按檔案名排序
            frame_files.sort()
            
            if not frame_files:
                raise ValueError(f"No frame files found in {frame_directory}")
            
            logger.info(f"Loading {len(frame_files)} frames from {frame_directory}")
            
            # 載入幀
            frames = []
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    # 轉換為 RGB（因為 cv2.imread 載入的是 BGR）
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    logger.warning(f"Failed to load frame: {frame_file}")
            
            logger.info(f"Successfully loaded {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error loading frames from directory {frame_directory}: {e}")
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
