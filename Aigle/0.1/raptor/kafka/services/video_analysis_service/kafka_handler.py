# services/video_analysis_service/kafka_handler.py

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from frame_extraction import FrameExtractor
from result_merger import ResultMerger 
from message_utils import MessageBuilder
from redis_manager import RedisStateManager
from datetime import datetime
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_SCENE_DETECTION_REQUEST,
    KAFKA_TOPIC_OCR_FRAME_REQUEST,
    KAFKA_TOPIC_FRAME_DESCRIPTION_REQUEST,  # 新增
    KAFKA_TOPIC_RESPONSE,
    KAFKA_TOPIC_DLQ,
    KAFKA_TOPIC_SCENE_DETECTION_RESULT,
    KAFKA_TOPIC_OCR_FRAME_RESULT,
    KAFKA_TOPIC_FRAME_DESCRIPTION_RESULT,  # 新增
    STATE_TIMEOUT,
    MAX_RETRY_COUNT,
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

class VideoAnalysisKafkaHandler:
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.result_merger = ResultMerger()
        self.redis_manager = RedisStateManager()  # 追蹤處理狀態
    
    async def start_consumer(self):
        """啟動 Kafka 消費者"""
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            KAFKA_TOPIC_SCENE_DETECTION_RESULT,
            KAFKA_TOPIC_OCR_FRAME_RESULT,
            KAFKA_TOPIC_FRAME_DESCRIPTION_RESULT,  # 新增
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
                await self.handle_video_analysis_request(data, producer)
            elif topic == KAFKA_TOPIC_SCENE_DETECTION_RESULT:
                await self.handle_scene_detection_result(data, producer)
            elif topic == KAFKA_TOPIC_OCR_FRAME_RESULT:
                await self.handle_ocr_frame_result(data, producer)
            elif topic == KAFKA_TOPIC_FRAME_DESCRIPTION_RESULT:  # 新增
                await self.handle_frame_description_result(data, producer)
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            await self.send_error_response(data, producer, str(e), "MESSAGE_PROCESSING_ERROR")
    
    async def handle_video_analysis_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理影片分析請求"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            request_id = payload["request_id"]
            video_file_path = parameters.get("video_file_path") or parameters.get("file_path")
            primary_filename = parameters.get("primary_filename")
            
            if not video_file_path or not os.path.exists(video_file_path):
                raise ValueError(f"Video file not found: {video_file_path}")
            
            if not primary_filename:
                raise ValueError("Primary filename is required")
            
            logger.info(f"Starting frame extraction for request: {request_id}")
            
            # 進行幀提取（現在是異步方法）
            extraction_result = await self.frame_extractor.extract_frames(
                video_path=video_file_path,
                request_id=request_id,
                filename=primary_filename
            )
            
            # 保存處理狀態到 Redis
            correlation_id = message["correlation_id"]
            state = {
                "original_message": message,
                "step": "frame_extraction_done",
                "frame_extraction_result": extraction_result,
                "scene_detection_result": None,
                "ocr_frame_result": None,
                "frame_description_result": None,
                "temp_cleanup_paths": [extraction_result["output_directory"]]
            }
        
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")
            
            # 發送場景檢測請求
            await self.send_scene_detection_request(message, producer, extraction_result)
            
            logger.info(f"Frame extraction completed and scene detection request sent for: {request_id}")
            
        except Exception as e:
            logger.error(f"Video analysis request failed: {e}")
            await self.send_error_response(message, producer, str(e), "FRAME_EXTRACTION_FAILED")
    
    async def send_scene_detection_request(
        self, 
        original_message: Dict[str, Any], 
        producer: AIOKafkaProducer,
        extraction_result: Dict[str, Any]
    ):
        """發送場景檢測請求"""
        payload = original_message["payload"]
        parameters = payload["parameters"]
        
        scene_detection_request = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="video_scene_detection_service",
            action="scene_detection",
            parameters={
                "primary_filename": parameters.get("primary_filename"),
                "frame_directory": extraction_result["output_directory"],
                "total_frames": extraction_result["total_frames_extracted"],
                "video_info": extraction_result["video_info"]
            },
            file_path=extraction_result["output_directory"]
        )
        
        await producer.send(KAFKA_TOPIC_SCENE_DETECTION_REQUEST, scene_detection_request)
        logger.info(f"Scene detection request sent for: {payload['request_id']}")
    
    async def handle_scene_detection_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理場景檢測結果"""
        correlation_id = message["correlation_id"]
        
        state = self.redis_manager.get_state(correlation_id)
        if state is None:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return
        payload = message["payload"]
        
        # 提取場景檢測結果的路徑資訊
        scene_result = {
            "status": payload["parameters"]["status"],
            "scenes_count": payload["parameters"].get("scenes_count", 0),
            "json_file_path": payload["file_path"],
            "scene_output_directory": payload["parameters"].get("scene_output_directory"),
            "diff_plot_path": payload["parameters"].get("diff_plot_path")
        }
        
        updates = {
            "scene_detection_result": scene_result,
            "step": "scene_detection_done"
        }
        if not self.redis_manager.update_state(correlation_id, updates):
            raise Exception("Failed to update state in Redis")
        
        logger.info(f"Scene detection result received for: {correlation_id}, scenes: {scene_result['scenes_count']}")
        state = self.redis_manager.get_state(correlation_id)
        
        # 並行發送 OCR 和 Frame Description 請求
        await self.send_parallel_frame_analysis_requests(state["original_message"], producer, state)
        
    async def send_parallel_frame_analysis_requests(
        self, 
        original_message: Dict[str, Any], 
        producer: AIOKafkaProducer,
        state: Dict[str, Any]
    ):
        """並行發送 OCR 和 Frame Description 請求"""
        payload = original_message["payload"]
        parameters = payload["parameters"]
        
        extraction_result = state["frame_extraction_result"]
        scene_result = state["scene_detection_result"]
        
        # 創建 OCR 請求
        ocr_frame_request = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="video_ocr_frame_service",
            action="ocr_frame",
            parameters={
                "primary_filename": parameters.get("primary_filename"),
                "frame_directory": extraction_result["output_directory"],
                "total_frames": extraction_result["total_frames_extracted"],
                "scene_json_file_path": scene_result["json_file_path"],
                "scene_output_directory": scene_result["scene_output_directory"]
            },
            file_path=extraction_result["output_directory"]
        )


        video_file_path = parameters.get("video_file_path") or parameters.get("file_path")
        
        # 創建 Frame Description 請求
        frame_description_request = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="video_frame_description_service",
            action="frame_description",
            parameters={
                "primary_filename": parameters.get("primary_filename"),
                "frame_directory": extraction_result["output_directory"],
                "total_frames": extraction_result["total_frames_extracted"],
                "scene_json_file_path": scene_result["json_file_path"],
                "scene_output_directory": scene_result["scene_output_directory"],
                "video_path": video_file_path
            },
            file_path=extraction_result["output_directory"]
        )
        
        # 並行發送兩個請求
        await asyncio.gather(
            producer.send(KAFKA_TOPIC_OCR_FRAME_REQUEST, ocr_frame_request),
            producer.send(KAFKA_TOPIC_FRAME_DESCRIPTION_REQUEST, frame_description_request)
        )
        
        logger.info(f"OCR and Frame Description requests sent for: {payload['request_id']}")
    
    async def handle_ocr_frame_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理 OCR 幀結果"""
        correlation_id = message["correlation_id"]
        
        state = self.redis_manager.get_state(correlation_id)
        if state is None:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return
        # 更新 Redis 狀態
        updates = {"ocr_frame_result": message["payload"]}
        if not self.redis_manager.update_state(correlation_id, updates):
            logger.error(f"Failed to update OCR result in Redis for: {correlation_id}")
            return
        state = self.redis_manager.get_state(correlation_id)
        
        logger.info(f"OCR frame result received for: {correlation_id}")
        
        # 檢查是否所有分析都完成
        await self.check_analysis_completion(producer, state, correlation_id)
    
    async def handle_frame_description_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理 Frame Description 結果"""
        correlation_id = message["correlation_id"]
                
        state = self.redis_manager.get_state(correlation_id)
        if state is None:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return
        
        # 更新 Redis 狀態
        updates = {"frame_description_result": message["payload"]}
        if not self.redis_manager.update_state(correlation_id, updates):
            logger.error(f"Failed to update frame description result in Redis for: {correlation_id}")
            return
        
        # 重新獲取最新狀態
        state = self.redis_manager.get_state(correlation_id)
        
        logger.info(f"Frame description result received for: {correlation_id}")
        
        # 檢查是否所有分析都完成
        await self.check_analysis_completion(producer, state, correlation_id)
    
    async def check_analysis_completion(self, producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """檢查分析是否完成，如果完成則發送最終結果"""
        if (state["scene_detection_result"] is not None and 
            state["ocr_frame_result"] is not None and
            state["frame_description_result"] is not None):  # 修改條件
            
            try:
                # 獲取原始請求資訊
                original_message = state["original_message"]
                payload = original_message["payload"]
                request_id = payload["request_id"]
                primary_filename = payload["parameters"].get("primary_filename")
                
                # 檢查是否有任何錯誤
                has_errors = False
                error_messages = []
                
                for service_name, result in [
                    ("scene_detection", state["scene_detection_result"]),
                    ("ocr_frame", state["ocr_frame_result"]),
                    ("frame_description", state["frame_description_result"])
                ]:
                    if isinstance(result, dict) and result.get("status") == "error":
                        has_errors = True
                        error_messages.append(f"{service_name}: {result.get('error', 'Unknown error')}")
                
                if has_errors:
                    error_message = "; ".join(error_messages)
                    await self.send_error_response(
                        original_message, 
                        producer, 
                        f"Video analysis failed: {error_message}", 
                        "ANALYSIS_FAILED"
                    )
                    return
                
                # 獲取三個結果的檔案路徑
                scene_json_path = state["scene_detection_result"]["json_file_path"]
                ocr_json_path = state["ocr_frame_result"]["file_path"]
                frame_description_json_path = state["frame_description_result"]["file_path"]
                
                # 檢查檔案路徑是否都存在
                if not all([scene_json_path, ocr_json_path, frame_description_json_path]):
                    raise ValueError("Missing result file paths from one or more services")
                
                # 合併結果（現在包含三個結果）
                merged_file_path = await self.result_merger.merge_all_video_results(
                    scene_json_path=scene_json_path,
                    ocr_json_path=ocr_json_path,
                    frame_description_json_path=frame_description_json_path,
                    request_id=request_id,
                    primary_filename=primary_filename
                )
                
                # 合併所有分析結果（包含合併後的檔案路徑）
                combined_results = {
                    "frame_extraction": {
                        "output_directory": state["frame_extraction_result"]["output_directory"],
                        "total_frames_extracted": state["frame_extraction_result"]["total_frames_extracted"],
                        "video_info": state["frame_extraction_result"]["video_info"]
                    },
                    "scene_detection": state["scene_detection_result"],
                    "ocr_frame": {
                        "status": state["ocr_frame_result"]["parameters"]["status"],
                        "result_file_path": state["ocr_frame_result"]["file_path"],
                        "total_scenes": state["ocr_frame_result"]["parameters"].get("total_scenes", 0),
                        "ocr_detected_count": state["ocr_frame_result"]["parameters"].get("ocr_detected_count", 0)
                    },
                    "frame_description": {
                        "status": state["frame_description_result"]["parameters"]["status"],
                        "result_file_path": state["frame_description_result"]["file_path"],
                        "total_descriptions": state["frame_description_result"]["parameters"].get("total_descriptions", 0),
                        "processing_service": "video_frame_description_service"
                    },
                    "merged_analysis": {
                        "merged_file_path": merged_file_path,
                        "format": "List of {frame_index, timestamp, text, frame_description} objects"
                    }
                }
                
                # 發送最終結果
                response = MessageBuilder.create_processing_response(
                    original_message=state["original_message"],
                    status="SUCCESS",
                    results=combined_results,
                    file_path=merged_file_path  # 主要結果檔案路徑
                )
                
                await producer.send(KAFKA_TOPIC_RESPONSE, response)
                
                # correlation_id = state["original_message"]["correlation_id"]
                logger.info(f"Video analysis completed successfully for: {correlation_id}")
                logger.info(f"Merged results saved to: {merged_file_path}")
                
                # 清理狀態和臨時檔案
                await self.cleanup_processing_state(correlation_id)
                
            except Exception as e:
                logger.error(f"Error in analysis completion: {e}")
                await self.send_error_response(
                    state["original_message"], 
                    producer, 
                    str(e), 
                    "ANALYSIS_COMPLETION_FAILED"
                )
    
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
            
            # 清理相關狀態
            correlation_id = original_message.get("correlation_id")
            if correlation_id and self.redis_manager.exists(correlation_id):
                await self.cleanup_processing_state(correlation_id)
                
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")
    
    async def cleanup_processing_state(self, correlation_id: str, keep_redis: bool = True):
        """
        清理處理狀態和臨時檔案
        
        Args:
            correlation_id: 關聯 ID
            keep_redis: 是否保留 Redis 資料 (True=保留讓它自然過期, False=立即刪除)
        """
        state = self.redis_manager.get_state(correlation_id)
        if state:
            cleanup_paths = state.get("temp_cleanup_paths", [])
            for path in cleanup_paths:
                self.frame_extractor.cleanup_frames(path)
            
            # ✅ 根據參數決定是否刪除 Redis
            if not keep_redis:
                self.redis_manager.delete_state(correlation_id)
                logger.info(f"Processing state deleted from Redis for: {correlation_id}")
            else:
                # 更新狀態為已完成,但保留在 Redis 中
                updates = {
                    "step": "completed",
                    "completed_at": datetime.now().isoformat()
                }
                self.redis_manager.update_state(correlation_id, updates)
                logger.info(f"Processing state kept in Redis (will expire) for: {correlation_id}")