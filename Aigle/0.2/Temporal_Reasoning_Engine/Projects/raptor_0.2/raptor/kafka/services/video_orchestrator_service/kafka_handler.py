# services/video_orchestrator_service/kafka_handler.py

import os
import json
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from api_client import SeaweedFSClient
from message_utils import MessageBuilder, create_final_result_message
from redis_manager import RedisStateManager
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_VIDEO_ANALYSIS_REQUEST,
    KAFKA_TOPIC_AUDIO_ANALYSIS_REQUEST,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SAVE_QDRANT_REQUEST,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT,
    KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT,
    KAFKA_TOPIC_SUMMARY_RESULT,
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

class VideoOrchestratorKafkaHandler:
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
            KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT,
            KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT,
            KAFKA_TOPIC_SUMMARY_RESULT,
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
        
        logger.info("Video Orchestrator Kafka consumers started...")
        
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
        """處理影片處理請求"""
        async for message in consumer:
            await self.handle_video_request(message.value, producer)
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_video_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理影片處理請求"""
        try:
            logger.info(f"Processing video request: {message.get('message_id')}")
            
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
            
            # 處理影片處理請求
            if message["payload"]["action"] == "video_processing":
                await self.handle_video_processing(message, producer)
            else:
                await self.send_error_response(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION"
                )
                
        except Exception as e:
            logger.error(f"Error processing video request: {e}")
            await self.send_to_dlq(producer, message, str(e))

    async def handle_video_processing(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理影片處理流程"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            # 提取必要資訊
            access_token = payload.get("access_token")
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            primary_filename = parameters.get("primary_filename")
            
            if not all([access_token, asset_path, version_id, primary_filename]):
                await self.send_error_response(
                    producer, message,
                    "Missing required parameters",
                    "MISSING_PARAMETERS"
                )
                return
            
            # 下載檔案
            logger.info(f"Downloading video file: {asset_path}/{version_id}/{primary_filename}")
            temp_file_path = await self.seaweedfs_client.download_file(
                access_token=access_token,
                asset_path=asset_path,
                version_id=version_id,
                filename=primary_filename
            )
            
            # 初始化處理狀態
            correlation_id = message.get("correlation_id")
            state = {
                "original_message": message,
                "temp_file_path": temp_file_path,
                "video_analysis_result": None,
                "audio_analysis_result": None,
                "summary_result": None,
                "save_result": None,
                "step": "parallel_analysis",
                "created_at": asyncio.get_event_loop().time()
            }
            
            # 存入 Redis
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")
            
            # 同時發送影片和音頻分析請求
            await self.send_parallel_analysis_requests(message, producer, temp_file_path, primary_filename)
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Video processing failed: {str(e)}",
                "PROCESSING_FAILED"
            )

    async def send_parallel_analysis_requests(self, message: Dict[str, Any], producer: AIOKafkaProducer, temp_file_path: str, primary_filename: str):
        """同時發送影片和音頻分析請求"""
        parameters = message["payload"]["parameters"]
        
        # 創建影片分析請求
        video_analysis_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="video_analysis_service",
            action="video_analysis",
            parameters={
                "file_type": parameters.get("file_type", "video"),
                "primary_filename": primary_filename,
                "video_file_path": temp_file_path  # 影片檔案路徑
            },
            temp_file_path=temp_file_path
        )
        
        # 創建音頻分析請求 (從影片中提取音頻)
        audio_analysis_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="audio_analysis_service",
            action="audio_analysis",
            parameters={
                "file_type": "video",  # 從影片提取音頻
                "primary_filename": primary_filename,
                "video_file_path": temp_file_path  # 影片檔案路徑，用於提取音頻
            },
            temp_file_path=temp_file_path
        )
        
        # 並行發送兩個分析請求
        await asyncio.gather(
            producer.send(KAFKA_TOPIC_VIDEO_ANALYSIS_REQUEST, video_analysis_request),
            producer.send(KAFKA_TOPIC_AUDIO_ANALYSIS_REQUEST, audio_analysis_request)
        )
        
        correlation_id = message.get("correlation_id")
        logger.info(f"Parallel analysis requests sent for: {correlation_id}")
        
    async def handle_service_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, topic: str):
        """處理各服務的結果"""
        try:
            payload = message["payload"]
            correlation_id = message.get("correlation_id")
                
            # 從 Redis 獲取狀態
            state = self.redis_manager.get_state(correlation_id)
            if not state:
                logger.warning(f"Unknown correlation ID: {correlation_id}")
                return

            results = payload.get("results", {})
            status = results.get("status")
            # 檢查結果狀態
            if status == "error":  
                error_info = results.get("error", {})
                error_message = error_info.get("message", "Unknown error")
                logger.error(f"Service error for {correlation_id}: {error_message}")
                # 清理並發送錯誤
                await self.handle_processing_error(producer, state, error_message, "SERVICE_ERROR")
                return
            
            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_VIDEO_ANALYSIS_RESULT:
                await self.handle_video_analysis_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_AUDIO_ANALYSIS_RESULT:
                await self.handle_audio_analysis_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_SUMMARY_RESULT:
                await self.handle_summary_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_SAVE_QDRANT_RESULT:
                await self.handle_save_result(message, producer, state, correlation_id)
                
        except Exception as e:
            logger.error(f"Error handling service result: {e}")
            logger.error(f"Message: {json.dumps(message, indent=2)}")

    async def handle_video_analysis_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理影片分析結果"""
        updates = {"video_analysis_result": message["payload"]}
        if not self.redis_manager.update_state(correlation_id, updates):
            raise Exception("Failed to update state in Redis")
        
        logger.info(f"Video analysis result received for: {state['original_message']['payload']['request_id']}")

        state = self.redis_manager.get_state(correlation_id)
        await self.check_analysis_completion(producer, state, correlation_id)

    async def handle_audio_analysis_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理音頻分析結果"""
        updates = {"audio_analysis_result": message["payload"]}
        if not self.redis_manager.update_state(correlation_id, updates):
            raise Exception("Failed to update state in Redis")
        
        logger.info(f"Audio analysis result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        state = self.redis_manager.get_state(correlation_id)
        await self.check_analysis_completion(producer, state, correlation_id)


    async def check_analysis_completion(self, producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """檢查分析是否完成，如果完成則發送摘要請求"""
        if (state["video_analysis_result"] is not None and 
            state["audio_analysis_result"] is not None and
            state["step"] == "parallel_analysis"):
            
            try:
                if not self.redis_manager.update_state(correlation_id, {"step": "summary"}):
                    raise Exception("Failed to update step in Redis")
                
                # 提取必要資訊
                original_message = state["original_message"]
                original_params = original_message["payload"]["parameters"]

                
                # 從分析結果中提取數據路徑
                video_result = state["video_analysis_result"]
                audio_result = state["audio_analysis_result"]
                
                video_merged_path = None
                audio_merged_path = None

                # 從視頻分析結果提取 merged_analysis 路徑
                video_results = video_result.get("metadata", {}).get("results", {})
                if "merged_analysis" in video_results:
                    video_merged_path = video_results["merged_analysis"].get("merged_file_path")
                
                # 從音頻分析結果提取 merged_analysis 路徑
                audio_results = audio_result.get("metadata", {}).get("results", {})
                if "merged_analysis" in audio_results:
                    audio_merged_path = audio_results["merged_analysis"].get("merged_file_path")
                
                # 檢查是否成功提取到路徑
                if not video_merged_path or not audio_merged_path:
                    logger.error(f"Missing merged analysis paths - Video: {video_merged_path}, Audio: {audio_merged_path}")
                    await self.handle_processing_error(
                        producer, state, 
                        "Missing merged analysis file paths", 
                        "MISSING_MERGED_PATHS"
                    )
                    return
                
                # 創建影片摘要請求，只傳送 merged_analysis 路徑
                summary_request = MessageBuilder.create_processing_request(
                    original_message=original_message,
                    target_service="video_summary_service",
                    action="video_summary",
                    parameters={
                        "filename": original_params.get("primary_filename"),
                        "asset_path": original_params.get("asset_path"),
                        "version_id": original_params.get("version_id"),
                        "status": original_params.get("status"),
                        "video_file_path": state["temp_file_path"],
                        # 只傳送 merged_analysis 的檔案路徑
                        "video_analysis_results": {
                            "merged_analysis": {
                                "merged_file_path": video_merged_path
                            }
                        },
                        "audio_analysis_results": {
                            "merged_analysis": {
                                "merged_file_path": audio_merged_path
                            }
                        }
                    },
                    temp_file_path=state["temp_file_path"]
                )
                
                await producer.send(KAFKA_TOPIC_SUMMARY_REQUEST, summary_request)
                logger.info(f"Summary request sent with merged analysis paths - Video: {video_merged_path}, Audio: {audio_merged_path} for: {correlation_id}")
                
            except Exception as e:
                logger.error(f"Error in analysis completion processing: {e}")
                await self.handle_processing_error(producer, state, str(e), "ANALYSIS_COMPLETION_FAILED")

    async def handle_summary_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理摘要結果"""
        updates = {
            "summary_result": message["payload"],
            "step": "save"
        }
        if not self.redis_manager.update_state(correlation_id, updates):
            raise Exception("Failed to update state in Redis")
        state = self.redis_manager.get_state(correlation_id)
        summary_result = state["summary_result"]
        results = summary_result.get("results", {})
        parameters = results.get("parameters", {})
        summary_result_path = parameters.get("summary_result_path")
    

        
        if not summary_result_path:
            logger.error(f"No summary_result_path found in summary result for request: {state['original_message']['payload']['request_id']}")
            logger.error(f"Summary result structure: {json.dumps(summary_result, indent=2)}")
            await self.handle_processing_error(producer, state, "Summary result path not found", "MISSING_SUMMARY_PATH")
            return
        
        logger.info(f"Summary result saved to: {summary_result_path}")
        
        original_payload = state["original_message"]["payload"]
        original_params = original_payload["parameters"]
        
        # 發送保存到 Qdrant 的請求
        save_request = MessageBuilder.create_processing_request(
            original_message=state["original_message"],
            target_service="video_save2qdrant_service",
            action="save_to_qdrant",
            parameters={
                "summary_result_path": summary_result_path,  # 摘要結果檔案路徑
                "filename": original_params.get("primary_filename"),
                "asset_path": original_params.get("asset_path"),
                "version_id": original_params.get("version_id"),
                "file_type": original_params.get("file_type", "video")
            },
            temp_file_path=state["temp_file_path"]  # 原始影片檔案路徑
        )
        
        await producer.send(KAFKA_TOPIC_SAVE_QDRANT_REQUEST, save_request)
        logger.info(f"Save to Qdrant request sent with summary_result_path: {summary_result_path} for request: {state['original_message']['payload']['request_id']}")

            
    async def handle_save_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理保存結果"""
        try:
            save_result = message["payload"]
            result_path = save_result.get("result_path")
            
            if not result_path:
                logger.error(f"No result_path found in save result for: {correlation_id}")
                await self.handle_processing_error(
                    producer, state, 
                    "Result path not found in save result", 
                    "MISSING_RESULT_PATH"
                )
                return
            
            # 讀取 JSON 結果檔案
            if not os.path.exists(result_path):
                logger.error(f"Result file not found: {result_path}")
                await self.handle_processing_error(
                    producer, state,
                    f"Result file not found: {result_path}",
                    "RESULT_FILE_NOT_FOUND"
                )
                return
            
            with open(result_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 提取 summary 和組合 text
            summary = ""
            text_parts = []
            
            for item in result_data:
                payload = item.get("payload", {})
                
                # 提取 summary (只會有一個)
                if "summary" in payload:
                    summary = payload["summary"]
                
                # 提取並組合 text
                if "text" in payload and "start_time" in payload and "end_time" in payload and "speaker" in payload:
                    start_time = payload["start_time"]
                    end_time = payload["end_time"]
                    speaker = payload["speaker"]
                    text = payload["text"]
                    
                    # 格式: "start_time-end_time": "speaker": "text"
                    text_part = f'"{start_time}-{end_time}sec" "{speaker}": "{text}"'
                    text_parts.append(text_part)
            
            
            combined_text = " ".join(text_parts)
            
            # 創建最終結果消息
            final_result = create_final_result_message(
                original_message=state["original_message"],
                video_analysis_result=state.get("video_analysis_result"),
                audio_analysis_result=state.get("audio_analysis_result"),
                summary_result=state.get("summary_result"),
                save_result=save_result
            )
            
            # 發送最終結果
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, final_result)
            logger.info(f"Final result sent for: {correlation_id}")
            # 清理資源
            if "temp_file_path" in state and state["temp_file_path"]:
                self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])
                logger.info(f"Cleaned up temp file: {state['temp_file_path']}")
            
            # 只更新 Redis 中的三個欄位
            simplified_state = {
                "step": "complete",
                "summary": summary,
                "text": combined_text
            }
            
            if not self.redis_manager.set_state(correlation_id, simplified_state):
                raise Exception("Failed to update state in Redis")
            
            logger.info(f"Updated Redis with summary and combined text for: {correlation_id}")
            logger.info(f"Summary length: {len(summary)}, Text parts count: {len(text_parts)}")
            logger.info(f"Video processing completed for: {correlation_id}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse result JSON file: {e}")
            await self.handle_processing_error(
                producer, state,
                f"Failed to parse result JSON: {str(e)}",
                "JSON_PARSE_ERROR"
            )
        except Exception as e:
            logger.error(f"Error in handle_save_result: {e}")
            await self.handle_processing_error(
                producer, state,
                f"Failed to process save result: {str(e)}",
                "SAVE_RESULT_PROCESSING_ERROR"
            )

    async def handle_processing_error(self, producer: AIOKafkaProducer, state: Dict[str, Any], error_message: str, error_code: str):
        """處理錯誤並清理資源"""
        correlation_id = state["original_message"].get("correlation_id")
        # 發送錯誤響應
        error_response = MessageBuilder.create_error_response(
            original_message=state["original_message"],
            error_message=error_message,
            error_code=error_code
        )
        await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
        
        # 清理資源
        self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])
        # self.redis_manager.delete_state(correlation_id)
        logger.info(f"Processing error handled for: {correlation_id}")

    def validate_message(self, message: Dict[str, Any]) -> bool:
        """驗證消息格式"""
        required_fields = [
            "message_id", "correlation_id", "timestamp", 
            "source_service", "target_service", "message_type", 
            "priority", "payload"
        ]
        
        for field in required_fields:
            if field not in message:
                logger.error(f"Missing required field: {field}")
                return False
        
        # 驗證 payload 必要欄位
        payload = message.get("payload", {})
        required_payload_fields = ["request_id", "action", "parameters"]
        
        for field in required_payload_fields:
            if field not in payload:
                logger.error(f"Missing required payload field: {field}")
                return False
        
        return True
    
    def is_message_expired(self, message: Dict[str, Any]) -> bool:
        """檢查消息是否過期"""
        try:
            timestamp_str = message.get("timestamp")
            ttl = message.get("ttl", 3600)
            
            if not timestamp_str:
                return False
            
            message_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            
            age_seconds = (current_time - message_time).total_seconds()
            
            return age_seconds > ttl
            
        except Exception as e:
            logger.error(f"Error checking message expiration: {e}")
            return False
    
    async def send_error_response(self, producer: AIOKafkaProducer, message: Dict[str, Any], error_message: str, error_code: str):
        """發送錯誤響應"""
        try:
            error_response = MessageBuilder.create_error_response(
                original_message=message,
                error_message=error_message,
                error_code=error_code
            )
            
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
            logger.info(f"Error response sent: {error_code} - {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")
    
    async def send_to_dlq(self, producer: AIOKafkaProducer, message: Dict[str, Any], error: str):
        """發送消息到 DLQ"""
        try:
            retry_count = message.get("retry_count", 0)
            dlq_message = MessageBuilder.create_dlq_message(
                original_message=message,
                error=error,
                final_retry_count=retry_count
            )
            
            await producer.send(KAFKA_TOPIC_DLQ, dlq_message)
            logger.info(f"Message sent to DLQ: {message.get('message_id')}")
            
        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {e}")
