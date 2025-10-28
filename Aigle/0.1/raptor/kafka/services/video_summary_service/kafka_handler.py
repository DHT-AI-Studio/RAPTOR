# services/video_summary_service/kafka_handler.py

import json
import uuid
import asyncio
import logging
import concurrent.futures
import os
from typing import Dict, Any
from datetime import datetime
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from video_summary import VideoSummary
from message_utils import MessageBuilder
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SUMMARY_RESULT,
    KAFKA_TOPIC_DLQ,
    SUMMARY_RESULTS_DIR,
    #OLLAMA_URL,
    #OLLAMA_MODEL,
    VIDEO_SUMMARY_CONFIG
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

class VideoSummaryKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.request_topic = KAFKA_TOPIC_SUMMARY_REQUEST
        self.result_topic = KAFKA_TOPIC_SUMMARY_RESULT
        self.dlq_topic = KAFKA_TOPIC_DLQ
        self.summarizer = VideoSummary(
            model_name=OLLAMA_MODEL,
            max_tokens_per_batch=VIDEO_SUMMARY_CONFIG["max_tokens_per_batch"],
            max_summary_length=VIDEO_SUMMARY_CONFIG["max_summary_length"],
            ollama_url=OLLAMA_URL
        )
        self.service_name = "video_summary_service"
        
        # 簡化線程池配置
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="VideoSummary"
        )
        
        logger.info("Video Summary Kafka Handler initialized")
        
    async def start_consumer(self):
        """啟動 Kafka Consumer"""
        consumer = AIOKafkaConsumer(
            self.request_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="video-summary-service-group",
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        
        await consumer.start()
        await producer.start()
        
        logger.info("Video Summary Kafka consumer started, waiting for messages...")
        
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
            
            # 處理視頻摘要請求
            if message["payload"]["action"] == "video_summary":
                summary_results = await self.handle_video_summary(message)
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

    async def handle_video_summary(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """處理視頻摘要請求"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self.summarize_video_sync,
                message["payload"],
                message
            )
            return result
            
        except Exception as e:
            logger.error(f"Video summary error: {e}")
            return {
                "request_id": message["payload"].get("request_id"),
                "action": "video_summary",
                "status": "error",
                "error": {
                    "code": "VIDEO_SUMMARY_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }

    def summarize_video_sync(self, request_payload: Dict[str, Any], kafka_message: Dict[str, Any] = None) -> Dict[str, Any]:
        """同步版本的視頻摘要方法"""
        try:
            request_id = request_payload["request_id"]
            parameters = request_payload.get("parameters", {})
            filename = parameters.get("filename", "unknown")
            asset_path = parameters.get("asset_path", "")
            version_id = parameters.get("version_id", "")
            status = parameters.get("status", "")
            
            # 獲取視頻和音頻分析結果路徑
            video_analysis_results = parameters.get("video_analysis_results", {})
            audio_analysis_results = parameters.get("audio_analysis_results", {})
            
            video_analysis_path = video_analysis_results.get("merged_analysis", {}).get("merged_file_path")
            audio_analysis_path = audio_analysis_results.get("merged_analysis", {}).get("merged_file_path")
            
            if not video_analysis_path or not audio_analysis_path:
                return {
                    "request_id": request_id,
                    "action": "video_summary",
                    "status": "error",
                    "error": {
                        "code": "MISSING_ANALYSIS_PATHS",
                        "message": "Missing video or audio analysis file paths",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 檢查檔案是否存在
            if not os.path.exists(video_analysis_path):
                return {
                    "request_id": request_id,
                    "action": "video_summary",
                    "status": "error",
                    "error": {
                        "code": "VIDEO_ANALYSIS_FILE_NOT_FOUND",
                        "message": f"Video analysis file not found: {video_analysis_path}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            if not os.path.exists(audio_analysis_path):
                return {
                    "request_id": request_id,
                    "action": "video_summary",
                    "status": "error",
                    "error": {
                        "code": "AUDIO_ANALYSIS_FILE_NOT_FOUND",
                        "message": f"Audio analysis file not found: {audio_analysis_path}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            logger.info(f"Processing video summary for: {filename}")
            logger.info(f"Video analysis path: {video_analysis_path}")
            logger.info(f"Audio analysis path: {audio_analysis_path}")
            
            # 處理摘要
            start_time = datetime.now()
            summary_text = self.summarizer.process_summary(video_analysis_path, audio_analysis_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 檢查摘要結果
            if summary_text.startswith("錯誤:") or summary_text.startswith("摘要生成失敗"):
                return {
                    "request_id": request_id,
                    "action": "video_summary",
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
                f"{request_id}_{filename}.json"
            )
            
            try:
                # 讀取原始音頻分析數據
                with open(audio_analysis_path, 'r', encoding='utf-8') as f:
                    original_audio_data = json.load(f)
                
                # 確保 original_audio_data 是列表
                if not isinstance(original_audio_data, list):
                    logger.error(f"Expected list format, got {type(original_audio_data)}")
                    return {
                        "request_id": request_id,
                        "action": "video_summary",
                        "status": "error",
                        "error": {
                            "code": "INVALID_DATA_FORMAT",
                            "message": f"Expected list format, got {type(original_audio_data).__name__}",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                
                # 創建摘要項目
                summary_id = str(uuid.uuid4())
                summary_item = {
                    "id": summary_id,
                    "payload": {
                        "filename": filename,
                        "type": "videos",
                        "upload_time": datetime.now().isoformat(),
                        "embedding_type": "summary",
                        "asset_path": asset_path,
                        "version_id": version_id,
                        "status": status,
                        "summary": summary_text
                    }
                }

                
                converted_audio_data = self.convert_audio_data_to_standard_format(
                original_audio_data, filename, asset_path, version_id, status
                )
                
                # 將摘要項目添加到列表開頭
                new_data = [summary_item] + converted_audio_data
                os.makedirs(os.path.dirname(summary_result_path), exist_ok=True)
                # 保存為新的 JSON 文件
                with open(summary_result_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Added summary item to the beginning. Total items: {len(new_data)}")
                logger.info(f"Video summary result saved to: {summary_result_path}")
                
            except Exception as json_error:
                logger.error(f"Error creating JSON summary result: {json_error}")
                return {
                    "request_id": request_id,
                    "action": "video_summary",
                    "status": "error",
                    "error": {
                        "code": "JSON_CREATION_FAILED",
                        "message": f"Failed to create JSON summary result: {str(json_error)}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 計算檔案大小
            file_size_bytes = os.path.getsize(summary_result_path)
            summary_length = len(summary_text)
            
            # 返回成功結果
            return {
                "request_id": request_id,
                "action": "video_summary",
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
            logger.error(f"Error generating video summary: {e}")
            return {
                "request_id": request_payload.get("request_id"),
                "action": "video_summary",
                "status": "error",
                "error": {
                    "code": "VIDEO_SUMMARY_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    def convert_audio_data_to_standard_format(self, original_audio_data: list, filename: str, asset_path: str, version_id: str, status: str) -> list:
        """將原始音頻數據轉換為標準格式"""
        converted_data = []
        current_time = datetime.now().isoformat()
        
        for index, item in enumerate(original_audio_data):
            # 檢查是否已經是標準格式
            if 'id' in item and 'payload' in item:
                # 已經是標準格式，直接添加
                converted_data.append(item)
                continue
            
            # 轉換舊格式為新格式
            item_id = str(uuid.uuid4())
            
            # 提取數據
            start_time = item.get('start_time', 0)
            end_time = item.get('end_time', 0)
            speaker = item.get('speaker', 'SPEAKER_00')
            text = item.get('text', '')
            audio_labels = item.get('audio_labels', ['Speech'])
            
            # 確保時間是字符串格式
            if isinstance(start_time, (int, float)):
                start_time = str(start_time)
            if isinstance(end_time, (int, float)):
                end_time = str(end_time)
            
            # 創建標準格式項目
            standard_item = {
                "id": item_id,
                "payload": {
                    "audio_id": f"{filename}_chunk_{index}",
                    "text": text,
                    "filename": filename,
                    "chunk_index": index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker": speaker,
                    "type": "videos",
                    "audio_labels": audio_labels,
                    "upload_time": current_time,
                    "embedding_type": "text",
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "status": status
                }
            }
            
            converted_data.append(standard_item)
        
        logger.info(f"Converted {len(original_audio_data)} audio items to standard format")
        return converted_data

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
