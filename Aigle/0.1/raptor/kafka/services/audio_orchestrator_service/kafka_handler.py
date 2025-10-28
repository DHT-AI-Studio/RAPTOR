# services/audio_orchestrator_service/kafka_handler.py

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
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_CLASSIFIER_REQUEST,
    KAFKA_TOPIC_DIARIZATION_REQUEST,
    KAFKA_TOPIC_RECOGNIZER_REQUEST,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SAVE_QDRANT_REQUEST,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_CLASSIFIER_RESULT,
    KAFKA_TOPIC_DIARIZATION_RESULT,
    KAFKA_TOPIC_RECOGNIZER_RESULT,
    KAFKA_TOPIC_SUMMARY_RESULT,
    KAFKA_TOPIC_SAVE_QDRANT_RESULT,
    KAFKA_TOPIC_DLQ,
    MERGED_FILE_DIR,
    SERVICE_NAME
)
import opencc
from redis_manager import RedisStateManager 
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

class AudioOrchestratorKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.group_id = KAFKA_GROUP_ID
        self.service_name = SERVICE_NAME
        self.seaweedfs_client = SeaweedFSClient()
        self.cc = opencc.OpenCC('s2t')
        os.makedirs(MERGED_FILE_DIR, exist_ok=True)
        # # 處理狀態儲存 (簡單的記憶體儲存)
        # self.processing_states = {}
        self.redis_manager = RedisStateManager() 

    def time_overlap(self, start1, end1, start2, end2):
        """計算時間重疊"""
        return max(0, min(end1, end2) - max(start1, start2))
    
    def get_audio_labels(self, start, end, classification_data, threshold=0.4):
        """
        根據時間範圍獲取音頻標籤
        Args:
            start: 開始時間
            end: 結束時間
            classification_data: 分類數據
            threshold: 概率閾值
        Returns:
            list: 符合條件的標籤列表（去重）
        """
        labels = set()  # 使用set來避免重複
        
        for segment in classification_data:
            # 檢查是否有時間重疊
            overlap = self.time_overlap(start, end, segment['segment_start'], segment['segment_end'])
            if overlap > 0:
                # 添加概率大於閾值的標籤
                for class_info in segment['top_classes']:
                    if class_info['probability'] > threshold:
                        labels.add(class_info['label'])
        
        return list(labels)
    
    def merge_all_data(self, segments, diarization, classification,filename, asset_path, version_id, status):
        """
        整合語音識別、說話人分離和音頻分類數據
        Args:
            segments: 語音識別結果
            diarization: 說話人分離結果
            classification: 音頻分類結果
        Returns:
            list: 整合後的結果
        """
        merged_all = []
        
        for index, seg in enumerate(segments):
            start = seg['start']
            end = seg['end']
            text = seg['text'].strip()
            text = self.cc.convert(text)

            # 計算與每個 speaker segment 的重疊時間
            overlaps = []
            for d in diarization:
                overlap = self.time_overlap(start, end, d['start'], d['end'])
                if overlap > 0:
                    overlaps.append((overlap, d['speaker']))
            
            # 取重疊最多的 speaker（如果有）
            if overlaps:
                speaker = max(overlaps, key=lambda x: x[0])[1]
            else:
                speaker = None
            
            # 獲取音頻標籤
            audio_labels = self.get_audio_labels(start, end, classification)
            
            merged_item = {
                "id": str(uuid.uuid4()),  # 生成 UUID
                "payload": {
                    "audio_id": f"{filename.split('.')[0]}_chunk_{index}",
                    "text": text,
                    "filename": filename,
                    "chunk_index": index,
                    "start_time": str(start),
                    "end_time": str(end),
                    "speaker": speaker,
                    "type": "audios",
                    "audio_labels": audio_labels,
                    "upload_time": datetime.now(timezone.utc).isoformat(),
                    "embedding_type": "text",
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "status": status
                }
            }
            
            merged_all.append(merged_item)
        
        return merged_all
        
    async def load_json_file(self, file_path: str) -> Any:
        """載入 JSON 檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise
    
    async def save_merged_data(self, merged_data: list, request_id: str) -> str:
        """保存合併後的資料到暫存檔案"""
        try:
            merged_file_path = os.path.join(MERGED_FILE_DIR, f"{request_id}_merged.json")
            
            with open(merged_file_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Merged data saved to: {merged_file_path}")
            return merged_file_path
            
        except Exception as e:
            logger.error(f"Failed to save merged data: {e}")
            raise
        
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
            KAFKA_TOPIC_CLASSIFIER_RESULT,
            KAFKA_TOPIC_DIARIZATION_RESULT,
            KAFKA_TOPIC_RECOGNIZER_RESULT,
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
        
        logger.info("Audio Orchestrator Kafka consumers started...")
        
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
        """處理音頻處理請求"""
        async for message in consumer:
            await self.handle_audio_request(message.value, producer)
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_audio_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理音頻處理請求"""
        try:
            logger.info(f"Processing audio request: {message.get('message_id')}")
            
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
            
            # 處理音頻處理請求
            if message["payload"]["action"] == "audio_processing":
                await self.handle_audio_processing(message, producer)
            else:
                await self.send_error_response(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION"
                )
                
        except Exception as e:
            logger.error(f"Error processing audio request: {e}")
            await self.send_to_dlq(producer, message, str(e))
    
    async def handle_audio_processing(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理音頻處理流程"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            # 提取必要資訊
            access_token = payload.get("access_token")
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            primary_filename = parameters.get("primary_filename")
            # status = parameters.get("status")
            
            if not all([access_token, asset_path, version_id, primary_filename]):
                await self.send_error_response(
                    producer, message,
                    "Missing required parameters",
                    "MISSING_PARAMETERS"
                )
                return
            
            # 下載檔案
            logger.info(f"Downloading audio file: {asset_path}/{version_id}/{primary_filename}")
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
                "classifier_result": None,
                "diarization_result": None,
                "recognizer_result": None,
                "summary_result": None,
                "save_result": None,
                "merged_file_path": None,
                "step": "parallel_processing",  # parallel_processing -> summary -> save -> complete
                "created_at": asyncio.get_event_loop().time()
            }
            # 保存到 Redis
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")
            
            # 同時發送三個並行處理請求
            await self.send_parallel_requests(message, producer, temp_file_path, primary_filename)
            
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Audio processing failed: {str(e)}",
                "PROCESSING_FAILED"
            )
    
    async def send_parallel_requests(self, message: Dict[str, Any], producer: AIOKafkaProducer, temp_file_path: str, primary_filename: str):
        """同時發送三個並行處理請求"""
        parameters = message["payload"]["parameters"]
        
        # 創建三個並行請求
        classifier_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="audio_classifier_service",
            action="audio_classification",
            parameters={
                "file_type": parameters.get("file_type", "audio"),
                "primary_filename": primary_filename
            },
            temp_file_path=temp_file_path
        )
        
        diarization_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="audio_diarization_service",
            action="audio_diarization",
            parameters={
                "file_type": parameters.get("file_type", "audio"),
                "primary_filename": primary_filename
            },
            temp_file_path=temp_file_path
        )
        
        recognizer_request = MessageBuilder.create_processing_request(
            original_message=message,
            target_service="audio_recognizer_service",
            action="audio_recognition",
            parameters={
                "file_type": parameters.get("file_type", "audio"),
                "primary_filename": primary_filename
            },
            temp_file_path=temp_file_path
        )
        
        # 同時發送三個請求
        await asyncio.gather(
            producer.send(KAFKA_TOPIC_CLASSIFIER_REQUEST, classifier_request),
            producer.send(KAFKA_TOPIC_DIARIZATION_REQUEST, diarization_request),
            producer.send(KAFKA_TOPIC_RECOGNIZER_REQUEST, recognizer_request)
        )
        
        correlation_id = message.get("correlation_id")
        logger.info(f"Parallel processing requests sent for: {correlation_id}")
    
    async def handle_service_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, topic: str):
        """處理各服務的結果"""
        try:
            payload = message["payload"]
            original_request_id = payload.get("original_request_id")
                
            if not original_request_id or not self.redis_manager.exists(original_request_id):
                logger.warning(f"Unknown request ID: {original_request_id}")
                return
            
            state = self.redis_manager.get_state(original_request_id)
            if not state:
                logger.error(f"Failed to retrieve state for: {original_request_id}")
                return
            # 檢查結果狀態
            if payload.get("status") == "error":
                logger.error(f"Service error for {original_request_id}: {payload.get('error')}")
                # 清理並發送錯誤
                self.seaweedfs_client.cleanup_temp_file(state["temp_file_path"])
                if state.get("merged_file_path"):  
                    self.cleanup_merged_file(state["merged_file_path"])
                # self.redis_manager.delete_state(original_request_id)
                
                error_response = MessageBuilder.create_error_response(
                    original_message=state["original_message"],
                    error_message=f"Service processing failed: {payload.get('error', {}).get('message')}",
                    error_code="SERVICE_ERROR"
                )
                await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
                return
            
            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_CLASSIFIER_RESULT:
                await self.handle_classifier_result(message, producer, state, original_request_id)
            elif topic == KAFKA_TOPIC_DIARIZATION_RESULT:
                await self.handle_diarization_result(message, producer, state, original_request_id)
            elif topic == KAFKA_TOPIC_RECOGNIZER_RESULT:
                await self.handle_recognizer_result(message, producer, state, original_request_id)
            elif topic == KAFKA_TOPIC_SUMMARY_RESULT:
                await self.handle_summary_result(message, producer, state, original_request_id)
            elif topic == KAFKA_TOPIC_SAVE_QDRANT_RESULT:
                await self.handle_save_result(message, producer, state, original_request_id)
                
        except Exception as e:
            logger.error(f"Error handling service result: {e}")
    
    async def handle_classifier_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理分類結果"""
    
        # state["classifier_result"] = message["payload"]
        self.redis_manager.update_state(correlation_id, {"classifier_result": message["payload"]})  # 更新 Redis
        logger.info(f"Classifier result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        await self.check_parallel_completion(producer, state, correlation_id)
    
    async def handle_diarization_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理說話人分離結果"""
        # state["diarization_result"] = message["payload"]
        
        self.redis_manager.update_state(correlation_id, {"diarization_result": message["payload"]})
        logger.info(f"Diarization result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        await self.check_parallel_completion(producer, state, correlation_id)
    
    async def handle_recognizer_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理語音識別結果"""
        # state["recognizer_result"] = message["payload"]
        
        self.redis_manager.update_state(correlation_id, {"recognizer_result": message["payload"]})
        logger.info(f"Recognizer result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        await self.check_parallel_completion(producer, state, correlation_id)
    
    async def check_parallel_completion(self, producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """檢查並行處理是否完成"""
        latest_state = self.redis_manager.get_state(correlation_id)
        if not latest_state:
            logger.error(f"State not found in Redis: {correlation_id}")
            return
        
        if (latest_state["classifier_result"] is not None and 
            latest_state["diarization_result"] is not None and 
            latest_state["recognizer_result"] is not None and
            latest_state["step"] == "parallel_processing"):
            
            try:
                self.redis_manager.update_state(correlation_id, {"step": "summary"})
                
                # 提取必要資訊
                original_message = latest_state["original_message"]
                original_params = original_message["payload"]["parameters"]
                
                # 從各結果中提取檔案路徑
                classifier_result_path = latest_state["classifier_result"].get("result_path")
                diarization_result_path = latest_state["diarization_result"].get("result_path")  
                recognition_result_path = latest_state["recognizer_result"].get("result_path")
                    
                filename = original_params.get("primary_filename")
                asset_path = original_params.get("asset_path")
                version_id = original_params.get("version_id")
                status = original_params.get("status")
                
                if not all([classifier_result_path, diarization_result_path, recognition_result_path]):
                    raise ValueError("Missing result file paths")
    
    
                
                logger.info(f"Extracted paths for {correlation_id}:")
                logger.info(f"  Classifier: {classifier_result_path}")
                logger.info(f"  Diarization: {diarization_result_path}")
                logger.info(f"  Recognition: {recognition_result_path}")
                
                classifier_data = await self.load_json_file(classifier_result_path)
                diarization_data = await self.load_json_file(diarization_result_path)
                recognizer_data = await self.load_json_file(recognition_result_path)
                
                # 提取需要的資料結構
                segments = recognizer_data
                diarization_segments = diarization_data
                classification_segments = classifier_data
                
                # 合併資料
                logger.info(f"Merging audio data for: {correlation_id}")
                merged_data = self.merge_all_data(segments, diarization_segments, classification_segments, filename, asset_path, version_id, status)
                
                # 保存合併後的資料
                merged_file_path = await self.save_merged_data(merged_data, correlation_id)
                self.redis_manager.update_state(correlation_id, {"merged_file_path": merged_file_path})
                
                # 創建簡化的摘要請求
                summary_request = {
                    "message_id": str(uuid.uuid4()),
                    "correlation_id": original_message["correlation_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_service": "audio_orchestrator_service",
                    "target_service": "audio_summary_service",
                    "message_type": "REQUEST",
                    "priority": original_message.get("priority", "MEDIUM"),
                    "payload": {
                        "request_id": str(uuid.uuid4()),
                        "original_request_id": correlation_id,
                        "user_id": original_message["payload"].get("user_id"),
                        "access_token": original_message["payload"].get("access_token"),
                        "action": "audio_summary",
                        "parameters": {
                            "filename": original_params.get("primary_filename"),
                            "asset_path": original_params.get("asset_path"),
                            "version_id": original_params.get("version_id"),
                            "file_path": merged_file_path,
                            "status": original_params.get("status")
                        },
                        "file_path": merged_file_path,
                        "metadata": {
                            "orchestrator_request_id": correlation_id,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "original_metadata": original_message["payload"].get("metadata", {})
                        }
                    },
                    "retry_count": 0,
                    "ttl": original_message.get("ttl", 3600)
                }
                
                await producer.send(KAFKA_TOPIC_SUMMARY_REQUEST, summary_request)
                logger.info(f"Summary request sent with merged data for: {correlation_id}")
                
            except Exception as e:
                logger.error(f"Error in parallel completion processing: {e}")
                # 發送錯誤響應
                error_response = MessageBuilder.create_error_response(
                    original_message=latest_state["original_message"],
                    error_message=f"Data merging failed: {str(e)}",
                    error_code="MERGE_FAILED"
                )
                await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
                
                # 清理資源
                self.seaweedfs_client.cleanup_temp_file(latest_state["temp_file_path"])
                if latest_state.get("merged_file_path"):
                    self.cleanup_merged_file(latest_state["merged_file_path"])
                # self.redis_manager.delete_state(correlation_id)

    def cleanup_merged_file(self, file_path: str):
        """清理合併檔案"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Merged file cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup merged file {file_path}: {e}")
    
    async def handle_summary_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """處理摘要結果"""
        # state["summary_result"] = message["payload"]
        # state["step"] = "save"
        updates = {
            "summary_result": message["payload"],
            "step": "save"
        }
        self.redis_manager.update_state(correlation_id, updates)
        # 從 Redis 獲取最新狀態
        latest_state = self.redis_manager.get_state(correlation_id)
        if not latest_state:
            logger.error(f"Failed to retrieve updated state for: {correlation_id}")
            return
        summary_result = latest_state["summary_result"]
        results = summary_result.get("results", {})
        parameters = results.get("parameters", {})
        summary_result_path = parameters.get("summary_result_path")
        
        if not summary_result_path:
            logger.error(f"No summary_result_path found in summary result for request: {state['original_message']['payload']['request_id']}")
            logger.error(f"Summary result structure: {json.dumps(summary_result, indent=2)}")
            # 發送錯誤響應
            error_response = MessageBuilder.create_error_response(
                original_message=latest_state["original_message"],
                error_message="Summary result path not found",
                error_code="MISSING_SUMMARY_PATH"
            )
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
            
            # 清理資源
            self.seaweedfs_client.cleanup_temp_file(latest_state["temp_file_path"])
            if latest_state.get("merged_file_path"):
                self.cleanup_merged_file(latest_state["merged_file_path"])
            # self.redis_manager.delete_state(correlation_id)
            return
        
        logger.info(f"Summary result path extracted: {summary_result_path}")
        
        original_payload = latest_state["original_message"]["payload"]
        original_params = original_payload["parameters"]
        
        # 發送保存請求，使用 summary_result_path 作為參數
        save_request = MessageBuilder.create_processing_request(
            original_message=latest_state["original_message"],
            target_service="audio_save2qdrant_service",
            action="save_to_qdrant",
            parameters={
                "summary_result_path": summary_result_path,
                "filename": original_params.get("primary_filename"),
                "asset_path": original_params.get("asset_path"),
                "version_id": original_params.get("version_id"),
            },
            temp_file_path=latest_state["temp_file_path"]
        )
        
        await producer.send(KAFKA_TOPIC_SAVE_QDRANT_REQUEST, save_request)
        logger.info(f"Save request sent with summary_result_path: {summary_result_path} for request: {state['original_message']['payload']['request_id']}")
        
    async def handle_save_result(self, message: Dict[str, Any], producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        
        """處理保存結果"""
        try:
            # 更新 Redis 狀態
            self.redis_manager.update_state(correlation_id, {
                "save_result": message["payload"],
                "step": "complete"
            })
            # 從 Redis 獲取最新狀態
            latest_state = self.redis_manager.get_state(correlation_id)
            if not latest_state:
                logger.error(f"Failed to retrieve final state for: {correlation_id}")
                return
            # 創建最終結果消息
            final_result = create_final_result_message(
                original_message=latest_state["original_message"],
                classifier_result=latest_state["classifier_result"],
                diarization_result=latest_state["diarization_result"],
                recognizer_result=latest_state["recognizer_result"],
                summary_result=latest_state["summary_result"],
                save_result=latest_state["save_result"]
            )
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, final_result)
            logger.info(f"Final result sent for: {correlation_id}")

            # 從 message 中提取 result_path
            result_path = message["payload"].get("result_path")
            if not result_path:
                logger.warning(f"Result path not found or doesn't exist: {result_path}")
                # 直接清理資源並返回
                self.seaweedfs_client.cleanup_temp_file(latest_state.get("temp_file_path"))
                if latest_state.get("merged_file_path"):
                    self.cleanup_merged_file(latest_state["merged_file_path"])
                return
            
    
            # 載入 result_path 的 JSON 檔案
            logger.info(f"Loading result file: {result_path}")
            result_data = await self.load_json_file(result_path)
        
            # 提取 summary 和組合 text
            summary = None
            text_parts = []
            
            for item in result_data:
                payload = item.get("payload", {})
                
                # 第一個項目是 summary
                if payload.get("embedding_type") == "summary":
                    summary = payload.get("summary")
                    logger.info(f"Summary extracted for: {correlation_id}")
                
                # 提取並組合 text (只處理 embedding_type 為 text 的項目)
                elif payload.get("embedding_type") == "text":
                    if all(k in payload for k in ["text", "start_time", "end_time", "speaker"]):
                        start_time = payload["start_time"]
                        end_time = payload["end_time"]
                        speaker = payload["speaker"]
                        text = payload["text"]
                        
                        # 格式: "start_time-end_time": "speaker": "text"
                        text_part = f'"{start_time}-{end_time}sec" "{speaker}": "{text}"'
                        text_parts.append(text_part)
        
            # 組合所有 text
            combined_text = " ".join(text_parts)
            
            logger.info(f"Extracted {len(text_parts)} text segments for: {correlation_id}")
            simplified_state = {
                "step": "complete",
                "summary": summary,
                "text": combined_text
            }
            if not self.redis_manager.set_state(correlation_id, simplified_state):
                logger.error(f"Failed to update simplified state in Redis for: {correlation_id}")
            else:
                logger.info(f"Redis state simplified for: {correlation_id}")
            # 清理資源
            self.seaweedfs_client.cleanup_temp_file(latest_state.get("temp_file_path"))
            if latest_state.get("merged_file_path"):
                self.cleanup_merged_file(latest_state["merged_file_path"])
            
            logger.info(f"Audio processing completed for: {correlation_id}")
            
        except Exception as e:
            logger.error(f"Error in handle_save_result: {e}")
            # 如果是在發送 final_result 之前出錯,才發送錯誤響應
            if 'final_result' not in locals():
                await self.send_error_response(
                    producer, state.get("original_message"),
                    f"Failed to process save result: {str(e)}",
                    "SAVE_RESULT_PROCESSING_FAILED"
                )
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
            from datetime import datetime, timezone
            
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
    
    async def send_error_response(
        self, 
        producer: AIOKafkaProducer, 
        message: Dict[str, Any], 
        error_message: str, 
        error_code: str
    ):
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
    
    async def send_to_dlq(
        self, 
        producer: AIOKafkaProducer, 
        message: Dict[str, Any], 
        error: str
    ):
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

