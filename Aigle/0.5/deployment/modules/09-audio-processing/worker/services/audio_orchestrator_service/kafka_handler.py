# services/audio_orchestrator_service/kafka_handler.py

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import opencc
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from api_client import ClaimOutcome, SeaweedFSClient
from capture_time import read_capture_time
from config import (
    ASSET_MANAGEMENT_URL,
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_CLASSIFIER_REQUEST,
    KAFKA_TOPIC_CLASSIFIER_RESULT,
    KAFKA_TOPIC_DIARIZATION_REQUEST,
    KAFKA_TOPIC_DIARIZATION_RESULT,
    KAFKA_TOPIC_DLQ,
    KAFKA_TOPIC_FINAL_RESULT,
    KAFKA_TOPIC_INDEX_COMPLETED,
    KAFKA_TOPIC_INDEXER_REQUEST,
    KAFKA_TOPIC_INDEXER_RESULT,
    KAFKA_TOPIC_RECOGNIZER_REQUEST,
    KAFKA_TOPIC_RECOGNIZER_RESULT,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_SUMMARY_REQUEST,
    KAFKA_TOPIC_SUMMARY_RESULT,
    MERGED_FILE_DIR,
    SERVICE_NAME,
)
from dotenv import load_dotenv
from message_utils import MessageBuilder, create_final_result_message
from redis_manager import RedisStateManager

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
        self.cc = opencc.OpenCC('s2twp')
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
    
    def merge_all_data(self, segments, diarization, classification, filename, asset_path, version_id, status, branch_id=""):
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
                    "source": filename.rsplit('.', 1)[-1].lower(),
                    "audio_labels": audio_labels,
                    "upload_time": datetime.now(timezone.utc).isoformat(),
                    "embedding_type": "text",
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "status": status,
                    "branch_id": branch_id
                }
            }
            
            merged_all.append(merged_item)
        
        return merged_all
        
    async def load_json_file(self, file_path: str) -> Any:
        """載入 JSON 檔案"""
        try:
            return await asyncio.to_thread(self._load_json_file_sync, file_path)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise

    @staticmethod
    def _load_json_file_sync(file_path: str) -> Any:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def save_merged_data(self, merged_data: list, request_id: str) -> str:
        """保存合併後的資料到暫存檔案"""
        try:
            merged_file_path = os.path.join(MERGED_FILE_DIR, f"{request_id}_merged.json")

            await asyncio.to_thread(self._save_merged_data_sync, merged_file_path, merged_data)

            logger.info(f"Merged data saved to: {merged_file_path}")
            return merged_file_path

        except Exception as e:
            logger.error(f"Failed to save merged data: {e}")
            raise

    @staticmethod
    def _save_merged_data_sync(merged_file_path: str, merged_data: list) -> None:
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    async def start_consumer(self):
        """啟動 Kafka Consumer"""
        # 建立 consumers
        request_consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            enable_auto_commit=False,  # VIE01-255: manual commit drives claim-based ack/redeliver
        )
        
        result_consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_CLASSIFIER_RESULT,
            KAFKA_TOPIC_DIARIZATION_RESULT,
            KAFKA_TOPIC_RECOGNIZER_RESULT,
            KAFKA_TOPIC_SUMMARY_RESULT,
            KAFKA_TOPIC_INDEXER_RESULT,
            KAFKA_TOPIC_INDEX_COMPLETED,
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
    
    def _claim_ids_from_message(self, message: dict[str, Any]) -> tuple[str, str, str, str, str] | None:
        """VIE01-255: pull (asset_path, version_id, correlation_id, user_id,
        branch_id) out of a request message, or None if it doesn't carry
        enough to claim/report status on."""
        payload = message.get("payload", {})
        parameters = payload.get("parameters", {})
        headers = payload.get("asset_managemant_download_header", {})
        asset_path = parameters.get("asset_path")
        version_id = parameters.get("version_id")
        correlation_id = message.get("correlation_id")
        user_id = headers.get("X-User-ID")
        branch_id = headers.get("X-Branch-ID")
        if not all([asset_path, version_id, correlation_id, user_id, branch_id]):
            return None
        return asset_path, version_id, correlation_id, user_id, branch_id

    async def process_requests(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理音頻處理請求"""
        async for message in consumer:
            ids = self._claim_ids_from_message(message.value)
            if ids is None:
                # Malformed message -- nothing to claim; let the existing
                # validate_message()-driven error path in handle_audio_request run.
                await self.handle_audio_request(message.value, producer)
                await consumer.commit()
                continue

            asset_path, version_id, correlation_id, user_id, branch_id = ids
            outcome = await self.seaweedfs_client.claim_processing(
                asset_path, version_id, correlation_id, user_id, branch_id
            )
            if outcome is ClaimOutcome.CONFLICT:
                logger.info(f"Claim 409 for {correlation_id}: discarding, no processing")
                await consumer.commit()
                continue
            if outcome is ClaimOutcome.TRANSIENT_ERROR:
                logger.warning(f"Claim failed (transient) for {correlation_id}: leaving uncommitted for redelivery")
                continue  # do NOT commit -- next poll/rebalance redelivers this message

            await self.handle_audio_request(message.value, producer)
            await consumer.commit()
    
    async def process_results(self, consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
        """處理各服務的結果"""
        async for message in consumer:
            await self.handle_service_result(message.value, producer, message.topic)
    
    async def handle_audio_request(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理音頻處理請求"""
        try:
            logger.info(f"Processing audio request: {message.get('message_id')}")
            
            # 驗證消息格式
            if not self.validate_message(message):
                await self.send_error_response(
                    producer, message,
                    "Invalid message format",
                    "INVALID_FORMAT",
                    message.get("correlation_id")
                )
                return

            # 檢查 TTL
            if self.is_message_expired(message):
                await self.send_error_response(
                    producer, message,
                    "Message expired",
                    "MESSAGE_EXPIRED",
                    message.get("correlation_id")
                )
                return

            # 檢查目標服務
            if message["target_service"] != self.service_name:
                await self.send_error_response(
                    producer, message,
                    f"Wrong target service: {message['target_service']}",
                    "WRONG_TARGET",
                    message.get("correlation_id")
                )
                return

            # 處理音頻處理請求
            if message["payload"]["action"] == "audio_processing":
                await self.handle_audio_processing(message, producer)
            else:
                await self.send_error_response(
                    producer, message,
                    f"Unknown action: {message['payload']['action']}",
                    "UNKNOWN_ACTION",
                    message.get("correlation_id")
                )
                
        except Exception as e:  # noqa: BLE001 -- dispatch spans validation/redis/kafka failures; already reports error downstream
            logger.error(f"Error processing audio request: {e}")
            # Must go through send_error_response (KAFKA_TOPIC_FINAL_RESULT), not
            # send_to_dlq (KAFKA_TOPIC_DLQ) — nothing consumes the DLQ topic, so
            # a failure sent there vanishes silently instead of reaching whatever
            # is waiting for this job's result.
            await self.send_error_response(
                producer, message, str(e), "DISPATCH_ERROR", message.get("correlation_id")
            )
    
    async def handle_audio_processing(self, message: dict[str, Any], producer: AIOKafkaProducer):
        """處理音頻處理流程"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            # 提取必要資訊
            download_headers = payload.get("asset_managemant_download_header", {})
            download_params = payload.get("asset_managemant_download_paramater", {})
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            primary_filename = parameters.get("primary_filename")
            download_url = f"{ASSET_MANAGEMENT_URL}/filedownload/{asset_path}/{version_id}"

            if not all([download_headers, asset_path, version_id, primary_filename]):
                await self.send_error_response(
                    producer, message,
                    "Missing required parameters",
                    "MISSING_PARAMETERS",
                    message.get("correlation_id")
                )
                return

            # 如果 Module 13 已存到 NFS，直接用本地路徑；否則退回 LakeFS 下載
            nfs_path = parameters.get("temp_file_path")
            if nfs_path and os.path.exists(nfs_path):
                temp_file_path = nfs_path
                logger.info(f"Using NFS file directly: {temp_file_path}")
            else:
                if nfs_path:
                    logger.warning(f"NFS path not found ({nfs_path}), falling back to download")
                logger.info(f"Downloading audio file via asset management API: {download_url}")
                temp_file_path = await self.seaweedfs_client.download_file(
                    download_url=download_url,
                    download_headers=download_headers,
                    download_params=download_params,
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
                "branch_id": parameters.get("branch_id", ""),
                "created_at": asyncio.get_event_loop().time()
            }
            # 保存到 Redis
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")
            
            # 同時發送三個並行處理請求
            await self.send_parallel_requests(message, producer, temp_file_path, primary_filename)
            
        except Exception as e:  # noqa: BLE001 -- download/redis/kafka failures mixed; already reports error downstream
            logger.error(f"Error in audio processing: {e}")
            await self.send_error_response(
                producer, message,
                f"Audio processing failed: {e!s}",
                "PROCESSING_FAILED",
                message.get("correlation_id")
            )
    
    async def send_parallel_requests(self, message: dict[str, Any], producer: AIOKafkaProducer, temp_file_path: str, primary_filename: str):
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
    
    async def handle_service_result(self, message: dict[str, Any], producer: AIOKafkaProducer, topic: str):
        """處理各服務的結果"""
        state = None
        # envelope field, not payload["original_request_id"] -- matches 10/11/12
        correlation_id = message.get("correlation_id")
        try:
            payload = message["payload"]
            if not correlation_id:
                logger.warning("Missing correlation_id in result message")
                return

            state = self.redis_manager.get_state(correlation_id)
            if not state:
                logger.warning(f"Unknown correlation_id: {correlation_id}")
                return

            # 檢查結果狀態 (also catches "failed", not just "error")
            message_type = message.get("message_type", "")
            service_status = payload.get("status", "")
            if message_type == "ERROR" or service_status in ("error", "failed"):
                # create_error_response() puts "error" at the top level of
                # payload; create_summary_result_message() (also used on this
                # same topic when summary_results["status"] is "error") nests
                # the whole results dict -- including its own "error" key --
                # one level deeper under "results". Check both so a
                # guardrail-block or other in-process failure doesn't fall
                # through to a message-less "Service processing failed: None".
                error_info = payload.get("error") or payload.get("results", {}).get("error") or {}
                error_message = error_info.get('message', 'Unknown service error')
                logger.error(f"Service error for {correlation_id}: {error_info}")
                await self.handle_processing_error(producer, state, f"Service processing failed: {error_message}", "SERVICE_ERROR")
                return

            # 根據 topic 處理不同的結果
            if topic == KAFKA_TOPIC_CLASSIFIER_RESULT:
                await self.handle_classifier_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_DIARIZATION_RESULT:
                await self.handle_diarization_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_RECOGNIZER_RESULT:
                await self.handle_recognizer_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_SUMMARY_RESULT:
                await self.handle_summary_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_INDEXER_RESULT:
                await self.handle_save_result(message, producer, state, correlation_id)
            elif topic == KAFKA_TOPIC_INDEX_COMPLETED:
                await self.handle_index_completed(message, producer, state, correlation_id)

        except Exception as e:  # noqa: BLE001 -- dispatch spans redis/file-IO/kafka failures; already reports error downstream
            logger.error(f"Error handling service result: {e}")
            # An exception here used to leave the asset stuck in "processing"
            # forever -- no failure ever got reported to Module 04, unlike
            # the sibling branch above (a downstream service explicitly
            # reporting "error"/"failed") which does call
            # handle_processing_error(). state.get("original_message") guards
            # the one case handle_processing_error() itself can't recover
            # from -- it needs that field too.
            if state and state.get("original_message"):
                try:
                    await self.handle_processing_error(producer, state, f"Unexpected error: {e}", "UNEXPECTED_ERROR")
                except Exception as inner_e:  # noqa: BLE001 -- last-resort: failure of the error-reporting path itself, already logged
                    logger.error(f"Also failed to report processing error: {inner_e}")
            else:
                logger.error(f"Cannot report failure for {correlation_id} -- state missing or has no original_message")

    async def handle_classifier_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理分類結果"""
    
        # state["classifier_result"] = message["payload"]
        self.redis_manager.update_state(correlation_id, {"classifier_result": message["payload"]})  # 更新 Redis
        logger.info(f"Classifier result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        await self.check_parallel_completion(producer, state, correlation_id)
    
    async def handle_diarization_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理說話人分離結果"""
        # state["diarization_result"] = message["payload"]
        
        self.redis_manager.update_state(correlation_id, {"diarization_result": message["payload"]})
        logger.info(f"Diarization result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        await self.check_parallel_completion(producer, state, correlation_id)
    
    async def handle_recognizer_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """處理語音識別結果"""
        # state["recognizer_result"] = message["payload"]
        
        self.redis_manager.update_state(correlation_id, {"recognizer_result": message["payload"]})
        logger.info(f"Recognizer result received for: {state['original_message']['payload']['request_id']}")
        
        # 檢查是否所有並行結果都完成
        await self.check_parallel_completion(producer, state, correlation_id)
    
    async def check_parallel_completion(self, producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
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
                branch_id = original_params.get("branch_id", "")

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
                merged_data = self.merge_all_data(segments, diarization_segments, classification_segments, filename, asset_path, version_id, status, branch_id)
                # P13: when the audio was recorded, as its container says -- Module 25 dates the
                # recording's "yesterday" / "last week" from it (after validating it).
                capture_time, capture_time_source = await asyncio.to_thread(
                    read_capture_time, latest_state.get("temp_file_path") or "")
                if capture_time:
                    for item in merged_data:
                        item["payload"]["capture_time"] = capture_time
                        item["payload"]["capture_time_source"] = capture_time_source
                
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
                            "status": original_params.get("status"),
                            "branch_id": original_params.get("branch_id", "")
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
                
            except Exception as e:  # noqa: BLE001 -- file-IO/JSON/dict merge/kafka failures mixed; already reports error downstream
                logger.error(f"Error in parallel completion processing: {e}")
                await self.handle_processing_error(producer, latest_state, f"Data merging failed: {e!s}", "MERGE_FAILED")

    def cleanup_merged_file(self, file_path: str):
        """清理合併檔案"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Merged file cleaned up: {file_path}")
        except OSError as e:
            logger.error(f"Failed to cleanup merged file {file_path}: {e}")
    
    async def handle_summary_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
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
            await self.handle_processing_error(producer, latest_state, "Summary result path not found", "MISSING_SUMMARY_PATH")
            return
        
        logger.info(f"Summary result path extracted: {summary_result_path}")

        # Pre-read summary file while it still exists (indexer deletes it after indexing)
        _summary_pre = None
        _text_parts_pre = []
        try:
            _result_data_pre = await asyncio.to_thread(self._load_json_file_sync, summary_result_path)
            for _item in _result_data_pre:
                _p = _item.get("payload", {})
                if _p.get("embedding_type") == "summary":
                    _summary_pre = _p.get("summary")
                elif _p.get("embedding_type") == "text" and all(k in _p for k in ["text", "start_time", "end_time", "speaker"]):
                    _text_parts_pre.append(
                        f'"{_p["start_time"]}-{_p["end_time"]}sec" "{_p["speaker"]}": "{_p["text"]}"'
                    )
        except (OSError, json.JSONDecodeError) as _read_err:
            logger.warning(f"Could not pre-read summary file: {_read_err}")
        self.redis_manager.update_state(correlation_id, {
            "summary": _summary_pre,
            "text": " ".join(_text_parts_pre)
        })

        original_payload = latest_state["original_message"]["payload"]
        original_params = original_payload["parameters"]
        
        # 發送保存請求，使用 summary_result_path 作為參數
        save_request = MessageBuilder.create_processing_request(
            original_message=latest_state["original_message"],
            target_service="audio_indexer_service",
            action="document_indexing",
            parameters={
                "summary_result_path": summary_result_path,
                "filename": original_params.get("primary_filename"),
                "asset_path": original_params.get("asset_path"),
                "version_id": original_params.get("version_id"),
            },
            temp_file_path=latest_state["temp_file_path"]
        )
        
        await producer.send(KAFKA_TOPIC_INDEXER_REQUEST, save_request)
        logger.info(f"Save request sent with summary_result_path: {summary_result_path} for request: {state['original_message']['payload']['request_id']}")
        
    async def handle_save_result(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):

        """處理保存結果 -- this module's own job is done here, but Module 25 has
        only just been asked to index, not confirmed to have finished -- step
        stays "indexing" and update_asset_status(active) waits for
        handle_index_completed() (VIE01-255 update_status() race fix)."""
        try:
            # 更新 Redis 狀態
            self.redis_manager.update_state(correlation_id, {
                "save_result": message["payload"],
                "step": "indexing"
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

            # Read pre-cached summary and text from Redis (indexer deletes the file before responding)
            summary = latest_state.get("summary")
            combined_text = latest_state.get("text", "")

            simplified_state = {
                "step": "indexing",
                "summary": summary,
                "text": combined_text,
                "branch_id": latest_state.get("branch_id", ""),
                # set_state() below replaces the whole stored value -- original_message
                # must survive the trim: handle_index_completed() needs it to flip
                # status later, and handle_processing_error() needs it on any
                # failure path from here on (including from handle_index_completed()).
                "original_message": latest_state["original_message"],
            }
            if not self.redis_manager.set_state(correlation_id, simplified_state):
                logger.error(f"Failed to update simplified state in Redis for: {correlation_id}")
            else:
                logger.info(f"Redis state simplified for: {correlation_id}")
            # 清理資源
            self.seaweedfs_client.cleanup_temp_file(latest_state.get("temp_file_path"))
            if latest_state.get("merged_file_path"):
                self.cleanup_merged_file(latest_state["merged_file_path"])

            logger.info(f"Audio processing saved (awaiting Module 25 indexing) for: {correlation_id}")

        except Exception as e:  # noqa: BLE001 -- redis/dict/kafka failures mixed; already reports error downstream
            logger.error(f"Error in handle_save_result: {e}")
            # 如果是在發送 final_result 之前出錯,才發送錯誤響應
            if 'final_result' not in locals():
                await self.send_error_response(
                    producer, state.get("original_message"),
                    f"Failed to process save result: {e!s}",
                    "SAVE_RESULT_PROCESSING_FAILED",
                    correlation_id
                )

    async def handle_index_completed(self, message: dict[str, Any], producer: AIOKafkaProducer, state: dict[str, Any], correlation_id: str):
        """Module 25 has actually finished indexing this asset (personal-index-completed) --
        only now is it safe to mark the job complete and flip the asset active
        (VIE01-255 update_status() race fix: this used to fire right after
        handle_save_result() merely enqueued the index-request)."""
        try:
            final_state = {
                "step": "complete",
                "summary": state.get("summary"),
                "text": state.get("text", ""),
                "branch_id": state.get("branch_id", "")
            }
            if not self.redis_manager.set_state(correlation_id, final_state):
                logger.error(f"Failed to update final state in Redis for: {correlation_id}")
            else:
                logger.info(f"Successfully updated final state in Redis for: {correlation_id}")

            ids = self._claim_ids_from_message(state["original_message"])
            if ids:
                asset_path, version_id, _cid, user_id, branch_id = ids
                await self.seaweedfs_client.update_asset_status(user_id, branch_id, asset_path, version_id, "active")

            logger.info(f"Audio processing completed for: {correlation_id}")

        except KeyError as e:
            logger.error(f"Error in handle_index_completed for {correlation_id}: {e}")
            await self.handle_processing_error(producer, state, str(e), "INDEX_COMPLETED_ERROR")

    async def handle_processing_error(self, producer: AIOKafkaProducer, state: dict[str, Any], error_message: str, error_code: str):
        """處理錯誤並清理資源"""
        correlation_id = state["original_message"].get("correlation_id")

        try:
            self.redis_manager.update_state(correlation_id, {
                "step": "error",
                "error_code": error_code,
                "error_message": error_message,
            })
        except TypeError as redis_err:
            logger.error(f"Failed to mark Redis state as error for {correlation_id}: {redis_err}")

        error_response = MessageBuilder.create_error_response(
            original_message=state["original_message"],
            error_message=error_message,
            error_code=error_code
        )
        await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)

        ids = self._claim_ids_from_message(state["original_message"])
        if ids:
            asset_path, version_id, _cid, user_id, branch_id = ids
            await self.seaweedfs_client.update_asset_status(user_id, branch_id, asset_path, version_id, "failed")

        self.seaweedfs_client.cleanup_temp_file(state.get("temp_file_path"))
        if state.get("merged_file_path"):
            self.cleanup_merged_file(state["merged_file_path"])

        logger.info(f"Processing error handled for: {correlation_id} (code={error_code})")

    def validate_message(self, message: dict[str, Any]) -> bool:
        """驗證消息格式"""
        required_fields = [
            "message_id", "correlation_id", "timestamp",
            "source_service", "target_service", "message_type",
            "priority", "payload", "retry_count", "ttl"
        ]

        if not all(field in message for field in required_fields):
            return False

        payload = message.get("payload", {})
        if payload.get("action") == "audio_processing":
            required_payload_fields = ["request_id", "action", "parameters"]
            if not all(field in payload for field in required_payload_fields):
                return False

            parameters = payload.get("parameters", {})
            required_param_fields = ["asset_path", "version_id", "primary_filename"]
            if not all(field in parameters for field in required_param_fields):
                return False

        return True

    def is_message_expired(self, message: dict[str, Any]) -> bool:
        """檢查消息是否過期"""
        try:
            timestamp = datetime.fromisoformat(message["timestamp"].replace('Z', '+00:00'))
            ttl = message.get("ttl", 3600)
            now = datetime.now(timezone.utc)
            return (now - timestamp).total_seconds() > ttl
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Error checking message expiration, treating as not expired: {e}")
            return False
    
    async def send_error_response(
        self,
        producer: AIOKafkaProducer,
        original_message: dict[str, Any],
        error_message: str,
        error_code: str,
        correlation_id: str | None = None
    ):
        """發送錯誤響應"""
        try:
            error_response = MessageBuilder.create_error_response(
                original_message=original_message,
                error_message=error_message,
                error_code=error_code
            )
            await producer.send(KAFKA_TOPIC_FINAL_RESULT, error_response)
            logger.warning(f"Error response sent: {error_response['message_id']}")

            if correlation_id:
                payload = original_message.get("payload", {})
                parameters = payload.get("parameters", {})
                branch_id = (
                    parameters.get("branch_id")
                    or payload.get("asset_managemant_download_header", {}).get("X-Branch-ID")
                    or ""
                )

                ids = self._claim_ids_from_message(original_message)
                if ids:
                    asset_path, version_id, _cid, user_id, req_branch_id = ids
                    await self.seaweedfs_client.update_asset_status(user_id, req_branch_id, asset_path, version_id, "failed")

                self.redis_manager.set_state(correlation_id, {
                    "step": "error",
                    "error_code": error_code,
                    "error_message": error_message,
                    "branch_id": branch_id,
                })
        except Exception as e:  # noqa: BLE001 -- last-resort fallback for the error-reporting path itself; already logged
            logger.error(f"Failed to send error response: {e}")
    
    async def send_to_dlq(self, producer: AIOKafkaProducer, original_message: dict[str, Any], error: str):
        """發送消息到 DLQ"""
        try:
            dlq_message = MessageBuilder.create_dlq_message(
                original_message=original_message,
                error=error,
                final_retry_count=original_message.get("retry_count", 0)
            )
            await producer.send(KAFKA_TOPIC_DLQ, dlq_message)
            logger.error(f"Message sent to DLQ: {dlq_message['message_id']}")
        except KafkaError as e:
            logger.error(f"Failed to send message to DLQ: {e}")

