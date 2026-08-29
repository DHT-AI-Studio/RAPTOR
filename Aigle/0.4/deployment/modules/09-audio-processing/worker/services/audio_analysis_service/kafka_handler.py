# services/audio_analysis_service/kafka_handler.py

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from audio_converter import AudioConverter
from result_merger import AudioResultMerger
from message_utils import MessageBuilder
from redis_manager import RedisStateManager
import time
from datetime import datetime, timezone
from config import (
    #KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPIC_REQUEST,
    KAFKA_TOPIC_CLASSIFIER_REQUEST,
    KAFKA_TOPIC_RECOGNIZER_REQUEST,
    KAFKA_TOPIC_DIARIZATION_REQUEST,
    KAFKA_TOPIC_RESPONSE,
    KAFKA_TOPIC_CLASSIFIER_RESULT,
    KAFKA_TOPIC_RECOGNIZER_RESULT,
    KAFKA_TOPIC_DIARIZATION_RESULT,
    STATE_TIMEOUT,
    MAX_RETRY_COUNT,
    SERVICE_NAME,
    AUDIO_MERGED_RESULTS_DIR,
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
from dotenv import load_dotenv
import os
# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

class AudioAnalysisKafkaHandler:
    def __init__(self):
        self.audio_converter = AudioConverter()
        self.result_merger = AudioResultMerger()
        self.redis_manager = RedisStateManager() 
    
    async def start_consumer(self):
        """啟動 Kafka 消費者"""
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            KAFKA_TOPIC_CLASSIFIER_RESULT,
            KAFKA_TOPIC_RECOGNIZER_RESULT,
            KAFKA_TOPIC_DIARIZATION_RESULT,
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
            self.redis_manager.close()
    
    async def handle_message(self, message, producer: AIOKafkaProducer):
        """處理接收到的消息"""
        topic = message.topic
        data = message.value
        
        try:
            if topic == KAFKA_TOPIC_REQUEST:
                await self.handle_audio_analysis_request(data, producer)
            elif topic == KAFKA_TOPIC_CLASSIFIER_RESULT:
                await self.handle_classifier_result(data, producer)
            elif topic == KAFKA_TOPIC_RECOGNIZER_RESULT:
                await self.handle_recognizer_result(data, producer)
            elif topic == KAFKA_TOPIC_DIARIZATION_RESULT:
                await self.handle_diarization_result(data, producer)
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            await self.send_error_response(data, producer, str(e), "MESSAGE_PROCESSING_ERROR")
    
    async def handle_audio_analysis_request(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理音頻分析請求"""
        try:
            payload = message["payload"]
            parameters = payload["parameters"]
            
            request_id = payload["request_id"]
            file_path = parameters.get("video_file_path") or parameters.get("file_path")
            primary_filename = parameters.get("primary_filename")
            
            if not file_path or not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            if not primary_filename:
                raise ValueError("Primary filename is required")
            
            logger.info(f"Starting audio analysis for request: {request_id}")
            
            # 音頻轉換（如果是影片則提取音頻；無音軌影片回傳 None）
            audio_file_path = self.audio_converter.extract_audio_if_video(file_path)

            if audio_file_path is None:
                logger.warning(f"No audio track in video, returning empty result: {file_path}")
                empty_response = MessageBuilder.create_processing_response(
                    original_message=message,
                    status="SUCCESS",
                    results={"no_audio": True, "merged_file_path": None},
                    file_path=None
                )
                await producer.send(KAFKA_TOPIC_RESPONSE, empty_response)
                logger.info(f"Empty audio result sent for: {request_id}")
                return

            audio_info = self.audio_converter.get_audio_info(audio_file_path)

            logger.info(f"Audio conversion completed: {audio_file_path}")
            logger.info(f"Audio info: duration={audio_info.get('duration', 0):.2f}s, "
                       f"sample_rate={audio_info.get('sample_rate', 0)}")

            # 按 moments 切割 ASR 音頻分段（若有）
            moments = parameters.get("moments", [])
            chunk_infos = []  # [{start_offset, path}, ...]
            if moments:
                base = os.path.splitext(audio_file_path)[0]
                for i, moment in enumerate(moments):
                    start = float(moment.get("start_time", 0))
                    end = float(moment.get("end_time", 0))
                    if end > start:
                        chunk_path = f"{base}_asr_chunk_{i}.wav"
                        self.audio_converter.extract_audio_segment(audio_file_path, start, end, chunk_path)
                        chunk_infos.append({"start_offset": start, "path": chunk_path})
                logger.info(f"Extracted {len(chunk_infos)} ASR audio chunks for: {request_id}")

            # 保存處理狀態
            correlation_id = message["correlation_id"]
            cleanup_paths = [audio_file_path] if audio_file_path != file_path else []
            cleanup_paths += [c["path"] for c in chunk_infos]
            state = {
                "original_message": message,
                "step": "audio_conversion_done",
                "audio_file_path": audio_file_path,
                "audio_info": audio_info,
                "classifier_result": None,
                "recognizer_result": None,
                "diarization_result": None,
                "temp_cleanup_paths": cleanup_paths,
                "recognizer_chunks_total": len(chunk_infos),
                "recognizer_chunk_results": [],
                "recognizer_chunk_request_map": {},
                "created_at": time.time()
            }
            if not self.redis_manager.set_state(correlation_id, state):
                raise Exception("Failed to save state to Redis")

            # 並行發送分析請求（recognizer 按 chunk 分段送）
            await self.send_parallel_analysis_requests(message, producer, audio_file_path, audio_info, chunk_infos)
            
            logger.info(f"Audio analysis requests sent for: {request_id}")
            
        except Exception as e:
            logger.error(f"Audio analysis request failed: {e}")
            await self.send_error_response(message, producer, str(e), "AUDIO_CONVERSION_FAILED")
    
    async def send_parallel_analysis_requests(
        self,
        original_message: Dict[str, Any],
        producer: AIOKafkaProducer,
        audio_file_path: str,
        audio_info: Dict[str, Any],
        chunk_infos: Optional[list] = None
    ):
        """並行發送音頻分析請求。
        classifier/diarization 送全段；recognizer 若有 chunk_infos 則送 N 段，否則送全段。
        """
        payload = original_message["payload"]
        parameters = payload["parameters"]
        primary_filename = parameters.get("primary_filename")
        correlation_id = original_message["correlation_id"]

        base_parameters = {
            "primary_filename": primary_filename,
            "file_path": audio_file_path,
            "audio_info": audio_info,
            "asset_path": parameters.get("asset_path"),
            "version_id": parameters.get("version_id")
        }

        tasks = []

        # Classifier — 全段
        classifier_msg = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="audio_classifier_service",
            action="audio_classification",
            parameters={**base_parameters, "classification_type": "segmented", "top_k": 10},
            file_path=audio_file_path
        )
        tasks.append(producer.send(KAFKA_TOPIC_CLASSIFIER_REQUEST, classifier_msg))

        # Diarization — 全段
        diarization_msg = MessageBuilder.create_processing_request(
            original_message=original_message,
            target_service="audio_diarization_service",
            action="audio_diarization",
            parameters={**base_parameters, "diarization_type": "basic", "min_speakers": 1, "max_speakers": 10},
            file_path=audio_file_path
        )
        tasks.append(producer.send(KAFKA_TOPIC_DIARIZATION_REQUEST, diarization_msg))

        # Recognizer — 分段或全段
        if chunk_infos:
            chunk_map = {}
            for chunk in chunk_infos:
                req = MessageBuilder.create_processing_request(
                    original_message=original_message,
                    target_service="audio_recognizer_service",
                    action="audio_recognition",
                    parameters={**base_parameters, "file_path": chunk["path"]},
                    file_path=chunk["path"]
                )
                chunk_map[req["payload"]["request_id"]] = chunk["start_offset"]
                tasks.append(producer.send(KAFKA_TOPIC_RECOGNIZER_REQUEST, req))
            # 記錄 request_id → start_offset 對應
            self.redis_manager.update_state(correlation_id, {"recognizer_chunk_request_map": chunk_map})
            logger.info(f"Sending {len(chunk_infos)} chunked ASR requests for: {payload['request_id']}")
        else:
            recognizer_msg = MessageBuilder.create_processing_request(
                original_message=original_message,
                target_service="audio_recognizer_service",
                action="audio_recognition",
                parameters=base_parameters,
                file_path=audio_file_path
            )
            tasks.append(producer.send(KAFKA_TOPIC_RECOGNIZER_REQUEST, recognizer_msg))

        await asyncio.gather(*tasks)
        logger.info(f"All audio analysis requests sent for: {payload['request_id']}")
    
    async def handle_classifier_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理音頻分類結果"""
        correlation_id = message["correlation_id"]
        
        state = self.redis_manager.get_state(correlation_id)
        if not state:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return
                
        # 檢查是否為錯誤響應
        if message["message_type"] == "ERROR" or message["payload"].get("status") == "error":
            logger.error(f"Audio classifier error: {message['payload'].get('error', 'Unknown error')}")
            classifier_result = {
                "status": "error",
                "error": message["payload"].get("error", "Unknown error"),
                "file_path": ""
            }
        else:
            # 成功響應
            classifier_result = message["payload"]
            
        if not self.redis_manager.update_state(correlation_id, {"classifier_result": classifier_result}):
            raise Exception("Failed to update state in Redis")
            
        logger.info(f"Audio classifier result received for: {correlation_id}")
        state = self.redis_manager.get_state(correlation_id)
        
        # 檢查是否所有分析都完成
        await self.check_analysis_completion(producer, state, correlation_id)
    
    async def handle_recognizer_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理語音識別結果（支援單段與分段兩種模式）"""
        correlation_id = message["correlation_id"]

        state = self.redis_manager.get_state(correlation_id)
        if not state:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return

        is_chunked = state.get("recognizer_chunks_total", 0) > 0

        if is_chunked:
            await self._handle_recognizer_chunk_result(message, producer, state, correlation_id)
            return

        # 非分段模式（純音檔或無 moments 的影片）
        if message["message_type"] == "ERROR" or message["payload"].get("status") == "error":
            logger.error(f"Audio recognizer error: {message['payload'].get('error', 'Unknown error')}")
            recognizer_result = {
                "status": "error",
                "error": message["payload"].get("error", "Unknown error"),
                "file_path": ""
            }
        else:
            recognizer_result = message["payload"]

        if not self.redis_manager.update_state(correlation_id, {"recognizer_result": recognizer_result}):
            raise Exception("Failed to update state in Redis")

        logger.info(f"Audio recognizer result received for: {correlation_id}")
        state = self.redis_manager.get_state(correlation_id)
        await self.check_analysis_completion(producer, state, correlation_id)

    async def _handle_recognizer_chunk_result(
        self,
        message: Dict[str, Any],
        producer: AIOKafkaProducer,
        state: Dict[str, Any],
        correlation_id: str
    ):
        """累積分段 ASR 結果，全部收齊後合併再繼續"""
        if message["message_type"] == "ERROR" or message["payload"].get("status") == "error":
            error_msg = message["payload"].get("error", "Unknown error")
            logger.error(f"ASR chunk error for {correlation_id}: {error_msg}")
            recognizer_result = {"status": "error", "error": error_msg, "file_path": ""}
            self.redis_manager.update_state(correlation_id, {"recognizer_result": recognizer_result})
            state = self.redis_manager.get_state(correlation_id)
            await self.check_analysis_completion(producer, state, correlation_id)
            return

        request_id = message["payload"].get("request_id")
        chunk_map = state.get("recognizer_chunk_request_map", {})
        start_offset = float(chunk_map.get(request_id, 0.0))
        result_path = message["payload"].get("result_path", "")

        current_results = state.get("recognizer_chunk_results", [])
        current_results.append({"start_offset": start_offset, "result_path": result_path})
        done = len(current_results)
        total = state["recognizer_chunks_total"]

        self.redis_manager.update_state(correlation_id, {"recognizer_chunk_results": current_results})
        logger.info(f"ASR chunk {done}/{total} received for: {correlation_id}")

        if done < total:
            return  # 等待其餘 chunk

        # 全部收齊：合併並補正 timestamp
        try:
            merged_path = self._merge_recognizer_chunks(current_results, correlation_id)
            recognizer_result = {"status": "success", "result_path": merged_path, "processing_time": 0}
            logger.info(f"All ASR chunks merged for: {correlation_id} → {merged_path}")
        except Exception as e:
            logger.error(f"Failed to merge ASR chunks for {correlation_id}: {e}")
            recognizer_result = {"status": "error", "error": str(e), "file_path": ""}

        self.redis_manager.update_state(correlation_id, {"recognizer_result": recognizer_result})
        state = self.redis_manager.get_state(correlation_id)
        await self.check_analysis_completion(producer, state, correlation_id)

    def _merge_recognizer_chunks(self, chunk_results: list, correlation_id: str) -> str:
        """
        讀取 N 個 ASR chunk JSON，將每段的 start/end 加上 start_offset，
        合併成與單段模式格式相同的 [{start, end, text, words}] 清單。
        """
        merged_segments = []
        for chunk in sorted(chunk_results, key=lambda c: c["start_offset"]):
            offset = chunk["start_offset"]
            result_path = chunk.get("result_path", "")
            if not result_path or not os.path.exists(result_path):
                logger.warning(f"Missing ASR chunk result, skipping: {result_path}")
                continue
            with open(result_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
            for seg in segments:
                merged_segments.append({
                    "start": seg["start"] + offset,
                    "end":   seg["end"]   + offset,
                    "text":  seg["text"],
                    "words": [
                        {**w, "start": w.get("start", 0) + offset, "end": w.get("end", 0) + offset}
                        for w in seg.get("words", [])
                    ]
                })

        os.makedirs(AUDIO_MERGED_RESULTS_DIR, exist_ok=True)
        merged_path = os.path.join(AUDIO_MERGED_RESULTS_DIR, f"{correlation_id}_asr_merged.json")
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged_segments, f, ensure_ascii=False, indent=2)

        return merged_path
    
    async def handle_diarization_result(self, message: Dict[str, Any], producer: AIOKafkaProducer):
        """處理說話人分離結果"""
        correlation_id = message["correlation_id"]
        
        state = self.redis_manager.get_state(correlation_id)
        if not state:
            logger.warning(f"No processing state found for correlation_id: {correlation_id}")
            return
        
        # 檢查是否為錯誤響應
        if message["message_type"] == "ERROR" or message["payload"].get("status") == "error":
            logger.error(f"Audio diarization error: {message['payload'].get('error', 'Unknown error')}")
            diarization_result = {
                "status": "error",
                "error": message["payload"].get("error", "Unknown error"),
                "file_path": ""
            }
        else:
            # 成功響應
            diarization_result = message["payload"]
        
        if not self.redis_manager.update_state(correlation_id, {"diarization_result": diarization_result}):
            raise Exception("Failed to update state in Redis")
        
        logger.info(f"Audio diarization result received for: {correlation_id}")

        state = self.redis_manager.get_state(correlation_id)
        
        # 檢查是否所有分析都完成
        await self.check_analysis_completion(producer, state, correlation_id)
    
    async def check_analysis_completion(self, producer: AIOKafkaProducer, state: Dict[str, Any], correlation_id: str):
        """檢查分析是否完成，如果完成則發送最終結果"""
        if (state["classifier_result"] is not None and 
            state["recognizer_result"] is not None and
            state["diarization_result"] is not None):
            
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
                    ("classifier", state["classifier_result"]),
                    ("recognizer", state["recognizer_result"]),
                    ("diarization", state["diarization_result"])
                ]:
                    if result.get("status") == "error":
                        has_errors = True
                        error_messages.append(f"{service_name}: {result.get('error', 'Unknown error')}")
                
                if has_errors:
                    # 如果有錯誤，發送錯誤響應
                    error_message = "; ".join(error_messages)
                    await self.send_error_response(
                        original_message, 
                        producer, 
                        f"Audio analysis failed: {error_message}", 
                        "ANALYSIS_FAILED"
                    )
                    return
                
                # 獲取三個結果的檔案路徑 - 使用各服務返回的路徑字段名
                classifier_path = state["classifier_result"].get("result_path", "")
                recognizer_path = state["recognizer_result"].get("result_path", "")
                diarization_path = state["diarization_result"].get("result_path", "")
                
                # 檢查檔案路徑是否都存在
                if not all([classifier_path, recognizer_path, diarization_path]):
                    raise ValueError("Missing result file paths from one or more services")
                
                # 合併結果
                merged_file_path = await self.result_merger.merge_audio_results(
                    classifier_result_path=classifier_path,
                    recognizer_result_path=recognizer_path,
                    diarization_result_path=diarization_path,
                    request_id=request_id,
                    primary_filename=primary_filename
                )

                # 合併所有分析結果（包含合併後的檔案路徑）
                combined_results = {
                    "audio_conversion": {
                        "original_file_path": state["audio_file_path"],
                        "audio_info": state["audio_info"]
                    },
                    "classification": {
                        "status": "success",
                        "result_file_path": classifier_path,
                        "processing_service": "audio_classifier_service",
                        "processing_time": state["classifier_result"].get("processing_time", 0)
                    },
                    "recognition": {
                        "status": "success",
                        "result_file_path": recognizer_path,
                        "processing_service": "audio_recognizer_service",
                        "processing_time": state["recognizer_result"].get("processing_time", 0)
                    },
                    "diarization": {
                        "status": "success",
                        "result_file_path": diarization_path,
                        "processing_service": "audio_diarization_service",
                        "processing_time": state["diarization_result"].get("processing_time", 0)
                    },
                    "merged_analysis": {
                        "merged_file_path": merged_file_path,
                        "format": "Combined audio analysis results with classification, recognition, and diarization"
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
                
                correlation_id = state["original_message"]["correlation_id"]
                logger.info(f"Audio analysis completed successfully for: {correlation_id}")
                logger.info(f"Merged results saved to: {merged_file_path}")
                
                if not self.redis_manager.update_state(correlation_id, {"step": "complete"}):
                    logger.warning(f"Failed to update completion state in Redis: {correlation_id}")
                
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
            
            # Must go to KAFKA_TOPIC_RESPONSE, not KAFKA_TOPIC_DLQ — video_orchestrator_service
            # only listens on the former, so sending errors to the DLQ meant it never learned
            # the job had failed and the correlation_id looked stuck forever instead.
            await producer.send(KAFKA_TOPIC_RESPONSE, error_response)
            logger.error(f"Error response sent: {error_message}")
            
            # 清理相關狀態
            correlation_id = original_message.get("correlation_id")
            if correlation_id:
                await self.cleanup_processing_state(correlation_id)
                
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")
    
    async def cleanup_processing_state(self, correlation_id: str, delete_state: bool = False):
        """清理處理狀態和臨時檔案
        
        Args:
            correlation_id: 關聯 ID
            delete_state: 是否刪除 Redis 狀態 (預設 False,讓 TTL 自然過期)
        """
        state = self.redis_manager.get_state(correlation_id)
        if not state:
            logger.warning(f"No state found for cleanup: {correlation_id}")
            return
        # 清理臨時檔案
        cleanup_paths = state.get("temp_cleanup_paths", [])
        for path in cleanup_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned up temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {path}: {e}")
        if delete_state:
            self.redis_manager.delete_state(correlation_id)
            logger.info(f"Redis state deleted for: {correlation_id}")
        else:
            logger.info(f"Temporary files cleaned up for: {correlation_id} (Redis state retained)")