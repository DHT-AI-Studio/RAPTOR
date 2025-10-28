# services/audio_classifier_service/audio_classification.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from processors.classifier import AudioClassifier
from config import (
    SUPPORTED_AUDIO_TYPES,
    CLASSIFICATION_RESULTS_DIR,
    PANNS_TOP_K,
    PANNS_SEGMENT_LENGTH
)

logger = logging.getLogger(__name__)

class AudioClassificationClient:
    def __init__(self):
        """初始化音頻分類客戶端"""
        self.audio_classifier = AudioClassifier()
        logger.info("Audio classification client initialized")
    
    def get_file_type(self, filename: str) -> str:
        """根據檔案名稱判斷檔案類型"""
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension in SUPPORTED_AUDIO_TYPES:
            return "audio"
        else:
            raise ValueError(f"Unsupported audio file type: {file_extension}")
    
    def classify_audio_sync(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步版本的音頻分類方法（用於線程池執行）
        
        Args:
            request_payload: 包含分類請求的 payload
            
        Returns:
            分類結果
        """
        try:
            # 提取請求參數
            request_id = request_payload["request_id"]
            parameters = request_payload.get("parameters", {})
            
            file_path = parameters.get("file_path")
            
            if not file_path:
                return {
                    "request_id": request_id,
                    "action": "audio_classification",
                    "status": "error",
                    "error": {
                        "code": "MISSING_FILE_PATH",
                        "message": "Missing file_path in parameters",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            audio_file_path = Path(file_path)
            
            if not audio_file_path.exists():
                return {
                    "request_id": request_id,
                    "action": "audio_classification",
                    "status": "error",
                    "error": {
                        "code": "FILE_NOT_FOUND",
                        "message": f"Audio file not found: {file_path}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 獲取檔案資訊
            filename = parameters.get("primary_filename", audio_file_path.name)
            asset_path = parameters.get("asset_path")  
            version_id = parameters.get("version_id")
            
            # 獲取分類參數
            classification_type = parameters.get("classification_type", "segmented")
            top_k = parameters.get("top_k", PANNS_TOP_K)
            segment_length_sec = parameters.get("segment_length_sec", PANNS_SEGMENT_LENGTH)
            
            logger.info(f"[SYNC] Processing audio classification: {filename} from: {file_path}")
            logger.debug(f"[SYNC] Asset path: {asset_path}, Version ID: {version_id}")
            logger.debug(f"[SYNC] Classification type: {classification_type}, top_k: {top_k}")
            
            # 判斷檔案類型
            file_type = self.get_file_type(filename)
            
            # 處理音頻檔案
            start_time = datetime.now()
            classification_result = self.audio_classifier.process_audio_file(
                str(audio_file_path), 
                request_id,
                classification_type=classification_type,
                top_k=top_k,
                segment_length_sec=segment_length_sec
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 檢查處理結果
            if "error" in classification_result:
                return {
                    "request_id": request_id,
                    "action": "audio_classification",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_FAILED",
                        "message": f"Audio classification failed: {classification_result.get('error')}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 生成分類結果檔案路徑
            classification_result_path = os.path.join(
                CLASSIFICATION_RESULTS_DIR,
                f"classification_result_{request_id}_{filename}.json"
            )
            
            # 準備保存的結果數據 
            if classification_type == "basic":
                # Basic 分類保存完整的分類結果
                result_data = classification_result.get("top_classes", [])
            elif classification_type == "segmented":
                # Segmented 分類保存段落結果
                result_data = classification_result.get("segments", [])
            else:
                # 預設保存完整結果
                result_data = classification_result
            
            
            # 保存結果到檔案
            with open(classification_result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[SYNC] Classification result saved to: {classification_result_path}")

            # 計算檔案大小
            file_size_bytes = os.path.getsize(classification_result_path)
            
            # 構建返回參數
            return_parameters = {
                "classification_result_path": classification_result_path,
                "filename": filename,
                "file_type": file_type,
                "classification_type": classification_type,
                "file_size_bytes": file_size_bytes,
                "processing_time_seconds": processing_time,
                "asset_path": asset_path,
                "version_id": version_id
            }
            
            # 根據分類類型添加特定參數
            if classification_type == "basic":
                return_parameters.update({
                    "top_k": top_k,
                    "top_class": classification_result.get("top_classes", [{}])[0].get("label", "unknown")
                })
            elif classification_type == "segmented":
                return_parameters.update({
                    "audio_duration": classification_result.get("audio_duration"),
                    "segment_length": classification_result.get("segment_length"),
                    "segments_count": len(classification_result.get("segments", []))
                })
            
            # 返回成功結果
            return {
                "request_id": request_id,
                "action": "audio_classification",
                "status": "success",
                "parameters": return_parameters,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "processing_method": "async_thread_pool",
                    "model_info": {
                        "model_type": "PANNs",
                        "device": self.audio_classifier.device
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"[SYNC] Error classifying audio: {e}")
            
            # 清理可能產生的檔案
            try:
                request_id = request_payload.get('request_id', 'unknown')
                filename = request_payload.get('parameters', {}).get('primary_filename', 'unknown')
                classification_result_path = os.path.join(
                    CLASSIFICATION_RESULTS_DIR,
                    f"classification_result_{request_id}_{filename}.json"
                )
                if os.path.exists(classification_result_path):
                    os.remove(classification_result_path)
                    logger.info(f"Cleaned up failed classification result file: {classification_result_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup classification result file: {cleanup_error}")
            
            return {
                "request_id": request_payload.get("request_id"),
                "action": "audio_classification",
                "status": "error",
                "error": {
                    "code": "CLASSIFICATION_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "error_type": type(e).__name__,
                    "processing_method": "async_thread_pool"
                }
            }
    
    async def classify_audio(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        異步版本的音頻分類方法（向後兼容）
        現在內部使用同步版本
        """
        import asyncio
        import concurrent.futures
        from config import ASYNC_PROCESSING_CONFIG
        
        loop = asyncio.get_event_loop()
        
        # 使用線程池執行同步版本
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=ASYNC_PROCESSING_CONFIG["max_workers"]
        ) as executor:
            try:
                # 設置超時
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, self.classify_audio_sync, request_payload),
                    timeout=ASYNC_PROCESSING_CONFIG["task_timeout"]
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Audio classification timeout for request: {request_payload.get('request_id')}")
                return {
                    "request_id": request_payload.get("request_id"),
                    "action": "audio_classification",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_TIMEOUT",
                        "message": f"Processing timeout after {ASYNC_PROCESSING_CONFIG['task_timeout']} seconds",
                        "timestamp": datetime.now().isoformat()
                    }
                }
