# services/audio_recognizer_service/audio_recognition.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from processors.recognizer import SpeechRecognizer
from config import (
    SUPPORTED_AUDIO_TYPES,
    RECOGNITION_RESULTS_DIR
)

logger = logging.getLogger(__name__)

class AudioRecognitionClient:
    def __init__(self):
        """初始化音頻識別客戶端"""
        self.speech_recognizer = SpeechRecognizer()
        logger.info("Audio recognition client initialized")
    
    def get_file_type(self, filename: str) -> str:
        """根據檔案名稱判斷檔案類型"""
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension in SUPPORTED_AUDIO_TYPES:
            return "audio"
        else:
            raise ValueError(f"Unsupported audio file type: {file_extension}")
    
    def recognize_audio_sync(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步版本的音頻識別方法（用於線程池執行）
        
        Args:
            request_payload: 包含識別請求的 payload
            
        Returns:
            識別結果
        """
        try:
            # 提取請求參數
            request_id = request_payload["request_id"]
            parameters = request_payload.get("parameters", {})
            
            file_path = parameters.get("file_path")
            
            if not file_path:
                return {
                    "request_id": request_id,
                    "action": "audio_recognition",
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
                    "action": "audio_recognition",
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
            
            logger.info(f"[SYNC] Processing audio: {filename} from: {file_path}")
            logger.debug(f"[SYNC] Asset path: {asset_path}, Version ID: {version_id}")
            
            # 判斷檔案類型
            file_type = self.get_file_type(filename)
            
            # 處理音頻檔案
            start_time = datetime.now()
            recognition_result = self.speech_recognizer.process_audio_file(
                str(audio_file_path), 
                request_id
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 檢查處理結果
            if recognition_result.get("status") == "error":
                return {
                    "request_id": request_id,
                    "action": "audio_recognition",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_FAILED",
                        "message": f"Audio recognition failed: {recognition_result.get('error')}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 生成識別結果檔案路徑
            recognition_result_path = os.path.join(
                RECOGNITION_RESULTS_DIR,
                f"recognition_result_{request_id}_{filename}.json"
            )
            
            # result_data=recognition_result
            # 只保留 segments 部分
            result_data = recognition_result.get("segments", [])
            
            
            # 保存結果到檔案
            with open(recognition_result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[SYNC] Recognition result saved to: {recognition_result_path}")

            # 計算檔案大小
            file_size_bytes = os.path.getsize(recognition_result_path)
            
            # 返回成功結果
            return {
                "request_id": request_id,
                "action": "audio_recognition",
                "status": "success",
                "parameters": {
                    "recognition_result_path": recognition_result_path,
                    "filename": filename,
                    "file_type": file_type,
                    "language": recognition_result.get("language"),
                    "text": recognition_result.get("text"),
                    "segments_count": len(recognition_result.get("segments", [])),
                    "duration": recognition_result.get("audio_info", {}).get("duration"),
                    "file_size_bytes": file_size_bytes,
                    "processing_time_seconds": processing_time,
                    "asset_path": asset_path,
                    "version_id": version_id
                },
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "processing_method": "async_thread_pool",
                    "model_info": recognition_result.get("model_info")
                }
            }
            
        except Exception as e:
            logger.error(f"[SYNC] Error recognizing audio: {e}")
            
            # 清理可能產生的檔案
            try:
                request_id = request_payload.get('request_id', 'unknown')
                filename = request_payload.get('parameters', {}).get('primary_filename', 'unknown')
                recognition_result_path = os.path.join(
                    RECOGNITION_RESULTS_DIR,
                    f"recognition_result_{request_id}_{filename}.json"
                )
                if os.path.exists(recognition_result_path):
                    os.remove(recognition_result_path)
                    logger.info(f"Cleaned up failed recognition result file: {recognition_result_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup recognition result file: {cleanup_error}")
            
            return {
                "request_id": request_payload.get("request_id"),
                "action": "audio_recognition",
                "status": "error",
                "error": {
                    "code": "RECOGNITION_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "error_type": type(e).__name__,
                    "processing_method": "async_thread_pool"
                }
            }
    
    async def recognize_audio(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        異步版本的音頻識別方法（向後兼容）
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
                    loop.run_in_executor(executor, self.recognize_audio_sync, request_payload),
                    timeout=ASYNC_PROCESSING_CONFIG["task_timeout"]
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Audio recognition timeout for request: {request_payload.get('request_id')}")
                return {
                    "request_id": request_payload.get("request_id"),
                    "action": "audio_recognition",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_TIMEOUT",
                        "message": f"Processing timeout after {ASYNC_PROCESSING_CONFIG['task_timeout']} seconds",
                        "timestamp": datetime.now().isoformat()
                    }
                }
