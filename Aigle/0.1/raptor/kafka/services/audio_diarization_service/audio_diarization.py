# services/audio_diarization_service/audio_diarization.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from processors.diarization import SpeakerDiarization
from config import (
    SUPPORTED_AUDIO_TYPES,
    DIARIZATION_RESULTS_DIR,
    DIARIZATION_MIN_SPEAKERS,
    DIARIZATION_MAX_SPEAKERS
)

logger = logging.getLogger(__name__)

class AudioDiarizationClient:
    def __init__(self):
        """初始化音頻說話人分離客戶端"""
        self.speaker_diarization = SpeakerDiarization()
        logger.info("Audio diarization client initialized")
    
    def get_file_type(self, filename: str) -> str:
        """根據檔案名稱判斷檔案類型"""
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension in SUPPORTED_AUDIO_TYPES:
            return "audio"
        else:
            raise ValueError(f"Unsupported audio file type: {file_extension}")
    
    def diarize_audio_sync(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步版本的音頻說話人分離方法（用於線程池執行）
        
        Args:
            request_payload: 包含分離請求的 payload
            
        Returns:
            分離結果
        """
        try:
            # 提取請求參數
            request_id = request_payload["request_id"]
            parameters = request_payload.get("parameters", {})
            
            file_path = parameters.get("file_path")
            
            if not file_path:
                return {
                    "request_id": request_id,
                    "action": "audio_diarization",
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
                    "action": "audio_diarization",
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
            
            # 獲取說話人分離參數
            diarization_type = parameters.get("diarization_type", "basic")
            min_speakers = parameters.get("min_speakers", DIARIZATION_MIN_SPEAKERS)
            max_speakers = parameters.get("max_speakers", DIARIZATION_MAX_SPEAKERS)
            transcription_segments = parameters.get("transcription_segments")
            
            logger.info(f"[SYNC] Processing audio diarization: {filename} from: {file_path}")
            logger.debug(f"[SYNC] Asset path: {asset_path}, Version ID: {version_id}")
            logger.debug(f"[SYNC] Diarization type: {diarization_type}, speakers: {min_speakers}-{max_speakers}")
            
            # 判斷檔案類型
            file_type = self.get_file_type(filename)
            
            # 處理音頻檔案
            start_time = datetime.now()
            diarization_result = self.speaker_diarization.process_audio_file(
                str(audio_file_path), 
                request_id,
                diarization_type=diarization_type,
                min_speakers=min_speakers if min_speakers > 0 else None,
                max_speakers=max_speakers if max_speakers > 0 else None,
                transcription_segments=transcription_segments
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 檢查處理結果
            if diarization_result.get("status") == "error":
                return {
                    "request_id": request_id,
                    "action": "audio_diarization",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_FAILED",
                        "message": f"Audio diarization failed: {diarization_result.get('error')}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            

            # 生成分離結果檔案路徑
            diarization_result_path = os.path.join(
                DIARIZATION_RESULTS_DIR,
                f"diarization_result_{request_id}_{filename}.json"
            )
            
            # 準備保存的結果數據 - 現在 diarization_result 已經是可序列化的
            # result_data = diarization_result
            result_data = diarization_result.get("diarization_result", [])
            
            # 保存結果到檔案 - 現在應該可以正常序列化
            with open(diarization_result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[SYNC] Diarization result saved to: {diarization_result_path}")

            # 計算檔案大小
            file_size_bytes = os.path.getsize(diarization_result_path)
            
            # 構建返回參數
            return_parameters = {
                "diarization_result_path": diarization_result_path,
                "filename": filename,
                "file_type": file_type,
                "diarization_type": diarization_type,
                "file_size_bytes": file_size_bytes,
                "processing_time_seconds": processing_time,
                "asset_path": asset_path,
                "version_id": version_id
            }
            
            # 根據分離類型添加特定參數
            return_parameters.update({
                "audio_duration": diarization_result.get("audio_duration"),
                "total_segments": diarization_result.get("total_segments", 0),
                "unique_speakers": diarization_result.get("unique_speakers", 0),
                "speakers": diarization_result.get("speakers", [])
            })
            
            # 返回成功結果
            return {
                "request_id": request_id,
                "action": "audio_diarization",
                "status": "success",
                "parameters": return_parameters,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "processing_method": "async_thread_pool",
                    "model_info": {
                        "model_type": "WhisperX_Diarization",
                        "device": self.speaker_diarization.device
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"[SYNC] Error in audio diarization: {e}")
            
            # 清理可能產生的檔案
            try:
                request_id = request_payload.get('request_id', 'unknown')
                filename = request_payload.get('parameters', {}).get('primary_filename', 'unknown')
                diarization_result_path = os.path.join(
                    DIARIZATION_RESULTS_DIR,
                    f"diarization_result_{request_id}_{filename}.json"
                )
                if os.path.exists(diarization_result_path):
                    os.remove(diarization_result_path)
                    logger.info(f"Cleaned up failed diarization result file: {diarization_result_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup diarization result file: {cleanup_error}")
            
            return {
                "request_id": request_payload.get("request_id"),
                "action": "audio_diarization",
                "status": "error",
                "error": {
                    "code": "DIARIZATION_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "error_type": type(e).__name__,
                    "processing_method": "async_thread_pool"
                }
            }
    
    async def diarize_audio(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        異步版本的音頻說話人分離方法（向後兼容）
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
                    loop.run_in_executor(executor, self.diarize_audio_sync, request_payload),
                    timeout=ASYNC_PROCESSING_CONFIG["task_timeout"]
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Audio diarization timeout for request: {request_payload.get('request_id')}")
                return {
                    "request_id": request_payload.get("request_id"),
                    "action": "audio_diarization",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_TIMEOUT",
                        "message": f"Processing timeout after {ASYNC_PROCESSING_CONFIG['task_timeout']} seconds",
                        "timestamp": datetime.now().isoformat()
                    }
                }
