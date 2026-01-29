# services/video_ocr_frame_service/message_utils.py

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from config import SERVICE_NAME

class MessageBuilder:
    @staticmethod
    def create_processing_response(
        original_message: Dict[str, Any],
        status: str,
        result_file_path: str,
        ocr_frames_dir: Optional[str] = None,
        total_scenes: Optional[int] = None,
        ocr_detected_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """創建處理響應消息（標準格式）"""
        original_payload = original_message["payload"]
        
        payload = {
            "request_id": original_payload["request_id"],
            "user_id": original_payload["user_id"],
            "action": f"{original_payload['action']}_result",
            "parameters": {
                "status": status,
                "processing_service": SERVICE_NAME
            },
            "file_path": result_file_path,
            "metadata": {
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "processing_service": SERVICE_NAME,
                "original_request_id": original_payload.get("request_id")
            }
        }
        
        # 添加 OCR 特定的路徑資訊
        if ocr_frames_dir:
            payload["parameters"]["ocr_frames_directory"] = ocr_frames_dir
        
        if total_scenes is not None:
            payload["parameters"]["total_scenes"] = total_scenes
            
        if ocr_detected_count is not None:
            payload["parameters"]["ocr_detected_count"] = ocr_detected_count
        
        if error_message:
            payload["parameters"]["error_message"] = error_message
        
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": original_message["correlation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": SERVICE_NAME,
            "target_service": original_message["source_service"],
            "message_type": "RESPONSE",
            "priority": "MEDIUM",
            "payload": payload,
            "retry_count": 0,
            "ttl": 3600
        }
    
    @staticmethod
    def create_error_response(
        original_message: Dict[str, Any],
        error_message: str,
        error_code: str
    ) -> Dict[str, Any]:
        """創建錯誤響應消息（標準格式）"""
        original_payload = original_message["payload"]
        
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": original_message["correlation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": SERVICE_NAME,
            "target_service": original_message["source_service"],
            "message_type": "ERROR",
            "priority": "HIGH",
            "payload": {
                "request_id": original_payload["request_id"],
                "user_id": original_payload["user_id"],
                "action": f"{original_payload['action']}_error",
                "parameters": {
                    "error_code": error_code,
                    "error_service": SERVICE_NAME
                },
                "file_path": "",
                "metadata": {
                    "error_at": datetime.now(timezone.utc).isoformat(),
                    "error_service": SERVICE_NAME,
                    "error_message": error_message
                }
            },
            "retry_count": 0,
            "ttl": 3600
        }
