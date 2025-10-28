# services/video_scene_detection_service/message_utils.py

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
        scene_output_dir: Optional[str] = None,
        diff_plot_path: Optional[str] = None,
        scenes_count: Optional[int] = None,
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
        
        # 添加場景檢測特定的路徑資訊
        if scene_output_dir:
            payload["parameters"]["scene_output_directory"] = scene_output_dir
        
        if diff_plot_path:
            payload["parameters"]["diff_plot_path"] = diff_plot_path
            
        if scenes_count is not None:
            payload["parameters"]["scenes_count"] = scenes_count
        
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
