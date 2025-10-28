# services/audio_analysis_service/message_utils.py

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from config import SERVICE_NAME

class MessageBuilder:
    @staticmethod
    def create_processing_request(
        original_message: Dict[str, Any],
        target_service: str,
        action: str,
        parameters: Dict[str, Any],
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """創建處理請求消息（標準格式）"""
        original_payload = original_message["payload"]
        
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": original_message["correlation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": SERVICE_NAME,
            "target_service": target_service,
            "message_type": "REQUEST",
            "priority": "MEDIUM",
            "payload": {
                "request_id": str(uuid.uuid4()),
                "user_id": original_payload["user_id"],
                "action": action,
                "parameters": parameters,
                "file_path": file_path or "",
                "metadata": {
                    "original_request_id": original_payload["request_id"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_service": SERVICE_NAME
                }
            },
            "retry_count": 0,
            "ttl": 3600
        }
    
    @staticmethod
    def create_processing_response(
        original_message: Dict[str, Any],
        status: str,
        results: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
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
            "file_path": file_path or "",
            "metadata": {
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "processing_service": SERVICE_NAME,
                "original_request_id": original_payload["request_id"]
            }
        }
        
        # 如果有結果資料，加入到 metadata 中
        if results:
            payload["metadata"]["results"] = results
        
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
