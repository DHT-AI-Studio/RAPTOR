# services/video_face_detection_service/message_utils.py

import uuid
from datetime import datetime, timezone
from typing import Any

from config import SERVICE_NAME


class MessageBuilder:
    @staticmethod
    def create_processing_response(original_message: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
        """創建處理響應消息（標準格式）"""
        original_payload = original_message["payload"]
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": original_message["correlation_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": SERVICE_NAME,
            "target_service": original_message["source_service"],
            "message_type": "RESPONSE",
            "priority": "MEDIUM",
            "payload": {
                "request_id": original_payload["request_id"],
                "user_id": original_payload.get("user_id"),
                "action": f"{original_payload['action']}_result",
                "status": "success",
                "results": results,
                "metadata": {
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "processing_service": SERVICE_NAME,
                    "original_request_id": original_payload.get("request_id"),
                },
            },
            "retry_count": 0,
            "ttl": 3600,
        }

    @staticmethod
    def create_error_response(original_message: dict[str, Any], error_message: str, error_code: str) -> dict[str, Any]:
        """創建錯誤響應消息（標準格式）"""
        original_payload = original_message.get("payload", {})
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": original_message.get("correlation_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": SERVICE_NAME,
            "target_service": original_message.get("source_service", ""),
            "message_type": "ERROR",
            "priority": "HIGH",
            "payload": {
                "request_id": original_payload.get("request_id"),
                "user_id": original_payload.get("user_id"),
                "action": f"{original_payload.get('action', 'face_detection')}_error",
                "status": "error",
                "error": {"code": error_code, "message": error_message},
                "metadata": {
                    "error_at": datetime.now(timezone.utc).isoformat(),
                    "error_service": SERVICE_NAME,
                    "error_message": error_message,
                },
            },
            "retry_count": 0,
            "ttl": 3600,
        }
