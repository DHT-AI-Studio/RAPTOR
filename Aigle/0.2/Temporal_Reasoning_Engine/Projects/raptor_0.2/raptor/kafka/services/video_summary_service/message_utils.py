# services/video_summary_service/message_utils.py

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class MessageBuilder:
    @staticmethod
    def create_message(
        source_service: str,
        target_service: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: str = "MEDIUM",
        correlation_id: Optional[str] = None,
        retry_count: int = 0,
        ttl: int = 3600
    ) -> Dict[str, Any]:
        """創建符合標準格式的消息"""
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": source_service,
            "target_service": target_service,
            "message_type": message_type,
            "priority": priority,
            "payload": payload,
            "retry_count": retry_count,
            "ttl": ttl
        }
    
    @staticmethod
    def create_response(
        original_message: Dict[str, Any],
        payload: Dict[str, Any],
        message_type: str = "RESPONSE"
    ) -> Dict[str, Any]:
        """基於原始消息創建響應消息"""
        return MessageBuilder.create_message(
            source_service="video_summary_service",
            target_service=original_message["source_service"],
            message_type=message_type,
            payload=payload,
            correlation_id=original_message["correlation_id"],
            priority=original_message.get("priority", "MEDIUM")
        )
        
    @staticmethod
    def create_summary_result_message(
        original_message: Dict[str, Any],
        summary_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """創建視頻摘要結果消息"""
        payload = {
            "request_id": original_message["payload"].get("request_id"),
            "original_request_id": original_message.get("correlation_id"),
            "user_id": original_message["payload"].get("user_id"),
            "action": "video_summary",
            "status": "success",
            "results": summary_results,
            "metadata": original_message["payload"].get("metadata", {})
        }
        
        return MessageBuilder.create_response(
            original_message=original_message,
            payload=payload
        )
    
    @staticmethod
    def create_error_response(
        original_message: Dict[str, Any],
        error_message: str,
        error_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """創建錯誤響應消息"""
        payload = {
            "request_id": original_message["payload"].get("request_id"),
            "action": original_message["payload"].get("action"),
            "status": "error",
            "error": {
                "message": error_message,
                "code": error_code,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "metadata": {
                "original_message_id": original_message["message_id"]
            }
        }
        
        return MessageBuilder.create_response(
            original_message=original_message,
            payload=payload,
            message_type="ERROR"
        )
    
    @staticmethod
    def create_dlq_message(
        original_message: Dict[str, Any],
        error: str,
        final_retry_count: int
    ) -> Dict[str, Any]:
        """創建 DLQ 消息"""
        payload = {
            "original_message": original_message,
            "error": error,
            "final_retry_count": final_retry_count,
            "dlq_timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "reason": "max_retries_exceeded"
            }
        }
        
        return MessageBuilder.create_message(
            source_service="video_summary_service",
            target_service="dlq_handler",
            message_type="ERROR",
            payload=payload,
            priority="HIGH",
            correlation_id=original_message["correlation_id"]
        )
