# services/image_processing_service/message_utils.py

import uuid
from datetime import datetime, timezone
from typing import Dict, Any

class MessageBuilder:
    @staticmethod
    def create_message(
        source_service: str,
        target_service: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: str = "MEDIUM",
        correlation_id: str = None,
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
            source_service="image_processing_service",
            target_service="image_orchestrator_service",
            message_type=message_type,
            payload=payload,
            correlation_id=original_message["correlation_id"],
            priority=original_message.get("priority", "MEDIUM")
        )
    
    @staticmethod
    def create_description_success_response(original_message: Dict[str, Any], result_path: str) -> Dict[str, Any]:
        """創建圖片描述成功響應消息"""
        payload = {
            "request_id": str(uuid.uuid4()),
            "original_request_id": original_message["payload"]["request_id"],
            "user_id": original_message["payload"]["user_id"],
            "action": "image_description_result",
            "status": "success",
            "results": {
                "parameters": {
                    "description_result_path": result_path,
                    "filename": original_message["payload"]["parameters"]["primary_filename"]
                }
            }
        }
        
        return MessageBuilder.create_response(
            original_message=original_message,
            payload=payload,
            message_type="RESPONSE"
        )
    
    @staticmethod
    def create_ocr_success_response(original_message: Dict[str, Any], result_path: str) -> Dict[str, Any]:
        """創建 OCR 成功響應消息"""
        payload = {
            "request_id": str(uuid.uuid4()),
            "original_request_id": original_message["payload"]["request_id"],
            "user_id": original_message["payload"]["user_id"],
            "action": "image_ocr_result",
            "status": "success",
            "results": {
                "parameters": {
                    "ocr_result_path": result_path,
                    "filename": original_message["payload"]["parameters"]["primary_filename"]
                }
            }
        }
        
        return MessageBuilder.create_response(
            original_message=original_message,
            payload=payload,
            message_type="RESPONSE"
        )
    
    @staticmethod
    def create_description_error_response(original_message: Dict[str, Any], error_message: str, error_code: str) -> Dict[str, Any]:
        """創建圖片描述錯誤響應消息"""
        payload = {
            "request_id": str(uuid.uuid4()),
            "original_request_id": original_message["payload"]["request_id"],
            "user_id": original_message["payload"]["user_id"],
            "action": "image_description_result",
            "status": "error",
            "error": {
                "message": error_message,
                "code": error_code
            }
        }
        
        return MessageBuilder.create_response(
            original_message=original_message,
            payload=payload,
            message_type="ERROR"
        )
    
    @staticmethod
    def create_ocr_error_response(original_message: Dict[str, Any], error_message: str, error_code: str) -> Dict[str, Any]:
        """創建 OCR 錯誤響應消息"""
        payload = {
            "request_id": str(uuid.uuid4()),
            "original_request_id": original_message["payload"]["request_id"],
            "user_id": original_message["payload"]["user_id"],
            "action": "image_ocr_result",
            "status": "error",
            "error": {
                "message": error_message,
                "code": error_code
            }
        }
        
        return MessageBuilder.create_response(
            original_message=original_message,
            payload=payload,
            message_type="ERROR"
        )
    
    @staticmethod
    def create_dlq_message(original_message: Dict[str, Any], error: str, final_retry_count: int) -> Dict[str, Any]:
        """創建 DLQ 消息"""
        payload = {
            "original_message": original_message,
            "error": error,
            "final_retry_count": final_retry_count,
            "dlq_timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "image_processing_service"
        }
        
        return MessageBuilder.create_message(
            source_service="image_processing_service",
            target_service="dlq_handler",
            message_type="ERROR",
            payload=payload,
            priority="HIGH",
            correlation_id=original_message["correlation_id"]
        )
