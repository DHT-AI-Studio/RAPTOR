# services/image_orchestrator_service/message_utils.py

import uuid
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging
logger = logging.getLogger(__name__)
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
            source_service="image_orchestrator_service",
            target_service=original_message["source_service"],
            message_type=message_type,
            payload=payload,
            correlation_id=original_message["correlation_id"],
            priority=original_message.get("priority", "MEDIUM")
        )
    
    @staticmethod
    def create_processing_request(
        original_message: Dict[str, Any],
        target_service: str,
        action: str,
        parameters: Dict[str, Any],
        temp_file_path: str
    ) -> Dict[str, Any]:
        """創建處理請求消息"""
        payload = {
            "request_id": str(uuid.uuid4()),
            "original_request_id": original_message["payload"].get("request_id"),
            "user_id": original_message["payload"].get("user_id"),
            "access_token": original_message["payload"].get("access_token"),
            "action": action,
            "parameters": {
                **parameters,
                "temp_file_path": temp_file_path,
                "asset_path": original_message["payload"]["parameters"].get("asset_path"),
                "version_id": original_message["payload"]["parameters"].get("version_id"),
                "status": original_message["payload"]["parameters"].get("status")
            },
            "metadata": {
                "orchestrator_request_id": original_message["payload"].get("request_id"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "original_metadata": original_message["payload"].get("metadata", {})
            }
        }
        
        return MessageBuilder.create_message(
            source_service="image_orchestrator_service",
            target_service=target_service,
            message_type="REQUEST",
            payload=payload,
            correlation_id=original_message["correlation_id"],
            priority=original_message.get("priority", "MEDIUM"),
            ttl=original_message.get("ttl", 3600)
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
            source_service="image_orchestrator_service",
            target_service="dlq_handler",
            message_type="ERROR",
            payload=payload,
            priority="HIGH",
            correlation_id=original_message["correlation_id"]
        )

class ImageResultMerger:
    @staticmethod
    def merge_results(
        description_result: Dict[str, Any],
        ocr_result: Dict[str, Any],
        original_message: Dict[str, Any]
    ) -> str:
        """合併圖片描述和OCR結果，返回合併後的JSON檔案路徑"""
        try:
            # 從原始消息中提取資訊
            original_params = original_message["payload"]["parameters"]
            filename = original_params.get("primary_filename")
            asset_path = original_params.get("asset_path")
            version_id = original_params.get("version_id")
            status = original_params.get("status")
            upload_time = original_message["payload"].get("metadata", {}).get("upload_timestamp", datetime.now(timezone.utc).isoformat())
            
            # 提取描述結果
            description_text = ""
            if description_result.get("results", {}).get("parameters", {}).get("description_result_path"):
                desc_path = description_result["results"]["parameters"]["description_result_path"]
                if os.path.exists(desc_path):
                    with open(desc_path, 'r', encoding='utf-8') as f:
                        desc_data = json.load(f)
                        description_text = desc_data.get("description", "")
                else:
                    logger.warning(f"Description result file not found: {desc_path}")
            
            # 提取OCR結果
            ocr_text = ""
            if ocr_result.get("results", {}).get("parameters", {}).get("ocr_result_path"):
                ocr_path = ocr_result["results"]["parameters"]["ocr_result_path"]
                if os.path.exists(ocr_path):
                    with open(ocr_path, 'r', encoding='utf-8') as f:
                        ocr_data = json.load(f)
                        ocr_text = ocr_data.get("extracted_text", "")
                else:
                    logger.warning(f"OCR result file not found: {ocr_path}")
            
            # 獲取檔案副檔名
            file_extension = filename.split('.')[-1].lower() if '.' in filename else "unknown"
            
            # 創建合併結果 - 按照指定格式
            merged_data = [
                {
                    "id": str(uuid.uuid4()),
                    "payload": {
                        "filename": filename,
                        "type": "images",  # 固定
                        "upload_time": upload_time,
                        "embedding_type": "summary",  # 固定
                        "asset_path": asset_path,
                        "version_id": version_id,
                        "status": status,
                        "summary": description_text
                    }
                },
                {
                    "id": str(uuid.uuid4()),
                    "payload": {
                        "document_id": f"{filename}_chunk_0",
                        "type": "images",  # 固定
                        "text": ocr_text,
                        "filename": filename,
                        "source": file_extension,
                        "chunk_index": 0,  # 固定
                        "page_numbers": [1],  # 固定
                        "element_types": ["text"],  # 固定
                        "char_count": len(ocr_text),
                        "upload_time": upload_time,
                        "embedding_type": "text",  # 固定
                        "asset_path": asset_path,
                        "version_id": version_id,
                        "status": status
                    }
                }
            ]
            
            # 保存合併結果到臨時檔案
            from config import TEMP_FILE_DIR
            os.makedirs(TEMP_FILE_DIR, exist_ok=True)
            
            merged_file_path = os.path.join(
                TEMP_FILE_DIR,
                f"{version_id}_merged_results.json"
            )
            
            with open(merged_file_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Merged results saved to: {merged_file_path}")
            logger.info(f"File exists: {os.path.exists(merged_file_path)}")
            logger.info(f"File size: {os.path.getsize(merged_file_path) if os.path.exists(merged_file_path) else 'N/A'} bytes")
            
            return merged_file_path
            
        except Exception as e:
            logger.error(f"Failed to merge image results: {str(e)}")
            raise ValueError(f"Failed to merge image results: {str(e)}")

def create_final_result_message(
    original_message: Dict[str, Any],
    description_result: Dict[str, Any],
    ocr_result: Dict[str, Any],
    save_result: Dict[str, Any]
) -> Dict[str, Any]:
    """創建最終結果消息"""
    payload = {
        "request_id": original_message["payload"].get("request_id"),
        "user_id": original_message["payload"].get("user_id"),
        "action": "image_processing_complete",
        "status": "success",
        "results": {
            "description": description_result.get("results", {}),
            "ocr": ocr_result.get("results", {}),
            "qdrant_save": save_result.get("results", {})
        },
        "metadata": {
            "processing_completed_at": datetime.now(timezone.utc).isoformat(),
            "asset_path": original_message["payload"]["parameters"].get("asset_path"),
            "version_id": original_message["payload"]["parameters"].get("version_id"),
            "primary_filename": original_message["payload"]["parameters"].get("primary_filename")
        }
    }
    
    return MessageBuilder.create_message(
        source_service="image_orchestrator_service",
        target_service=original_message["source_service"],
        message_type="RESPONSE",
        payload=payload,
        correlation_id=original_message["correlation_id"],
        priority=original_message.get("priority", "MEDIUM")
    )
