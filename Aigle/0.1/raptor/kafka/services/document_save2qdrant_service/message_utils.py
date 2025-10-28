# services/document_save2qdrant_service/message_utils.py

import uuid
from datetime import datetime, timezone
from typing import Dict, Any

def create_response_message(
    original_message: Dict[str, Any],
    status: str,
    results: Dict[str, Any] = None,
    error: str = None
) -> Dict[str, Any]:
    """創建回應訊息"""
    
    payload = original_message.get("payload", {})
    
    response_payload = {
        "request_id": payload.get("request_id"),
        "original_request_id": payload.get("original_request_id"),
        "user_id": payload.get("user_id"),
        "action": payload.get("action"),
        "status": status,
        "filename": payload.get("parameters", {}).get("filename"),
        "asset_path": payload.get("parameters", {}).get("asset_path"),
        "version_id": payload.get("parameters", {}).get("version_id"),
        "temp_file_path": payload.get("parameters", {}).get("temp_file_path"),
        "result_path": payload.get("parameters", {}).get("summary_result_path"),
        "metadata": payload.get("metadata", {})
    }
    
    if results:
        response_payload["results"] = results
    
    if error:
        response_payload["error"] = error
    
    return {
        "message_id": str(uuid.uuid4()),
        "correlation_id": original_message.get("correlation_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_service": "document_save2qdrant_service",
        "target_service": original_message.get("source_service"),
        "message_type": "RESPONSE",
        "priority": original_message.get("priority", "MEDIUM"),
        "payload": response_payload,
        "retry_count": 0,
        "ttl": 3600
    }
