"""
Service for producing messages to Kafka.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from aiokafka import AIOKafkaProducer
from fastapi import HTTPException, status

from app.core.config import Settings

_logger = logging.getLogger(__name__)


class KafkaService:
    def __init__(self, producer: AIOKafkaProducer, settings: Settings):
        self.producer = producer
        self.settings = settings

    def get_file_type(self, filename: str) -> str:
        """Determines the file type based on the extension."""
        if not filename:
            return "unknown"
        ext = filename.lower().split('.')[-1]
        
        document_exts = ['pdf', 'doc', 'docx', 'txt', 'html', 'htm', 'csv', 'xlsx', 'xls', 'ppt', 'pptx']
        image_exts = ['jpg', 'jpeg', 'png']
        video_exts = ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', '3gp']
        audio_exts = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'wma', 'm4a']
        
        if ext in document_exts:
            return "document"
        elif ext in image_exts:
            return "image"
        elif ext in video_exts:
            return "video"
        elif ext in audio_exts:
            return "audio"
        else:
            return "unknown"

    def create_processing_request_message(
        self,
        original_message: Dict[str, Any],
        asset_path: str,
        version_id: str,
        file_type: str,
        user_id: str,
        branch_id: str,
        filename: str,
        status: str,
        temp_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates the Kafka message for a processing request."""
        service_map = {
            "document": "document_orchestrator_service",
            "image": "image_orchestrator_service",
            "video": "video_orchestrator_service",
            "audio": "audio_orchestrator_service",
        }
        action_map = {
            "document": "document_processing",
            "image": "image_processing",
            "video": "video_processing",
            "audio": "audio_processing",
        }
        
        target_service = service_map.get(file_type, "unknown_service")
        action = action_map.get(file_type, "unknown_processing")

        message = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": "api_gateway",
            "target_service": target_service,
            "message_type": "REQUEST",
            "priority": "MEDIUM",
            "payload": {
                "request_id": str(uuid.uuid4()),
                "user_id": original_message.get("user_id", "unknown"),
                "asset_managemant_download_header": {"X-User-ID": user_id, "X-Branch-ID": branch_id},
                "asset_managemant_download_paramater": {"return_file_content": "false"},
                "action": action,
                "parameters": {
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "primary_filename": filename,
                    "file_type": file_type,
                    "status": status,
                    "branch_id": branch_id,
                    **({"temp_file_path": temp_file_path} if temp_file_path else {}),
                },
                "file_path": f"{asset_path}/{version_id}/{filename}",
                "metadata": {
                    "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_metadata": original_message.get("metadata", {}),
                },
            },
            "retry_count": 0,
            "ttl": 3600,
        }
        return message

    async def send_processing_request(
        self,
        upload_result: Dict[str, Any],
        user_id: str,
        branch_id: str,
        processing_mode: str | None,
        temp_file_path: Optional[str] = None,
    ):
        """Sends a file processing request to the appropriate Kafka topic."""
        asset_path = upload_result.get("asset_path")
        version_id = upload_result.get("version_id")
        filename = upload_result.get("primary_filename")
        upload_status = upload_result.get("status")

        if not all([asset_path, version_id, filename, upload_status]):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload result is missing necessary information.")

        file_type = self.get_file_type(filename)
        if file_type == "unknown":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file type for file: {filename}")

        target_topic = self.settings.kafka_topics.get(file_type)
        if not target_topic:
            _logger.error(f"No Kafka topic configured for file type: {file_type}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error: Kafka topic not configured.")

        original_message = {
            "user_id": user_id,
            "metadata": {
                "upload_source": "api_gateway",
                "client_ip": "N/A", # Can be extracted from request if needed
            }
        }
        if processing_mode:
            original_message["metadata"]["processing_mode"] = processing_mode

        message = self.create_processing_request_message(
            original_message=original_message,
            asset_path=asset_path,
            version_id=version_id,
            file_type=file_type,
            user_id=user_id,
            branch_id=branch_id,
            filename=filename,
            status=upload_status,
            temp_file_path=temp_file_path,
        )

        try:
            _logger.info(f"Sending message to Kafka topic '{target_topic}'", extra={"message_id": message["message_id"]})
            # payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
            await self.producer.send_and_wait(target_topic, message)
            return {"message_id": message["message_id"], "correlation_id": message["correlation_id"]}
        except Exception as e:
            _logger.exception("Failed to send message to Kafka", exc_info=e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to send processing request.")
