"""
Asset management router (v0.2).
Extends asset.py with module 02's full API surface.
Auth: Keycloak JWT via get_current_user. Kafka token: raw JWT from bearer.
"""
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import asyncio
import aiofiles
from aiokafka import AIOKafkaProducer  # type: ignore[import]
from fastapi import (
    APIRouter, Depends, File, Form, HTTPException,
    Request, UploadFile, status,
)
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import Settings, get_settings
from app.services.kafka_service import KafkaService
from app.services.storage_service import StorageService, User

_logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# Dependencies
# ============================================================================

def get_kafka_producer(request: Request) -> AIOKafkaProducer:
    producer = getattr(request.app.state, "kafka_producer", None)
    if not producer:
        raise RuntimeError("Kafka producer is not initialised")
    return producer


_NFS_UPLOAD_DIR = "/tmp/media_processing/uploads"


def _user_from_payload(payload: Dict[str, Any]) -> User:
    sub = payload["sub"]
    return User(user_id=sub, branch_id=sub)


async def _save_to_nfs(primary_file: UploadFile, upload_result: Dict[str, Any]) -> Optional[str]:
    """Save the uploaded file to the shared NFS volume so processing workers
    can read it directly without re-downloading from LakeFS."""
    version_id = upload_result.get("version_id", "")
    filename = upload_result.get("primary_filename") or primary_file.filename or "upload"
    ext = os.path.splitext(filename)[1] or ""
    dest_dir = os.path.join(_NFS_UPLOAD_DIR, version_id)
    dest_path = os.path.join(dest_dir, f"{uuid.uuid4()}{ext}")

    for attempt in range(1, 4):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            await primary_file.seek(0)
            async with aiofiles.open(dest_path, "wb") as f:
                while True:
                    chunk = await primary_file.read(65536)
                    if not chunk:
                        break
                    await f.write(chunk)
            size = os.path.getsize(dest_path)
            if size == 0:
                raise IOError("NFS write produced an empty file")
            _logger.info(f"Saved upload to NFS: {dest_path} ({size} bytes)")
            return dest_path
        except Exception as exc:
            _logger.warning(f"NFS save attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                await asyncio.sleep(1.0)

    _logger.error("NFS save failed after 3 attempts, processing will fall back to LakeFS download")
    return None


# ============================================================================
# Upload + Analysis (upload → Kafka)
# ============================================================================

@router.post(
    "/fileupload_analysis",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Upload file and trigger AI analysis",
    description="""
Upload a single file and trigger the corresponding AI analysis pipeline.

**Supported file types:**
- **document** (PDF, Word, CSV…) — OCR / text extraction / embedding
- **image** (JPG, PNG…) — object detection / scene description / embedding
- **audio** (MP3, WAV…) — speech recognition (ASR) / speaker diarization
- **video** (MP4, MOV…) — frame description / ASR / OCR

**Duplicate handling:**
If the same file already exists (detected by content hash), the analysis step is skipped and `processing_result.status` will be `skipped`.

**Rate limit:** 50 requests/minute
""",
)
@limiter.limit("50/minute")
async def upload_asset_analysis(
    request: Request,
    primary_file: UploadFile = File(...),
    archive_date_or_ttl: Optional[str] = Form(None),
    destroy_date_or_ttl: Optional[str] = Form(None),
    processing_mode: Optional[str] = Form("default"),
    svc: StorageService = Depends(get_storage_service),
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    settings: Settings = Depends(get_settings),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    kafka_service = KafkaService(producer, settings, getattr(request.app.state, "redis_client", None))

    upload_result = await svc.upload_file(
        primary_file=primary_file,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl,
        user=user,
    )

    existence_info = upload_result.get("existence_info", {})
    if existence_info.get("exists", False):
        return {
            "upload_result": upload_result,
            "processing_result": {
                "status": "skipped",
                "reason": "duplicate_file",
                "message": existence_info.get("message", "File already exists, skipping processing"),
            },
        }

    temp_file_path = await _save_to_nfs(primary_file, upload_result)

    processing_result = await kafka_service.send_processing_request(
        upload_result=upload_result,
        user_id=user.user_id,
        branch_id=user.branch_id,
        processing_mode=processing_mode,
        temp_file_path=temp_file_path,
    )
    return {"upload_result": upload_result, "processing_result": processing_result}


@router.post(
    "/fileupload_analysis_batch",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Batch upload files and trigger AI analysis",
    description="""
Upload multiple files at once and trigger AI analysis for each one.

Uploads and analysis are processed concurrently (up to 4 by default). Each file is handled independently — a failure on one does not affect the others.

**Rate limit:** 10 requests/minute
""",
)
@limiter.limit("10/minute")
async def upload_assets_analysis_batch(
    request: Request,
    primary_files: List[UploadFile] = File(...),
    archive_date_or_ttl: Optional[str] = Form(None),
    destroy_date_or_ttl: Optional[str] = Form(None),
    processing_mode: Optional[str] = Form("default"),
    concurrency: int = Form(4),
    svc: StorageService = Depends(get_storage_service),
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    settings: Settings = Depends(get_settings),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    kafka_service = KafkaService(producer, settings, getattr(request.app.state, "redis_client", None))

    upload_result = await svc.upload_files_batch(
        primary_files=primary_files,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl,
        user=user,
        concurrency=min(concurrency, settings.batch_upload_concurrency),
    )

    file_by_index = {i: f for i, f in enumerate(primary_files)}

    processing_results = []
    for success_item in upload_result.get("successes", []):
        result_data = success_item.get("result", {})
        existence_info = result_data.get("existence_info", {})
        idx = success_item.get("index")

        if existence_info.get("exists", False):
            processing_results.append({
                "filename": success_item.get("filename"),
                "status": "skipped",
                "reason": "duplicate_file",
                "message": existence_info.get("message", "File already exists"),
            })
            continue

        temp_file_path = None
        if idx is not None and idx in file_by_index:
            temp_file_path = await _save_to_nfs(file_by_index[idx], result_data)

        try:
            kafka_result = await kafka_service.send_processing_request(
                upload_result=result_data,
                user_id=user.user_id,
                branch_id=user.branch_id,
                processing_mode=processing_mode,
                temp_file_path=temp_file_path,
            )
            processing_results.append({
                "filename": success_item.get("filename"),
                "status": "kafka_sent",
                "kafka_result": kafka_result,
            })
        except Exception as e:
            _logger.error(f"Kafka send failed for {success_item.get('filename')}", exc_info=e)
            processing_results.append({
                "filename": success_item.get("filename"),
                "status": "kafka_failed",
                "error": str(e),
            })

    return {
        "upload_summary": {
            "total": upload_result.get("total"),
            "success_count": upload_result.get("success_count"),
            "failure_count": upload_result.get("failure_count"),
        },
        "upload_successes": upload_result.get("successes", []),
        "upload_failures": upload_result.get("failures", []),
        "processing_results": processing_results,
    }


# ============================================================================
# Asset CRUD
# ============================================================================

@router.post("/fileupload", tags=["Asset"], status_code=status.HTTP_200_OK, include_in_schema=False)
@limiter.limit("100/minute")
async def upload_asset(
    request: Request,
    primary_file: UploadFile = File(...),
    associated_files: Optional[List[UploadFile]] = File(None),
    archive_date_or_ttl: Optional[Union[int, str]] = Form(None),
    destroy_date_or_ttl: Optional[Union[int, str]] = Form(None),
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    result = await svc.upload_file(
        primary_file=primary_file,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl,
        user=user,
    )
    # Upload associated files if provided
    if associated_files:
        await svc.add_associated_files(
            asset_path=result.get("asset_path", ""),
            associated_files=associated_files,
            user=user,
            primary_version_id=result.get("version_id"),
        )
    return result


@router.post("/add-associated-files/{asset_path:path}", tags=["Asset"], status_code=status.HTTP_200_OK, include_in_schema=False)
async def add_associated_files(
    asset_path: str,
    associated_files: List[UploadFile] = File(...),
    primary_version_id: Optional[str] = Form(None),
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.add_associated_files(
        asset_path=asset_path,
        associated_files=associated_files,
        user=user,
        primary_version_id=primary_version_id,
    )


@router.get(
    "/filedownload/{asset_path:path}/{version_id}",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Get file download link",
    description="Obtain `asset_path` and `version_id` from the upload response or `/users/commits`. Set `return_file_content=true` to return the raw file bytes; otherwise a pre-signed URL is returned.",
)
async def download_asset(
    asset_path: str,
    version_id: str,
    return_file_content: bool = False,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.download_asset(
        asset_path=asset_path,
        version_id=version_id,
        user=user,
        return_file_content=return_file_content,
    )


@router.get(
    "/fileversions/{asset_path:path}/{filename}",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="List file version history",
    description="Returns all versions of a file at the given path, including `version_id`, upload time, and size.",
)
async def list_versions(
    asset_path: str,
    filename: str,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    result = await svc.list_versions(asset_path=asset_path, filename=filename, user=user)
    return result if isinstance(result, dict) else {"versions": result}


@router.get(
    "/users/commits",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="List my upload history",
    description="Returns all upload records for the current user. Supports filtering by keyword and date range, with pagination. Results include `asset_path` and `version_id` for use with other asset endpoints.",
)
async def get_user_commits(
    keyword: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.get_user_commits(
        user=user,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/filearchive/{asset_path:path}/{version_id}",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Archive a file version",
    description="Mark a specific version as archived. Archived files are excluded from regular search results but remain accessible by version ID.",
)
async def archive_asset(
    asset_path: str,
    version_id: str,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.archive_asset(asset_path=asset_path, version_id=version_id, user=user)


@router.post(
    "/delfile/{asset_path:path}/{version_id}",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Delete a file version",
    description="Permanently delete a specific file version and its associated embedding indexes. **This operation cannot be undone.**",
)
async def delete_asset(
    asset_path: str,
    version_id: str,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.delete_asset(asset_path=asset_path, version_id=version_id, user=user)


@router.post(
    "/file-expiration/{asset_path:path}/{version_id}",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Update file expiration",
    description="""
Update the archive or destroy date for a specific file version.

- `archive_date_or_ttl` — ISO 8601 date string or TTL in seconds; the file will be archived at this time
- `destroy_date_or_ttl` — ISO 8601 date string or TTL in seconds; the file will be automatically deleted at this time
""",
)
async def update_asset_expiration(
    asset_path: str,
    version_id: str,
    archive_date_or_ttl: Optional[Union[int, str]] = Form(None),
    destroy_date_or_ttl: Optional[Union[int, str]] = Form(None),
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.update_expiration(
        asset_path=asset_path,
        version_id=version_id,
        user=user,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl,
    )


@router.post("/fileupload_batch", tags=["Asset"], status_code=status.HTTP_200_OK, include_in_schema=False)
@limiter.limit("10/minute")
async def upload_assets_batch(
    request: Request,
    primary_files: List[UploadFile] = File(...),
    archive_date_or_ttl: Optional[Union[int, str]] = Form(None),
    destroy_date_or_ttl: Optional[Union[int, str]] = Form(None),
    concurrency: int = Form(4),
    svc: StorageService = Depends(get_storage_service),
    settings: Settings = Depends(get_settings),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.upload_files_batch(
        primary_files=primary_files,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl,
        user=user,
        concurrency=min(concurrency, settings.batch_upload_concurrency),
    )
