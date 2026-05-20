"""
Asset management router (v0.2).
Extends asset.py with module 02's full API surface.
Auth: Keycloak JWT via get_current_user. Kafka token: raw JWT from bearer.
"""
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union

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
    try:
        version_id = upload_result.get("version_id", "")
        filename = upload_result.get("primary_filename") or primary_file.filename or "upload"
        ext = os.path.splitext(filename)[1] or ""
        dest_dir = os.path.join(_NFS_UPLOAD_DIR, version_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{uuid.uuid4()}{ext}")

        await primary_file.seek(0)
        async with aiofiles.open(dest_path, "wb") as f:
            while True:
                chunk = await primary_file.read(65536)
                if not chunk:
                    break
                await f.write(chunk)

        _logger.info(f"Saved upload to NFS: {dest_path}")
        return dest_path
    except Exception as exc:
        _logger.warning(f"NFS save failed, processing will fall back to LakeFS download: {exc}")
        return None


# ============================================================================
# Upload + Analysis (upload → Kafka)
# ============================================================================

@router.post("/fileupload_analysis", tags=["Asset"], status_code=status.HTTP_200_OK)
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
    kafka_service = KafkaService(producer, settings)

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


@router.post("/fileupload_analysis_batch", tags=["Asset"], status_code=status.HTTP_200_OK)
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
    kafka_service = KafkaService(producer, settings)

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


@router.get("/filedownload/{asset_path:path}/{version_id}", tags=["Asset"], status_code=status.HTTP_200_OK)
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


@router.get("/fileversions/{asset_path:path}/{filename}", tags=["Asset"], status_code=status.HTTP_200_OK)
async def list_versions(
    asset_path: str,
    filename: str,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    result = await svc.list_versions(asset_path=asset_path, filename=filename, user=user)
    return result if isinstance(result, dict) else {"versions": result}


@router.get("/users/commits", tags=["Asset"], status_code=status.HTTP_200_OK)
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


@router.post("/filearchive/{asset_path:path}/{version_id}", tags=["Asset"], status_code=status.HTTP_200_OK)
async def archive_asset(
    asset_path: str,
    version_id: str,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.archive_asset(asset_path=asset_path, version_id=version_id, user=user)


@router.post("/delfile/{asset_path:path}/{version_id}", tags=["Asset"], status_code=status.HTTP_200_OK)
async def delete_asset(
    asset_path: str,
    version_id: str,
    svc: StorageService = Depends(get_storage_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.delete_asset(asset_path=asset_path, version_id=version_id, user=user)


@router.post("/file-expiration/{asset_path:path}/{version_id}", tags=["Asset"], status_code=status.HTTP_200_OK)
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
