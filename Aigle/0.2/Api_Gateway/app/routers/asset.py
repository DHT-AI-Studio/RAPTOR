"""
API router for handling direct file uploads and proxying them to the object storage service.
"""
import logging
from typing import Any, Dict, Optional, List, Union

import httpx
from aiokafka import AIOKafkaProducer  # type: ignore[import]
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
    status,
)
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.dependencies import get_current_user
from app.core.config import Settings, get_settings
from app.services.kafka_service import KafkaService
from app.services.storage_service import StorageService

_logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Dependency to get the application's HTTP client."""
    client = getattr(request.app.state, "http_client", None)
    if not client:
        raise RuntimeError("HTTP client is not initialised")
    return client


def get_kafka_producer(request: Request) -> AIOKafkaProducer:
    """Dependency to get the application's Kafka producer."""
    producer = getattr(request.app.state, "kafka_producer", None)
    if not producer:
        raise RuntimeError("Kafka producer is not initialised")
    return producer


@router.post("/fileupload", tags=["file-upload"], status_code=status.HTTP_200_OK)
@limiter.limit("100/minute")
async def upload_file_to_storage(
    request: Request,
    primary_file: UploadFile = File(...),
    archive_ttl: Optional[str] = Form(30),
    destroy_ttl: Optional[str] = Form(30),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Handles file uploads by proxying them to the backend object storage service.
    Uses streaming upload to minimize memory usage.

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/fileupload"

    with open(file_path, "rb") as f:
        files = {"primary_file": f}
        data = {
            "archive_ttl": archive_ttl,
            "destroy_ttl": destroy_ttl
        }
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = requests.post(url, files=files, data=data, headers=headers)

    response.raise_for_status()
    result = response.json()
    ```
    """

    storage_service = StorageService(client, settings)

    upload_result = await storage_service.upload_file(
        primary_file=primary_file,
        archive_ttl=archive_ttl,
        destroy_ttl=destroy_ttl,
    )

    _ = current_user  # ensure dependency is evaluated

    return upload_result

@router.post("/fileupload_batch", tags=["file-upload"], status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def upload_files_to_storage_batch(
    request: Request,
    primary_files: List[UploadFile] = File(..., description="Multiple files to upload"),
    archive_ttl: Optional[str] = Form(30),
    destroy_ttl: Optional[str] = Form(30),
    concurrency: int = Form(4, description="Max concurrent uploads", ge=1, le=16),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Batch upload multiple files to the backend object storage service with bounded concurrency.
    
    **Parameters:**
    - `primary_files`: Multiple files to upload
    - `archive_ttl`: Archive time-to-live (default: 30 days)
    - `destroy_ttl`: Destroy time-to-live (default: 30 days)
    - `concurrency`: Max concurrent uploads (1-16, default: 4)
    
    **Returns:**
    - Batch upload result with per-file success/failure status

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/fileupload_batch"

    files = []
    try:
        for file_path in file_paths:
            files.append(("primary_files", open(file_path, "rb")))

        data = {
            "archive_ttl": archive_ttl,
            "destroy_ttl": destroy_ttl,
            "concurrency": concurrency
        }
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = requests.post(url, files=files, data=data, headers=headers)
        response.raise_for_status()
        result = response.json()
    finally:
        for _, file_obj in files:
            file_obj.close()
    ```
    """
    storage_service = StorageService(client, settings)

    result = await storage_service.upload_files_batch(
        primary_files=primary_files,
        archive_ttl=archive_ttl,
        destroy_ttl=destroy_ttl,
        concurrency=min(concurrency, settings.batch_upload_concurrency),
    )

    _ = current_user  # ensure dependency is evaluated
    return result

@router.post("/fileupload_analysis_batch", tags=["file-upload"], status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def upload_files_to_storage_analysis_batch(
    request: Request,
    primary_files: List[UploadFile] = File(..., description="Multiple files to upload and analyze"),
    archive_ttl: Optional[str] = Form(30),
    destroy_ttl: Optional[str] = Form(30),
    processing_mode: Optional[str] = Form("default"),
    concurrency: int = Form(4, description="Max concurrent uploads", ge=1, le=16),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Batch upload multiple files to the backend object storage service with bounded concurrency,
    and automatically send each successfully uploaded file to Kafka for processing/analysis.
    
    **Parameters:**
    - `primary_files`: Multiple files to upload
    - `archive_ttl`: Archive time-to-live (default: 30 days)
    - `destroy_ttl`: Destroy time-to-live (default: 30 days)
    - `processing_mode`: Processing mode for analysis (default: 'default')
    - `concurrency`: Max concurrent uploads (1-16, default: 4)
    
    **Returns:**
    - Batch upload result with per-file success/failure status
    - Kafka processing results for successfully uploaded files

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/fileupload_analysis_batch"

    files = []
    try:
        for file_path in file_paths:
            files.append(("primary_files", open(file_path, "rb")))

        data = {
            "processing_mode": processing_mode,
            "archive_ttl": archive_ttl,
            "destroy_ttl": destroy_ttl,
            "concurrency": concurrency
        }
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = requests.post(url, files=files, data=data, headers=headers)
        response.raise_for_status()
        result = response.json()
    finally:
        for _, file_obj in files:
            file_obj.close()
    ```
    """
    storage_service = StorageService(client, settings)

    # 批次上傳檔案
    upload_result = await storage_service.upload_files_batch(
        primary_files=primary_files,
        archive_ttl=archive_ttl,
        destroy_ttl=destroy_ttl,
        concurrency=min(concurrency, settings.batch_upload_concurrency),
    )

    user_id = current_user["sub"]
    
    # Extract the raw token for forwarding
    auth_header = request.headers.get("Authorization", "")
    _, _, access_token = auth_header.partition(" ")

    # 為每個成功上傳的檔案發送 Kafka 處理請求
    kafka_service = KafkaService(producer, settings)
    processing_results = []
    
    for success_item in upload_result.get("successes", []):
        result_data = success_item.get("result", {})
        change_status = result_data.get("change_status", {})
        
        # 檢查 change_status，如果 changed 為 False 則跳過 Kafka 處理
        if not change_status.get("changed", True):
            _logger.info(
                f"Skipping Kafka processing for duplicate file: {success_item.get('filename')} - "
                f"{change_status.get('message', 'File already exists')}"
            )
            processing_results.append({
                "filename": success_item.get("filename"),
                "status": "skipped",
                "reason": "duplicate_file",
                "message": change_status.get("message", "File already exists, skipping processing"),
            })
            continue
        
        try:
            processing_result = await kafka_service.send_processing_request(
                upload_result=result_data,
                user_id=user_id,
                access_token=await storage_service.get_access_token(),
                processing_mode=processing_mode,
            )
            processing_results.append({
                "filename": success_item.get("filename"),
                "status": "kafka_sent",
                "kafka_result": processing_result,
            })
        except Exception as e:
            _logger.error(
                f"Failed to send Kafka message for file {success_item.get('filename')}",
                exc_info=e,
            )
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

@router.post("/fileupload_analysis", tags=["file-upload"], status_code=status.HTTP_200_OK)
@limiter.limit("50/minute")
async def upload_file_to_storage_analysis(
    request: Request,
    primary_file: UploadFile = File(...),
    archive_ttl: Optional[str] = Form(30),
    destroy_ttl: Optional[str] = Form(30),
    processing_mode: Optional[str] = Form("default"),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Handles file uploads and schedules downstream analysis via Kafka after upload succeeds.

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/fileupload_analysis"

    with open(file_path, "rb") as f:
        files = {"primary_file": f}
        data = {
            "processing_mode": processing_mode,
            "archive_ttl": archive_ttl,
            "destroy_ttl": destroy_ttl
        }
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = requests.post(url, files=files, data=data, headers=headers)

    response.raise_for_status()
    result = response.json()
    ```
    """

    storage_service = StorageService(client, settings)

    upload_result = await storage_service.upload_file(
        primary_file=primary_file,
        archive_ttl=archive_ttl,
        destroy_ttl=destroy_ttl,
    )

    user_id = current_user["sub"]

    # 檢查 change_status，如果 changed 為 False 則跳過 Kafka 處理
    change_status = upload_result.get("change_status", {})
    if not change_status.get("changed", True):
        _logger.info(
            f"Skipping Kafka processing for duplicate file: {upload_result.get('primary_filename')} - "
            f"{change_status.get('message', 'File already exists')}"
        )
        return {
            "upload_result": upload_result,
            "processing_result": {
                "status": "skipped",
                "reason": "duplicate_file",
                "message": change_status.get("message", "File already exists, skipping processing"),
            },
        }

    kafka_service = KafkaService(producer, settings)
    processing_result = await kafka_service.send_processing_request(
        upload_result=upload_result,
        user_id=user_id,
        access_token=await storage_service.get_access_token(),  
        processing_mode=processing_mode,
    )

    return {
        "upload_result": upload_result,
        "processing_result": processing_result,
    }


@router.get("/fileversions", tags=["asset-management"], status_code=status.HTTP_200_OK)
async def list_versions(
    asset_path: str = Query(..., description="Asset path"),
    filename: str = Query(..., description="Filename"),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Union[Dict[str, Any], List[Any]]:
    """
    List all versions of an asset.
    
    Parameters:
    - asset_path: The asset path identifier
    - filename: The filename to list versions for

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/fileversions"
    params = {
        "asset_path": asset_path,
        "filename": filename
    }
    response = requests.get(url, params=params, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    storage_service = StorageService(client, settings)
    
    try:
        result = await storage_service.list_versions(
            asset_path=asset_path,
            filename=filename,
        )
        # 確保返回正確的格式
        if isinstance(result, list):
            return {"versions": result}
        elif isinstance(result, dict):
            return result
        else:
            return {"data": result}
    except HTTPException:
        raise
    except Exception as exc:
        _logger.error(f"Unexpected error in list_versions: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )

@router.get("/filedownload", tags=["asset-management"], status_code=status.HTTP_200_OK)
async def download_asset(
    asset_path: str = Query(..., description="Asset path"),
    version_id: str = Query(..., description="Version ID"),
    return_file_content: bool = Query(False, description="Return file content"),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Download an asset from object storage.
    
    Parameters:
    - asset_path: The asset path identifier
    - version_id: The version ID to download
    - return_file_content: Whether to return the file content in the response

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/filedownload"
    params = {
        "asset_path": asset_path,
        "version_id": version_id,
        "return_file_content": return_file_content
    }
    response = requests.get(url, params=params, headers=self._get_headers())
    response.raise_for_status()

    if return_file_content:
        content = response.content
    else:
        result = response.json()
    ```
    """
    storage_service = StorageService(client, settings)
    
    try:
        result = await storage_service.download_asset(
            asset_path=asset_path,
            version_id=version_id,
            return_file_content=return_file_content,
        )
        # 確保返回字典
        if isinstance(result, dict):
            return result
        else:
            return {"data": result}
    except HTTPException:
        raise
    except Exception as exc:
        _logger.error(f"Unexpected error in download_asset: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )

@router.post("/filearchive", tags=["asset-management"], status_code=status.HTTP_200_OK)
async def archive_asset(
    asset_path: str = Query(..., description="Asset path"),
    version_id: str = Query(..., description="Version ID"),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Archive an asset in object storage.
    
    Parameters:
    - asset_path: The asset path identifier
    - version_id: The version ID to archive

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/filearchive"
    params = {
        "asset_path": asset_path,
        "version_id": version_id
    }
    response = requests.post(url, params=params, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    storage_service = StorageService(client, settings)
    
    try:
        result = await storage_service.archive_asset(
            asset_path=asset_path,
            version_id=version_id,
        )
        # 確保返回字典
        if isinstance(result, dict):
            return result
        else:
            return {"data": result}
    except HTTPException:
        raise
    except Exception as exc:
        _logger.error(f"Unexpected error in archive_asset: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )

@router.post("/delfile", tags=["asset-management"], status_code=status.HTTP_200_OK)
async def delete_asset(
    asset_path: str = Query(..., description="Asset path"),
    version_id: str = Query(..., description="Version ID"),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_http_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Delete an archived asset from object storage.
    
    Parameters:
    - asset_path: The asset path identifier
    - version_id: The version ID to delete

    **Usage:**
    ```python
    url = f"{base_url}/api/v1/asset/delfile"
    params = {
        "asset_path": asset_path,
        "version_id": version_id
    }
    response = requests.post(url, params=params, headers=self._get_headers())
    response.raise_for_status()
    result = response.json()
    ```
    """
    storage_service = StorageService(client, settings)
    
    try:
        result = await storage_service.delete_asset(
            asset_path=asset_path,
            version_id=version_id,
        )
        # 確保返回字典
        if isinstance(result, dict):
            return result
        else:
            return {"data": result}
    except HTTPException:
        raise
    except Exception as exc:
        _logger.error(f"Unexpected error in delete_asset: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )