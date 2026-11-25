"""
Asset management router (v0.2).
Extends asset.py with module 02's full API surface.
Auth: Keycloak JWT via get_current_user. Kafka token: raw JWT from bearer.
"""
import asyncio
import logging
import os
import uuid
from typing import Any

import aiofiles
from aiokafka import AIOKafkaProducer  # type: ignore[import]
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import Response
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.dependencies import get_current_user, get_storage_service
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


_NFS_UPLOAD_DIR = "/tmp/media_processing/uploads"  # nosec B108 -- fixed base directory constant; the actual per-upload filename below is uuid.uuid4()-random


def _user_from_payload(payload: dict[str, Any]) -> User:
    sub = payload["sub"]
    return User(user_id=sub, branch_id=sub)


async def _save_to_nfs(primary_file: UploadFile, upload_result: dict[str, Any]) -> str | None:
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
                raise OSError("NFS write produced an empty file")
            _logger.info(f"Saved upload to NFS: {dest_path} ({size} bytes)")
            return dest_path
        except Exception as exc:  # noqa: BLE001 -- spans filesystem I/O and upload-stream errors across retries, logged per attempt
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
    summary="[Legacy] Upload file and trigger AI analysis",
    description="""
Upload a single file and trigger the corresponding AI analysis pipeline.

**Legacy:** combines upload and analysis in one call. Prefer `POST /fileupload` to upload only, then trigger analysis separately when ready.

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
    archive_date_or_ttl: str | None = Form(None),
    destroy_date_or_ttl: str | None = Form(None),
    processing_mode: str | None = Form("default"),
    svc: StorageService = Depends(get_storage_service),
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    settings: Settings = Depends(get_settings),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
        # Best-effort: confirm/activate any clone_point() data waiting on
        # this exact asset_path/version_id (VIE01-214's cloned_from fast
        # path), so this legacy one-shot endpoint doesn't leave a clone
        # stuck at "uploaded" forever the way it used to at "unregistered".
        # A 409 here just means the global_active dedup case (already
        # active, nothing to confirm) -- not an error worth surfacing.
        try:
            register_result = await svc.register_asset(upload_result["asset_path"], upload_result["version_id"], user)
            if register_result.get("outcome") == "registered":
                # upload_result was captured before this call flipped the
                # row to "active" -- refresh it so the response reflects
                # what actually happened, not a stale pre-confirm snapshot.
                upload_result["status"] = "active"
        except HTTPException:
            pass
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
    summary="[Legacy] Batch upload files and trigger AI analysis",
    description="""
Upload multiple files at once and trigger AI analysis for each one.

**Legacy:** combines upload and analysis in one call. Prefer `POST /fileupload` to upload only, then trigger analysis separately when ready.

Uploads and analysis are processed concurrently (up to 4 by default). Each file is handled independently — a failure on one does not affect the others.

**Rate limit:** 10 requests/minute
""",
)
@limiter.limit("10/minute")
async def upload_assets_analysis_batch(
    request: Request,
    primary_files: list[UploadFile] = File(...),
    archive_date_or_ttl: str | None = Form(None),
    destroy_date_or_ttl: str | None = Form(None),
    processing_mode: str | None = Form("default"),
    concurrency: int = Form(4),
    svc: StorageService = Depends(get_storage_service),
    producer: AIOKafkaProducer = Depends(get_kafka_producer),
    settings: Settings = Depends(get_settings),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
            # Best-effort: same clone_point() confirm/activate as the
            # single-file endpoint above -- see its comment for why.
            try:
                register_result = await svc.register_asset(result_data["asset_path"], result_data["version_id"], user)
                if register_result.get("outcome") == "registered":
                    # result_data is the same dict object referenced from
                    # upload_result["successes"] -- mutating it in place
                    # keeps the response in sync with the actual flip.
                    result_data["status"] = "active"
            except HTTPException:
                pass
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

@router.post(
    "/fileupload",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Upload file (no analysis triggered)",
    description="""
Upload a single file to storage only. No AI analysis is triggered.

Use this to batch-upload files and decide later which ones to send for analysis, via `POST /fileupload_analysis` or `POST /fileupload_analysis_batch`.

Need associated files too? Upload the primary file here first, then call `POST /add-associated-files/{asset_path}` -- a single
`List[UploadFile]` form field doesn't survive this gateway's multipart proxying intact (only the last file arrives), so it isn't
accepted inline here; the dedicated endpoint doesn't share that problem.

**Rate limit:** 100 requests/minute
""",
)
@limiter.limit("100/minute")
async def upload_asset(
    request: Request,
    primary_file: UploadFile = File(...),
    archive_date_or_ttl: int | str | None = Form(None),
    destroy_date_or_ttl: int | str | None = Form(None),
    svc: StorageService = Depends(get_storage_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.upload_file(
        primary_file=primary_file,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl,
        user=user,
    )


@router.post("/add-associated-files/{asset_path:path}", tags=["Asset"], status_code=status.HTTP_200_OK, include_in_schema=False)
async def add_associated_files(
    asset_path: str,
    associated_files: list[UploadFile] = File(...),
    primary_version_id: str | None = Form(None),
    svc: StorageService = Depends(get_storage_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user = _user_from_payload(current_user)
    result = await svc.list_versions(asset_path=asset_path, filename=filename, user=user)
    return result if isinstance(result, dict) else {"versions": result}


@router.get(
    "/users/commits",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="List my upload history",
    description="Returns all upload records for the current user. Supports filtering by keyword, date range, and status, with pagination. "
                "`status` is repeatable (e.g. `?status=uploaded&status=failed`); omitted means all 6 statuses "
                "(uploaded/processing/active/failed/unregistered/archived) are returned, not just active. "
                "Results include `asset_path` and `version_id` for use with other asset endpoints.",
)
async def get_user_commits(
    keyword: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 10,
    status_filter: list[str] | None = Query(None, alias="status"),
    svc: StorageService = Depends(get_storage_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.get_user_commits(
        user=user,
        keyword=keyword,
        statuses=status_filter,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
    )


class ScanFile(BaseModel):
    path: str
    checksum: str


class ScanRequestBody(BaseModel):
    files: list[ScanFile]


@router.post(
    "/scan",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Batch duplicate-content check",
    description="MTR-8 / VIE01-307. For each (path, checksum) pair, reports whether that "
                "content already exists -- no upload. Empty `files` returns an empty result, "
                "not an error. 400 on a malformed checksum.",
)
async def scan_assets(
    body: ScanRequestBody,
    svc: StorageService = Depends(get_storage_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.scan_assets(files=[f.model_dump() for f in body.files], user=user)


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
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.archive_asset(asset_path=asset_path, version_id=version_id, user=user)


@router.get(
    "/thumbnail/{asset_path:path}/{version_id}",
    tags=["Asset"],
    summary="Extract a video thumbnail via NVDEC at a given timestamp",
    description="MTR-7 / VIE01-306. Decodes a single JPEG frame near `second` "
                "directly from the asset's presigned URL (NVDEC hardware decode, "
                "HTTP Range -- no full-file download). Cached 7 days per "
                "asset_path/version_id/second. 404 if the asset/version doesn't "
                "exist, 409 if it isn't `active`, 400 if `second` is beyond the "
                "video's duration.",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def get_thumbnail(
    asset_path: str,
    version_id: str,
    second: float = Query(..., description="Timestamp in seconds to extract the thumbnail from"),
    svc: StorageService = Depends(get_storage_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    user = _user_from_payload(current_user)
    jpeg_bytes = await svc.get_thumbnail(asset_path=asset_path, version_id=version_id, second=second, user=user)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.post(
    "/unregister/{asset_path:path}/{version_id}",
    tags=["Asset"],
    status_code=status.HTTP_200_OK,
    summary="Unregister an active file version (filter-only)",
    description="MTR-5 / VIE01-305: flip an active version's status to "
                "'unregistered' so it drops out of search/index results "
                "without deleting any embedding, vector, or graph data. "
                "Only legal from 'active' -- 409 otherwise. Reversible via "
                "POST /processing/process-file (there is no standalone "
                "/register route), which fast-paths straight back to "
                "'active' with no re-analysis.",
)
async def unregister_asset(
    asset_path: str,
    version_id: str,
    svc: StorageService = Depends(get_storage_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.unregister_asset(asset_path=asset_path, version_id=version_id, user=user)


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
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
    archive_date_or_ttl: int | str | None = Form(None),
    destroy_date_or_ttl: int | str | None = Form(None),
    svc: StorageService = Depends(get_storage_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
    primary_files: list[UploadFile] = File(...),
    archive_date_or_ttl: int | str | None = Form(None),
    destroy_date_or_ttl: int | str | None = Form(None),
    concurrency: int = Form(4),
    svc: StorageService = Depends(get_storage_service),
    settings: Settings = Depends(get_settings),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user = _user_from_payload(current_user)
    return await svc.upload_files_batch(
        primary_files=primary_files,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl,
        user=user,
        concurrency=min(concurrency, settings.batch_upload_concurrency),
    )
