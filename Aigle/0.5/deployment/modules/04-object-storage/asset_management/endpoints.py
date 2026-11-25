import asyncio
import logging

# from datetime import datetime, timedelta, timezone
import os
from contextlib import asynccontextmanager
from typing import Literal
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .client import AssetManager
from .config import settings
from .database import Database
from .maintenance_mode import MaintenanceModeMiddleware
from .models import (
    AssetMetadataResponse,
    ModerationFlags,
    PresignBatchRequest,
    PresignBatchResponse,
    PresignBatchResult,
    ScanRequest,
    ScanResponse,
    User,
    model_to_response,
)
from .object_store import ObjectStore
from .search_sync import SearchSync
from .thumbnail_service import ThumbnailService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

db = Database()
object_store = ObjectStore()
search_sync = SearchSync()
manager = AssetManager(db=db, object_store=object_store, search_sync=search_sync)
thumbnail_service = ThumbnailService(object_store=object_store, db=db)
scheduler = AsyncIOScheduler()

TIMEZONE = settings.timezone
AUTO_DAILY_ARCHIVE_TIME = settings.auto_daily_archive_time
AUTO_DAILY_DESTROY_TIME = settings.auto_daily_destroy_time
ARCHIVE_HOUR, ARCHIVE_MINUTE = settings.auto_daily_archive_hour_minute
DESTROY_HOUR, DESTROY_MINUTE = settings.auto_daily_destroy_hour_minute


# -------------------- User dependency --------------------
async def get_user(
    user_id: str = Header(..., alias="X-User-ID"),
    branch_id: str = Header(..., alias="X-Branch-ID")
) -> User:
    """
    Resolve the current authenticated user context from request headers.

    This dependency is designed for internal service usage behind an API Gateway.
    Authentication (e.g. JWT / OAuth2 / Keycloak) is expected to be handled by the
    Gateway layer. After successful authentication and authorization, the Gateway
    must inject the following headers:

    - X-User-ID:
        The unique identifier of the authenticated user (usually a UUID).
        This value represents the user who performs the operation.

    - X-Branch-ID:
        The logical workspace or namespace where the operation is performed.
        This can be:
            - the user's own UUID (private workspace)
            - a group UUID (shared workspace)

    The asset service does not distinguish between private or group workspaces.
    All access control decisions are delegated to the Gateway.

    Returns:
        User:
            A lightweight user context containing:
            - user_id: the authenticated user identifier
            - branch_id: the target workspace identifier

    Raises:
        HTTPException(400):
            If required headers are missing.
    """
    if not user_id or not branch_id:
        raise HTTPException(
            status_code=400,
            detail="Missing X-User-ID or X-Branch-ID headers"
        )

    return User(user_id=user_id, branch_id=branch_id)


# -------------------- Scheduler tasks --------------------
async def run_auto_archive():
    try:
        archived_assets = await manager.auto_archive()
        # archived_assets = await manager.auto_archive(datetime.now()+timedelta(days=90)) # For testing
        logger.info(f"Auto-archive completed: {len(archived_assets)} assets archived")
    except Exception as e:  # noqa: BLE001 -- scheduled job; db/network errors from get_assets_to_archive must never crash the scheduler
        logger.error(f"Auto-archive failed: {e!s}")

async def run_auto_destroy():
    try:
        destroyed_assets = await manager.auto_destroy()
        # destroyed_assets = await manager.auto_destroy(datetime.now()+timedelta(days=180)) # For testing
        logger.info(f"Auto-destroy completed: {len(destroyed_assets)} assets destroyed")
    except Exception as e:  # noqa: BLE001 -- scheduled job; db/network errors from get_assets_to_destroy must never crash the scheduler
        logger.error(f"Auto-destroy failed: {e!s}")


# -------------------- App lifespan --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await object_store.initialize()
    await search_sync.initialize()
    if object_store._redis:
        try:
            await object_store._redis.ping()
            logger.info("Redis cache connected for presigned URLs")
        except RedisError as _e:
            logger.warning(f"Redis ping failed, presigned URL caching disabled: {_e}")
            object_store._redis = None

    # Dedicated connection for the maintenance-mode write-freeze check (VIE01-369) --
    # deliberately separate from object_store._redis, which PRESIGN_URL_CACHE_ENABLED=false
    # disables independently of this.
    app.state.redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=True,
    )

    scheduler.start()
    scheduler.add_job(
        run_auto_archive,
        CronTrigger(hour=ARCHIVE_HOUR, minute=ARCHIVE_MINUTE, timezone=ZoneInfo(TIMEZONE)),
        # CronTrigger(hour="11", minute="50", timezone=ZoneInfo(TIMEZONE)), # For testing
        id="auto_archive",
        name=f"Daily auto-archive task at {AUTO_DAILY_ARCHIVE_TIME}"
    )
    scheduler.add_job(
        run_auto_destroy,
        CronTrigger(hour=DESTROY_HOUR, minute=DESTROY_MINUTE, timezone=ZoneInfo(TIMEZONE)),
        # CronTrigger(hour="12", minute="00", timezone=ZoneInfo(TIMEZONE)), # For testing
        id="auto_destroy",
        name=f"Daily auto-destroy task at {AUTO_DAILY_DESTROY_TIME}"
    )
    logger.info("Scheduler started with auto-archive and auto-destroy tasks")

    yield

    scheduler.shutdown()
    await db.close()
    await object_store.close()
    await search_sync.close()
    await app.state.redis_client.aclose()


# -------------------- FastAPI app --------------------
app = FastAPI(lifespan=lifespan)
app.add_middleware(MaintenanceModeMiddleware)


# -------------------- Global exception handler --------------------
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """
    VIE01-214: a pydantic ValidationError (e.g. an AssetMetadata/
    AssetMetadataResponse constructed with a status outside AssetStatus's
    Literal) is a client-caused bad-input problem, not a server fault —
    reject with 400, not the generic 500 below. Registered ahead of the
    catch-all Exception handler so this more specific type wins.
    """
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid data: {exc!s}"}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Intercept all unexpected exceptions.
    If it's a normal HTTPException (like 400, 404), FastAPI will handle it first and won't reach here.
    """
    logger.error(f"Unexpected Error on {request.url.path}: {exc!s}", exc_info=True)  # noqa: LOG014 -- invoked from Starlette's own except block, traceback is real
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {exc!s}"}
    )


# -------------------- File APIs --------------------
@app.post("/fileupload", summary="Upload a new asset")
async def upload_asset(
    primary_file: UploadFile = File(...),
    associated_files: list[UploadFile] = File([], description='If you do not want to upload files, uncheck "Send empty value".'),
    archive_date_or_ttl: int | str | None = Form(
        None,
        description=(
            "Archive date or TTL. "
            "Accepts an absolute date (YYYY-MM-DD or YYYY/MM/DD), "
            "an ISO datetime (YYYY-MM-DDTHH:MM:SS), "
            "or a relative TTL such as '30d', '2w', '6m', '1y'. "
            "If a number is provided, it is treated as days from now. "
            "Default is 'None' (never archive)."
        ),
    ),
    destroy_date_or_ttl: int | str | None = Form(
        None,
        description=(
            "Destroy date or TTL, relative to the archive date. "
            "Accepts the same formats as archive_date_or_ttl. "
            "For example, '1y' means one year after the archive date. "
            "Default is 'None' (never destroy)."
        ),
    ),
    user: User = Depends(get_user)
):
    read_tasks = [primary_file.read()] + [f.read() for f in associated_files]
    file_datas = await asyncio.gather(*read_tasks)

    primary_data = file_datas[0]
    associated = [(data, f.filename) for data, f in zip(file_datas[1:], associated_files)]

    return await manager.upload_files(
        username=user.user_id,
        branch=user.branch_id,
        primary_file=(primary_data, primary_file.filename),
        associated_files=associated,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl
    )
    

@app.post("/add-associated-files/{asset_path:path}", summary="Add associated files to an existing asset", response_model=AssetMetadataResponse)
async def add_associated_files(
    asset_path: str,
    associated_files: list[UploadFile] = File(...),
    primary_version_id: str | None = Form(None),
    user: User = Depends(get_user)
):
    file_datas = await asyncio.gather(*[f.read() for f in associated_files])
    associated = [(data, f.filename) for data, f in zip(file_datas, associated_files)]

    return await manager.add_associated_files(
        username=user.user_id,
        branch=user.branch_id,
        asset_path=asset_path,
        associated_files=associated,
        primary_version_id=primary_version_id
    )


@app.post("/identity-photo/{identity_id}", summary="Store a person's seed photo", include_in_schema=False)
async def upload_identity_photo(
    identity_id: str,
    photo: UploadFile = File(...),
    user: User = Depends(get_user),
):
    """Internal: called by Module 13 when a user sets a person's photo (dev plan v1.4
    §4.5). Writes only under the identity/ prefix, at a key derived from the
    caller's branch and the identity_id -- the caller never chooses the key."""
    return await manager.upload_identity_photo(
        username=user.user_id,
        branch=user.branch_id,
        identity_id=identity_id,
        data=await photo.read(),
        filename=photo.filename,
    )


@app.get("/presign-url/{key:path}/{version_id}", summary="Get presigned URL for a known file key")
async def presign_url(
    key: str,
    version_id: str,
    user: User = Depends(get_user),
):
    """Lightweight endpoint: returns only the presigned URL for a specific LakeFS key+version.
    Skips DB lookup and associated-file fetching — use when the full file path is already known.
    """
    url = await object_store.generate_presigned_url(key, version_id)
    return {"url": url}


@app.post("/presign-urls-batch", summary="Get presigned URLs for many known file keys", response_model=PresignBatchResponse)
async def presign_urls_batch(
    req: PresignBatchRequest,
    user: User = Depends(get_user),
):
    """Batched /presign-url: resolves many (key, version_id) pairs in as few LakeFS calls as
    possible (object_store.batch_generate_presigned_urls_by_key()) -- items under the same directory
    in the same commit cost one LakeFS list_objects call, not one per key. Best-effort: an item this
    couldn't resolve is simply absent from `resolved`, same trust boundary as /presign-url (no DB
    lookup or ownership check -- the caller is expected to have already verified it).
    """
    resolved = await object_store.batch_generate_presigned_urls_by_key([(i.key, i.version_id) for i in req.items])
    return PresignBatchResponse(resolved=[
        PresignBatchResult(key=k, version_id=v, url=r["url"], content_type=r["content_type"])
        for (k, v), r in resolved.items()
    ])


@app.get("/filedownload/{asset_path:path}/{version_id}", summary="Download an asset")
async def download_asset(
    asset_path: str,
    version_id: str,
    return_file_content: bool = False,
    user: User = Depends(get_user)
):
    return await manager.retrieve_asset(
        username=user.user_id,
        branch=user.branch_id,
        asset_path=asset_path,
        version_id=version_id,
        return_file_content=return_file_content
    )


@app.post("/filearchive/{asset_path:path}/{version_id}", summary="Archive an asset", response_model=AssetMetadataResponse)
async def archive_asset(asset_path: str, version_id: str, user: User = Depends(get_user)):
    return await manager.archive(
        username=user.user_id,
        branch=user.branch_id,
        asset_path=asset_path,
        version_id=version_id
    )


@app.get("/thumbnail/{asset_path:path}/{version_id}", summary="Extract a video thumbnail via NVDEC at a given timestamp")
async def get_thumbnail(asset_path: str, version_id: str, second: float = Query(..., description="Timestamp in seconds to extract the thumbnail from"), user: User = Depends(get_user)):
    asset_path = manager._sanitize_path(asset_path)
    jpeg_bytes = await thumbnail_service.get_thumbnail(
        asset_path=asset_path,
        version_id=version_id,
        branch=user.branch_id,
        second=second,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.post("/scan", summary="Batch duplicate-content check", response_model=ScanResponse)
async def scan_assets(req: ScanRequest, user: User = Depends(get_user)):
    if not req.files:
        return ScanResponse(results=[])
    results = await manager.scan_assets(
        files=[(f.path, f.checksum) for f in req.files],
        branch=user.branch_id,
        concurrency=settings.scan_concurrency,
    )
    return ScanResponse(results=results)


@app.post("/unregister/{asset_path:path}/{version_id}", summary="Unregister an active asset (filter-only -- no data deleted)", response_model=AssetMetadataResponse)
async def unregister_asset(asset_path: str, version_id: str, user: User = Depends(get_user)):
    return await manager.unregister_asset(
        username=user.user_id,
        branch=user.branch_id,
        asset_path=asset_path,
        version_id=version_id
    )


@app.post("/delfile/{asset_path:path}/{version_id}", summary="Delete an archived asset", response_model=AssetMetadataResponse)
async def destroy_asset(asset_path: str, version_id: str, user: User = Depends(get_user)):
    return await manager.destroy_v2(
        username=user.user_id,
        branch=user.branch_id,
        asset_path=asset_path,
        version_id=version_id
    )


@app.post("/register/{asset_path:path}/{version_id}", summary="Register an asset for analysis", include_in_schema=False)
async def register_asset(asset_path: str, version_id: str, user: User = Depends(get_user)):
    """Internal endpoint called by Module 13's Register API (POST /processing/process-file).

    Returns the current status branch so the gateway knows whether to
    publish Kafka: {"outcome": "not_found" | "conflict" | "registered" |
    "needs_processing", ...}. Never itself writes "processing" -- see
    POST /claim-processing for that.
    """
    result = await manager.register_asset(branch=user.branch_id, asset_path=asset_path, version_id=version_id)
    if result["outcome"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}/{version_id}")
    if result["outcome"] == "conflict":
        raise HTTPException(status_code=409, detail=f"Asset is already {result['status']}")
    return {
        "outcome": result["outcome"],
        "metadata": model_to_response(result["metadata"], AssetMetadataResponse),
    }


@app.post("/claim-processing/{asset_path:path}/{version_id}", summary="Atomically claim a processing job", include_in_schema=False)
async def claim_processing(asset_path: str, version_id: str, correlation_id: str, user: User = Depends(get_user)):
    """Internal endpoint called by the processing orchestrator that picks up
    a Kafka job -- not by end users or the gateway. Atomically transitions
    'uploaded'/'failed' -> 'processing' under this correlation_id; a retry
    with the *same* correlation_id re-claims idempotently (200); a
    concurrent claim by a different correlation_id, or a status that
    disallows claiming, gets 409.
    """
    claimed = await manager.db.claim_processing(asset_path, version_id, user.branch_id, correlation_id)
    if claimed is None:
        raise HTTPException(status_code=409, detail="Asset is not claimable (already claimed by another job, or in a non-claimable status)")
    return claimed


@app.post("/update-status/{asset_path:path}/{version_id}", summary="Report a claimed job's outcome (active/failed)", include_in_schema=False)
async def update_asset_status(asset_path: str, version_id: str, status: Literal["active", "failed"], user: User = Depends(get_user)):
    """Internal endpoint called by the processing orchestrator (09-12) once
    a claimed job finishes, success or failure -- not by end users or the
    gateway. Writes commit_history.status and best-effort syncs Module 25.

    The response echoes whatever update_status() actually wrote, not the
    caller's requested `status` -- a claimed "active" can be downgraded to
    "failed" by that call's own Module 25 double-check, and a caller trusting
    this response's `status` field deserves the real outcome, not an echo of
    what it asked for.
    """
    written_status = await manager.update_status(
        username=user.user_id, branch=user.branch_id,
        asset_path=asset_path, version_id=version_id, status=status,
    )
    return {"asset_path": asset_path, "version_id": version_id, "status": written_status}


@app.post("/update-moderation/{asset_path:path}/{version_id}", summary="Store an image's moderation verdict (CM-2)", include_in_schema=False)
async def update_asset_moderation(asset_path: str, version_id: str, flags: ModerationFlags, user: User = Depends(get_user)):
    """Internal endpoint called by Module 10's image orchestrator once Module 23's
    image guard has judged an ingested image -- not by end users or the gateway,
    like /update-status. Stored as the asset's moderation_flags, which the asset
    metadata responses carry."""
    data = flags.model_dump(mode="json")
    await manager.update_moderation(
        username=user.user_id, branch=user.branch_id,
        asset_path=asset_path, version_id=version_id, flags=data,
    )
    return {"asset_path": asset_path, "version_id": version_id, "moderation_flags": data}


def _require_internal_token(x_internal_token: str = Header(None, alias="X-Internal-Token")) -> None:
    if not settings.gdpr_internal_token or x_internal_token != settings.gdpr_internal_token:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Internal-Token")


@app.delete("/internal/gdpr/user/{user_id}", dependencies=[Depends(_require_internal_token)])
async def gdpr_erase_user(user_id: str):
    """GD-1 / VIE01-246: called only by Module 33 (the GDPR erasure
    orchestrator), never by end users or Module 13's gateway."""
    deleted_count = await manager.gdpr_erase_user(branch=user_id)
    return {"user_id": user_id, "assets_deleted": deleted_count}


@app.get("/internal/gdpr/user/{user_id}/summary", dependencies=[Depends(_require_internal_token)])
async def gdpr_summarize_user(user_id: str):
    """GD-4 / VIE01-249: count-only file summary for Module 33's DSAR
    assembler -- deliberately not /users/commits (full metadata rows, paged)
    or GD-2's file fetch; a DSAR only needs how many files and what kinds.
    total_bytes isn't here: commit_history records no file size."""
    summary = await manager.gdpr_summarize_user(branch=user_id)
    return {"user_id": user_id, **summary}


@app.get("/internal/gdpr/user/{user_id}/files", dependencies=[Depends(_require_internal_token)])
async def gdpr_list_user_files(
    user_id: str,
    expires_in: int = Query(259200, ge=60, le=604800, description="Download link lifetime in seconds (default 72h, max 7 days)"),
):
    """GD-2 / VIE01-247: the export's files/index.json -- one entry per
    stored file with name, type, size, uploaded_at and a download link
    valid for expires_in (Module 33 passes the export's own TTL). Metadata
    only: the export ZIP never carries file bytes."""
    return {"user_id": user_id, "files": await manager.gdpr_list_files(branch=user_id, expires_in=expires_in)}


@app.get("/fileversions/{asset_path:path}/{filename}", summary="List versions of an asset")
async def list_versions(asset_path: str, filename: str, user: User = Depends(get_user)):
    key = os.path.normpath(f"{asset_path}/{filename}").replace('\\', '/')
    return await manager.list_file_versions(
        username=user.user_id,
        key=key,
        branch=user.branch_id
    )


@app.get("/users/commits", summary="Get commit history with filters")
async def get_user_commits(
    keyword: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 1,
    page_size: int = 10,
    status: list[str] | None = Query(None, description="Repeatable; e.g. ?status=uploaded&status=failed. Omitted = all 6 statuses."),
    user: User = Depends(get_user)
):
    commits = await manager.get_user_commits(
        username=user.user_id,
        branch=user.branch_id,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
        statuses=status,
    )
    return commits
    

@app.post("/file-expiration/{asset_path:path}/{version_id}", summary="Update asset expiration dates", response_model=AssetMetadataResponse)
async def update_asset_expiration(
    asset_path: str,
    version_id: str,
    archive_date_or_ttl: int | str | None = Form(
        None,
        description=(
            "Archive date or TTL. "
            "Accepts an absolute date (YYYY-MM-DD or YYYY/MM/DD), "
            "an ISO datetime (YYYY-MM-DDTHH:MM:SS), "
            "or a relative TTL such as '30d', '2w', '6m', '1y'. "
            "If a number is provided, it is treated as days from now."
            "If set to 'no_change', the archive date will not change. "
            "Default is 'None' (never archive)."
        ),
    ),
    destroy_date_or_ttl: int | str | None = Form(
        None,
        description=(
            "Destroy date or TTL, relative to the archive date. "
            "Accepts the same formats as archive_date_or_ttl. "
            "For example, '1y' means one year after the archive date."
            "If set to 'no_change', the destroy date will not change. "
            "Default is 'None' (never destroy)."
        ),
    ),
    user: User = Depends(get_user)
):
    # if not archive_date_or_ttl and not destroy_date_or_ttl:
    #     raise HTTPException(status_code=400, detail="At least one of archive_date_or_ttl or destroy_date_or_ttl must be provided")

    logger.info(f"archive_date_or_ttl: {archive_date_or_ttl}, destroy_date_or_ttl: {destroy_date_or_ttl}")
    archive_date_or_ttl = archive_date_or_ttl if archive_date_or_ttl else None
    destroy_date_or_ttl = destroy_date_or_ttl if destroy_date_or_ttl else None

    return await manager.update_expiration(
        username=user.user_id,
        branch=user.branch_id,
        asset_path=asset_path,
        version_id=version_id,
        archive_date_or_ttl=archive_date_or_ttl,
        destroy_date_or_ttl=destroy_date_or_ttl
    )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
