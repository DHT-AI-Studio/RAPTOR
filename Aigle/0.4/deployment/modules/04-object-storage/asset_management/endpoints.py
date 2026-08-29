from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Optional, Union
# from datetime import datetime, timedelta, timezone
import os
import logging
import asyncio
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .client import AssetManager
from .database import Database
from .object_store import ObjectStore
from .search_sync import SearchSync
from .models import AssetMetadataResponse, User
from .config import settings

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
    except Exception as e:
        logger.error(f"Auto-archive failed: {str(e)}")

async def run_auto_destroy():
    try:
        destroyed_assets = await manager.auto_destroy()
        # destroyed_assets = await manager.auto_destroy(datetime.now()+timedelta(days=180)) # For testing
        logger.info(f"Auto-destroy completed: {len(destroyed_assets)} assets destroyed")
    except Exception as e:
        logger.error(f"Auto-destroy failed: {str(e)}")


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
        except Exception as _e:
            logger.warning(f"Redis ping failed, presigned URL caching disabled: {_e}")
            object_store._redis = None

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


# -------------------- FastAPI app --------------------
app = FastAPI(lifespan=lifespan)


# -------------------- Global exception handler --------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Intercept all unexpected exceptions.
    If it's a normal HTTPException (like 400, 404), FastAPI will handle it first and won't reach here.
    """
    logger.error(f"Unexpected Error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )


# -------------------- File APIs --------------------
@app.post("/fileupload", summary="Upload a new asset")
async def upload_asset(
    primary_file: UploadFile = File(...),
    associated_files: List[UploadFile] = File([], description='If you do not want to upload files, uncheck "Send empty value".'),
    archive_date_or_ttl: Optional[Union[int, str]] = Form(
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
    destroy_date_or_ttl: Optional[Union[int, str]] = Form(
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
    associated_files: List[UploadFile] = File(...),
    primary_version_id: Optional[str] = Form(None),
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


@app.post("/delfile/{asset_path:path}/{version_id}", summary="Delete an archived asset", response_model=AssetMetadataResponse)
async def destroy_asset(asset_path: str, version_id: str, user: User = Depends(get_user)):
    return await manager.destroy_v2(
        username=user.user_id,
        branch=user.branch_id,
        asset_path=asset_path,
        version_id=version_id
    )


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
    keyword: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    user: User = Depends(get_user)
):
    commits = await manager.get_user_commits(
        username=user.user_id,
        branch=user.branch_id,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size
    )
    return commits
    

@app.post("/file-expiration/{asset_path:path}/{version_id}", summary="Update asset expiration dates", response_model=AssetMetadataResponse)
async def update_asset_expiration(
    asset_path: str,
    version_id: str,
    archive_date_or_ttl: Optional[Union[int, str]] = Form(
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
    destroy_date_or_ttl: Optional[Union[int, str]] = Form(
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
