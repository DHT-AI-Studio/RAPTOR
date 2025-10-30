from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from jose import JWTError, jwt
import uuid
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import os
import logging
# import uvicorn
import asyncio
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from .client import AssetManager
from .database import Database
from .object_store import ObjectStore
from .vector_store import VectorStore
from .models import Token, User, AssetMetadata
from .config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

db = Database()
object_store = ObjectStore()
vector_store = VectorStore()

manager = AssetManager(db=db, object_store=object_store, vector_store=vector_store)
scheduler = AsyncIOScheduler()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = settings.jwt_secret_key
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
TIMEZONE = settings.timezone
AUTO_DAILY_ARCHIVE_TIME = settings.auto_daily_archive_time
AUTO_DAILY_DESTROY_TIME = settings.auto_daily_destroy_time
ARCHIVE_HOUR, ARCHIVE_MINUTE = settings.auto_daily_archive_hour_minute
DESTROY_HOUR, DESTROY_MINUTE = settings.auto_daily_destroy_hour_minute


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)
    expire = now_utc + (expires_delta or timedelta(minutes=15))
    to_encode.update(
        {
            "exp": int(expire.timestamp()),
            "iat": int(now_utc.timestamp()),
            "jti": str(uuid.uuid4())
        }
    )
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        branch: str = payload.get("branch")
        permissions: List[str] = payload.get("permissions", [])
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return User(username=username, branch=branch, permissions=permissions)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Scheduler tasks
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await object_store.initialize()
    await vector_store.initialize()
    # Start the scheduler
    scheduler.start()
    # Schedule auto_archive to run daily
    scheduler.add_job(
        run_auto_archive,
        CronTrigger(hour=ARCHIVE_HOUR, minute=ARCHIVE_MINUTE, timezone=ZoneInfo(TIMEZONE)),
        id="auto_archive",
        name=f"Daily auto-archive task at {AUTO_DAILY_ARCHIVE_TIME}"
    )
    # Schedule auto_destroy to run daily
    scheduler.add_job(
        run_auto_destroy,
        CronTrigger(hour=DESTROY_HOUR, minute=DESTROY_MINUTE, timezone=ZoneInfo(TIMEZONE)),
        id="auto_destroy",
        name=f"Daily auto-destroy task at {AUTO_DAILY_DESTROY_TIME}"
    )
    logger.info("Scheduler started with auto-archive and auto-destroy tasks")

    yield

    # Shutdown the scheduler
    scheduler.shutdown()
    await db.close()
    await object_store.close()
    await vector_store.close()


#########################      API endpoints      ############################

app = FastAPI(lifespan=lifespan)

@app.post("/users", summary="Create a new user and assign a new branch to the user")
async def create_admin_user(user: User):
    if not user.username:
        raise HTTPException(status_code=400, detail="Username is required")
    if not user.password:
        raise HTTPException(status_code=400, detail="Password is required")
    try:
        await db.create_admin_user(user.username, user.password)
        return {"status": "success", "username": user.username}
    except Exception as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/token", summary="Create a new access token for the user", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await db.get_user_by_name(form_data.username)
    if not user or not await db.verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": user.username, "branch": user.branch, "permissions": user.permissions},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer", username=user.username, branch=user.branch)


@app.post("/shared-users", summary="Create a new shared user to access your branch")
async def create_shared_user(user: User, admin_user: User = Depends(get_current_user)):
    if "admin" not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Only admins can create shared users")
    if not user.username:
        raise HTTPException(status_code=400, detail="Username is required")
    if not user.password:
        raise HTTPException(status_code=400, detail="Password is required")
    if not user.permissions:
        raise HTTPException(status_code=400, detail="Permissions are required")
    try:
        if "admin" in user.permissions:
            raise HTTPException(status_code=400, detail="Shared users cannot have admin permissions. Available permissions: upload, download, list, archive, destroy")
        await db.create_user(user.username, user.password, admin_user.branch, user.permissions)
    except Exception as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    logger.info(f"User {user.username} created with permissions {user.permissions}")
    return {"status": "success", "username": user.username}


@app.delete("/shared-users", summary="Delete a shared user who can access your branch")
async def delete_shared_user(user: User, admin_user: User = Depends(get_current_user)):
    if "admin" not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Only admins can delete shared users")
    if not user.username:
        raise HTTPException(status_code=400, detail="Username is required")
    try:
        user = await db.get_user_by_name(user.username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if "admin" in user.permissions:
            raise HTTPException(status_code=400, detail="User is not a shared user. Only shared users can be deleted by this endpoint.")
        if user.branch != admin_user.branch:
            raise HTTPException(status_code=403, detail="User is not a shared user of your branch. Only shared users of your branch can be deleted by this endpoint.")
        await db.delete_user_by_name(user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    logger.info(f"Shared user {user.username} deleted successfully")
    return {"status": "success"}


@app.put("/shared-users", summary="Change a shared user's permissions")
async def change_shared_user(user: User, admin_user: User = Depends(get_current_user)):
    if "admin" not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Only admins can change shared users's permissions")
    if not user.username:
        raise HTTPException(status_code=400, detail="Username is required")
    if not user.permissions:
        raise HTTPException(status_code=400, detail="Permissions are required")
    new_permissions = user.permissions
    try:
        user = await db.get_user_by_name(user.username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if "admin" in user.permissions:
            raise HTTPException(status_code=400, detail="User is not a shared user. Only shared users can be deleted by this endpoint.")
        if user.branch != admin_user.branch:
            raise HTTPException(status_code=403, detail="User is not a shared user of your branch. Only shared users of your branch can be deleted by this endpoint.")
        await db.change_shared_user_permission(user.username, new_permissions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    logger.info(f"Shared user {user.username} deleted successfully")
    return {"status": "success"}


@app.post("/fileupload", summary="Upload a new asset", response_model=AssetMetadata)
async def upload_asset(
    primary_file: UploadFile = File(...),
    associated_files: List[UploadFile] = File([]),
    archive_ttl: Optional[int] = Form(None),
    destroy_ttl: Optional[int] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        read_tasks = [primary_file.read()] + [f.read() for f in associated_files]
        file_datas = await asyncio.gather(*read_tasks)

        primary_data = file_datas[0]
        primary_filename = primary_file.filename

        associated = [
            (data, file.filename) 
            for data, file in zip(file_datas[1:], associated_files)
        ]

        return await manager.upload_files(
            username=current_user.username,
            branch=current_user.branch,
            primary_file=(primary_data, primary_filename),
            associated_files=associated,
            archive_ttl=archive_ttl,
            destroy_ttl=destroy_ttl
        )
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/add-associated-files/{asset_path:path}", summary="Add associated files to an existing asset", response_model=AssetMetadata)
async def add_associated_files(
    asset_path: str,
    associated_files: List[UploadFile] = File(...),
    primary_version_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        file_datas = await asyncio.gather(*[f.read() for f in associated_files])
        filename_list = [f.filename for f in associated_files]
        associated = [(data, filename) for data, filename in zip(file_datas, filename_list)]

        return await manager.add_associated_files(
            username=current_user.username,
            branch=current_user.branch,
            asset_path=asset_path,
            associated_files=associated,
            primary_version_id=primary_version_id
        )

    except Exception as e:
        logger.error(f"Failed to add associated files to {asset_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add associated files: {str(e)}")


@app.get("/filedownload/{asset_path:path}/{version_id}", summary="Download an asset")
async def download_asset(asset_path: str, version_id: str, return_file_content: bool = False, current_user: dict = Depends(get_current_user)):
    try:
        return await manager.retrieve_asset(current_user.username, current_user.branch, asset_path, version_id, return_file_content)
    except Exception as e:
        logger.error(f"Failed to retrieve asset {asset_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve asset: {str(e)}")


@app.post("/filearchive/{asset_path:path}/{version_id}", summary="Archive an asset", response_model=AssetMetadata)
async def archive_asset(asset_path: str, version_id: str, current_user: dict = Depends(get_current_user)):
    try:
        return await manager.archive(current_user.username, current_user.branch, asset_path, version_id)
    except Exception as e:
        logger.error(f"Failed to archive asset {asset_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to archive asset: {str(e)}")


@app.post("/delfile/{asset_path:path}/{version_id}", summary="Delete an archived asset", response_model=AssetMetadata)
async def destroy_asset(asset_path: str, version_id: str, current_user: dict = Depends(get_current_user)):
    try:
        return await manager.destroy(current_user.username, current_user.branch, asset_path, version_id)
    except Exception as e:
        logger.error(f"Failed to destroy asset {asset_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to destroy asset: {str(e)}")


@app.get("/fileversions/{asset_path:path}/{filename}", summary="List versions of an asset")
async def list_versions(asset_path: str, filename: str, current_user: dict = Depends(get_current_user)):
    try:
        key = os.path.normpath(f"{asset_path}/{filename}").replace('\\', '/')
        return await manager.list_file_versions(current_user.username, key, current_user.branch)
    except Exception as e:
        logger.error(f"Failed to list versions for {key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
