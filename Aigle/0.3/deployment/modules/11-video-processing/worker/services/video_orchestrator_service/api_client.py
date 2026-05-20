# services/video_orchestrator_service/api_client.py

import aiohttp
import asyncio
import logging
import os
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from config import (
    TEMP_FILE_DIR,
    ASSET_MANAGEMENT_URL,
)
from dotenv import load_dotenv
# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
logger = logging.getLogger(__name__)

class SeaweedFSClient:
    def __init__(self):
        os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    
    async def download_file(
        self,
        download_url: str,
        download_headers: Dict[str, str],
        download_params: Dict[str, Any],
        filename: str
    ) -> str:
        import aiofiles
        import uuid

        try:
            timeout = aiohttp.ClientTimeout(total=1800, connect=30)  # 30 min for large files
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"Step 1: Getting presigned URL from {download_url}")
                async with session.get(
                    download_url,
                    headers=download_headers,
                    params=download_params
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Asset API returned {resp.status}: {await resp.text()}")
                    asset_info = await resp.json()
                    presigned_url = asset_info["primary_file"]["url"]
                    logger.info(f"Step 1 OK: Got presigned URL")

                logger.info(f"Step 2: Downloading file from presigned URL")
                async with session.get(presigned_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"File download returned {resp.status}")
                    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
                    _ext = os.path.splitext(filename)[1] or ".mp4"
                    temp_file_path = os.path.join(TEMP_FILE_DIR, f"tmp_{uuid.uuid4()}{_ext}")
                    async with aiofiles.open(temp_file_path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            await f.write(chunk)
                    logger.info(f"Step 2 OK: File saved to {temp_file_path}")
                    return temp_file_path

        except Exception as e:
            logger.error(f"download_file failed: {e}")
            raise
    
    async def register_keyframes(
        self,
        user_id: str,
        branch_id: str,
        asset_path: str,
        version_id: str,
        keyframe_paths: list,
    ) -> None:
        """Upload keyframe JPEGs as associated files on the video asset.
        Fails gracefully — never aborts the pipeline.
        """
        if not keyframe_paths:
            return

        existing = [p for p in keyframe_paths if os.path.exists(p)]
        if not existing:
            logger.warning("register_keyframes: no keyframe files found on disk, skipping")
            return

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                form = aiohttp.FormData()
                form.add_field("primary_version_id", version_id)
                for path in existing:
                    with open(path, "rb") as f:
                        form.add_field(
                            "associated_files",
                            f.read(),
                            filename=os.path.basename(path),
                            content_type="image/jpeg",
                        )
                async with session.post(
                    f"{ASSET_MANAGEMENT_URL}/add-associated-files/{asset_path}",
                    headers={"X-User-ID": user_id, "X-Branch-ID": branch_id},
                    data=form,
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Registered {len(existing)} keyframes on {asset_path}/{version_id}")
                    else:
                        body = await resp.text()
                        logger.warning(f"Keyframe registration returned {resp.status}: {body[:200]}")
        except Exception as e:
            logger.warning(f"Keyframe registration failed (non-fatal): {e}")

    def cleanup_temp_file(self, file_path: str):
        """清理臨時檔案"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary video file cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup temporary video file {file_path}: {e}")
