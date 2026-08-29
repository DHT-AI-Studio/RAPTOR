# services/document_orchestrator_service/api_client.py

import aiohttp
import asyncio
import logging
import os
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from config import (
    #SEAWEEDFS_BASE_URL,
    SEAWEEDFS_TIMEOUT,
    SEAWEEDFS_RETRY_COUNT,
    TEMP_FILE_DIR
)
from dotenv import load_dotenv
import os
# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
logger = logging.getLogger(__name__)

class SeaweedFSClient:
    def __init__(self):
        self.timeout = SEAWEEDFS_TIMEOUT
        self.retry_count = SEAWEEDFS_RETRY_COUNT
        
        # 確保臨時目錄存在
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
            async with aiohttp.ClientSession() as session:
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
                    _ext = os.path.splitext(filename)[1] or ".pdf"
                    temp_file_path = os.path.join(TEMP_FILE_DIR, f"tmp_{uuid.uuid4()}{_ext}")
                    async with aiofiles.open(temp_file_path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            await f.write(chunk)
                    logger.info(f"Step 2 OK: File saved to {temp_file_path}")
                    return temp_file_path

        except Exception as e:
            logger.error(f"download_file failed: {e}")
            raise
    
    def cleanup_temp_file(self, file_path: str):
        """清理臨時檔案"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup temporary file {file_path}: {e}")