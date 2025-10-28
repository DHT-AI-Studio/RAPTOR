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
SEAWEEDFS_BASE_URL = os.getenv("SEAWEEDFS_BASE_URL")

logger = logging.getLogger(__name__)

class SeaweedFSClient:
    def __init__(self):
        self.base_url = SEAWEEDFS_BASE_URL
        self.timeout = SEAWEEDFS_TIMEOUT
        self.retry_count = SEAWEEDFS_RETRY_COUNT
        
        # 確保臨時目錄存在
        os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    
    async def download_file(
        self, 
        access_token: str, 
        asset_path: str, 
        version_id: str,
        filename: str
    ) -> str:
        """從 SeaweedFS 下載檔案到本地臨時檔案，返回本地檔案路徑"""
        
        for attempt in range(self.retry_count):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # 先獲取檔案資訊和 URL
                    async with session.get(
                        f"{self.base_url}/filedownload/{asset_path}/{version_id}?return_file_content=false",
                        headers={"Authorization": f"Bearer {access_token}"}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # 檢查主要檔案
                            primary_file = result.get("primary_file")
                            if not primary_file:
                                raise ValueError("No primary file in response")
                            
                            # 檢查檔案名是否匹配
                            if primary_file["filename"] != filename:
                                raise ValueError(f"Primary file {primary_file['filename']} does not match requested {filename}")
                            
                            # 使用 URL 下載實際檔案內容
                            file_url = primary_file["url"]
                            
                            # 下載檔案內容
                            async with session.get(file_url) as file_response:
                                if file_response.status == 200:
                                    file_content = await file_response.read()
                                    
                                    # 創建臨時檔案
                                    temp_file_path = os.path.join(
                                        TEMP_FILE_DIR, 
                                        f"{version_id}_{filename}"
                                    )
                                    
                                    # 寫入檔案
                                    with open(temp_file_path, 'wb') as f:
                                        f.write(file_content)
                                    
                                    logger.info(f"File downloaded successfully: {temp_file_path}")
                                    return temp_file_path
                                else:
                                    raise ValueError(f"Failed to download file content: {file_response.status}")
                            
                        elif response.status in [401, 403]:
                            # 認證錯誤不重試
                            logger.error(f"Authentication failed for download: {response.status}")
                            raise ValueError(f"Authentication failed: {response.status}")
                            
                        elif response.status == 404:
                            # 檔案不存在不重試
                            logger.error(f"File not found: {asset_path}/{version_id}")
                            raise ValueError(f"File not found: {asset_path}/{version_id}")
                            
                        else:
                            # 其他錯誤可重試
                            logger.warning(f"Download failed with status {response.status}, attempt {attempt + 1}")
                            if attempt == self.retry_count - 1:
                                raise ValueError(f"Download failed with status {response.status}")
                            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Network error on download attempt {attempt + 1}: {e}")
                if attempt == self.retry_count - 1:
                    raise ValueError(f"Network error: {e}")
                
                # 指數退避
                await asyncio.sleep(2 ** attempt)
        
        raise ValueError("Maximum retry attempts exceeded for download")
    
    async def download_file_with_content(
        self, 
        access_token: str, 
        asset_path: str, 
        version_id: str,
        filename: str
    ) -> str:
        """使用 return_file_content=true 下載檔案（如果 API 支援 base64 內容）"""
        
        for attempt in range(self.retry_count):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # 設置 return_file_content=true 來獲取檔案內容
                    async with session.get(
                        f"{self.base_url}/filedownload/{asset_path}/{version_id}?return_file_content=true",
                        headers={"Authorization": f"Bearer {access_token}"}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # 檢查主要檔案
                            primary_file = result.get("primary_file")
                            if not primary_file:
                                raise ValueError("No primary file in response")
                            
                            # 檢查檔案名是否匹配
                            if primary_file["filename"] != filename:
                                raise ValueError(f"Primary file {primary_file['filename']} does not match requested {filename}")
                            
                            # 檢查是否有 base64 內容
                            if "content" not in primary_file:
                                # 如果沒有內容，回退到 URL 下載
                                logger.info("No base64 content, falling back to URL download")
                                return await self.download_file(access_token, asset_path, version_id, filename)
                            
                            # 解碼檔案內容
                            import base64
                            file_content = base64.b64decode(primary_file["content"])
                            
                            # 創建臨時檔案
                            temp_file_path = os.path.join(
                                TEMP_FILE_DIR, 
                                f"{version_id}_{filename}"
                            )
                            
                            # 寫入檔案
                            with open(temp_file_path, 'wb') as f:
                                f.write(file_content)
                            
                            logger.info(f"File downloaded successfully with content: {temp_file_path}")
                            return temp_file_path
                            
                        elif response.status in [401, 403]:
                            # 認證錯誤不重試
                            logger.error(f"Authentication failed for download: {response.status}")
                            raise ValueError(f"Authentication failed: {response.status}")
                            
                        elif response.status == 404:
                            # 檔案不存在不重試
                            logger.error(f"File not found: {asset_path}/{version_id}")
                            raise ValueError(f"File not found: {asset_path}/{version_id}")
                            
                        else:
                            # 其他錯誤可重試
                            logger.warning(f"Download failed with status {response.status}, attempt {attempt + 1}")
                            if attempt == self.retry_count - 1:
                                raise ValueError(f"Download failed with status {response.status}")
                            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Network error on download attempt {attempt + 1}: {e}")
                if attempt == self.retry_count - 1:
                    raise ValueError(f"Network error: {e}")
                
                # 指數退避
                await asyncio.sleep(2 ** attempt)
        
        raise ValueError("Maximum retry attempts exceeded for download")
    
    def cleanup_temp_file(self, file_path: str):
        """清理臨時檔案"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup temporary file {file_path}: {e}")