"""
Service for interacting with the downstream object storage.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from fastapi import HTTPException, UploadFile, status

from app.core.config import Settings

_logger = logging.getLogger(__name__)


class TokenCache:
    """Simple in-memory token cache with TTL."""
    
    def __init__(self, ttl: int = 3600):
        self.token: Optional[str] = None
        self.expires_at: float = 0
        self.ttl = ttl
    
    def is_valid(self) -> bool:
        """Check if cached token is still valid."""
        return self.token is not None and time.time() < self.expires_at
    
    def set(self, token: str) -> None:
        """Set token with TTL."""
        self.token = token
        self.expires_at = time.time() + self.ttl
    
    def get(self) -> Optional[str]:
        """Get token if valid."""
        if self.is_valid():
            return self.token
        return None


class StorageService:
    def __init__(self, client: httpx.AsyncClient, settings: Settings, auth_header: Optional[str] = None):
        self.client = client
        self.settings = settings
        self.base_url = str(self.settings.object_storage_url).rstrip("/")
        self.auth_header = auth_header  # 用戶 JWT token
        self.token_cache = TokenCache(ttl=settings.storage_token_cache_ttl)

    async def get_access_token(self) -> Optional[str]:
        """
        Fetch an access token for downstream storage requests.
        Uses local cache to reduce redundant token requests.
        """
        # 檢查快取
        cached_token = self.token_cache.get()
        if cached_token:
            _logger.debug("Using cached object storage access token")
            return cached_token
        
        token_url = f"{self.base_url}/token"
        try:
            response = await self.client.post(
                token_url,
                data={"username": "user1", "password": "dht888888"},
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            _logger.warning("Object storage token request timed out", extra={"downstream_url": token_url})
            return None
        except httpx.HTTPError as exc:
            _logger.error(
                "Failed to obtain object storage token",
                extra={"downstream_url": token_url, "error": str(exc)},
            )
            return None

        result = response.json()
        token = result.get("access_token")
        if not token:
            _logger.error(
                "Object storage token response missing access_token",
                extra={"downstream_url": token_url},
            )
            return None

        # 快取 token
        self.token_cache.set(token)
        _logger.debug("Obtained and cached object storage access token", extra={"downstream_url": token_url})
        return token

    async def upload_file(
        self,
        primary_file: UploadFile,
        archive_ttl: Optional[str],
        destroy_ttl: Optional[str],
    ) -> Dict[str, Any]:
        """
        Uploads a file to the object storage service.
        Returns only the result (simplified return type).
        """
        access_token = await self.get_access_token()
        return await self.upload_file_stream(
            primary_file=primary_file,
            archive_ttl=archive_ttl,
            destroy_ttl=destroy_ttl,
            access_token=access_token,
        )

    async def upload_file_stream(
        self,
        primary_file: UploadFile,
        archive_ttl: Optional[str],
        destroy_ttl: Optional[str],
        access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stream a single file to object storage without loading entire file into memory.
        Allows passing a pre-obtained access_token for batch operations.
        """
        downstream_url = f"{self.base_url}/fileupload"

        headers: Dict[str, str] = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            _logger.warning("Proceeding without object storage token", extra={"downstream_url": downstream_url})

        form_fields: Dict[str, str] = {}
        if archive_ttl:
            form_fields["archive_ttl"] = str(archive_ttl)
        if destroy_ttl:
            form_fields["destroy_ttl"] = str(destroy_ttl)

        # Reset file pointer and stream from file-like object
        try:
            primary_file.file.seek(0)
        except (AttributeError, OSError):
            pass

        files = {
            "primary_file": (
                primary_file.filename or "uploaded_file",
                primary_file.file,  # Stream from file-like object
                primary_file.content_type or "application/octet-stream",
            )
        }

        try:
            _logger.info(
                "Proxying file upload to object storage (streaming)",
                extra={
                    "downstream_url": downstream_url,
                    "uploaded_file": primary_file.filename,  # Changed from 'filename' to 'uploaded_file'
                },
            )
            response = await self.client.post(
                downstream_url,
                data=form_fields,
                files=files,
                headers=headers,
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            _logger.info(
                "Object storage upload succeeded",
                extra={
                    "downstream_url": downstream_url,
                    "uploaded_file": primary_file.filename,  # Changed from 'filename' to 'uploaded_file'
                    "asset_path": result.get("asset_path"),
                    "version_id": result.get("version_id"),
                },
            )
            return result

        except httpx.TimeoutException as exc:
            _logger.warning("Object storage request timed out", extra={"downstream_url": downstream_url})
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to object storage service timed out.",
            ) from exc
        except httpx.RequestError as exc:
            _logger.error(
                "Failed to connect to object storage",
                extra={"downstream_url": downstream_url, "error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not connect to the object storage service.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            _logger.warning(
                "Object storage returned an error",
                extra={
                    "downstream_url": downstream_url,
                    "uploaded_file": primary_file.filename,  # Changed from 'filename' to 'uploaded_file'
                    "status_code": exc.response.status_code,
                    "response": exc.response.text,
                },
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.json(),
            ) from exc

    async def upload_files_batch(
        self,
        primary_files: List[UploadFile],
        archive_ttl: Optional[str],
        destroy_ttl: Optional[str],
        *,
        concurrency: int = 4,
    ) -> Dict[str, Any]:
        """
        Batch upload multiple files concurrently with bounded concurrency and token reuse.
        Returns successes and failures per-file.
        """
        # Reuse one token for the whole batch (best effort)
        access_token = await self.get_access_token()

        sem = asyncio.Semaphore(max(1, concurrency))

        async def _worker(index: int, f: UploadFile) -> Dict[str, Any]:
            async with sem:
                try:
                    result = await self.upload_file_stream(
                        primary_file=f,
                        archive_ttl=archive_ttl,
                        destroy_ttl=destroy_ttl,
                        access_token=access_token,
                    )
                    return {"index": index, "uploaded_file": f.filename, "status": "success", "result": result}  # Changed 'filename' to 'uploaded_file'
                except HTTPException as exc:
                    return {
                        "index": index,
                        "uploaded_file": f.filename,  # Changed 'filename' to 'uploaded_file'
                        "status": "failed",
                        "error": exc.detail,
                        "status_code": exc.status_code,
                    }
                except Exception as exc:
                    _logger.exception("Unexpected error during batch upload", exc_info=exc, extra={"target_file": f.filename})  # Changed 'filename' to 'target_file'
                    return {
                        "index": index,
                        "uploaded_file": f.filename,  # Changed 'filename' to 'uploaded_file'
                        "status": "failed",
                        "error": str(exc),
                        "status_code": 500,
                    }

        tasks = [_worker(i, f) for i, f in enumerate(primary_files)]
        outcomes = await asyncio.gather(*tasks)

        successes = [o for o in outcomes if o.get("status") == "success"]
        failures = [o for o in outcomes if o.get("status") == "failed"]

        return {
            "total": len(primary_files),
            "concurrency": concurrency,
            "successes": successes,
            "failures": failures,
            "success_count": len(successes),
            "failure_count": len(failures),
        }


    async def list_versions(
        self,
        asset_path: str,
        filename: str,
    ) -> Dict[str, Any]:
        """
        List all versions of an asset from object storage.
        使用路徑參數格式: /fileversions/{asset_path}/{filename}
        """
        # URL 編碼路徑參數（safe="" 表示編碼所有特殊字符）
        encoded_asset_path = quote(asset_path, safe="")
        encoded_filename = quote(filename, safe="")
        downstream_url = f"{self.base_url}/fileversions/{encoded_asset_path}/{encoded_filename}"

        headers: Dict[str, str] = {}
        access_token = await self.get_access_token()
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            _logger.warning("Proceeding without object storage token", extra={"downstream_url": downstream_url})

        try:
            _logger.info(
                "Fetching asset versions from object storage",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "target_filename": filename,
                },
            )
            response = await self.client.get(
                downstream_url,
                headers=headers,
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            _logger.info(
                "Object storage asset versions fetch succeeded",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "target_filename": filename,
                    "response_status": response.status_code,
                },
            )
            return result

        except httpx.TimeoutException as exc:
            _logger.warning("Object storage request timed out", extra={"downstream_url": downstream_url})
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to object storage service timed out.",
            ) from exc
        except httpx.RequestError as exc:
            _logger.error(
                "Failed to connect to object storage",
                extra={"downstream_url": downstream_url, "error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not connect to the object storage service.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            _logger.warning(
                "Object storage returned an error",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "target_filename": filename,
                    "status_code": exc.response.status_code,
                    "response_body": exc.response.text[:500],
                },
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Object storage error: {exc.response.text}",
            ) from exc

    async def download_asset(
        self,
        asset_path: str,
        version_id: str,
        return_file_content: bool = False,
    ) -> Dict[str, Any]:
        """
        Download an asset from object storage.
        使用路徑參數格式: /filedownload/{asset_path}/{version_id}
        """
        encoded_asset_path = quote(asset_path, safe="")
        encoded_version_id = quote(version_id, safe="")
        downstream_url = f"{self.base_url}/filedownload/{encoded_asset_path}/{encoded_version_id}"

        headers: Dict[str, str] = {}
        access_token = await self.get_access_token()
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            _logger.warning("Proceeding without object storage token", extra={"downstream_url": downstream_url})

        # 將 return_file_content 作為查詢參數
        params = {}
        if return_file_content:
            params["return_file_content"] = "true"

        try:
            _logger.info(
                "Downloading asset from object storage",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "return_file_content": return_file_content,
                },
            )
            response = await self.client.get(
                downstream_url,
                params=params,
                headers=headers,
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            _logger.info(
                "Object storage asset download succeeded",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "response_status": response.status_code,
                },
            )
            return result

        except httpx.TimeoutException as exc:
            _logger.warning("Object storage request timed out", extra={"downstream_url": downstream_url})
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to object storage service timed out.",
            ) from exc
        except httpx.RequestError as exc:
            _logger.error(
                "Failed to connect to object storage",
                extra={"downstream_url": downstream_url, "error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not connect to the object storage service.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            _logger.warning(
                "Object storage returned an error",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "status_code": exc.response.status_code,
                    "response_body": exc.response.text[:500],
                },
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Object storage error: {exc.response.text}",
            ) from exc

    async def archive_asset(
        self,
        asset_path: str,
        version_id: str,
    ) -> Dict[str, Any]:
        """
        Archive an asset in object storage.
        使用路徑參數格式: /filearchive/{asset_path}/{version_id}
        """
        encoded_asset_path = quote(asset_path, safe="")
        encoded_version_id = quote(version_id, safe="")
        downstream_url = f"{self.base_url}/filearchive/{encoded_asset_path}/{encoded_version_id}"

        headers: Dict[str, str] = {}
        access_token = await self.get_access_token()
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            _logger.warning("Proceeding without object storage token", extra={"downstream_url": downstream_url})

        try:
            _logger.info(
                "Archiving asset in object storage",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                },
            )
            response = await self.client.post(
                downstream_url,
                headers=headers,
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            _logger.info(
                "Object storage asset archiving succeeded",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "response_status": response.status_code,
                },
            )
            return result

        except httpx.TimeoutException as exc:
            _logger.warning("Object storage request timed out", extra={"downstream_url": downstream_url})
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to object storage service timed out.",
            ) from exc
        except httpx.RequestError as exc:
            _logger.error(
                "Failed to connect to object storage",
                extra={"downstream_url": downstream_url, "error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not connect to the object storage service.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            _logger.warning(
                "Object storage returned an error",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "status_code": exc.response.status_code,
                    "response_body": exc.response.text[:500],
                },
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Object storage error: {exc.response.text}",
            ) from exc

    async def delete_asset(
        self,
        asset_path: str,
        version_id: str,
    ) -> Dict[str, Any]:
        """
        Delete an archived asset from object storage.
        使用路徑參數格式: /delfile/{asset_path}/{version_id}
        """
        encoded_asset_path = quote(asset_path, safe="")
        encoded_version_id = quote(version_id, safe="")
        downstream_url = f"{self.base_url}/delfile/{encoded_asset_path}/{encoded_version_id}"

        headers: Dict[str, str] = {}
        access_token = await self.get_access_token()
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        else:
            _logger.warning("Proceeding without object storage token", extra={"downstream_url": downstream_url})

        try:
            _logger.info(
                "Proxying asset deletion to object storage",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                },
            )
            response = await self.client.post(
                downstream_url,
                headers=headers,
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            _logger.info(
                "Object storage asset deletion succeeded",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "response_status": response.status_code,
                },
            )
            return result

        except httpx.TimeoutException as exc:
            _logger.warning("Object storage request timed out", extra={"downstream_url": downstream_url})
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to object storage service timed out.",
            ) from exc
        except httpx.RequestError as exc:
            _logger.error(
                "Failed to connect to object storage",
                extra={"downstream_url": downstream_url, "error": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not connect to the object storage service.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            _logger.warning(
                "Object storage returned an error",
                extra={
                    "downstream_url": downstream_url,
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "status_code": exc.response.status_code,
                    "response_body": exc.response.text[:500],
                },
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Object storage error: {exc.response.text}",
            ) from exc