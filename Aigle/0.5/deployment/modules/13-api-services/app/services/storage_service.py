"""
Service for interacting with the asset management service (module 02).
All calls pass X-User-ID / X-Branch-ID headers derived from the caller's
Keycloak JWT — no internal token fetch.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import Settings

_logger = logging.getLogger(__name__)


class User(BaseModel):
    user_id: str
    branch_id: str


def _user_headers(user: User) -> dict[str, str]:
    """Build X-User-ID / X-Branch-ID headers from a User object."""
    return {
        "X-User-ID": user.user_id,
        "X-Branch-ID": user.branch_id,
    }


class StorageService:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.base_url = settings.asset_management_url

    # ── Upload ────────────────────────────────────────────────────────────

    async def upload_file(
        self,
        primary_file: UploadFile,
        archive_date_or_ttl: str | None,
        destroy_date_or_ttl: str | None,
        user: User,
    ) -> dict[str, Any]:
        downstream_url = f"{self.base_url}/fileupload"
        data = {}
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)

        try:
            primary_file.file.seek(0)
        except (AttributeError, OSError):
            pass

        files = {
            "primary_file": (
                primary_file.filename or "uploaded_file",
                primary_file.file,
                primary_file.content_type or "application/octet-stream",
            )
        }

        try:
            resp = await self.client.post(
                downstream_url,
                data=data,
                files=files,
                headers=_user_headers(user),
                timeout=self.settings.upload_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail="Request to asset service timed out.") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                detail="Could not connect to asset service.") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=exc.response.json()) from exc

    async def upload_files_batch(
        self,
        primary_files: list[UploadFile],
        archive_date_or_ttl: str | None,
        destroy_date_or_ttl: str | None,
        user: User,
        concurrency: int = 4,
    ) -> dict[str, Any]:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _worker(index: int, f: UploadFile) -> dict[str, Any]:
            async with sem:
                try:
                    result = await self.upload_file(
                        primary_file=f,
                        archive_date_or_ttl=archive_date_or_ttl,
                        destroy_date_or_ttl=destroy_date_or_ttl,
                        user=user,
                    )
                    return {"index": index, "filename": f.filename, "status": "success", "result": result}
                except HTTPException as exc:
                    return {"index": index, "filename": f.filename, "status": "failed",
                            "error": exc.detail, "status_code": exc.status_code}
                except Exception as exc:
                    _logger.exception("Unexpected error during batch upload", exc_info=exc)
                    return {"index": index, "filename": f.filename, "status": "failed",
                            "error": str(exc), "status_code": 500}

        outcomes = await asyncio.gather(*[_worker(i, f) for i, f in enumerate(primary_files)])
        successes = [o for o in outcomes if o["status"] == "success"]
        failures = [o for o in outcomes if o["status"] == "failed"]

        return {
            "total": len(primary_files),
            "concurrency": concurrency,
            "successes": successes,
            "failures": failures,
            "success_count": len(successes),
            "failure_count": len(failures),
        }

    async def add_associated_files(
        self,
        asset_path: str,
        associated_files: list[UploadFile],
        user: User,
        primary_version_id: str | None = None,
    ) -> dict[str, Any]:
        files = [
            ("associated_files", (f.filename, await f.read(), f.content_type))
            for f in associated_files
        ]
        data = {}
        if primary_version_id:
            data["primary_version_id"] = primary_version_id

        try:
            resp = await self.client.post(
                f"{self.base_url}/add-associated-files/{quote(asset_path, safe='')}",
                files=files,
                data=data,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=exc.response.json()) from exc

    async def upload_identity_photo(
        self,
        identity_id: str,
        filename: str | None,
        data: bytes,
        content_type: str | None,
        user: User,
    ) -> dict[str, Any]:
        """A person's seed photo (dev plan v1.4 §4.5). Module 04 picks the key itself
        (identity/{branch}/{identity_id}.{ext}); returns {image_path, image_version_id}."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/identity-photo/{quote(identity_id, safe='')}",
                files={"photo": (filename or "photo", data, content_type or "application/octet-stream")},
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    # ── Download / metadata ───────────────────────────────────────────────────

    async def get_presign_url(
        self,
        key: str,
        version_id: str,
        user: User,
    ) -> str:
        """Fast path: get presigned URL for a known LakeFS key+version without DB lookup or associated-file fetching."""
        url = f"{self.base_url}/presign-url/{quote(key, safe='')}/{quote(version_id, safe='')}"
        try:
            resp = await self.client.get(
                url,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json().get("url")
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail="Asset service timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def get_presign_urls_batch(
        self,
        items: list[tuple[str, str]],
        user: User,
    ) -> dict[tuple[str, str], str]:
        """Batched get_presign_url(): resolves many (key, version_id) pairs in as few LakeFS calls
        as Module 04 can manage (POST /presign-urls-batch). Best-effort -- an unreachable or erroring
        Module 04 just means an empty result here, never an exception: callers fall back to
        get_presign_url() per key for anything missing, same contract as Module 04's own batch
        helper. Returns {(key, version_id): url} for whatever was resolved."""
        if not items:
            return {}
        try:
            resp = await self.client.post(
                f"{self.base_url}/presign-urls-batch",
                json={"items": [{"key": k, "version_id": v} for k, v in items]},
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPError as exc:
            _logger.warning(f"Batch presign failed, falling back per-key: {exc}")
            return {}
        return {(r["key"], r["version_id"]): r["url"] for r in body.get("resolved", [])}

    async def download_asset(
        self,
        asset_path: str,
        version_id: str,
        user: User,
        return_file_content: bool = False,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/filedownload/{quote(asset_path, safe='')}/{quote(version_id, safe='')}"
        try:
            resp = await self.client.get(
                url,
                params={"return_file_content": str(return_file_content).lower()},
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail="Request to asset service timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def list_versions(
        self,
        asset_path: str,
        filename: str,
        user: User,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/fileversions/{quote(asset_path, safe='')}/{quote(filename, safe='')}"
        try:
            resp = await self.client.get(
                url,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail="Request to asset service timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def get_user_commits(
        self,
        user: User,
        keyword: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        page_size: int = 10,
        statuses: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            params = {k: v for k, v in {
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
                "page": page,
                "page_size": page_size,
            }.items() if v is not None}
            if statuses:
                # httpx repeats the key for a list value -- ?status=a&status=b,
                # matching module 04's Query(None) (repeatable) parameter.
                params["status"] = statuses
            resp = await self.client.get(
                f"{self.base_url}/users/commits",
                params=params,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def scan_assets(
        self,
        files: list[dict[str, str]],
        user: User,
    ) -> dict[str, Any]:
        """MTR-8 / VIE01-307: batch duplicate-content check -- no upload,
        just a checksum lookup."""
        url = f"{self.base_url}/scan"
        try:
            resp = await self.client.post(
                url,
                json={"files": files},
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def archive_asset(
        self,
        asset_path: str,
        version_id: str,
        user: User,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/filearchive/{quote(asset_path, safe='')}/{quote(version_id, safe='')}"
        try:
            resp = await self.client.post(
                url,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def get_thumbnail(
        self,
        asset_path: str,
        version_id: str,
        second: float,
        user: User,
    ) -> bytes:
        """MTR-7 / VIE01-306. Longer timeout than the default proxy calls --
        a cache miss runs ffprobe + an NVDEC ffmpeg decode on module 04's
        side (up to ~45s combined there), not just a quick DB round trip."""
        url = f"{self.base_url}/thumbnail/{quote(asset_path, safe='')}/{quote(version_id, safe='')}"
        try:
            resp = await self.client.get(
                url,
                params={"second": second},
                headers=_user_headers(user),
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def unregister_asset(
        self,
        asset_path: str,
        version_id: str,
        user: User,
    ) -> dict[str, Any]:
        """MTR-5 / VIE01-305: active -> unregistered (filter-only, no data
        deleted). Module 04 owns the status check (409 if not active) and
        the synchronous Module 25 status flip."""
        url = f"{self.base_url}/unregister/{quote(asset_path, safe='')}/{quote(version_id, safe='')}"
        try:
            resp = await self.client.post(
                url,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def register_asset(
        self,
        asset_path: str,
        version_id: str,
        user: User,
    ) -> dict[str, Any]:
        """VIE01-215: run the Register status-branching logic (module 04
        owns it -- it's the one with the DB row and the Module 25 call).
        Raises 404/409 straight through via HTTPStatusError below."""
        url = f"{self.base_url}/register/{quote(asset_path, safe='')}/{quote(version_id, safe='')}"
        try:
            resp = await self.client.post(
                url,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def delete_asset(
        self,
        asset_path: str,
        version_id: str,
        user: User,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/delfile/{quote(asset_path, safe='')}/{quote(version_id, safe='')}"
        try:
            resp = await self.client.post(
                url,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc

    async def update_expiration(
        self,
        asset_path: str,
        version_id: str,
        user: User,
        archive_date_or_ttl: str | None = None,
        destroy_date_or_ttl: str | None = None,
    ) -> dict[str, Any]:
        data = {}
        if archive_date_or_ttl is not None:
            data["archive_date_or_ttl"] = str(archive_date_or_ttl)
        if destroy_date_or_ttl is not None:
            data["destroy_date_or_ttl"] = str(destroy_date_or_ttl)

        try:
            resp = await self.client.post(
                f"{self.base_url}/file-expiration/{quote(asset_path, safe='')}/{quote(version_id, safe='')}",
                data=data,
                headers=_user_headers(user),
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Asset service error: {exc.response.text}") from exc
