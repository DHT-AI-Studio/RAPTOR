# services/image_orchestrator_service/api_client.py

import asyncio
import logging
import os
from enum import Enum
from typing import Any

import aiohttp
from config import (
    ASSET_MANAGEMENT_URL,
    SEAWEEDFS_RETRY_COUNT,
    #SEAWEEDFS_BASE_URL,
    SEAWEEDFS_TIMEOUT,
    TEMP_FILE_DIR,
    WRITE_FREEZE_RETRY_COUNT,
)
from dotenv import load_dotenv

# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
logger = logging.getLogger(__name__)


class ClaimOutcome(Enum):
    """VIE01-255: claim_processing()'s tri-state result -- the caller
    (kafka_handler.process_requests) branches its commit decision on this."""
    CLAIMED = "claimed"
    CONFLICT = "conflict"                # 409 -> caller acks + discards, no processing
    TRANSIENT_ERROR = "transient_error"  # 5xx or network/timeout -> caller must NOT ack


class SeaweedFSClient:
    def __init__(self):
        self.timeout = SEAWEEDFS_TIMEOUT
        self.retry_count = SEAWEEDFS_RETRY_COUNT
        
        # 確保臨時目錄存在
        os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    
    async def download_file(
        self,
        download_url: str,
        download_headers: dict[str, str],
        download_params: dict[str, Any],
        filename: str,
        max_retries: int = SEAWEEDFS_RETRY_COUNT,
    ) -> str:
        """Resumable: a mid-download failure (confirmed live -- presigned
        URLs fetched via the public S3 endpoint can time out/truncate)
        retries from the byte offset already on disk via an HTTP Range
        request, instead of restarting the whole file. The presigned URL
        itself is re-fetched (Step 1) on every attempt, not cached across
        retries -- it's cheap and its TTL (20min) could otherwise run out
        mid-retry-loop on a slow connection.
        """
        import uuid

        import aiofiles

        os.makedirs(TEMP_FILE_DIR, exist_ok=True)
        _ext = os.path.splitext(filename)[1] or ".jpg"
        temp_file_path = os.path.join(TEMP_FILE_DIR, f"tmp_{uuid.uuid4()}{_ext}")

        last_exc: Exception | None = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(1, max_retries + 1):
                existing_size = os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 0
                try:
                    logger.info(f"Step 1 (attempt {attempt}/{max_retries}): Getting presigned URL from {download_url}")
                    async with session.get(download_url, headers=download_headers, params=download_params) as resp:
                        if resp.status != 200:
                            raise Exception(f"Asset API returned {resp.status}: {await resp.text()}")
                        presigned_url = (await resp.json())["primary_file"]["url"]

                    range_headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
                    logger.info(
                        f"Step 2 (attempt {attempt}/{max_retries}): Downloading from presigned URL"
                        + (f" (resuming from byte {existing_size})" if existing_size else "")
                    )
                    async with session.get(presigned_url, headers=range_headers) as resp:
                        if existing_size and resp.status == 200:
                            # Server didn't honor the Range request (no partial-
                            # content support) -- start over clean rather than
                            # appending a full response onto what's already there.
                            logger.warning("Range request not honored (got 200, not 206) -- restarting from byte 0")
                            existing_size = 0
                        elif resp.status not in (200, 206):
                            raise Exception(f"File download returned {resp.status}")
                        mode = "ab" if existing_size else "wb"
                        async with aiofiles.open(temp_file_path, mode) as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                await f.write(chunk)
                    logger.info(f"Step 2 OK: File saved to {temp_file_path}")
                    return temp_file_path

                except Exception as e:  # noqa: BLE001 -- also catches the bare Exception raised above for non-2xx status, retried per-attempt
                    last_exc = e
                    logger.warning(
                        f"download_file attempt {attempt}/{max_retries} failed: {e} -- "
                        f"{os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 0} bytes on disk so far"
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(min(2 ** attempt, 30))

        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
        logger.error(f"download_file failed after {max_retries} attempts: {last_exc}")
        raise last_exc
    
    def cleanup_temp_file(self, file_path: str):
        """清理臨時檔案"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file cleaned up: {file_path}")
        except OSError as e:
            logger.error(f"Failed to cleanup temporary file {file_path}: {e}")

    async def claim_processing(
        self, asset_path: str, version_id: str, correlation_id: str,
        user_id: str, branch_id: str,
    ) -> ClaimOutcome:
        """VIE01-255: POST /claim-processing/{asset_path}/{version_id}?correlation_id=...
        Never raises -- every failure mode maps to a ClaimOutcome."""
        url = f"{ASSET_MANAGEMENT_URL}/claim-processing/{asset_path}/{version_id}"
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    headers={"X-User-ID": user_id, "X-Branch-ID": branch_id},
                    params={"correlation_id": correlation_id},
                ) as resp:
                    if resp.status == 200:
                        return ClaimOutcome.CLAIMED
                    if resp.status == 409:
                        logger.info(f"Claim 409 for {asset_path}/{version_id} (already claimed/non-claimable)")
                        return ClaimOutcome.CONFLICT
                    logger.warning(f"Claim failed ({resp.status}) for {asset_path}/{version_id}: {(await resp.text())[:200]}")
                    return ClaimOutcome.TRANSIENT_ERROR
        except aiohttp.ClientError as e:
            logger.warning(f"Claim request failed (network/timeout) for {asset_path}/{version_id}: {e}")
            return ClaimOutcome.TRANSIENT_ERROR

    async def update_asset_status(
        self, user_id: str, branch_id: str, asset_path: str, version_id: str, status: str,
        max_retries: int = WRITE_FREEZE_RETRY_COUNT,
    ) -> None:
        """Write back the final status (active/failed) once processing finishes.
        Retries like download_file() -- a 5xx here (e.g. VIE01-369's write-freeze
        during a coordinated --backup) is almost always transient, and this write
        happens after the expensive processing work is already done, with no
        Kafka redelivery to fall back on (the result-topic consumer auto-commits).
        Still fails gracefully after exhausting retries -- never aborts the
        pipeline, matching register_keyframes()'s posture."""
        last_status: int | None = None
        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{ASSET_MANAGEMENT_URL}/update-status/{asset_path}/{version_id}",
                        headers={"X-User-ID": user_id, "X-Branch-ID": branch_id},
                        params={"status": status},
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"Updated status={status} for {asset_path}/{version_id}")
                            return
                        last_status = resp.status
                        body = await resp.text()
                        logger.warning(f"Status update attempt {attempt}/{max_retries} returned {resp.status}: {body[:200]}")
                        if 400 <= resp.status < 500:
                            break  # a 4xx is a permanent, not transient, failure -- retrying won't help
            except aiohttp.ClientError as e:
                logger.warning(f"Status update attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 90))
        logger.warning(f"Status update failed (non-fatal) after {attempt} attempts (last status: {last_status})")

    async def update_moderation(
        self, user_id: str, branch_id: str, asset_path: str, version_id: str, flags: dict[str, Any],
        max_retries: int = WRITE_FREEZE_RETRY_COUNT,
    ) -> None:
        """CM-2: write the image's moderation_flags to its Module 04 metadata.
        Retries like update_asset_status() -- same reasoning (write-freeze is
        transient, no Kafka redelivery here). Fails gracefully after exhausting
        retries: the same flags still reach Module 25 with the summary entry."""
        last_status: int | None = None
        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{ASSET_MANAGEMENT_URL}/update-moderation/{asset_path}/{version_id}",
                        headers={"X-User-ID": user_id, "X-Branch-ID": branch_id},
                        json=flags,
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"Updated moderation_flags ({flags.get('verdict')}) for {asset_path}/{version_id}")
                            return
                        last_status = resp.status
                        body = await resp.text()
                        logger.warning(f"Moderation update attempt {attempt}/{max_retries} returned {resp.status}: {body[:200]}")
                        if 400 <= resp.status < 500:
                            break  # a 4xx is a permanent, not transient, failure -- retrying won't help
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Moderation update attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 90))
        logger.warning(f"Moderation update failed (non-fatal) after {attempt} attempts (last status: {last_status})")
