# services/video_orchestrator_service/api_client.py

import asyncio
import logging
import os
from enum import Enum
from typing import Any

import aiohttp
from assoc_lock import associated_files_lock
from config import (
    ASSET_MANAGEMENT_URL,
    SEAWEEDFS_RETRY_COUNT,
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


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class ClaimOutcome(Enum):
    """VIE01-255: claim_processing()'s tri-state result -- the caller
    (kafka_handler.process_requests) branches its commit decision on this."""
    CLAIMED = "claimed"
    CONFLICT = "conflict"                # 409 -> caller acks + discards, no processing
    TRANSIENT_ERROR = "transient_error"  # 5xx or network/timeout -> caller must NOT ack


class SeaweedFSClient:
    def __init__(self):
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
        URLs fetched via the public S3 endpoint can time out/truncate, see
        VIE01-xxx) retries from the byte offset already on disk via an HTTP
        Range request, instead of restarting the whole file. The presigned
        URL itself is re-fetched (Step 1) on every attempt, not cached
        across retries -- it's cheap and its TTL (20min) could otherwise
        run out mid-retry-loop on a slow connection.
        """
        import uuid

        import aiofiles

        os.makedirs(TEMP_FILE_DIR, exist_ok=True)
        _ext = os.path.splitext(filename)[1] or ".mp4"
        temp_file_path = os.path.join(TEMP_FILE_DIR, f"tmp_{uuid.uuid4()}{_ext}")

        last_exc: Exception | None = None
        timeout = aiohttp.ClientTimeout(total=1800, connect=30)  # 30 min for large files
        async with aiohttp.ClientSession(timeout=timeout) as session:
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

                except Exception as e:  # noqa: BLE001 -- retry loop also catches its own explicit non-200 Exception above; aiohttp/file-I/O failure modes vary, already logged
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
    
    async def register_keyframes(
        self,
        user_id: str,
        branch_id: str,
        asset_path: str,
        version_id: str,
        keyframe_paths: list,
        max_retries: int = WRITE_FREEZE_RETRY_COUNT,
    ) -> None:
        """Upload keyframe JPEGs as associated files on the video asset.
        Retries like download_file() -- a 5xx here (e.g. VIE01-369's write-freeze
        during a coordinated --backup) is almost always transient, and this write
        happens after the expensive processing work is already done, with no
        Kafka redelivery to fall back on (the result-topic consumer auto-commits).
        Still fails gracefully after exhausting retries -- never aborts the pipeline.
        Holds associated_files_lock while uploading, so the face-detection service's
        representative upload can't overwrite Module 04's list (see assoc_lock.py).
        """
        if not keyframe_paths:
            return

        existing = [p for p in keyframe_paths if os.path.exists(p)]
        if not existing:
            logger.warning("register_keyframes: no keyframe files found on disk, skipping")
            return

        last_status: int | None = None
        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=120)
                async with associated_files_lock(branch_id, asset_path, version_id), \
                        aiohttp.ClientSession(timeout=timeout) as session:
                    form = aiohttp.FormData()
                    form.add_field("primary_version_id", version_id)
                    for path in existing:
                        base = os.path.basename(path)
                        moment_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
                        ext = os.path.splitext(base)[1] or ".jpg"
                        stem = os.path.splitext(base)[0]
                        unique_filename = f"{moment_dir}_{stem}{ext}"
                        file_bytes = await asyncio.to_thread(_read_file_bytes, path)
                        form.add_field(
                            "associated_files",
                            file_bytes,
                            filename=unique_filename,
                            content_type="image/jpeg",
                        )
                    async with session.post(
                        f"{ASSET_MANAGEMENT_URL}/add-associated-files/{asset_path}",
                        headers={"X-User-ID": user_id, "X-Branch-ID": branch_id},
                        data=form,
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"Registered {len(existing)} keyframes on {asset_path}/{version_id}")
                            return
                        last_status = resp.status
                        body = await resp.text()
                        logger.warning(f"Keyframe registration attempt {attempt}/{max_retries} returned {resp.status}: {body[:200]}")
                        if 400 <= resp.status < 500:
                            break  # a 4xx is a permanent, not transient, failure -- retrying won't help
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                logger.warning(f"Keyframe registration attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 90))
        logger.warning(f"Keyframe registration failed (non-fatal) after {attempt} attempts (last status: {last_status})")

    def cleanup_temp_file(self, file_path: str):
        """清理臨時檔案"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary video file cleaned up: {file_path}")
        except OSError as e:
            logger.error(f"Failed to cleanup temporary video file {file_path}: {e}")

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
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Claim request failed (network/timeout) for {asset_path}/{version_id}: {e}")
            return ClaimOutcome.TRANSIENT_ERROR

    async def update_moderation(
        self, user_id: str, branch_id: str, asset_path: str, version_id: str, flags: dict[str, Any],
        max_retries: int = WRITE_FREEZE_RETRY_COUNT,
    ) -> None:
        """CM-3: write the video's moderation rollup to its Module 04 metadata (the
        same POST /update-moderation CM-2 uses for images). Retries like
        register_keyframes() -- same reasoning (write-freeze is transient, no
        Kafka redelivery here). Fails gracefully after exhausting retries: the
        same flags still reach Module 25 via the indexer."""
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

    async def update_asset_status(
        self, user_id: str, branch_id: str, asset_path: str, version_id: str, status: str,
        max_retries: int = WRITE_FREEZE_RETRY_COUNT,
    ) -> None:
        """Write back the final status (active/failed) once processing finishes.
        Retries like register_keyframes() -- same reasoning (write-freeze is
        transient, no Kafka redelivery here). Still fails gracefully after
        exhausting retries -- never aborts the pipeline."""
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
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Status update attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 90))
        logger.warning(f"Status update failed (non-fatal) after {attempt} attempts (last status: {last_status})")
