# services/video_face_detection_service/asset_client.py
# Module 04 (Asset Management) 呼叫：取得原始上傳影片的本服務專用副本、上傳代表圖

import asyncio
import logging
import os
import shutil
from typing import Any

import aiofiles
import aiohttp
from assoc_lock import associated_files_lock
from config import (
    ASSET_MANAGEMENT_URL,
    FACE_REPRESENTATIVE_BATCH_SIZE,
    WRITE_FREEZE_RETRY_COUNT,
)

logger = logging.getLogger(__name__)


async def fetch_video(
    dest_path: str,
    source_path: str | None,
    asset_path: str,
    version_id: str,
    download_headers: dict[str, str],
    download_params: dict[str, Any],
) -> str:
    """
    orchestrator 的暫存檔還在時直接 hard link（同一個 NFS volume，不複製資料），
    之後 orchestrator 刪掉它的暫存檔也不影響這份；
    link 失敗時改用複製，檔案已不在時照 orchestrator 的方式從 Module 04 下載。
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if source_path and os.path.exists(source_path):
        try:
            os.link(source_path, dest_path)
            logger.info(f"Linked source video {source_path} -> {dest_path}")
            return dest_path
        except OSError as e:
            logger.warning(f"Hard link failed ({e}), copying instead")
            try:
                shutil.copyfile(source_path, dest_path)
                return dest_path
            except OSError as copy_err:
                logger.warning(f"Copy failed ({copy_err}), downloading from asset management")

    timeout = aiohttp.ClientTimeout(total=1800, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            f"{ASSET_MANAGEMENT_URL}/filedownload/{asset_path}/{version_id}",
            headers=download_headers,
            params=download_params,
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Asset API returned {resp.status}: {await resp.text()}")
            presigned_url = (await resp.json())["primary_file"]["url"]

        async with session.get(presigned_url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"File download returned {resp.status}")
            async with aiofiles.open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    await f.write(chunk)
    logger.info(f"Downloaded {asset_path}/{version_id} -> {dest_path}")
    return dest_path


async def upload_representatives(
    user_id: str,
    branch_id: str,
    asset_path: str,
    version_id: str,
    files: list[tuple[str, bytes]],
) -> dict[str, str]:
    """
    代表圖批次上傳（v1.4 §4.3 第 8 步）：作為影片的 associated files，與 orchestrator 上傳 keyframe 用同一個 endpoint。
    每 FACE_REPRESENTATIVE_BATCH_SIZE 張一個請求（Module 04 一個請求最多收 1000 個檔案），
    每個請求在 Module 04 內部合成一次 commit（v1.4 §3.4 的批次路徑）。
    每批各拿一次 associated_files_lock，避免和 keyframe 上傳互相覆蓋 Module 04 的清單（見 assoc_lock.py）。
    檔案存在 {asset_path}/{filename}。回傳 {filename: version_id}，只含這次上傳的檔案。
    某一批失敗時停止：前面已有批次成功就回傳那些（已 commit 的圖照樣可用），一批都沒成功才丟出例外。
    """
    stored: dict[str, str] = {}
    for start in range(0, len(files), FACE_REPRESENTATIVE_BATCH_SIZE):
        batch = files[start:start + FACE_REPRESENTATIVE_BATCH_SIZE]
        try:
            stored.update(await _upload_batch(user_id, branch_id, asset_path, version_id, batch))
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            if not stored:
                raise
            logger.warning(f"Representative upload stopped after {len(stored)}/{len(files)} files: {e}")
            break
    return stored


async def _upload_batch(
    user_id: str,
    branch_id: str,
    asset_path: str,
    version_id: str,
    files: list[tuple[str, bytes]],
    max_retries: int = WRITE_FREEZE_RETRY_COUNT,
) -> dict[str, str]:
    """一個請求上傳一批代表圖。回傳 {filename: version_id}，只含這批的檔案。
    重試邏輯跟 orchestrator 的 download_file() 一樣 -- 這裡的 5xx（例如 VIE01-369
    的 write-freeze）幾乎都是暫時的,而這個寫入發生在昂貴的偵測/分群工作都做完
    之後,這個 consumer 是 auto-commit,沒有 Kafka 重新投遞可以依賴。重試次數
    用完後仍然丟出例外,交給呼叫端(upload_representatives)決定要不要停止。"""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            form = aiohttp.FormData()
            form.add_field("primary_version_id", version_id)
            for filename, data in files:
                form.add_field("associated_files", data, filename=filename, content_type="image/jpeg")
            timeout = aiohttp.ClientTimeout(total=120)
            async with associated_files_lock(branch_id, asset_path, version_id), \
                    aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{ASSET_MANAGEMENT_URL}/add-associated-files/{asset_path}",
                    headers={"X-User-ID": user_id, "X-Branch-ID": branch_id},
                    data=form,
                ) as resp:
                    if resp.status != 200:
                        body_text = (await resp.text())[:200]
                        if 400 <= resp.status < 500:
                            # a 4xx is a permanent, not transient, failure -- raise immediately,
                            # don't burn the retry budget on something that will never succeed
                            raise RuntimeError(f"Representative upload returned {resp.status}: {body_text}")
                        last_exc = RuntimeError(f"Representative upload returned {resp.status}: {body_text}")
                        logger.warning(f"Representative batch upload attempt {attempt}/{max_retries} failed: {last_exc}")
                        if attempt < max_retries:
                            await asyncio.sleep(min(2 ** attempt, 90))
                        continue
                    body = await resp.json()
            wanted = {filename for filename, _ in files}
            return {name: ver for name, ver in body.get("associated_filenames", []) if name in wanted}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exc = e
            logger.warning(f"Representative batch upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 90))
    raise last_exc
