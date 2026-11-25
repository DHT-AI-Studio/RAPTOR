# services/video_orchestrator_service/assoc_lock.py
# 同一支影片的 associated files 一次只讓一方寫。
#
# Module 04 的 add-associated-files 是「讀出清單 → 加上新檔案 → 整份寫回」：同一支影片
# 兩個請求同時進來，後寫的會把先寫的那批從清單上蓋掉。檔案還在 LakeFS，但 Module 04 之後
# 找不到它們，刪除影片時也不會刪（人臉代表圖就會殘留）。
# Module 11 裡會寫這份清單的有兩個：orchestrator 上傳 keyframe、人臉服務上傳代表圖。
# 兩邊呼叫前都先拿這把 Redis 鎖。
#
# 本檔在 video_orchestrator_service 與 video_face_detection_service 各有一份，內容相同
# （Module 11 的服務之間不共用套件）。改一份就要改另一份。

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

LOCK_TTL_SEC = 180      # 上傳逾時 120 秒；拿鎖的一方當掉時，鎖最晚這麼久後自己釋放
LOCK_WAIT_SEC = 180     # 最多等這麼久；等不到就不拿鎖照常上傳，不讓上傳卡住
_POLL_SEC = 0.5

# 只刪自己的鎖：鎖已過期、被別人拿走時不能誤刪
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"

_client = None


def _redis():
    global _client
    if _client is None:
        _client = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _client


def lock_key(branch_id: str, asset_path: str, version_id: str) -> str:
    """與 Module 04 清單的主鍵（asset_path, version_id, branch）一一對應。"""
    return f"asset_assoc_lock:{branch_id}:{asset_path}:{version_id}"


@asynccontextmanager
async def associated_files_lock(branch_id: str, asset_path: str, version_id: str,
                                client=None, ttl: int = LOCK_TTL_SEC, wait: float = LOCK_WAIT_SEC):
    """
    async with associated_files_lock(...) as held: 呼叫 add-associated-files
    held 為 False 表示沒拿到鎖（Redis 連不上或等太久），呼叫端照常上傳：
    上傳本來就是「失敗不影響流程」，鎖只是降低互相覆蓋的機會，不能反過來讓上傳停擺。
    """
    r = client or _redis()
    key = lock_key(branch_id, asset_path, version_id)
    token = uuid.uuid4().hex
    held = False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait
    try:
        while True:
            if await r.set(key, token, nx=True, ex=ttl):
                held = True
                break
            if loop.time() >= deadline:
                logger.warning(f"Waited {wait}s for {key}; uploading without the lock")
                break
            await asyncio.sleep(_POLL_SEC)
    except (aioredis.RedisError, OSError) as e:
        logger.warning(f"Associated-files lock unavailable ({e}); uploading without it")

    try:
        yield held
    finally:
        if held:
            try:
                await r.eval(_RELEASE, 1, key, token)
            except (aioredis.RedisError, OSError) as e:
                logger.warning(f"Failed to release {key} (expires in {ttl}s): {e}")
