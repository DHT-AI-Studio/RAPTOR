# services/video_face_detection_service/consent_client.py
# GD-11：使用者有沒有同意 face_detection（GDPR Art. 9 生物特徵，預設 false）。
#
# 查不到就當作沒有同意（fail-closed），跟 Module 13 的 People Search、Module 34 的錄音一樣：
# 同意狀態不明時，不處理生物特徵資料。人臉偵測不在影片主流程的完成條件裡，跳過不影響影片本身。

import asyncio
import logging

import aiohttp
from config import GD_INTERNAL_TOKEN, PRIVACY_COMPLIANCE_URL

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=5)


async def face_consent_denied(user_id: str) -> str | None:
    """None：已同意。否則回傳不處理的原因：
    "no_consent"（沒同意，或 Keycloak 沒有這個使用者）／"consent_unavailable"（Module 33 查不到）。"""
    headers = {"X-User-ID": user_id, "X-Internal-Token": GD_INTERNAL_TOKEN}
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.get(f"{PRIVACY_COMPLIANCE_URL.rstrip('/')}/consent", headers=headers) as resp:
                if resp.status == 404:
                    return "no_consent"
                resp.raise_for_status()
                payload = await resp.json()
        granted = any(p["id"] == "face_detection" and p["granted"] is True for p in payload["purposes"])
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"face_detection consent check for {user_id} failed, skipping face detection: {e}")
        return "consent_unavailable"
    return None if granted else "no_consent"
