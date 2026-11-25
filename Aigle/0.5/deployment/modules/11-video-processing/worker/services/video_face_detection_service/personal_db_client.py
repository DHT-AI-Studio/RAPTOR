# services/video_face_detection_service/personal_db_client.py
# 把一支影片的軌跡與分群結果寫入 Module 25（v1.4 §4.5「Trace-and-cluster write」）。
#
# 使用者資料庫以 payload 的 user_id 為鍵，和 video_indexer_service 寫內容索引時相同，
# 人臉資料才會和同一支影片的內容在同一個資料庫。
# 人臉可能比內容索引先完成，新使用者的資料庫這時還不存在，所以寫入前先呼叫冪等的 /internal/db/init。
# trace_id / group_id 是固定的，Module 25 會整支影片覆蓋舊資料，重試不會產生重複。

import asyncio
import logging
from typing import Any

import aiohttp
from config import PERSONAL_DB_SERVICE_URL, WRITE_FREEZE_RETRY_COUNT

logger = logging.getLogger(__name__)


def _floats(values) -> list:
    return [float(v) for v in values]


def build_face_traces_request(analysis, version_id: str, asset_path: str,
                              filename: str | None = None) -> dict[str, Any]:
    """FaceAnalysis → POST /personal/index/face-traces 的 body。
    代表圖沒上傳成功的軌跡（representative_version_id 為 None）不帶圖片路徑。"""
    traces = []
    for t in analysis.traces:
        rep = t.representative
        uploaded = t.representative_version_id is not None
        traces.append({
            "trace_id": t.trace_id,
            "start_sec": float(t.start_sec),
            "end_sec": float(t.end_sec),
            "embedding": _floats(t.embedding),
            "frames": [{"frame_no": int(f.frame_no), "timestamp_sec": float(f.timestamp_sec),
                        "bbox": _floats(f.bbox), "confidence": float(f.confidence)} for f in t.frames],
            "representative": {
                "frame_no": int(rep.frame.frame_no),
                "timestamp_sec": float(rep.frame.timestamp_sec),
                "bbox": _floats(rep.frame.bbox),
                "quality": float(rep.quality.score),
                "min_side": float(rep.min_side),
                # Module 04 把 associated file 存在 {asset_path}/{filename}
                "image_path": f"{asset_path}/face_{t.trace_id}.jpg" if uploaded else None,
                "image_version_id": t.representative_version_id,
            },
        })
    groups = [{"group_id": g.group_id, "embedding": _floats(g.embedding),
               "trace_ids": [t.trace_id for t in g.traces]} for g in analysis.groups]
    return {"version_id": version_id, "asset_path": asset_path, "filename": filename,
            "traces": traces, "groups": groups}


async def write_face_traces(
    user_id: str, body: dict[str, Any], max_retries: int = WRITE_FREEZE_RETRY_COUNT,
) -> dict[str, Any]:
    """確保使用者資料庫存在後寫入，回傳 Module 25 的回應。
    重試邏輯跟 asset_client.py 的 _upload_batch() 一樣 -- 這裡的 5xx(例如
    --backup 的 write-freeze 窗口)幾乎都是暫時的，而這個寫入是整個人臉偵測
    工作（下載、偵測、追蹤、分群、代表圖上傳）全部做完才跑的最後一步：不重試
    直接丟例外，會讓這整個 job 被標成 FAILED、所有前面的工作都白做，Kafka
    這個 consumer 是 auto-commit，也不會真的重新投遞。trace_id/group_id 固定、
    Module 25 整支影片覆蓋舊資料（見檔案頂部註解），重試是安全的。重試次數用
    完後仍然丟出例外 -- 保留原本「真的失敗就讓這個 job 明確報錯」的設計。"""
    headers = {"X-User-ID": user_id, "X-Branch-ID": user_id}
    timeout = aiohttp.ClientTimeout(total=600, connect=30)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with aiohttp.ClientSession(base_url=PERSONAL_DB_SERVICE_URL, timeout=timeout, headers=headers) as session:
                async with session.post("/internal/db/init") as resp:
                    if resp.status != 200:
                        body_text = (await resp.text())[:200]
                        if 400 <= resp.status < 500:
                            # permanent failure -- raise immediately, don't burn the retry budget
                            raise RuntimeError(f"Personal DB init returned {resp.status}: {body_text}")
                        last_exc = RuntimeError(f"Personal DB init returned {resp.status}: {body_text}")
                        logger.warning(f"write_face_traces attempt {attempt}/{max_retries} failed: {last_exc}")
                        if attempt < max_retries:
                            await asyncio.sleep(min(2 ** attempt, 90))
                        continue
                async with session.post("/personal/index/face-traces", json=body) as resp:
                    if resp.status != 201:
                        body_text = (await resp.text())[:200]
                        if 400 <= resp.status < 500:
                            raise RuntimeError(f"Personal DB face-traces returned {resp.status}: {body_text}")
                        last_exc = RuntimeError(f"Personal DB face-traces returned {resp.status}: {body_text}")
                        logger.warning(f"write_face_traces attempt {attempt}/{max_retries} failed: {last_exc}")
                        if attempt < max_retries:
                            await asyncio.sleep(min(2 ** attempt, 90))
                        continue
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exc = e
            logger.warning(f"write_face_traces attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 90))
    raise last_exc
