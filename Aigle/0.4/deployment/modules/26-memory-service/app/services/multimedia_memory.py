from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.config import settings
from core.paths import user_dir as _ensure_user_dir
from services.memvid_store import normalize_scores, sync_append, sync_search, sync_timeline

MediaType = Literal["video", "audio", "image"]

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class VideoIndexRequest(BaseModel):
    asset_path: str = Field(..., description="影片檔在儲存系統中的路徑（如 SeaweedFS/S3 key）")
    version_id: str = Field(..., description="該資產的版本 ID，用於區分同一路徑的不同版本")
    start_sec: float = Field(..., description="片段在影片中的起始秒數")
    end_sec: float = Field(..., description="片段在影片中的結束秒數")
    transcription: str = Field(..., description="該片段的逐字稿文字，用於 embedding 及全文搜尋")
    session_id: str = Field("", description="觸發索引的 session ID（可省略，僅作追溯用途）")
    context_query: str = Field(
        "", description="使用者當時查詢的問題（可省略），有助於之後語意檢索命中同類問題"
    )


class AudioIndexRequest(BaseModel):
    asset_path: str = Field(..., description="音訊檔在儲存系統中的路徑")
    version_id: str = Field(..., description="該資產的版本 ID")
    start_sec: float = Field(..., description="片段起始秒數")
    end_sec: float = Field(..., description="片段結束秒數")
    transcription: Optional[str] = Field(
        None, description="逐字稿文字；省略時會自動呼叫 Module 07 ASR 產生"
    )
    session_id: str = Field("", description="觸發索引的 session ID（可省略）")
    context_query: str = Field("", description="使用者當時查詢的問題（可省略）")


class ImageIndexRequest(BaseModel):
    asset_path: str = Field(..., description="圖片檔在儲存系統中的路徑")
    version_id: str = Field(..., description="該資產的版本 ID")
    ocr_text: str = Field("", description="圖片 OCR 辨識出的文字（可省略）")
    description: str = Field("", description="圖片的文字描述（可省略），與 ocr_text 一併做 embedding")
    session_id: str = Field("", description="觸發索引的 session ID（可省略）")


class IndexResponse(BaseModel):
    frame_id: str = Field(..., description="新寫入 frame 的 ID")
    asset_path: str = Field(..., description="已索引的資產路徑")
    media_type: str = Field(..., description="媒體類型：video | audio | image")


class MediaSearchRequest(BaseModel):
    query: str = Field(..., description="查詢文字，做 BM25 + bge-m3 語意向量混合搜尋")
    top_k: int = Field(5, ge=1, le=50, description="回傳筆數上限")
    media_type: Optional[MediaType] = Field(
        None, description="限定搜尋的媒體類型：video | audio | image；省略則搜尋全部類型"
    )


class MediaSearchHit(BaseModel):
    asset_path: str = Field(..., description="命中資產的儲存路徑")
    version_id: str = Field(..., description="命中資產的版本 ID")
    start_sec: Optional[float] = Field(None, description="片段起始秒數（video/audio 才有）")
    end_sec: Optional[float] = Field(None, description="片段結束秒數（video/audio 才有）")
    text_snippet: str = Field(..., description="命中的文字片段（逐字稿 / OCR / 描述）")
    score: float = Field(..., description="正規化後的混合相關性分數（0~1）")
    media_type: str = Field(..., description="媒體類型：video | audio | image")
    session_id: str = Field(..., description="索引當時記錄的 session ID")
    timestamp: Optional[float] = Field(None, description="索引寫入時間（Unix timestamp）")


# ── Service ───────────────────────────────────────────────────────────────────

class MultimediaMemoryService:
    def __init__(self, storage_root: str | None = None) -> None:
        self._root = Path(storage_root or settings.storage_root)

    def _media_dir(self, user_id: str) -> Path:
        return self._root / f"user_{user_id}" / "media"

    def _media_dir_create(self, user_id: str) -> Path:
        _ensure_user_dir(self._root, user_id)
        d = self._media_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
        return d

    def _mv2_path(self, user_id: str, media_type: str, asset_path: str) -> str:
        h = hashlib.md5(asset_path.encode()).hexdigest()[:16]
        return str(self._media_dir_create(user_id) / f"{media_type}_{h}.mv2")

    async def index_video(self, user_id: str, data: VideoIndexRequest) -> IndexResponse:
        text = f"{data.context_query}\n{data.transcription}".strip() if data.context_query else data.transcription
        frame = {
            "title": f"video: {data.asset_path} [{data.start_sec:.1f}-{data.end_sec:.1f}s]",
            "label": "video_moment",
            "text": text,
            "timestamp": time.time(),
            "asset_path": data.asset_path,
            "version_id": data.version_id,
            "start_sec": data.start_sec,
            "end_sec": data.end_sec,
            "transcription": data.transcription,
            "session_id": data.session_id,
            "context_query": data.context_query,
            "media_type": "video",
        }
        path = self._mv2_path(user_id, "video", data.asset_path)
        frame_id = await asyncio.to_thread(sync_append, path, frame)
        return IndexResponse(frame_id=frame_id, asset_path=data.asset_path, media_type="video")

    async def index_audio(self, user_id: str, data: AudioIndexRequest, transcription: str) -> IndexResponse:
        text = f"{data.context_query}\n{transcription}".strip() if data.context_query else transcription
        frame = {
            "title": f"audio: {data.asset_path} [{data.start_sec:.1f}-{data.end_sec:.1f}s]",
            "label": "audio_clip",
            "text": text,
            "timestamp": time.time(),
            "asset_path": data.asset_path,
            "version_id": data.version_id,
            "start_sec": data.start_sec,
            "end_sec": data.end_sec,
            "transcription": transcription,
            "session_id": data.session_id,
            "context_query": data.context_query,
            "media_type": "audio",
        }
        path = self._mv2_path(user_id, "audio", data.asset_path)
        frame_id = await asyncio.to_thread(sync_append, path, frame)
        return IndexResponse(frame_id=frame_id, asset_path=data.asset_path, media_type="audio")

    async def index_image(self, user_id: str, data: ImageIndexRequest) -> IndexResponse:
        text = f"{data.description}\n{data.ocr_text}".strip() or data.asset_path
        frame = {
            "title": f"image: {data.asset_path}",
            "label": "image_memory",
            "text": text,
            "timestamp": time.time(),
            "asset_path": data.asset_path,
            "version_id": data.version_id,
            "ocr_text": data.ocr_text,
            "description": data.description,
            "session_id": data.session_id,
            "media_type": "image",
        }
        path = self._mv2_path(user_id, "image", data.asset_path)
        frame_id = await asyncio.to_thread(sync_append, path, frame)
        return IndexResponse(frame_id=frame_id, asset_path=data.asset_path, media_type="image")

    async def search(
        self, user_id: str, req: MediaSearchRequest, *, _raw_scores: bool = False
    ) -> list[MediaSearchHit]:
        """`_raw_scores=True` skips the per-batch normalization and returns
        unscaled SDK scores — used by GlobalSearchService to normalize once
        across all scopes together instead of once per scope."""
        media_dir = self._media_dir(user_id)
        if not media_dir.exists():
            return []
        pattern = f"{req.media_type}_*.mv2" if req.media_type else "*.mv2"
        paths = [str(p) for p in media_dir.glob(pattern)]
        if not paths:
            return []

        tasks = [
            asyncio.to_thread(sync_search, path, req.query, req.top_k * 4)
            for path in paths
        ]
        shard_results = await asyncio.gather(*tasks)

        candidates: list[dict] = []
        for hits in shard_results:
            candidates.extend(hits)

        candidates.sort(key=lambda h: h.get("score", float("-inf")), reverse=True)
        candidates = candidates[: req.top_k]

        raw_scores = [h.get("score", 0.0) for h in candidates]
        scores = raw_scores if _raw_scores else normalize_scores(raw_scores)
        return [
            MediaSearchHit(
                asset_path=h.get("asset_path", ""),
                version_id=h.get("version_id", ""),
                start_sec=h.get("start_sec"),
                end_sec=h.get("end_sec"),
                text_snippet=h.get("text") or h.get("snippet", ""),
                score=score,
                media_type=h.get("media_type", ""),
                session_id=h.get("session_id", ""),
                timestamp=float(h["timestamp"]) if h.get("timestamp") else None,
            )
            for h, score in zip(candidates, scores)
        ]

    async def list_all(self, user_id: str) -> list[dict]:
        """Return all indexed media frames across all shards, sorted by recency.

        No embedding/search involved — this is for cheap token-budget estimation
        (see CompactMemoryService.evaluate_session), not ranked retrieval.
        """
        media_dir = self._media_dir(user_id)
        if not media_dir.exists():
            return []
        paths = [str(p) for p in media_dir.glob("*.mv2")]
        if not paths:
            return []

        tasks = [asyncio.to_thread(sync_timeline, path) for path in paths]
        timelines = await asyncio.gather(*tasks)

        all_frames: list[dict] = []
        for timeline in timelines:
            all_frames.extend(timeline)

        all_frames.sort(key=lambda f: f.get("timestamp", 0), reverse=True)
        return all_frames
