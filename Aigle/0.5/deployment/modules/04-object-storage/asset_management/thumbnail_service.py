"""
MTR-7 / VIE01-306: on-demand video thumbnail extraction.

Delegates actual ffprobe/ffmpeg NVDEC work to the thumbnail-worker sidecar
(its own GPU-scoped container, see docker-compose.yml) -- this process has
no GPU access. "Not blanket Module 04" = only the worker gets GPU/CUDA.
"""
import logging

import httpx
import redis.asyncio as aioredis
from fastapi import HTTPException
from redis.exceptions import RedisError

from .config import settings
from .database import Database
from .object_store import ObjectStore

logger = logging.getLogger(__name__)

THUMBNAIL_CACHE_TTL_SECONDS = 7 * 24 * 3600  # VIE01-306: 7-day TTL
WORKER_TIMEOUT_SECONDS = 80.0  # worker's own ffprobe(15s)+NVDEC(30s)+CPU-fallback(20s) total 65s, plus headroom


class ThumbnailService:
    def __init__(self, object_store: ObjectStore, db: Database):
        self.object_store = object_store
        self.db = db
        # Separate Redis client from ObjectStore's presign cache -- that one
        # uses decode_responses=True (text URLs); thumbnails are raw JPEG
        # bytes, so this client must stay in binary mode.
        try:
            self._redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                socket_connect_timeout=1,
                decode_responses=False,
            )
        except Exception as e:  # noqa: BLE001 -- Redis() ctor failures aren't typed, fail-open to no cache
            logger.warning(f"Redis unavailable, thumbnail caching disabled: {e}")
            self._redis = None

    @staticmethod
    def _cache_key(asset_path: str, version_id: str, second: float) -> str:
        return f"thumbnail:{asset_path}:{version_id}:{second}"

    async def _extract_via_worker(self, url: str, second: float) -> bytes:
        """Passes the worker's own 400/502 straight through."""
        try:
            async with httpx.AsyncClient(timeout=WORKER_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{settings.thumbnail_worker_url}/extract",
                    json={"url": url, "second": second},
                )
        except httpx.RequestError as e:
            logger.error(f"thumbnail-worker unreachable: {e}")
            raise HTTPException(status_code=502, detail="Thumbnail worker unreachable")

        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return resp.content

    async def get_thumbnail(self, asset_path: str, version_id: str, branch: str, second: float) -> bytes:
        if second < 0:
            raise HTTPException(status_code=400, detail="second must be >= 0")

        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found",
            )
        if metadata.status != "active":
            raise HTTPException(
                status_code=409,
                detail=f"Asset {asset_path}/{version_id} is {metadata.status}, not active -- cannot generate thumbnail",
            )

        cache_key = self._cache_key(asset_path, version_id, second)
        if self._redis:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    return cached
            except RedisError as e:
                logger.warning(f"Thumbnail cache read failed, falling back to ffmpeg: {e}")

        key = f"{metadata.asset_path}/{metadata.primary_filename}"
        presigned_url = await self.object_store.generate_presigned_url(key, version_id=version_id)

        jpeg_bytes = await self._extract_via_worker(presigned_url, second)

        if self._redis:
            try:
                await self._redis.setex(cache_key, THUMBNAIL_CACHE_TTL_SECONDS, jpeg_bytes)
            except RedisError as e:
                logger.warning(f"Failed to cache thumbnail: {e}")

        return jpeg_bytes
