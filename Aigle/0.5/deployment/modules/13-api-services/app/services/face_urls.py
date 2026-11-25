"""Presigned URLs for face images in Module 25 responses (dev plan v1.4 §4.5/§4.6).

Module 25 returns face images as {image_path, image_version_id}: a representative
frame Module 11 uploaded as one of the video's associated files, or a seed photo.
image_version_id is the LakeFS commit the file was written in -- the ref Module 04's
/presign-url takes -- so each image needs exactly one presign call.

A long video's pending-persons carries a picture per trace -- thousands of them.
Module 04's /presign-urls-batch resolves most of them in a handful of LakeFS calls
(one representative-image batch upload lands in one commit, see Module 11's
video_face_detection_service), so it's tried first; whatever it couldn't resolve
(a seed photo under its own identity/ prefix, a failed group, ...) falls back to
/presign-url directly, capped at face_image_presign_concurrency so that fallback
can't itself re-create the pool-queueing timeout this all started from.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.config import get_settings
from app.services.storage_service import StorageService, User

_logger = logging.getLogger(__name__)


def _images(node: Any, out: list[dict]) -> None:
    if isinstance(node, dict):
        if node.get("image_path") and node.get("image_version_id"):
            out.append(node)
        for value in node.values():
            _images(value, out)
    elif isinstance(node, list):
        for value in node:
            _images(value, out)


async def add_image_urls(data: Any, storage: StorageService, user_id: str) -> None:
    """In place: add `image_url` next to every image_path/image_version_id pair in
    `data`. A failed or unresolved image leaves image_url null rather than failing
    the whole response."""
    images: list[dict] = []
    _images(data, images)
    if not images:
        return
    user = User(user_id=user_id, branch_id=user_id)
    keys = sorted({(i["image_path"], i["image_version_id"]) for i in images})

    urls: dict[tuple[str, str], Any] = await storage.get_presign_urls_batch(keys, user)
    missing = [k for k in keys if k not in urls]
    if missing:
        limit = asyncio.Semaphore(get_settings().face_image_presign_concurrency)

        async def _one(key: tuple[str, str]):
            try:
                async with limit:
                    return key, await storage.get_presign_url(key[0], key[1], user)
            except Exception as exc:  # noqa: BLE001 -- get_presign_url() can raise HTTPException or httpx errors; one image's failure must not block the batch, logged
                _logger.warning("face image URL failed for %s@%s: %s", key[0], key[1], exc)
                return key, None

        urls.update(await asyncio.gather(*[_one(k) for k in missing]))

    for i in images:
        i["image_url"] = urls.get((i["image_path"], i["image_version_id"]))
