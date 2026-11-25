"""Turn the asset identifiers the platform hands out into a presigned URL.

The asset service has two spellings for the same file:

* `asset_path` + `version_id` -- what the REST API, `UploadAssetResponse.asset_id`, search results
  and `SubmitJob` use. `asset_path` has NO file name (`video/mp4/clip`); the name is the version's
  `primary_filename`.
* the LakeFS object key, `asset_path/primary_filename` (`video/mp4/clip/clip.mp4`) -- what its
  `/presign-url/{key}/{version_id}` endpoint wants.

DownloadAsset and StreamVideoSegment used to pass `asset_path` straight in as the key, so the
identifier the platform itself returned (for example from UploadAsset) could not be used to read the
asset back (`INTERNAL`, 404). They now accept either spelling.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from fastapi import HTTPException
from interceptors.errors import NotFoundError, ValidationError

# Looking an asset up is two idempotent reads (metadata, presigned URL). On the test host they
# sometimes failed for a moment, mostly right after an upload (about 3 in 60 downloads answered
# UNAVAILABLE and worked when repeated), so a transient failure is retried before it reaches the
# caller. Nothing has been sent to the client yet at this point, so a retry cannot duplicate or
# cut anything.
_RETRY_DELAYS_S = (0.3, 1.0)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, HTTPException):
        return exc.status_code in (502, 503, 504)
    return isinstance(exc, httpx.TransportError)


async def _with_retries(call):
    for delay in _RETRY_DELAYS_S:
        try:
            return await call()
        except Exception as exc:  # only transient errors are swallowed, the rest re-raised
            if not _is_transient(exc):
                raise
            await asyncio.sleep(delay)
    return await call()


@dataclass(frozen=True)
class ResolvedAsset:
    url: str
    filename: str


async def _primary_filename(storage_service, asset_path: str, version_id: str, user) -> str | None:
    try:
        payload = await _with_retries(lambda: storage_service.download_asset(asset_path, version_id, user))
    except HTTPException as exc:
        if exc.status_code == 404:
            return None  # not a platform asset_path; the caller may have sent the full key
        raise
    metadata = payload.get("metadata", payload) if isinstance(payload, dict) else {}
    return metadata.get("primary_filename") or None


async def resolve_asset(storage_service, asset_path: str, version_id: str, user) -> ResolvedAsset:
    """Presigned URL and file name for an asset, from either identifier spelling.

    The platform's `asset_path` is tried first (one metadata lookup, then the presign); a full
    object key still works as before.
    """
    if not version_id:
        # The asset service answers a presign request with no version by redirecting (HTTP 307), which
        # used to reach the caller as an unexplained INVALID_ARGUMENT. Say what is missing instead.
        raise ValidationError("version_id is required together with asset_path (UploadAssetResponse.version_id).")
    candidates: list[tuple[str, str]] = []
    filename = await _primary_filename(storage_service, asset_path, version_id, user)
    if filename:
        candidates.append((f"{asset_path}/{filename}", filename))
    candidates.append((asset_path, asset_path.rsplit("/", 1)[-1]))

    for key, name in candidates:
        try:
            url = await _with_retries(lambda key=key: storage_service.get_presign_url(key, version_id, user))
        except HTTPException as exc:
            if exc.status_code == 404:
                continue
            raise
        if url:
            return ResolvedAsset(url=url, filename=name)
    raise NotFoundError(f"Asset not found: {asset_path} (version {version_id}).")
