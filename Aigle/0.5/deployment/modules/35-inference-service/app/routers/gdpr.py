"""Internal GDPR endpoints, called only by Module 33 (the privacy-compliance
service) -- never by end users or Module 13's gateway: GD-1 / VIE01-246
erasure and GD-2 / VIE01-247's count for the export manifest.

Authenticates the caller via the shared X-Internal-Token secret, the same way
Modules 04/22/25/26's /internal/gdpr/* endpoints do, and takes an explicit
user_id since there's no session of the user's own here.
"""
from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request

from app.errors import ServiceError
from app.schemas import GdprDeleteResponse, GdprSummaryResponse
from app.services.voice.speaker_cache import SpeakerCache

router = APIRouter(prefix="/internal/gdpr", tags=["GDPR (internal)"])


def require_internal_token(
    request: Request, x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
) -> None:
    expected = request.app.state.settings.gdpr_internal_token
    if not expected or not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
        raise ServiceError(401, "invalid_internal_token")


def _cache(request: Request, user_id: str) -> SpeakerCache:
    settings = request.app.state.settings
    return SpeakerCache(request.app.state.voice_redis, user_id,
                        settings.speaker_cache_ttl_s, settings.speaker_cache_max)


@router.delete("/user/{user_id}", dependencies=[Depends(require_internal_token)],
               summary="Erase all speaker embeddings for a user (GD-1)", response_model=GdprDeleteResponse)
async def delete_user(request: Request, user_id: str) -> dict[str, object]:
    """Delete all of the user's voice-clone speaker embeddings. Always 200 --
    a user who never cloned a voice has nothing to delete, which isn't a
    failure for erasure."""
    return {"user_id": user_id, "speaker_embeddings_deleted": await _cache(request, user_id).purge()}


@router.get("/user/{user_id}/summary", dependencies=[Depends(require_internal_token)],
            summary="Count speaker embeddings for GDPR export manifest (GD-2)", response_model=GdprSummaryResponse)
async def summarize_user(request: Request, user_id: str) -> dict[str, object]:
    """Count only, never the embeddings: GD-2's export notes their existence
    in manifest.json (speaker_embeddings_cached) -- opaque model tensors
    aren't exportable personal data in any readable form."""
    return {"user_id": user_id, "speaker_embeddings": await _cache(request, user_id).count()}
