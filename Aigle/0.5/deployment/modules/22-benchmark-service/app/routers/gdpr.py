"""GD-5 / VIE01-250: internal GDPR endpoint, called only by Module 33 (the
privacy-compliance service) when a user withdraws benchmark consent -- never
by end users or Module 13's gateway.

Authenticates the *caller* (Module 33) via the shared X-Internal-Token
secret, the same way Modules 04/25/26's /internal/gdpr/* endpoints do, and
takes an explicit user_id since there's no session of the user's own here.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from app.core.config import get_settings
from app.services import schema_store

router = APIRouter(prefix="/internal/gdpr", tags=["GDPR (internal)"])


def require_internal_token(x_internal_token: str = Header(None, alias="X-Internal-Token")) -> None:
    expected = get_settings().gdpr_internal_token
    if not expected or x_internal_token != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Internal-Token")


@router.delete("/benchmark/user/{user_id}", dependencies=[Depends(require_internal_token)])
async def delete_user_benchmark_data(user_id: str) -> dict[str, object]:
    """Delete the user's runs and the schemas only they use; detach (clear
    the owner of) schemas other users still have runs on. See
    schema_store.delete_user_benchmark_data."""
    counts = await schema_store.delete_user_benchmark_data(user_id)
    return {"user_id": user_id, **counts}
