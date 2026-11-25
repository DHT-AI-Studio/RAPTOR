"""PRIV-X3 / VIE01-357: rectification (GDPR Art. 16), restriction of
processing (Art. 18) and objection to processing (Art. 21).

Rectification covers the Keycloak profile only. The ticket also names an
"ArcadeDB user vertex" to propagate to, but no such vertex exists: Module
25's per-user database holds only content extracted from the user's files
(chunks, sources, entities, faces, temporal facts), never a profile -- a
user's name and email are stored in Keycloak alone.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.models.consent import ConsentPurposeId

# Field name in the request -> Keycloak user representation key.
RECTIFIABLE_FIELDS = {"email": "email", "firstName": "firstName", "lastName": "lastName"}

# Named by the ticket's objection list but with no consent purpose (and so
# nothing to stop) in this codebase yet -- rejected with this explanation.
UNSUPPORTED_OBJECTION_PURPOSES = {
    "speaker_cloning": "Module 35's voice cloning has no consent purpose yet",
}


class RectifiableField(str, Enum):
    EMAIL = "email"
    FIRST_NAME = "firstName"
    LAST_NAME = "lastName"


class RectifyRequest(BaseModel):
    field: RectifiableField
    value: str = Field(min_length=1, max_length=255)


class RectifyResponse(BaseModel):
    fields_updated: list[str]
    propagation_status: str = "complete"


class RestrictRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


class LiftRestrictionRequest(BaseModel):
    user_id: str = Field(min_length=1)


class RestrictedResponse(BaseModel):
    # The Keycloak attribute gdpr_processing_restricted, for Module 13's middleware.
    restricted: bool


class RestrictionStatus(BaseModel):
    user_id: str
    restricted: bool
    restricted_at: datetime | None = None
    lifted_at: datetime | None = None
    reason: str | None = None


class ObjectRequest(BaseModel):
    # A plain string, so the two unsupported purposes get a specific 422.
    processing_purpose: str


class ObjectResponse(BaseModel):
    purposes_stopped: list[ConsentPurposeId]
    # Objection stops future processing only; erasing past data is a
    # separate erasure request (GD-1) or consent withdrawal (GD-5).
    data_deleted: bool = False
