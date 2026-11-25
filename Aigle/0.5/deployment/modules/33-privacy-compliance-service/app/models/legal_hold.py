"""PRIV-X4 / VIE01-358: legal holds on erasure (GDPR Art. 17(3)).

A hold keeps a user's data while it must be retained for legal compliance or
legal claims: an erasure requested during it is DEFERRED, and runs once the
hold ends (the daily expiry job). The request body is the ticket's.
"""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class LegalHoldRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1, max_length=255)       # e.g. "regulatory_audit"
    hold_until: date                                        # the hold ends at the start of this day (UTC)
    authority: str = Field(min_length=1, max_length=255)    # e.g. "DPO"


class LegalHold(BaseModel):
    id: int
    user_id: str
    reason: str
    authority: str
    placed_at: datetime
    hold_until: date
    released_at: datetime | None = None


class LegalHoldList(BaseModel):
    holds: list[LegalHold]
