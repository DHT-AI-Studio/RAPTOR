"""Per-user database lifecycle endpoints (PA-2 / VIE01-189).

Identity comes from the headers Module 13 injects from the caller's JWT:
`X-User-ID` carries the `sub` claim. `X-Branch-ID` is accepted as a fallback so
the 0.3 callers that only send that header keep working. Each user maps to one
ArcadeDB database `user_{sub}`.

Module 13 exposes these as `/api/0.4/personal-db/*`; this service is internal
and never reachable from outside the raptor network.
"""
from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.audit import record_deletion
from app.services.schema_init import initialize_schema

logger = logging.getLogger("personal_db.database")
router = APIRouter(prefix="/internal/db", tags=["database lifecycle"])
client = ArcadeDBClient()


def _require_user(x_user_id: str | None, x_branch_id: str | None) -> str:
    user_id = x_user_id or x_branch_id
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-ID header")
    return user_id


class InitResponse(BaseModel):
    user_id: str
    database: str
    created: bool                  # False on the second call — init is idempotent
    status: str = "ready"


class RecordCounts(BaseModel):
    chunks: int
    entities: int
    sources: int
    temporal_facts: int
    by_type: Dict[str, int]        # {documents: n, videos: n, images: n, audios: n}


class StatusResponse(BaseModel):
    user_id: str
    db_exists: bool
    record_counts: RecordCounts


class DeleteResponse(BaseModel):
    user_id: str
    database: str
    deleted: bool


_EMPTY_COUNTS = RecordCounts(chunks=0, entities=0, sources=0, temporal_facts=0, by_type={})


async def _collect_counts(db: str) -> RecordCounts:
    """Count what the database holds. Missing vertex types count as zero rather
    than raising — a schema that predates a later type must still report."""

    async def _count(vtype: str) -> int:
        try:
            rows = await client.query(db, f"SELECT count(*) AS c FROM {vtype}")
            return rows[0]["c"] if rows else 0
        except Exception:
            return 0

    by_type: Dict[str, int] = {}
    try:
        for r in await client.query(db, "SELECT type, count(*) AS c FROM Chunk GROUP BY type"):
            if r.get("type"):
                by_type[r["type"]] = r["c"]
    except Exception:
        pass

    return RecordCounts(
        chunks=await _count("Chunk"),
        entities=await _count("Entity"),
        sources=await _count("Source"),
        temporal_facts=await _count("TemporalFact"),
        by_type=by_type,
    )


@router.post("/init", response_model=InitResponse)
async def init_db(
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Create the user's database and initialize its schema.

    Idempotent: Module 13 calls this on every user's first request of a session,
    so a second call must return 200 with `created: false` and leave the existing
    database untouched.
    """
    user_id = _require_user(x_user_id, x_branch_id)
    db = db_name_for(user_id)
    created = False
    if not await client.database_exists(db):
        await client.create_database(db)
        created = True
    await initialize_schema(client, db)   # idempotent
    return InitResponse(user_id=user_id, database=db, created=created)


@router.get("/status", response_model=StatusResponse)
async def db_status(
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Report whether the user's database exists, and what it holds.

    A missing database is a normal state (the user has simply never been
    provisioned), so it answers 200 with `db_exists: false` and zeroed counts
    rather than 404 — the caller is asking a question, not addressing a resource.
    """
    user_id = _require_user(x_user_id, x_branch_id)
    db = db_name_for(user_id)
    if not await client.database_exists(db):
        return StatusResponse(user_id=user_id, db_exists=False, record_counts=_EMPTY_COUNTS)
    return StatusResponse(user_id=user_id, db_exists=True, record_counts=await _collect_counts(db))


@router.delete("", response_model=DeleteResponse)
async def delete_db(
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_branch_id: str = Header(None, alias="X-Branch-ID"),
):
    """Drop the user's entire database and everything in it.

    Module 13 restricts this to the owner of the database (RBAC); this service
    trusts the identity headers because it is not routable from outside.

    The deletion is logged to PostgreSQL *before* ArcadeDB is touched, and a
    failure to log aborts the delete with 503. The alternative — destroying the
    data and carrying on when the audit is unavailable — would make the audit
    table unable to answer the one question it exists for.
    """
    user_id = _require_user(x_user_id, x_branch_id)
    db = db_name_for(user_id)

    exists = await client.database_exists(db)
    counts = await _collect_counts(db) if exists else _EMPTY_COUNTS

    if settings.audit_required:
        try:
            await record_deletion(user_id, db, counts.model_dump())
        except Exception as exc:
            logger.error("[db] audit write failed for %s, refusing delete: %s", db, exc)
            raise HTTPException(
                status_code=503,
                detail="Deletion audit log unavailable; delete refused. Retry later.",
            )

    if exists:
        await client.drop_database(db)
    return DeleteResponse(user_id=user_id, database=db, deleted=exists)
