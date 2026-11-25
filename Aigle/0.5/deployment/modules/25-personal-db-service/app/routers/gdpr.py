"""GD-1 / VIE01-246: internal erasure endpoints, called only by Module 33
(the privacy-compliance orchestrator), never by end users or Module 13.

Distinct from DELETE /internal/db (database.py) -- that endpoint trusts
X-User-ID/X-Branch-ID headers Module 13 injects for a user deleting their
own database. This one authenticates the *caller* (Module 33) via a shared
secret instead, and accepts an explicit user_id since an admin-initiated
erasure has no session of the erased user's own to derive headers from.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from app.core.config import settings
from app.routers.database import _collect_counts, client
from app.services.arcadedb_client import db_name_for
from app.services.audit import record_deletion
from app.services.identity_store import FACE_LAYER, purge_face_layer

logger = logging.getLogger("personal_db.gdpr")
router = APIRouter(prefix="/internal/gdpr", tags=["gdpr"])


def require_internal_token(x_internal_token: str = Header(None, alias="X-Internal-Token")) -> None:
    if not settings.gdpr_internal_token or x_internal_token != settings.gdpr_internal_token:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Internal-Token")


@router.delete("/user/{user_id}", dependencies=[Depends(require_internal_token)])
async def erase_user(user_id: str):
    db = db_name_for(user_id)

    exists = await client.database_exists(db)
    counts = await _collect_counts(db) if exists else None

    if settings.audit_required and exists:
        try:
            await record_deletion(user_id, db, counts.model_dump())
        except Exception as exc:  # noqa: BLE001 -- any failure to write the audit (Postgres down, DNS, timeout, a bad pool) must refuse the irreversible erasure
            logger.error("[gdpr] audit write failed for %s, refusing delete: %s", db, exc)
            raise HTTPException(status_code=503, detail="Deletion audit log unavailable; delete refused. Retry later.")

    if exists:
        await client.drop_database(db)
    return {"user_id": user_id, "database": db, "deleted": exists}


@router.delete("/faces/user/{user_id}", dependencies=[Depends(require_internal_token)])
async def erase_faces(user_id: str):
    """GD-11: Module 33's cascade when the user withdraws face_detection consent --
    deletes only the face layer of the user's database (see
    identity_store.purge_face_layer); the database and its content index stay.
    Always 200 with the counts deleted: a user with no database, or no faces, has
    nothing to delete, which isn't a failure for the cascade. Idempotent, so Module
    33's retry is safe."""
    db = db_name_for(user_id)
    exists = await client.database_exists(db)
    deleted = await purge_face_layer(client, db) if exists else {key: 0 for _, key in FACE_LAYER}
    logger.info("[gdpr] face layer erased for %s: %s", db, deleted)
    return {"user_id": user_id, "database": db, "db_exists": exists, "deleted": deleted}


async def _count_documents(db: str) -> int:
    """Distinct document versions with anything indexed: the union of
    Source vertices tagged media_type='documents' and the version_ids of
    documents-type Chunks. Neither side alone is complete -- live data has
    document Chunks with no tagged Source (Source.media_type is optional,
    and only _link_source sets it), and index_source_summary can create a
    document's Source before any of its Chunks exist. Only version_id is
    read; a missing vertex type counts as none, same as _collect_counts."""

    async def _version_ids(sql: str) -> set:
        try:
            return {r["version_id"] for r in await client.query(db, sql) if r.get("version_id")}
        except Exception as e:  # noqa: BLE001 -- missing vertex type or a real query error are both reported as empty, logged
            logger.warning(f"version_id query failed, reporting none (missing type or real query error): {e}")
            return set()

    return len(
        await _version_ids("SELECT version_id FROM Source WHERE media_type = 'documents'")
        | await _version_ids("SELECT DISTINCT(version_id) AS version_id FROM Chunk WHERE type = 'documents'")
    )


async def _count_pii_flagged(db: str):
    """GD-6 / VIE01-251: Chunks flagged at ingestion as containing PII
    (Module 12's scan -> pii_flag). None when the count itself fails, so a
    query error is never reported as "no PII"."""
    try:
        rows = await client.query(db, "SELECT count(*) AS n FROM Chunk WHERE pii_flag = true")
    except Exception as exc:  # noqa: BLE001 -- missing type or a real query error both report None, logged
        logger.warning("[gdpr] pii_flagged count failed for %s: %s", db, exc)
        return None
    return int(rows[0].get("n", 0)) if rows else 0


@router.get("/user/{user_id}/summary", dependencies=[Depends(require_internal_token)])
async def summarize_user(user_id: str):
    """GD-4 / VIE01-249: count-only summary for Module 33's DSAR assembler
    -- never row contents (that's /export's job, and GD-2's cost).

    documents counts indexed documents, not their chunks -- a data subject
    with 3 PDFs should read "documents: 3", not the few hundred search
    chunks those PDFs were split into. See _count_documents. moments stays
    a Chunk count (videos/images/audios), since a moment IS a segment.
    pii_flagged (GD-6) counts Chunks flagged as containing PII at ingestion.

    indexed_items is internal-only, not a DSAR report field: Module 33's
    ">10k indexed items" sync/async decision. It counts every indexed
    record (all chunks + entities), since that's what a large account's
    cost scales with -- `documents` can't stand in for it now that it
    counts documents rather than their chunks (20 PDFs can be 50k chunks).
    Both counts already come from _collect_counts, so no extra query."""
    db = db_name_for(user_id)
    if not await client.database_exists(db):
        return {
            "user_id": user_id, "db_exists": False,
            "documents": 0, "moments": 0, "entities": 0, "pii_flagged": 0, "indexed_items": 0,
        }

    counts = await _collect_counts(db)
    return {
        "user_id": user_id,
        "db_exists": True,
        "documents": await _count_documents(db),
        "moments": sum(counts.by_type.get(t, 0) for t in ("videos", "images", "audios")),
        "entities": counts.entities,
        "pii_flagged": await _count_pii_flagged(db),
        "indexed_items": counts.chunks + counts.entities,
    }


@router.get("/user/{user_id}/export",dependencies=[Depends(require_internal_token)])
async def export_user(user_id: str):
    """GD-2 / VIE01-247: JSON-serializable content for the export ZIP's
    content/ section.

    The ticket's documents.json/entities.json/moments.json split assumes a
    per-content-type vertex schema (Document/Moment) that no longer exists
    in this codebase -- Module 25 unified those into one `Chunk` vertex
    type (see schema_init.py's own comment on why). Reproduced here as
    close to the ticket's intent as the real schema allows: `documents`
    and `moments` are both Chunk rows, split by their `type` property
    (documents vs videos/images/audios); `entities` is the real Entity
    vertex type, matching the ticket as-is.
    """
    db = db_name_for(user_id)
    if not await client.database_exists(db):
        return {"documents": [], "moments": [], "entities": []}

    chunks = await client.query(db, "SELECT * FROM Chunk")
    documents = [c for c in chunks if c.get("type") == "documents"]
    moments = [c for c in chunks if c.get("type") in ("videos", "images", "audios")]
    entities = await client.query(db, "SELECT * FROM Entity")

    return {"documents": documents, "moments": moments, "entities": entities}
