"""PA-10 — database lifecycle: create, empty stats, delete."""
from __future__ import annotations

import pytest

from app.services.arcadedb_client import db_name_for
from app.services.schema_init import initialize_schema

pytestmark = pytest.mark.asyncio


async def test_create_stats_empty_delete(arcade):
    branch = "ittest_life"
    db = db_name_for(branch)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)

    # create + schema (what POST /internal/db/init does)
    await arcade.create_database(db)
    await initialize_schema(arcade, db)
    assert await arcade.database_exists(db)

    # fresh DB → all vertex counts are zero
    for vtype in ("Chunk", "Entity", "Source", "TemporalFact"):
        rows = await arcade.query(db, f"SELECT count(*) AS c FROM {vtype}")
        assert rows[0]["c"] == 0

    # delete drops the whole database
    await arcade.drop_database(db)
    assert not await arcade.database_exists(db)


async def test_init_is_idempotent(arcade):
    branch = "ittest_life_idem"
    db = db_name_for(branch)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)
    try:
        await arcade.create_database(db)
        await initialize_schema(arcade, db)
        await initialize_schema(arcade, db)          # re-run must not raise
        assert await arcade.database_exists(db)
    finally:
        await arcade.drop_database(db)


# ── VIE01-189 subtasks ────────────────────────────────────────────────────────

async def test_identifier_sanitization_rejects_special_characters():
    """A `sub` carrying SQL/path metacharacters must not reach ArcadeDB intact.

    Pure unit test — no server needed. Everything outside [A-Za-z0-9_] collapses
    to `_`, so the name can neither close a quoted string, terminate a statement,
    nor escape a REST path segment.
    """
    assert db_name_for("3f2b9c1a-77de-4a10-9b21-0c5e8d4f6a33") == \
        "user_3f2b9c1a_77de_4a10_9b21_0c5e8d4f6a33"
    assert db_name_for("ev'il; DROP DATABASE x--") == "user_ev_il__DROP_DATABASE_x"
    assert db_name_for("../../etc/passwd") == "user_etc_passwd"
    assert db_name_for("") == "user_anon"

    for raw in ("a-b", "a.b", "a/b", "a b", "a'b", 'a"b', "a;b", "a--b", "a\\b", "a\x00b"):
        name = db_name_for(raw)
        assert name.startswith("user_")
        assert name[5:].replace("_", "").isalnum(), name


async def test_init_endpoint_is_idempotent_over_http(arcade):
    """POST /internal/db/init twice for the same user → 200, created flips to false,
    and exactly one database exists afterwards."""
    import httpx
    from app.main import app

    user_id = "ittest-life-http"          # dashes on purpose: sanitized to user_ittest_life_http
    db = db_name_for(user_id)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            headers = {"X-User-ID": user_id}

            first = await http.post("/internal/db/init", headers=headers)
            assert first.status_code == 200, first.text
            assert first.json() == {"user_id": user_id, "database": db,
                                    "created": True, "status": "ready"}

            second = await http.post("/internal/db/init", headers=headers)
            assert second.status_code == 200, second.text
            assert second.json()["created"] is False

            # no duplicate database was made
            names = await arcade.list_databases()
            assert names.count(db) == 1

            status = await http.get("/internal/db/status", headers=headers)
            assert status.status_code == 200
            body = status.json()
            assert body["user_id"] == user_id
            assert body["db_exists"] is True
            assert body["record_counts"]["chunks"] == 0
    finally:
        await arcade.drop_database(db)


async def test_status_reports_missing_database_without_404(arcade):
    """A user who has never been provisioned is a normal 200 answer, not an error."""
    import httpx
    from app.main import app

    user_id = "ittest-life-absent"
    db = db_name_for(user_id)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        resp = await http.get("/internal/db/status", headers={"X-User-ID": user_id})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "user_id": user_id,
        "db_exists": False,
        "record_counts": {"chunks": 0, "entities": 0, "sources": 0,
                          "temporal_facts": 0, "by_type": {}},
    }


# ── VIE01-189 deletion audit ──────────────────────────────────────────────────

async def test_delete_refuses_when_audit_unavailable(arcade, monkeypatch):
    """The whole point of a mandatory audit: if it cannot be written, the
    irreversible action does not happen. 503, and the database is still there."""
    import httpx
    from app.main import app
    from app.core.config import settings
    from app.routers import database as database_router

    user_id = "ittest-audit-down"
    db = db_name_for(user_id)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)
    await arcade.create_database(db)
    await initialize_schema(arcade, db)

    async def _boom(*_a, **_kw):
        raise RuntimeError("postgres unreachable")

    monkeypatch.setattr(settings, "audit_required", True)
    monkeypatch.setattr(database_router, "record_deletion", _boom)

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            resp = await http.delete("/internal/db", headers={"X-User-ID": user_id})

        assert resp.status_code == 503, resp.text
        assert await arcade.database_exists(db), "database was dropped despite the audit failing"
    finally:
        await arcade.drop_database(db)


async def test_delete_records_audit_row_then_drops(arcade, monkeypatch):
    """Happy path: one audit row, carrying the counts as they were *before* the
    drop, and the database is gone afterwards."""
    import httpx
    from app.main import app
    from app.core.config import settings
    from app.routers import database as database_router

    user_id = "ittest-audit-ok"
    db = db_name_for(user_id)
    if await arcade.database_exists(db):
        await arcade.drop_database(db)
    await arcade.create_database(db)
    await initialize_schema(arcade, db)

    recorded: list[tuple] = []

    async def _capture(uid, dbname, counts):
        recorded.append((uid, dbname, counts))

    monkeypatch.setattr(settings, "audit_required", True)
    monkeypatch.setattr(database_router, "record_deletion", _capture)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        resp = await http.delete("/internal/db", headers={"X-User-ID": user_id})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"user_id": user_id, "database": db, "deleted": True}
    assert not await arcade.database_exists(db)

    assert len(recorded) == 1
    uid, dbname, counts = recorded[0]
    assert (uid, dbname) == (user_id, db)
    assert counts["chunks"] == 0 and "by_type" in counts
