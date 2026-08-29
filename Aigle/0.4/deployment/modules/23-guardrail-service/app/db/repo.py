"""Raw asyncpg repository functions for guardrail_policies storage."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

import asyncpg

from app.engine.policy_targets import COLUMN_BY_DROPDOWN_NAME, CONTENT_COLUMNS

_DETAIL_COLUMNS = ", ".join(["id", "name", "version", *CONTENT_COLUMNS, "is_active", "created_at"])


async def upsert_policy_content(pool: asyncpg.Pool, name: str, version: str, target: str, content: str) -> asyncpg.Record:
    """Create a policy version, or set one target's content on an existing
    (name, version) — this is how a single policy accumulates per-guard-model
    prompt overrides across multiple uploads (one call per target)."""
    column = COLUMN_BY_DROPDOWN_NAME[target]
    async with pool.acquire() as conn, conn.transaction():
        existing = await conn.fetchrow(
            "SELECT id FROM guardrail_policies WHERE name = $1 AND version = $2 LIMIT 1",
            name, version,
        )
        if existing is not None:
            return await conn.fetchrow(
                f"UPDATE guardrail_policies SET {column} = $1 WHERE id = $2 "
                "RETURNING id, name, version, created_at",
                content, existing["id"],
            )
        if column == "raw_content":
            return await conn.fetchrow(
                "INSERT INTO guardrail_policies (name, version, raw_content) VALUES ($1, $2, $3) "
                "RETURNING id, name, version, created_at",
                name, version, content,
            )
        return await conn.fetchrow(
            f"INSERT INTO guardrail_policies (name, version, raw_content, {column}) "
            "VALUES ($1, $2, $3, $4) RETURNING id, name, version, created_at",
            name, version, "", content,
        )


async def update_policy_content(pool: asyncpg.Pool, policy_id: UUID, fields: dict[str, str]) -> asyncpg.Record | None:
    """Directly overwrite one or more content columns on an existing policy
    row by id. `fields` keys must be a subset of CONTENT_COLUMNS (enforced by
    the caller — app.engine.policy_store — via a closed Pydantic model, so
    this never interpolates arbitrary column names). Returns None if the
    policy doesn't exist."""
    set_clauses: list[str] = []
    params: list = []
    for column, value in fields.items():
        params.append(value)
        set_clauses.append(f"{column} = ${len(params)}")
    params.append(policy_id)
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            f"UPDATE guardrail_policies SET {', '.join(set_clauses)} WHERE id = ${len(params)} "
            f"RETURNING {_DETAIL_COLUMNS}",
            *params,
        )


async def list_policies(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT id, name, version, is_active, created_at "
            "FROM guardrail_policies ORDER BY created_at DESC"
        )


async def get_policy(pool: asyncpg.Pool, policy_id: UUID) -> asyncpg.Record | None:
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            f"SELECT {_DETAIL_COLUMNS} FROM guardrail_policies WHERE id = $1",
            policy_id,
        )


async def get_active_policy(pool: asyncpg.Pool) -> asyncpg.Record | None:
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            f"SELECT {_DETAIL_COLUMNS} FROM guardrail_policies WHERE is_active = true LIMIT 1"
        )


async def activate_policy(pool: asyncpg.Pool, policy_id: UUID) -> asyncpg.Record | None:
    """Deactivate whatever's currently active and activate `policy_id`.

    Existence is checked *before* deactivating anything, inside one transaction,
    so a bogus id can never leave the system with zero active policies.
    """
    async with pool.acquire() as conn, conn.transaction():
        exists = await conn.fetchval("SELECT 1 FROM guardrail_policies WHERE id = $1", policy_id)
        if not exists:
            return None
        await conn.execute("UPDATE guardrail_policies SET is_active = false WHERE is_active = true")
        return await conn.fetchrow(
            f"UPDATE guardrail_policies SET is_active = true WHERE id = $1 RETURNING {_DETAIL_COLUMNS}",
            policy_id,
        )


async def deactivate_policy(
    pool: asyncpg.Pool, policy_id: UUID
) -> tuple[Literal["ok", "not_found", "not_active"], asyncpg.Record | None]:
    async with pool.acquire() as conn, conn.transaction():
        is_active = await conn.fetchval("SELECT is_active FROM guardrail_policies WHERE id = $1", policy_id)
        if is_active is None:
            return "not_found", None
        if not is_active:
            return "not_active", None
        row = await conn.fetchrow(
            f"UPDATE guardrail_policies SET is_active = false WHERE id = $1 RETURNING {_DETAIL_COLUMNS}",
            policy_id,
        )
        return "ok", row


async def delete_policy_status(pool: asyncpg.Pool, policy_id: UUID) -> Literal["ok", "not_found", "active"]:
    async with pool.acquire() as conn, conn.transaction():
        is_active = await conn.fetchval("SELECT is_active FROM guardrail_policies WHERE id = $1", policy_id)
        if is_active is None:
            return "not_found"
        if is_active:
            return "active"
        await conn.execute("DELETE FROM guardrail_policies WHERE id = $1", policy_id)
        return "ok"
