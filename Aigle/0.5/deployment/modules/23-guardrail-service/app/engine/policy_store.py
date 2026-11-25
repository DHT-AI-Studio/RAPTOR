"""Policy Store — CRUD over the Postgres-backed guardrail_policies table.

Postgres is the source of truth. `activate_policy`/`update_policy_content`
also write the newly active policy into Redis (`guardrail:active_policy`) so
`policy_engine.py`'s evaluation path can read it without a DB round-trip.

Postgres commits *before* the Redis write is attempted (activate_policy's
transaction closes inside app/db/repo.py, well before this module ever
touches Redis), so a Redis failure here can never roll back — or even be
noticed by — the already-committed DB change. If that Redis write silently
fails (a transient reconnect, a momentary timeout, ...), a stale value would
otherwise sit in the cache forever: `policy_engine._get_active_policy()`
only ever falls back to Postgres on a cache *miss*, so a stale-but-present
key is never re-checked. `_refresh_active_cache` below closes that gap two
ways — best-effort delete-on-failure (so the very next check call sees a
clean miss and re-warms immediately) plus a short TTL on every write (so
even a failed delete self-heals within ACTIVE_POLICY_CACHE_TTL seconds) —
and never raises, since a cache-layer hiccup must not turn an
already-committed Postgres write into a 500.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from uuid import UUID

from fastapi import HTTPException

from app.db import connection, redis_conn, repo
from app.engine.models import PolicyCreateResponse, PolicyDetail, PolicySummary
from app.services.default_policy import (
    DEFAULT_POLICY_NAME,
    DEFAULT_POLICY_VERSION,
    DEFAULT_STANDARD_POLICY,
)
from app.services.policy_translator import translate_policy_to_prompts

_logger = logging.getLogger(__name__)

ACTIVE_POLICY_CACHE_KEY = "guardrail:active_policy"
ACTIVE_POLICY_CACHE_TTL = 60   # seconds — bounds how long a desynced cache can stay wrong


async def _refresh_active_cache(detail: PolicyDetail) -> None:
    """Best-effort: write the active policy into Redis with a bounded TTL.
    Never raises — Postgres already has the correct data; on failure this
    falls back to deleting the (now possibly stale) key so the next read is
    a clean cache-miss instead of silently serving old content."""
    try:
        redis = await redis_conn.get_redis()
        await redis.set(ACTIVE_POLICY_CACHE_KEY, detail.model_dump_json(), ex=ACTIVE_POLICY_CACHE_TTL)
    except Exception:
        _logger.exception(
            "guardrail policy: failed to refresh active-policy Redis cache for id=%s — "
            "deleting the stale key so the next check call re-warms from Postgres", detail.id,
        )
        try:
            redis = await redis_conn.get_redis()
            await redis.delete(ACTIVE_POLICY_CACHE_KEY)
        except Exception:
            _logger.exception(
                "guardrail policy: failed to delete stale active-policy cache key too — "
                "it will self-heal after TTL=%ss", ACTIVE_POLICY_CACHE_TTL,
            )


def _row_to_detail(row) -> PolicyDetail:
    return PolicyDetail(
        id=row["id"], name=row["name"], version=row["version"],
        raw_content=row["raw_content"],
        content_llama_guard=row["content_llama_guard"],
        content_granite_guardian=row["content_granite_guardian"],
        content_gpt_oss_safeguard=row["content_gpt_oss_safeguard"],
        is_active=row["is_active"], created_at=row["created_at"],
    )


async def create_policy(name: str, version: str, raw_content: str, target: str = "original") -> PolicyCreateResponse:
    """Create a policy version, or set one `target`'s content on an existing
    (name, version) — see app/db/repo.py's upsert_policy_content."""
    pool = await connection.get_pool()
    row = await repo.upsert_policy_content(pool, name, version, target, raw_content)
    return PolicyCreateResponse(
        id=row["id"], name=row["name"], version=row["version"], created_at=row["created_at"],
    )


async def update_policy_content(policy_id: UUID, fields: dict[str, str]) -> PolicyDetail:
    """Directly overwrite one or more content columns on an existing policy
    (see app/db/repo.py's update_policy_content). If this policy is the
    active one, the Redis cache is refreshed so /policy/check/* sees the new
    content on the very next request — otherwise a stale cached copy would
    keep serving the old content until some other write happened to refresh it."""
    pool = await connection.get_pool()
    row = await repo.update_policy_content(pool, policy_id, fields)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    detail = _row_to_detail(row)

    if detail.is_active:
        await _refresh_active_cache(detail)

    return detail


async def list_policies() -> list[PolicySummary]:
    pool = await connection.get_pool()
    rows = await repo.list_policies(pool)
    return [
        PolicySummary(
            id=r["id"], name=r["name"], version=r["version"],
            is_active=r["is_active"], created_at=r["created_at"],
        )
        for r in rows
    ]


async def get_policy(policy_id: UUID) -> PolicyDetail:
    pool = await connection.get_pool()
    row = await repo.get_policy(pool, policy_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return _row_to_detail(row)


async def activate_policy(policy_id: UUID) -> PolicyDetail:
    pool = await connection.get_pool()
    row = await repo.activate_policy(pool, policy_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    detail = _row_to_detail(row)

    detail = await _compose_and_persist_prompts(pool, detail)
    await _refresh_active_cache(detail)

    return detail


async def _compose_and_persist_prompts(pool, detail: PolicyDetail) -> PolicyDetail:
    """If `detail.raw_content` is the Standard Policy Format, compose each
    guard model's own prompt (app.services.policy_translator) and persist it
    to that model's content_* column — but only where the column is still
    empty. A column the caller already filled in by hand (a manual per-model
    override, uploaded via target=llama-guard/granite-guardian/gpt-oss-safeguard)
    IS that policy's final prompt already and is never overwritten. Composed
    prompts are never cleared on deactivation — deactivate_policy/delete_policy
    don't touch these columns, so they stay in place as a record of what was
    actually sent the last time this policy was active.
    """
    composed = translate_policy_to_prompts(detail.raw_content)
    if composed is None:
        return detail

    fields = {
        column: json.dumps(dataclasses.asdict(template))
        for column, template in composed.items()
        if not getattr(detail, column)
    }
    if not fields:
        return detail

    row = await repo.update_policy_content(pool, detail.id, fields)
    if row is None:   # policy vanished between activation and this write — keep the pre-compose detail
        return detail
    return _row_to_detail(row)


async def get_active_policy() -> PolicyDetail:
    pool = await connection.get_pool()
    row = await repo.get_active_policy(pool)
    if row is None:
        raise HTTPException(status_code=404, detail="No active policy")
    return _row_to_detail(row)


async def ensure_default_policy_active() -> PolicyDetail:
    """Guarantee an active policy exists, falling back to the built-in
    default Standard Policy (app/services/default_policy.py) when nothing
    else is active. Called both at startup (app/main.py's lifespan) and
    after deactivate_policy() below leaves nothing active — same invariant,
    same code path, either way.

    If the default policy already exists (it was created once before, and
    possibly hand-edited by an admin since), that existing row is reused and
    reactivated as-is — its content is never overwritten. It's only created
    fresh, with the built-in 13-category content, the first time this is
    ever called."""
    pool = await connection.get_pool()
    row = await repo.get_policy_by_name_version(pool, DEFAULT_POLICY_NAME, DEFAULT_POLICY_VERSION)
    if row is None:
        created = await create_policy(DEFAULT_POLICY_NAME, DEFAULT_POLICY_VERSION, json.dumps(DEFAULT_STANDARD_POLICY))
        policy_id = created.id
    else:
        policy_id = row["id"]
    return await activate_policy(policy_id)


async def deactivate_policy(policy_id: UUID) -> PolicyDetail:
    pool = await connection.get_pool()
    status, row = await repo.deactivate_policy(pool, policy_id)
    if status == "not_found":
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    if status == "not_active":
        raise HTTPException(status_code=409, detail=f"Policy '{policy_id}' is not currently active")
    detail = _row_to_detail(row)

    try:
        redis = await redis_conn.get_redis()
        await redis.delete(ACTIVE_POLICY_CACHE_KEY)
    except Exception:
        _logger.exception(
            "guardrail policy: failed to clear active-policy Redis cache on deactivate of id=%s — "
            "it will self-heal after TTL=%ss", policy_id, ACTIVE_POLICY_CACHE_TTL,
        )

    # Deactivating leaves zero active policies — fall back to the default,
    # unless the thing just deactivated WAS the default (that's the one way
    # to deliberately end up with no active policy at all; don't immediately
    # re-activate it and defeat that).
    if detail.name != DEFAULT_POLICY_NAME:
        await ensure_default_policy_active()

    return detail   # reports the policy that was just deactivated, not the fallback


async def delete_policy(policy_id: UUID) -> None:
    pool = await connection.get_pool()
    status = await repo.delete_policy_status(pool, policy_id, protected_name=DEFAULT_POLICY_NAME)
    if status == "not_found":
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    if status == "active":
        raise HTTPException(status_code=409, detail="Cannot delete the active policy — activate another one first")
    if status == "protected":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete '{DEFAULT_POLICY_NAME}' — it is the built-in fallback policy",
        )
