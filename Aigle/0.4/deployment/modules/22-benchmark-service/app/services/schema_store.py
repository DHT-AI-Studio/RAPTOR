"""PostgreSQL CRUD for benchmark marking schemas (BM-3).

Table ``benchmark_schemas`` is created by 03-database init 001_init.sql:
    id UUID, name, version, pipeline, definition JSONB, created_at
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.db import db
from app.models.schema import BenchmarkSchema

logger = logging.getLogger(__name__)


async def create_schema(schema: BenchmarkSchema) -> Dict[str, Any]:
    """Persist a validated schema; return {id, name, version, created_at}."""
    row = await db.pool.fetchrow(
        """
        INSERT INTO benchmark_schemas (name, version, pipeline, definition)
        VALUES ($1, $2, $3, $4)
        RETURNING id, name, version, created_at
        """,
        schema.name,
        schema.version,
        schema.target_pipeline.value,
        json.dumps(schema.model_dump(mode="json")),
    )
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "version": row["version"],
        "created_at": row["created_at"],
    }


async def list_schemas(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Paginated list of schema summaries, newest first."""
    rows = await db.pool.fetch(
        """
        SELECT id, name, version, pipeline, created_at
        FROM benchmark_schemas
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit,
        offset,
    )
    return [
        {
            "id": str(r["id"]),
            "name": r["name"],
            "version": r["version"],
            "target_pipeline": r["pipeline"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


async def get_schema(schema_id: str) -> Optional[Dict[str, Any]]:
    """Full schema definition, or None if not found."""
    row = await _fetchrow_by_id(
        "SELECT id, name, version, pipeline, definition, created_at FROM benchmark_schemas WHERE id = $1",
        schema_id,
    )
    if row is None:
        return None
    definition = row["definition"]
    if isinstance(definition, str):
        definition = json.loads(definition)
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "version": row["version"],
        "target_pipeline": row["pipeline"],
        "created_at": row["created_at"],
        "definition": definition,
    }


async def delete_schema(schema_id: str) -> bool:
    """Delete a schema and its runs; return True if a schema row was removed."""
    if not _is_uuid(schema_id):
        return False
    async with db.pool.acquire() as conn:
        async with conn.transaction():
            # Remove child runs first (benchmark_runs.schema_id FK).
            await conn.execute("DELETE FROM benchmark_runs WHERE schema_id = $1", schema_id)
            result = await conn.execute("DELETE FROM benchmark_schemas WHERE id = $1", schema_id)
    # asyncpg returns e.g. "DELETE 1"
    return result.rsplit(" ", 1)[-1] != "0"


async def schema_exists(schema_id: str) -> bool:
    row = await _fetchrow_by_id("SELECT 1 FROM benchmark_schemas WHERE id = $1", schema_id)
    return row is not None


def partition_cases(cases: List[Dict[str, Any]], holdout_ratio: float = 0.3,
                    seed: int = 1234) -> Optional[tuple]:
    """Deterministically split test cases into (dev, held-out).

    Holds out ~``holdout_ratio`` of the cases (at least 1), always keeping at
    least one dev case. Returns None if there are fewer than 2 cases. Pure /
    side-effect-free so it is unit-testable without a DB.
    """
    import random

    if len(cases) < 2:
        return None
    order = list(range(len(cases)))
    random.Random(seed).shuffle(order)
    n_holdout = max(1, round(len(cases) * holdout_ratio))
    n_holdout = min(n_holdout, len(cases) - 1)  # keep at least one dev case
    holdout_idx = set(order[:n_holdout])
    dev = [c for i, c in enumerate(cases) if i not in holdout_idx]
    holdout = [c for i, c in enumerate(cases) if i in holdout_idx]
    return dev, holdout


async def split_for_holdout(schema_id: str, holdout_ratio: float = 0.3,
                            seed: int = 1234) -> Optional[tuple]:
    """Split a schema's test cases into a dev schema + a held-out schema.

    The optimizer runs against the dev schema; the held-out schema (untouched by
    the loop) is used once at the end to detect eval-overfitting. Deterministic
    given ``seed``. Returns (dev_schema_id, holdout_schema_id), or None if there
    are too few cases to split (< 2).
    """
    original = await get_schema(schema_id)
    if original is None:
        return None
    definition = original["definition"]
    partitioned = partition_cases(list(definition.get("test_cases", [])), holdout_ratio, seed)
    if partitioned is None:
        return None  # too few cases to split
    dev_cases, holdout_cases = partitioned

    base_name = original["name"]
    dev = await _create_derived(definition, dev_cases, f"{base_name} [dev]")
    holdout = await _create_derived(definition, holdout_cases, f"{base_name} [holdout]")
    return dev, holdout


async def _create_derived(definition: Dict[str, Any], cases: List[Dict[str, Any]],
                          name: str) -> str:
    """Create a schema identical to ``definition`` but with a subset of cases."""
    d = dict(definition)
    d["test_cases"] = cases
    d["name"] = name
    schema = BenchmarkSchema.model_validate(d)
    created = await create_schema(schema)
    return created["id"]


# ── helpers ──────────────────────────────────────────────────────────

def _is_uuid(value: str) -> bool:
    import uuid

    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


async def _fetchrow_by_id(query: str, schema_id: str):
    """fetchrow guarding against malformed (non-UUID) ids."""
    if not _is_uuid(schema_id):
        return None
    return await db.pool.fetchrow(query, schema_id)
