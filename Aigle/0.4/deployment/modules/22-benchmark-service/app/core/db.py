"""Shared connection handles for the benchmark service.

Holds a single asyncpg pool (schema definitions + run history) and one
redis.asyncio client (live run state). Both are created during the FastAPI
lifespan and shared by schema_store / run_manager.
"""
from __future__ import annotations

import logging
from typing import Optional

import asyncpg
import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Database:
    """Container for the process-wide asyncpg pool and redis client."""

    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        settings = get_settings()

        await self._ensure_database(settings)
        self.pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            min_size=1,
            max_size=10,
        )
        await self._ensure_tables()
        logger.info(
            "Connected to PostgreSQL %s:%s/%s",
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
        )

        self.redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
        )
        await self.redis.ping()
        logger.info("Connected to Redis %s:%s", settings.redis_host, settings.redis_port)

    async def _ensure_database(self, settings) -> None:
        """Create the benchmark database if it does not yet exist.

        The canonical path is 03-database init 001_init.sql on a fresh volume;
        this bootstrap covers the common case where PostgreSQL was already
        initialized before Module 22 existed (init script won't re-run).
        """
        try:
            conn = await asyncpg.connect(
                host=settings.postgres_host, port=settings.postgres_port,
                user=settings.postgres_user, password=settings.postgres_password,
                database=settings.postgres_db,
            )
            await conn.close()
            return
        except asyncpg.InvalidCatalogNameError:
            pass  # database missing → create it below

        admin = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database="postgres",
        )
        try:
            await admin.execute(f'CREATE DATABASE "{settings.postgres_db}"')
            logger.info("Created PostgreSQL database %s", settings.postgres_db)
        finally:
            await admin.close()

    async def _ensure_tables(self) -> None:
        """Idempotently create benchmark tables (mirrors 001_init.sql)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_schemas (
                    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name       VARCHAR(255) NOT NULL,
                    version    VARCHAR(50)  DEFAULT '1.0',
                    pipeline   VARCHAR(50)  NOT NULL,
                    definition JSONB        NOT NULL,
                    created_at TIMESTAMPTZ  DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_bs_created ON benchmark_schemas (created_at DESC);

                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    schema_id            UUID REFERENCES benchmark_schemas(id) ON DELETE CASCADE,
                    status               VARCHAR(50)  DEFAULT 'queued',
                    aggregate_score      FLOAT,
                    scores_per_dimension JSONB,
                    scores_per_case      JSONB,
                    config_override      JSONB,
                    started_at           TIMESTAMPTZ,
                    completed_at         TIMESTAMPTZ,
                    created_at           TIMESTAMPTZ  DEFAULT NOW(),
                    mlflow_run_id        VARCHAR(255)
                );
                CREATE INDEX IF NOT EXISTS idx_br_schema  ON benchmark_runs (schema_id);
                CREATE INDEX IF NOT EXISTS idx_br_created ON benchmark_runs (created_at DESC);

                -- Auto-tuning experiments (AUTOTUNE). One row per optimization goal.
                -- Per-iteration (config -> score) history is reused from benchmark_runs
                -- via config_override + aggregate_score, so no separate iteration table.
                CREATE TABLE IF NOT EXISTS experiments (
                    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    goal             TEXT,
                    status           VARCHAR(50)  DEFAULT 'planning',
                    plan             JSONB,
                    eval_schema_id   UUID REFERENCES benchmark_schemas(id) ON DELETE SET NULL,
                    budget            JSONB,
                    iterations_done   INTEGER      DEFAULT 0,
                    best_run_id       UUID,
                    best_score        FLOAT,
                    best_config       JSONB,
                    holdout_schema_id UUID,
                    holdout_score     FLOAT,
                    error             TEXT,
                    created_at        TIMESTAMPTZ  DEFAULT NOW(),
                    completed_at      TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS idx_exp_status  ON experiments (status);
                CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments (created_at DESC);
                -- Older deployments: add newer columns if missing.
                ALTER TABLE experiments ADD COLUMN IF NOT EXISTS holdout_schema_id   UUID;
                ALTER TABLE experiments ADD COLUMN IF NOT EXISTS holdout_score       FLOAT;
                ALTER TABLE experiments ADD COLUMN IF NOT EXISTS generated_schema_id UUID;
                ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mlflow_run_id VARCHAR(255);
                """
            )

    async def disconnect(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None


# Process-wide singleton, populated in main.lifespan().
db = Database()
