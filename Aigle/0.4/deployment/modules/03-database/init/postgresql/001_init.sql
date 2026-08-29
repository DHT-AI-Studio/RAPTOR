-- Raptor PostgreSQL initialization
-- Runs automatically on first container startup via docker-entrypoint-initdb.d

-- ==================== Create per-service databases ====================
CREATE DATABASE mlflow;           -- 07-ai-ml-services
CREATE DATABASE asset_management; -- 04-object-storage
CREATE DATABASE gateway;          -- 09-api-services
CREATE DATABASE vision_tre;       -- 18-vision-tre
CREATE DATABASE tkg;              -- 19-tkg-query-service
CREATE DATABASE benchmark;        -- 22-benchmark-service
CREATE DATABASE personal_db;      -- 25-personal-db-service
CREATE DATABASE guardrails;       -- 23-guardrail-service

-- ==================== asset_management ====================
\connect asset_management

CREATE TABLE IF NOT EXISTS filemeta (
  dirhash   BIGINT NOT NULL,
  name      VARCHAR(1024) NOT NULL,
  directory VARCHAR(1024),
  meta      BYTEA,
  PRIMARY KEY (dirhash, name)
);

-- ==================== gateway ====================
\connect gateway

CREATE TABLE IF NOT EXISTS api_keys (
    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hashed_key TEXT NOT NULL UNIQUE,
    owner_user_id TEXT NOT NULL,
    name TEXT,
    roles TEXT[] DEFAULT '{}',
    rate_limit_per_minute INTEGER DEFAULT 60,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hashed ON api_keys (hashed_key);
CREATE INDEX IF NOT EXISTS idx_api_keys_owner ON api_keys (owner_user_id);

CREATE TABLE IF NOT EXISTS registered_agents (
    agent_id TEXT PRIMARY KEY,
    agent_card JSONB NOT NULL,
    last_seen TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_trusted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS query_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    query TEXT,
    intent TEXT,
    latency_ms INTEGER,
    sources JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_qa_user ON query_audit (user_id);
CREATE INDEX IF NOT EXISTS idx_qa_created ON query_audit (created_at DESC);

-- ==================== tkg ====================
\connect tkg

CREATE TABLE IF NOT EXISTS temporal_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity TEXT NOT NULL,
    relation TEXT NOT NULL,
    value TEXT NOT NULL,
    time_start TIMESTAMPTZ,
    time_end TIMESTAMPTZ,
    confidence FLOAT DEFAULT 1.0,
    source_document_id TEXT,
    source_chunk_index INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tf_entity ON temporal_facts (entity);
CREATE INDEX IF NOT EXISTS idx_tf_relation ON temporal_facts (relation);
CREATE INDEX IF NOT EXISTS idx_tf_time ON temporal_facts (time_start, time_end);
CREATE INDEX IF NOT EXISTS idx_tf_source ON temporal_facts (source_document_id);

-- ==================== benchmark (22-benchmark-service) ====================
\connect benchmark

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
    mlflow_run_id        VARCHAR(255),
    -- Submitter's JWT sub, captured once at POST /runs time by Module 13's
    -- typed proxy (same pattern as asset.py's upload -> Kafka job payload).
    -- Used as the default X-Branch-ID for target_pipeline=search test cases
    -- that don't set their own input.branch_id/input.user_id.
    branch_id            VARCHAR(255)
);
CREATE INDEX IF NOT EXISTS idx_br_schema  ON benchmark_runs (schema_id);
CREATE INDEX IF NOT EXISTS idx_br_created ON benchmark_runs (created_at DESC);

-- Auto-tuning experiments (AUTOTUNE). One row per NL optimization goal;
-- per-iteration (config -> score) history is reused from benchmark_runs.
CREATE TABLE IF NOT EXISTS experiments (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    goal             TEXT,
    status           VARCHAR(50)  DEFAULT 'planning',
    plan             JSONB,
    eval_schema_id   UUID REFERENCES benchmark_schemas(id) ON DELETE SET NULL,
    budget           JSONB,
    iterations_done  INTEGER      DEFAULT 0,
    best_run_id       UUID,
    best_score        FLOAT,
    best_config       JSONB,
    holdout_schema_id   UUID,
    holdout_score       FLOAT,
    generated_schema_id UUID,
    error             TEXT,
    created_at        TIMESTAMPTZ  DEFAULT NOW(),
    completed_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_exp_status  ON experiments (status);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments (created_at DESC);

-- ==================== personal_db ====================
\connect personal_db

-- Deletion audit for the per-user ArcadeDB databases (VIE01-189).
-- Dropping a user database is irreversible and leaves nothing behind to inspect,
-- so record_counts captures what the database held immediately before the drop —
-- afterwards there is no way to answer "how much was lost".
CREATE TABLE IF NOT EXISTS personal_db_audit (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       TEXT NOT NULL,
    database      TEXT NOT NULL,
    action        TEXT NOT NULL,          -- 'delete'
    record_counts JSONB,                  -- contents at the moment of deletion
    deleted_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pda_user ON personal_db_audit (user_id);
CREATE INDEX IF NOT EXISTS idx_pda_deleted ON personal_db_audit (deleted_at DESC);

-- Consumer deduplication for `personal-index-requests` (VIE01-190).
-- event_id = sha256(asset_path + version_id + user_id). The primary key is the
-- dedup: an INSERT that conflicts means this asset version was already indexed
-- for this user, so the message is acknowledged and dropped rather than
-- reprocessed. This survives restarts, which the per-chunk Redis key (7-day TTL)
-- does not — the two layers catch different things and both stay.
-- `status` is what makes the row a claim rather than a tombstone: a message that
-- is claimed and then fails all its attempts is marked 'failed', which lets a
-- later replay re-claim it. Without that, a DLQ'd event would be permanently
-- indistinguishable from a successfully indexed one and could never be retried.
CREATE TABLE IF NOT EXISTS personal_index_events (
    event_id     CHAR(64) PRIMARY KEY,   -- sha256 hex
    user_id      TEXT NOT NULL,
    asset_path   TEXT,
    version_id   TEXT,
    source_module TEXT,                  -- 09-audio | 10-image | 11-video | 12-document
    status       TEXT NOT NULL DEFAULT 'processed',   -- processed | failed
    error        TEXT,                   -- set when the message was routed to the DLQ
    processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pie_user ON personal_index_events (user_id);
CREATE INDEX IF NOT EXISTS idx_pie_processed ON personal_index_events (processed_at DESC);
