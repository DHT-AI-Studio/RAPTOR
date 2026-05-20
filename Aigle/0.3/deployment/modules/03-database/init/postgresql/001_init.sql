-- Raptor PostgreSQL initialization
-- Runs automatically on first container startup via docker-entrypoint-initdb.d

-- ==================== Create per-service databases ====================
CREATE DATABASE mlflow;           -- 07-ai-ml-services
CREATE DATABASE asset_management; -- 04-object-storage
CREATE DATABASE gateway;          -- 09-api-services
CREATE DATABASE vision_tre;       -- 18-vision-tre
CREATE DATABASE tkg;              -- 19-tkg-query-service

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
