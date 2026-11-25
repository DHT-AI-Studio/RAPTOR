-- GD-1 / VIE01-246: Cross-Store Right-to-Erasure Orchestrator (Module 33)
--
-- Like every other service database in 001_init.sql, this only runs
-- automatically on a brand-new Postgres data volume (docker-entrypoint-initdb.d
-- scripts never re-run against an already-initialized one) -- on an existing
-- deployment, apply it manually: `psql -h $POSTGRES_HOST -U $POSTGRES_USER -f 002_gdpr.sql`

CREATE DATABASE privacy_compliance; -- 33-privacy-compliance-service

\connect privacy_compliance

CREATE TABLE IF NOT EXISTS gdpr_erasure_jobs (
    job_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        VARCHAR(255) NOT NULL,
    requested_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    requested_by   VARCHAR(255) NOT NULL,
    status         VARCHAR(50)  NOT NULL DEFAULT 'pending',
    store_statuses JSONB        NOT NULL DEFAULT '{}',
    completed_at   TIMESTAMPTZ,
    notes          TEXT
);

CREATE INDEX IF NOT EXISTS idx_gdpr_erasure_jobs_user_id      ON gdpr_erasure_jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_gdpr_erasure_jobs_status       ON gdpr_erasure_jobs (status);
CREATE INDEX IF NOT EXISTS idx_gdpr_erasure_jobs_requested_at ON gdpr_erasure_jobs (requested_at DESC);

-- GD-5 / VIE01-250: consent history. Append-only legal record of every
-- consent change (GDPR Art. 7(1): the controller must be able to demonstrate
-- consent) -- kept 7 years per GD-3's seeded gdpr_consent_history policy
-- (auto_delete=false). Existing deployments get the same DDL from
-- 006_gdpr_consent.sql, since this file only runs on a fresh volume.
CREATE TABLE IF NOT EXISTS gdpr_consent_history (
    id             BIGSERIAL PRIMARY KEY,
    user_id        VARCHAR(255) NOT NULL,
    purpose        VARCHAR(64)  NOT NULL,
    action         VARCHAR(16)  NOT NULL CHECK (action IN ('grant', 'withdraw')),
    timestamp      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- SHA-256 of the request's X-Forwarded-For -- never the raw IP.
    ip_hash        CHAR(64),
    policy_version VARCHAR(32),
    -- user | registration_default | first_read_default
    source         VARCHAR(32)  NOT NULL DEFAULT 'user'
);
CREATE INDEX IF NOT EXISTS idx_gdpr_consent_history_user_purpose
    ON gdpr_consent_history (user_id, purpose, timestamp DESC);

-- Immutable: rows can only ever be appended.
CREATE OR REPLACE FUNCTION gdpr_consent_history_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'gdpr_consent_history is append-only (% not allowed)', TG_OP;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE TRIGGER trg_gdpr_consent_history_no_update_delete
    BEFORE UPDATE OR DELETE ON gdpr_consent_history
    FOR EACH ROW EXECUTE FUNCTION gdpr_consent_history_append_only();
CREATE OR REPLACE TRIGGER trg_gdpr_consent_history_no_truncate
    BEFORE TRUNCATE ON gdpr_consent_history
    FOR EACH STATEMENT EXECUTE FUNCTION gdpr_consent_history_append_only();

-- GD-7 / VIE01-252: data access audit log (GDPR Art. 30). One row per
-- request to a personal-data route (Module 13's audit middleware), written
-- by Module 33. Retention: GD-3's gdpr_access_log policy (365 days,
-- auto_delete) keyed on `timestamp`. Replaces GD-3's placeholder table of
-- the same name (004_gdpr_retention.sql); existing deployments get this
-- shape from 007_gdpr_access_log.sql.
CREATE TABLE IF NOT EXISTS gdpr_access_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         VARCHAR(255),                -- whose data (NULL: no single subject, e.g. an admin listing)
    accessor_id     VARCHAR(255) NOT NULL,       -- JWT sub of the requester
    endpoint        VARCHAR(512) NOT NULL,
    method          VARCHAR(16)  NOT NULL,
    data_categories VARCHAR(64)[] NOT NULL DEFAULT '{}',
    purpose         VARCHAR(64),
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    ip_hash         VARCHAR(64)                  -- SHA-256 of X-Forwarded-For, never the raw IP
);
CREATE INDEX IF NOT EXISTS idx_gdpr_access_log_timestamp      ON gdpr_access_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_gdpr_access_log_user_timestamp ON gdpr_access_log (user_id, timestamp DESC);

-- PRIV-X3 / VIE01-357: restriction of processing (GDPR Art. 18) -- the
-- ticket's (id, user_id, restricted_at, lifted_at, reason). One row per
-- restriction; lifted_at stays NULL while it is in force, and the partial
-- unique index allows at most one such open row per user. Kept as a legal
-- record, like gdpr_consent_history.
CREATE TABLE IF NOT EXISTS gdpr_restriction_log (
    id            BIGSERIAL PRIMARY KEY,
    user_id       VARCHAR(255) NOT NULL,
    restricted_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    lifted_at     TIMESTAMPTZ,
    reason        TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gdpr_restriction_log_open
    ON gdpr_restriction_log (user_id) WHERE lifted_at IS NULL;

-- PRIV-X3 / VIE01-357: objection to processing (GDPR Art. 21). One row per
-- objection.
CREATE TABLE IF NOT EXISTS gdpr_objection_log (
    id                 BIGSERIAL PRIMARY KEY,
    user_id            VARCHAR(255) NOT NULL,
    processing_purpose VARCHAR(64)  NOT NULL,
    objected_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_gdpr_objection_log_user
    ON gdpr_objection_log (user_id, objected_at DESC);

-- PRIV-X4 / VIE01-358: legal holds (GDPR Art. 17(3)) -- the ticket's (id,
-- user_id, reason, authority, placed_at, hold_until, released_at). While a
-- hold is open (released_at NULL, hold_until not reached) an erasure request
-- is deferred; the daily expiry job sets released_at and runs it. At most one
-- open hold per user. Kept as a legal record, like gdpr_restriction_log.
CREATE TABLE IF NOT EXISTS gdpr_legal_holds (
    id          BIGSERIAL PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    reason      VARCHAR(255) NOT NULL,
    authority   VARCHAR(255) NOT NULL,
    placed_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    hold_until  DATE         NOT NULL,
    released_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gdpr_legal_holds_open
    ON gdpr_legal_holds (user_id) WHERE released_at IS NULL;
