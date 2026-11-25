-- GD-5 / VIE01-250: Consent Management (Module 33 + Module 22)
--
-- The ticket puts gdpr_consent_history in 002_gdpr.sql (fresh volumes);
-- this file applies the same, idempotent DDL to an existing deployment,
-- plus the benchmark_schemas.owner_id column (Module 22 also adds that one
-- itself at startup). Apply manually:
-- `psql -h $POSTGRES_HOST -U $POSTGRES_USER -f 006_gdpr_consent.sql`

\connect privacy_compliance
-- GD-5 / VIE01-250: consent history. Append-only legal record of every
-- consent change (GDPR Art. 7(1): the controller must be able to demonstrate
-- consent) -- kept 7 years per GD-3's seeded gdpr_consent_history policy
-- (auto_delete=false). Same DDL as 002_gdpr.sql's GD-5 block.
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

\connect benchmark

ALTER TABLE benchmark_schemas ADD COLUMN IF NOT EXISTS owner_id VARCHAR(255);
CREATE INDEX IF NOT EXISTS idx_bs_owner ON benchmark_schemas (owner_id);
