-- PRIV-X3 / VIE01-357: Rectification, Restriction and Objection (Module 33)
--
-- 002_gdpr.sql defines these tables for fresh volumes; this file applies the
-- same, idempotent DDL to an existing deployment. Apply manually:
-- `psql -h $POSTGRES_HOST -U $POSTGRES_USER -f 008_gdpr_rights.sql`

\connect privacy_compliance

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
