-- PRIV-X4 / VIE01-358: Legal Holds and Erasure Exceptions (Module 33)
--
-- 002_gdpr.sql defines this table for fresh volumes; this file applies the
-- same, idempotent DDL to an existing deployment. Apply manually:
-- `psql -h $POSTGRES_HOST -U $POSTGRES_USER -f 010_gdpr_legal_hold.sql`
-- Module 33 also runs this DDL at start-up (app/services/legal_hold_state.py),
-- so an erasure never fails on a missing table.

\connect privacy_compliance

-- One row per legal hold (GDPR Art. 17(3)). While a hold is open (released_at
-- NULL, hold_until not reached) an erasure request is deferred; the daily
-- expiry job sets released_at and runs it. At most one open hold per user.
-- Kept as a legal record, like gdpr_restriction_log.
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
