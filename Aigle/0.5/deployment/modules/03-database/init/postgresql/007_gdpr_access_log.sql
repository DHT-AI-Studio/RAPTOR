-- GD-7 / VIE01-252: Data Access Audit Log (Modules 13 + 33)
--
-- 002_gdpr.sql defines gdpr_access_log for fresh volumes; this file gives an
-- existing deployment the same shape. GD-3 created the table as an empty
-- placeholder (id BIGSERIAL, user_id, action, timestamp) that nothing ever
-- wrote to, so it's replaced rather than altered -- but only while it is
-- still that empty placeholder: a placeholder holding rows aborts here
-- instead of losing them. Idempotent. Apply manually:
-- `psql -h $POSTGRES_HOST -U $POSTGRES_USER -f 007_gdpr_access_log.sql`

\connect privacy_compliance

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'gdpr_access_log')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'gdpr_access_log' AND column_name = 'accessor_id') THEN
        IF EXISTS (SELECT 1 FROM gdpr_access_log) THEN
            RAISE EXCEPTION 'gdpr_access_log placeholder is not empty -- migrate its rows by hand first';
        END IF;
        DROP TABLE gdpr_access_log;
    END IF;
END
$$;

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
