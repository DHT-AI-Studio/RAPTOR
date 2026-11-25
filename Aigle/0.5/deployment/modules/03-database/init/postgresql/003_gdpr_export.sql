-- GD-2 / VIE01-247: Cross-Store Data Portability Export (Module 33)
--
-- Reuses gdpr_erasure_jobs (per the ticket's explicit instruction) for both
-- erasure and export jobs, distinguished by `type`. Like 002_gdpr.sql, this
-- only runs automatically on a brand-new Postgres data volume -- on an
-- existing deployment, apply manually:
-- `psql -h $POSTGRES_HOST -U $POSTGRES_USER -d privacy_compliance -f 003_gdpr_export.sql`

\connect privacy_compliance

ALTER TABLE gdpr_erasure_jobs ADD COLUMN IF NOT EXISTS type VARCHAR(20) NOT NULL DEFAULT 'erasure';
-- Export-specific result fields. NULL for erasure jobs.
ALTER TABLE gdpr_erasure_jobs ADD COLUMN IF NOT EXISTS download_urls JSONB;
ALTER TABLE gdpr_erasure_jobs ADD COLUMN IF NOT EXISTS size_bytes BIGINT;
ALTER TABLE gdpr_erasure_jobs ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_gdpr_erasure_jobs_type ON gdpr_erasure_jobs (type);
