-- GD-4 / VIE01-249: Data Subject Access Request (Module 33)
--
-- Reuses gdpr_erasure_jobs (type = 'dsar') for the async path only -- an
-- account with more than GD_DSAR_ASYNC_THRESHOLD indexed items, or one
-- whose synchronous assembly overruns its budget. `result` holds the
-- finished DSAR report; NULL for erasure/export jobs, and cleared again
-- past GD_DSAR_RESULT_TTL_HOURS (reusing GD-2's expires_at column).
--
-- Like 002-004, this only runs automatically on a brand-new Postgres data
-- volume -- on an existing deployment, apply manually:
-- `psql -h $POSTGRES_HOST -U $POSTGRES_USER -d privacy_compliance -f 005_gdpr_dsar.sql`

\connect privacy_compliance

ALTER TABLE gdpr_erasure_jobs ADD COLUMN IF NOT EXISTS result JSONB;
