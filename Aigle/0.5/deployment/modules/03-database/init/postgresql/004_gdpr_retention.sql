-- GD-3 / VIE01-248: Data Retention Policy Engine (Module 33)
--
-- Like 002_gdpr.sql / 003_gdpr_export.sql, this only runs automatically on a
-- brand-new Postgres data volume -- on an existing deployment, apply
-- manually:
-- `psql -h $POSTGRES_HOST -U $POSTGRES_USER -d privacy_compliance -f 004_gdpr_retention.sql`

\connect privacy_compliance

CREATE TABLE IF NOT EXISTS gdpr_retention_policies (
    data_category  VARCHAR(64) PRIMARY KEY,
    retention_days INT NOT NULL,
    auto_delete    BOOLEAN NOT NULL,
    legal_basis    VARCHAR(255),
    notes          TEXT,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by     VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS gdpr_retention_log (
    id            BIGSERIAL PRIMARY KEY,
    run_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    data_category VARCHAR(64) NOT NULL,
    rows_deleted  INT NOT NULL,
    duration_ms   INT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gdpr_retention_log_category_run_at
    ON gdpr_retention_log (data_category, run_at DESC);

-- gdpr_access_log itself is defined in 002_gdpr.sql (GD-7 / VIE01-252 gave
-- GD-3's placeholder table its real shape); only its retention policy row
-- is seeded below.

-- Default retention policies, seeded on first startup (idempotent: a row
-- whose data_category already exists -- e.g. an operator already changed it
-- via PUT /gdpr/retention/policy -- is left untouched).
INSERT INTO gdpr_retention_policies
    (data_category, retention_days, auto_delete, legal_basis, notes, updated_by)
VALUES
    ('benchmark_runs', 90, TRUE, 'GDPR Art. 5.1.e', 'Benchmark runs', 'system'),
    ('guardrail_violations', 90, TRUE, 'GDPR Art. 5.1.e', 'Guardrail violation records', 'system'),
    -- CM-5 / VIE01-408: distinct from guardrail_violations above -- this
    -- table has a branch_id column and is also erasable on demand (GD-1).
    -- Split by status: this row only covers 'cleared' (reviewed, false
    -- positive -- no ongoing value). 'flagged'/'removed' share the longer
    -- moderation_flags_actionable row below.
    ('moderation_flags', 90, TRUE, 'GDPR Art. 5.1.e', 'Automated content moderation flags -- cleared only', 'system'),
    -- 'flagged' (still pending review) and 'removed' (confirmed violation)
    -- kept longer than a cleared/false-positive flag: dispute defense and
    -- repeat-offender detection need the record of what was actually
    -- enforced, same "Legal record" precedent as gdpr_erasure_jobs below.
    ('moderation_flags_actionable', 1095, TRUE, 'Legal record', 'Automated content moderation flags -- flagged (pending) and removed (confirmed violation)', 'system'),
    ('memvid_sessions', 365, TRUE, 'GDPR Art. 5.1.e', 'Conversation memory sessions (MemVID)', 'system'),
    ('gdpr_consent_history', 2555, FALSE, 'Legal record', 'GDPR consent history (7 years) -- no consent_history table exists in this codebase yet; seeded for completeness, never enforced (auto_delete=false)', 'system'),
    ('gdpr_access_log', 365, TRUE, 'GDPR Art. 5.1.e', 'Data access audit log', 'system'),
    ('gdpr_erasure_jobs', 1095, FALSE, 'Legal record', 'Erasure job records (3 years)', 'system'),
    -- GD-10 / VIE01-416: neither row is purged by Module 33's retention engine
    -- (not in ENFORCEABLE_CATEGORIES); each note says who enforces it.
    ('voice_recordings', 90, TRUE, 'GDPR Art. 5.1.e', 'Module 34 voice session recordings (WAV in SeaweedFS + realtime_audio.rt_audio_recordings) -- Module 34 deletes them itself after RT_RECORDING_RETENTION_DAYS (default 90); changing this row does not change Module 34', 'system'),
    ('loki_logs', 30, TRUE, 'GDPR Art. 5.1.e', 'Module 14 Loki logs -- Loki''s compactor deletes them itself after limits_config.retention_period (720h = 30 days, 14-monitoring/config/loki/loki-config.yaml); changing this row does not change Loki', 'system')
ON CONFLICT (data_category) DO NOTHING;
