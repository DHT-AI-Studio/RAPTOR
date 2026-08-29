"""DDL for Policy storage tables."""
from __future__ import annotations

DDL = """
DROP TABLE IF EXISTS policy_domains;
DROP TABLE IF EXISTS policy_rules;

CREATE TABLE IF NOT EXISTS guardrail_policies (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    raw_content TEXT NOT NULL DEFAULT '',
    is_active   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migration for a guardrail_policies table left over from the
-- prior 'definition JSONB' shape — safe to run on every startup. On a fresh
-- install these are all no-ops (the CREATE TABLE above already has the
-- final shape).
ALTER TABLE guardrail_policies ADD COLUMN IF NOT EXISTS raw_content TEXT NOT NULL DEFAULT '';
ALTER TABLE guardrail_policies DROP COLUMN IF EXISTS definition;
ALTER TABLE guardrail_policies ALTER COLUMN raw_content DROP DEFAULT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_guardrail_policies_one_active
    ON guardrail_policies (is_active) WHERE is_active;

-- Per-guard-model policy prompt overrides for /policy/check/llm/*. raw_content
-- remains the "original"/shared content (used by every other check family
-- unchanged, and as the fallback here when a model-specific column is NULL).
ALTER TABLE guardrail_policies ADD COLUMN IF NOT EXISTS content_llama_guard TEXT;
ALTER TABLE guardrail_policies ADD COLUMN IF NOT EXISTS content_granite_guardian TEXT;
ALTER TABLE guardrail_policies ADD COLUMN IF NOT EXISTS content_gpt_oss_safeguard TEXT;

-- GB-6: audit trail of every block/redact/flag decision the GB-4 checker makes.
-- policy_id is nullable + ON DELETE SET NULL (not CASCADE) so deleting a policy
-- never deletes the audit history it produced.
CREATE TABLE IF NOT EXISTS guardrail_violations (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id    UUID REFERENCES guardrail_policies(id) ON DELETE SET NULL,
    module       TEXT,
    direction    TEXT NOT NULL,
    category     TEXT,
    action_taken TEXT NOT NULL,
    request_id   TEXT,
    content      TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_guardrail_violations_created_at ON guardrail_violations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_guardrail_violations_module     ON guardrail_violations (module);
CREATE INDEX IF NOT EXISTS idx_guardrail_violations_category   ON guardrail_violations (category);
"""
