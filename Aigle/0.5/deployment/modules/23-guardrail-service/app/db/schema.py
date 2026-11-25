"""DDL for Policy storage tables."""
from __future__ import annotations

from raptor_audit import AUDIT_EVENTS_DDL

_POLICY_AND_VIOLATIONS_DDL = """
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
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_guardrail_violations_created_at ON guardrail_violations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_guardrail_violations_module     ON guardrail_violations (module);
CREATE INDEX IF NOT EXISTS idx_guardrail_violations_category   ON guardrail_violations (category);

-- CF-4 (VIE01-234): each violation row links to its audit_events row (GDPR
-- erasure traceability); the foreign key is added below, once audit_events
-- exists.
ALTER TABLE guardrail_violations ADD COLUMN IF NOT EXISTS event_id UUID;

-- GRD-X4 (VIE01-354): no content column. Before CF-4 it held the raw screened
-- text, after CF-4 its SHA-256 -- which the linked audit_events.content_hash
-- already keeps. Dropping it removes both the duplicate and any raw text left
-- from before CF-4.
ALTER TABLE guardrail_violations DROP COLUMN IF EXISTS content;
CREATE INDEX IF NOT EXISTS idx_guardrail_violations_event_id   ON guardrail_violations (event_id);
"""

# CM-5 (VIE01-408): automated flagging system -- a mutable, admin-reviewable
# record per flagged content item, distinct from guardrail_violations (a
# write-once audit log of every check call, with no notion of "this specific
# asset/moment/chunk"). Module 23 is a stateless classifier with no content
# identity of its own, so the caller (module 10/11/12, whichever one holds
# the asset/moment/chunk id) supplies it when posting the flag.
_MODERATION_FLAGS_DDL = """
CREATE TABLE IF NOT EXISTS moderation_flags (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- The tenant this content belongs to -- Module 25's ArcadeDB is one database
    -- per branch, so a consumer reacting to this row (search exclusion) can't
    -- find the right database without it; there is no cross-tenant lookup.
    branch_id      TEXT NOT NULL,
    -- asset_path + version_id identify the real, queryable Module 04 asset this content
    -- belongs to -- always required, on every flag, regardless of content type. Without
    -- both, an admin reviewing a flag has only an opaque moment_id (a one-way hash) or
    -- chunk_id (a bare UUID) to look at -- neither means anything on its own, and
    -- neither can be turned back into something Module 04 can serve (every one of its
    -- endpoints needs {asset_path}/{version_id} together; there is no version_id-only
    -- lookup). moment_id/chunk_id below are an optional, finer-grained sub-reference
    -- WITHIN this asset (which moment, which chunk) -- not an alternative to it.
    version_id     TEXT NOT NULL,
    asset_path     TEXT NOT NULL,
    -- At most one of these may be set (never both) -- a single flag is about one moment
    -- OR one chunk of asset_path/version_id above, never both granularities at once.
    -- Neither set at all means the flag is about the whole asset (e.g. an image).
    moment_id      TEXT,
    chunk_id       TEXT,
    category       TEXT NOT NULL,
    score          DOUBLE PRECISION,
    model          TEXT NOT NULL,
    policy_version TEXT,
    status         TEXT NOT NULL DEFAULT 'flagged',
    event_id       UUID,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT moderation_flags_status_chk
        CHECK (status IN ('flagged', 'cleared', 'removed')),
    CONSTRAINT moderation_flags_sub_ref_chk
        CHECK (NOT (moment_id IS NOT NULL AND chunk_id IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_moderation_flags_branch_id ON moderation_flags (branch_id);
CREATE INDEX IF NOT EXISTS idx_moderation_flags_version_id ON moderation_flags (version_id);
CREATE INDEX IF NOT EXISTS idx_moderation_flags_asset_path ON moderation_flags (asset_path);
CREATE INDEX IF NOT EXISTS idx_moderation_flags_moment_id ON moderation_flags (moment_id) WHERE moment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_moderation_flags_chunk_id  ON moderation_flags (chunk_id)  WHERE chunk_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_moderation_flags_status    ON moderation_flags (status);
CREATE INDEX IF NOT EXISTS idx_moderation_flags_created_at ON moderation_flags (created_at DESC);
"""

_MODERATION_FLAGS_EVENT_ID_FK_DDL = """
-- Same pattern as guardrail_violations' own fk_guardrail_violations_event_id:
-- ON DELETE SET NULL keeps audit_events' retention independent of this table.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'fk_moderation_flags_event_id'
                     AND conrelid = 'moderation_flags'::regclass) THEN
        UPDATE moderation_flags f SET event_id = NULL
         WHERE event_id IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM audit_events e WHERE e.event_id = f.event_id);
        ALTER TABLE moderation_flags
            ADD CONSTRAINT fk_moderation_flags_event_id
            FOREIGN KEY (event_id) REFERENCES audit_events (event_id) ON DELETE SET NULL;
    END IF;
END
$$;
"""

_EVENT_ID_FK_DDL = """
-- GRD-X4 (VIE01-354): guardrail_violations.event_id is a foreign key to
-- audit_events. ON DELETE SET NULL keeps the two tables' retention independent:
-- purging an audit event leaves its violation row, unlinked; deleting a
-- violation row never touches the event. Added once -- an event_id that points
-- to no event (none should, CF-4 writes both in one transaction) is cleared
-- first, so the constraint can always be created.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'fk_guardrail_violations_event_id'
                     AND conrelid = 'guardrail_violations'::regclass) THEN
        UPDATE guardrail_violations v SET event_id = NULL
         WHERE event_id IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM audit_events e WHERE e.event_id = v.event_id);
        ALTER TABLE guardrail_violations
            ADD CONSTRAINT fk_guardrail_violations_event_id
            FOREIGN KEY (event_id) REFERENCES audit_events (event_id) ON DELETE SET NULL;
    END IF;
END
$$;
"""

# All three parts are fixed constants (this module's own DDL plus raptor_audit's own
# constant) -- no caller input ever reaches this string, so the concatenation bandit's
# B608 flags here is not a SQL-injection vector. CI-846.
DDL = (_POLICY_AND_VIOLATIONS_DDL + AUDIT_EVENTS_DDL + _EVENT_ID_FK_DDL
       + _MODERATION_FLAGS_DDL + _MODERATION_FLAGS_EVENT_ID_FK_DDL)  # nosec B608
