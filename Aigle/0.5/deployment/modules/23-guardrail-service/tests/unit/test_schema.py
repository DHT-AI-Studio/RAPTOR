"""Module 23's startup DDL (app/db/schema.py): the GRD-X4 / VIE01-354 foreign key
from guardrail_violations.event_id to audit_events. Executed against a real
PostgreSQL in the PR's verification; here, its shape."""
from app.db.schema import DDL


def test_event_id_is_a_foreign_key_to_audit_events_with_on_delete_set_null():
    assert "FOREIGN KEY (event_id) REFERENCES audit_events (event_id) ON DELETE SET NULL" in DDL


def test_foreign_key_comes_after_both_tables_exist():
    fk = DDL.index("ADD CONSTRAINT fk_guardrail_violations_event_id")
    assert DDL.index("CREATE TABLE IF NOT EXISTS audit_events") < fk
    assert DDL.index("ADD COLUMN IF NOT EXISTS event_id") < fk


def test_foreign_key_is_added_only_once():
    """Startup runs the DDL every time: the constraint is guarded by a pg_constraint check."""
    assert "WHERE conname = 'fk_guardrail_violations_event_id'" in DDL


def test_guardrail_violations_has_no_content_column():
    """GRD-X4: the content column (raw text before CF-4, a duplicate hash after) is gone --
    new tables never get it and existing ones drop it."""
    table = DDL[DDL.index("CREATE TABLE IF NOT EXISTS guardrail_violations"):]
    table = table[:table.index(");")]
    assert "content" not in table
    assert "ALTER TABLE guardrail_violations DROP COLUMN IF EXISTS content;" in DDL
