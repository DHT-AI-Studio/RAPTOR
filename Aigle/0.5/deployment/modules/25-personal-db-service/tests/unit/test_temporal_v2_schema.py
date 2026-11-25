"""TemporalFact v2 schema (TMP-8): what schema_init declares and what the normaliser
writes for every fact. No ArcadeDB -- tests/integration/test_temporal_v2_schema.py runs
the same statements on a real one.
"""
from __future__ import annotations

import re

from app.core.config import Settings, settings
from app.services import temporal_backfill
from app.services.schema_init import _ddl
from app.services.temporal_normalize import NORMALIZER_VERSION, normalize_fact_dates

V2_PROPERTIES = {
    "s_state": "STRING", "e_state": "STRING", "s_lo": "LONG", "s_hi": "LONG", "e_lo": "LONG", "e_hi": "LONG",
    "timeline_id": "STRING", "axis": "STRING", "time_unit": "STRING", "timezone": "STRING",
    "normalizer_version": "STRING", "extent": "STRING", "precision": "STRING", "anchor": "STRING",
    "date_text": "STRING", "source_sentence": "STRING", "evidence_offsets": "STRING",
    "confirmed_active_at": "LONG", "derived": "BOOLEAN", "support_ids": "LIST", "network_version": "STRING",
    "media_start_ms": "LONG", "media_end_ms": "LONG",
}
DDL = _ddl(settings.vector_dim, settings.vector_similarity)


def test_every_v2_property_is_declared_with_its_type():
    declared = dict(m.groups() for stmt in DDL
                    if (m := re.match(r"CREATE PROPERTY TemporalFact\.(\w+) IF NOT EXISTS (\w+)$", stmt)))
    assert {name: declared.get(name) for name in V2_PROPERTIES} == V2_PROPERTIES


def test_v1_projections_are_still_declared():
    assert "CREATE PROPERTY TemporalFact.time_start IF NOT EXISTS STRING" in DDL
    assert "CREATE PROPERTY TemporalFact.time_end IF NOT EXISTS STRING" in DDL


def test_the_three_v2_indexes_are_declared():
    for columns in ("(timeline_id, s_lo)", "(timeline_id, e_hi)", "(entity_id, s_lo)"):
        assert f"CREATE INDEX IF NOT EXISTS ON TemporalFact {columns} NOTUNIQUE" in DDL


def test_every_statement_is_safe_to_apply_again():
    assert all("IF NOT EXISTS" in stmt for stmt in DDL)


def test_normaliser_states_the_axis_and_unit_of_its_ranges():
    d = normalize_fact_dates("2011", None, "Asia/Taipei")
    assert (d["timeline_id"], d["axis"], d["time_unit"]) == ("world", "world", "ms")
    assert NORMALIZER_VERSION.startswith("tmp8-")       # a new version: the backfill redoes older rows


def test_migration_writes_only_what_a_legacy_row_can_establish():
    written = set(normalize_fact_dates("2011", None, "Asia/Taipei"))
    assert written == set(temporal_backfill._DATE_PROPS)
    assert not written & {"extent", "anchor", "date_text", "source_sentence", "evidence_offsets",
                          "confirmed_active_at", "derived", "support_ids", "network_version",
                          "media_start_ms", "media_end_ms"}


def test_v2_flag_is_off_by_default_and_read_from_the_environment(monkeypatch):
    assert Settings.model_fields["temporal_v2"].default is False
    monkeypatch.setenv("PD_TEMPORAL_V2", "true")
    assert Settings().temporal_v2 is True
