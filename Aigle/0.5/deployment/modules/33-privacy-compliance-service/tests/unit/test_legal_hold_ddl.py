"""PRIV-X4 / VIE01-358: the gdpr_legal_holds DDL Module 33 runs at start-up
is the same as 03-database's init SQL -- 010_gdpr_legal_hold.sql for existing
deployments and 002_gdpr.sql for fresh volumes.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_legal_hold_ddl.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from app.services.legal_hold_state import LEGAL_HOLDS_DDL

INIT_SQL = Path(__file__).resolve().parents[3] / "03-database" / "init" / "postgresql"


def _statements(sql: str) -> list[str]:
    """The legal-hold statements, comments and psql meta-commands dropped,
    whitespace collapsed."""
    sql = re.sub(r"--[^\n]*|^\\[^\n]*", "", sql, flags=re.MULTILINE)
    found = [" ".join(s.split()) for s in sql.split(";")]
    return [s for s in found if "gdpr_legal_holds" in s]


@pytest.mark.parametrize("name", ["010_gdpr_legal_hold.sql", "002_gdpr.sql"])
def test_startup_ddl_matches_the_init_sql(name):
    expected = _statements(LEGAL_HOLDS_DDL)
    assert len(expected) == 2
    assert _statements((INIT_SQL / name).read_text(encoding="utf-8")) == expected
