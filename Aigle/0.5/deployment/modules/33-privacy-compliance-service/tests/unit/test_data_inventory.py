"""PRIV-X1 / VIE01-355: docs/_internal/gdpr-data-inventory.yaml -- the authoritative
scope for GDPR erasure -- is complete and well-formed.

Also a regression guard: a PostgreSQL table created with a user/owner
column in 03-database's init SQL must be listed in the inventory, so a new
table holding personal data can't land without one.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_data_inventory.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml  # installed with uvicorn[standard]

REPO = Path(__file__).resolve().parents[5]
INVENTORY = yaml.safe_load((REPO / "docs" / "_internal" / "gdpr-data-inventory.yaml").read_text(encoding="utf-8"))
STORES = INVENTORY["stores"]
BY_ID = {s["id"]: s for s in STORES}

REQUIRED_FIELDS = ("store", "module", "data_categories", "owner_field", "isolation", "retention_default",
                   "erasure_method", "erasure_story", "backup_covered", "v04_status")
ISOLATION = {"per_user_store", "per_user_rows", "shared_unkeyed", "system"}
V04_STATUS = {"implemented", "partially_implemented", "not_implemented", "structural_only", "legacy"}
ERASURE_STORIES = {"GD-1", "GD-3", "GD-5", "PRIV-X5", "open", "none_required"}


def test_ids_are_unique():
    ids = [s["id"] for s in STORES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("entry", STORES, ids=lambda s: s["id"])
def test_entry_has_every_required_field(entry):
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    assert not missing, f"{entry['id']} missing {missing}"
    assert set(entry["data_categories"]) <= set(INVENTORY["category_vocabulary"])
    assert entry["isolation"] in ISOLATION
    assert entry["v04_status"] in V04_STATUS
    assert entry["erasure_story"] in ERASURE_STORIES
    assert isinstance(entry["backup_covered"], bool)


@pytest.mark.parametrize("store, prefix", [
    ("Keycloak", "keycloak"), ("SeaweedFS", "seaweedfs_"), ("LakeFS", "lakefs_"), ("ArcadeDB", "arcadedb_"),
    ("MemVID", "memvid"), ("PostgreSQL", "pg_"), ("Redis", "redis_"), ("Loki", "loki_"), ("Kafka", "kafka_"),
    ("Prometheus", "prometheus"), ("MLflow", "mlflow_"), ("NFS/temp files", "nfs_"),
])
def test_ticket_minimum_stores_are_covered(store, prefix):
    assert any(i.startswith(prefix) for i in BY_ID), f"no inventory entry for {store}"


def test_ticket_named_additions_are_present():
    assert BY_ID["module34_voice_recordings"]["data_categories"] == ["biometric"]
    assert BY_ID["module35_speaker_embeddings"]["data_categories"] == ["biometric"]
    assert BY_ID["arcadedb_faces"]["data_categories"] == ["biometric"]
    assert "72h" in BY_ID["seaweedfs_gdpr_exports"]["retention_default"]


def test_every_uncovered_backup_is_a_priv_x5_open_item():
    uncovered = {s["id"] for s in STORES if s["backup_covered"] is False}
    listed = {item["store_id"] for item in INVENTORY["priv_x5_open_items"]}
    assert uncovered == listed


def test_discrepancies_point_at_real_entries():
    assert {d["store_id"] for d in INVENTORY["ticket_discrepancies"]} <= set(BY_ID)


_OWNER_COLUMN = re.compile(r"^\s*(user_id|branch_id|owner_id|owner_user_id|accessor_id)\s", re.MULTILINE)


def _tables_with_owner_columns():
    sql_dir = REPO / "deployment" / "modules" / "03-database" / "init" / "postgresql"
    for sql_file in sorted(sql_dir.glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8")
        for match in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\);", sql, re.DOTALL):
            if _OWNER_COLUMN.search(match.group(2)):
                yield sql_file.name, match.group(1)


def test_every_owned_postgres_table_is_inventoried():
    listed = " ".join(s["store"] for s in STORES)
    missing = [f"{table} ({source})" for source, table in _tables_with_owner_columns()
               if not re.search(rf"\b{table}\b", listed)]
    assert not missing, f"tables with a user/owner column missing from the inventory: {missing}"


def test_owned_table_scan_finds_known_tables():
    """Guard the guard: the scan above really sees the init SQL."""
    found = {table for _, table in _tables_with_owner_columns()}
    assert {"benchmark_runs", "gdpr_erasure_jobs"} <= found
