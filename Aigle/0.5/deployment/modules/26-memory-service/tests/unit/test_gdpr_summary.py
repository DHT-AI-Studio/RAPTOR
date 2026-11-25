"""
GD-4 / VIE01-249: GET /internal/gdpr/user/{user_id}/summary -- the
count-only memory summary Module 33's DSAR assembler calls.

Run from the 26-memory-service/ directory:
    python -m pytest tests/unit/test_gdpr_summary.py -v
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MEM_REDIS_HOST", "localhost")
os.environ.setdefault("MEM_STORAGE_ROOT", "/tmp/gd4_test")
os.environ.setdefault("MEM_GDPR_INTERNAL_TOKEN", "test-internal-token")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

import pytest
from core.config import settings
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers import gdpr as gdpr_router
from routers.management import ManagementService, get_management_service

# Same import-order fragility test_api.py documents for storage_root: this
# module's own os.environ.setdefault above only takes effect if it's the
# first to import core.config in the whole pytest run. Overwrite the
# singleton attribute directly so this module's token is authoritative
# regardless of import order.
settings.gdpr_internal_token = "test-internal-token"


def _svc(root: Path) -> ManagementService:
    svc = ManagementService.__new__(ManagementService)
    svc._root = root
    return svc


def _write_store(user_dir: Path, stem: str, frames: dict) -> None:
    """A .mv2 placeholder + its .meta.json sidecar -- the summary only ever
    reads the sidecar, never the .mv2 itself (that's the export's cost)."""
    (user_dir / f"{stem}.mv2").write_bytes(b"")
    (user_dir / f"{stem}.meta.json").write_text(json.dumps(frames))


@pytest.mark.asyncio
async def test_summary_for_user_with_no_memory(tmp_path):
    assert await _svc(tmp_path).get_gdpr_summary("ghost") == {
        "user_id": "ghost", "sessions": 0, "long_term_facts": 0, "oldest_session": None,
    }


@pytest.mark.asyncio
async def test_summary_counts_sessions_facts_and_oldest_turn(tmp_path):
    user_dir = tmp_path / "user_u1"
    user_dir.mkdir()
    # Session "a" spans two shards -- one session, not two.
    _write_store(user_dir, "session_a_0", {"1": {"frame_type": "turn", "timestamp": 1_700_000_500.0}})
    _write_store(user_dir, "session_a_1", {"2": {"frame_type": "turn", "timestamp": 1_700_000_900.0}})
    _write_store(user_dir, "session_b_0", {
        "1": {"frame_type": "turn", "timestamp": 1_700_000_100.0},  # the oldest turn anywhere
        "2": {"frame_type": "summary", "timestamp": 1_700_000_200.0},
    })
    # 4 facts (fact/preference/entity) across 2 shards; the 2 "conversation"
    # frames are long-term memory but not facts, so they mustn't count.
    _write_store(user_dir, "long_term_0", {
        "1": {"frame_type": "fact"}, "2": {"frame_type": "preference"},
        "3": {"frame_type": "conversation"}, "4": {"frame_type": "entity"},
    })
    _write_store(user_dir, "long_term_1", {"5": {"frame_type": "fact"}, "6": {"frame_type": "conversation"}})

    summary = await _svc(tmp_path).get_gdpr_summary("u1")
    assert summary["sessions"] == 2
    assert summary["long_term_facts"] == 4
    assert summary["oldest_session"] == datetime.fromtimestamp(1_700_000_100.0, tz=timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_long_term_facts_matches_get_facts(tmp_path):
    """The DSAR count and GET /memory/longterm/facts must agree -- same
    frames, same filter -- including after a soft delete."""
    from services.long_term_memory import FACT_FRAME_TYPES

    user_dir = tmp_path / "user_u3"
    user_dir.mkdir()
    frames = {
        "1": {"frame_type": "fact"}, "2": {"frame_type": "conversation"},
        "3": {"frame_type": "entity"}, "4": {"frame_type": "preference"},
    }
    _write_store(user_dir, "long_term_0", frames)
    expected = sum(1 for m in frames.values() if m["frame_type"] in FACT_FRAME_TYPES)
    assert (await _svc(tmp_path).get_gdpr_summary("u3"))["long_term_facts"] == expected == 3

    # Soft delete (memvid_store.sync_remove) drops the sidecar entry only.
    del frames["1"]
    (user_dir / "long_term_0.meta.json").write_text(json.dumps(frames))
    assert (await _svc(tmp_path).get_gdpr_summary("u3"))["long_term_facts"] == 2


@pytest.mark.asyncio
async def test_summary_falls_back_to_mtime_without_timestamps(tmp_path):
    user_dir = tmp_path / "user_u2"
    user_dir.mkdir()
    _write_store(user_dir, "session_x_0", {"1": {"frame_type": "turn"}})
    os.utime(user_dir / "session_x_0.mv2", (1_600_000_000, 1_600_000_000))

    summary = await _svc(tmp_path).get_gdpr_summary("u2")
    assert summary["sessions"] == 1
    assert summary["oldest_session"] == datetime.fromtimestamp(1_600_000_000, tz=timezone.utc).isoformat()


def _client(tmp_path) -> TestClient:
    app = FastAPI()
    app.include_router(gdpr_router.router)
    app.dependency_overrides[get_management_service] = lambda: _svc(tmp_path)
    return TestClient(app)


def test_summary_endpoint_requires_internal_token(tmp_path):
    r = _client(tmp_path).get("/internal/gdpr/user/u1/summary")
    assert r.status_code == 401


def test_summary_endpoint_returns_counts(tmp_path):
    r = _client(tmp_path).get(
        "/internal/gdpr/user/u1/summary", headers={"X-Internal-Token": settings.gdpr_internal_token},
    )
    assert r.status_code == 200
    assert r.json() == {"user_id": "u1", "sessions": 0, "long_term_facts": 0, "oldest_session": None}
