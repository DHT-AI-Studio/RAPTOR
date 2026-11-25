"""GD-6 / VIE01-251: Module 12's pii_detected -> Chunk.pii_flag -> the
search result's pii_warning.

Pure unit test (no ArcadeDB): the Kafka entry mapping and the shared result
builder every search endpoint -- hybrid, vector, bm25 -- goes through.

Run from the 25-personal-db-service/ directory:
    python -m pytest tests/unit/test_pii_flag.py -v
"""
from __future__ import annotations

from app.services import searcher
from app.services.kafka_consumer import _map_entry


def _entry(**extra):
    return {"text": "Contact jane.doe@example.com", "type": "documents", "version_id": "v1",
            "asset_path": "a/b.pdf", "branch_id": "u1", **extra}


def test_pii_detected_sets_pii_flag():
    req = _map_entry(_entry(pii_detected=True), "c1", None)
    assert req.pii_flag is True
    assert req.model_dump(exclude_none=True)["pii_flag"] is True  # written by index_chunk()


def test_clean_or_unscanned_chunk_gets_no_flag():
    for ep in (_entry(pii_detected=False), _entry()):     # scanned clean / fail-open (never scanned)
        req = _map_entry(ep, "c1", None)
        assert req.pii_flag is None
        assert "pii_flag" not in req.model_dump(exclude_none=True)


def test_search_result_annotates_flagged_chunk_without_hiding_it():
    rows = [
        {"chunk_id": "c1", "text": "Contact jane.doe@example.com", "pii_flag": True, "@rid": "#1:0", "distance": 0.1},
        {"chunk_id": "c2", "text": "Quarterly revenue grew.", "pii_flag": None, "@rid": "#1:1", "distance": 0.2},
        {"version_id": "v9", "summary": "doc summary", "@rid": "#2:0", "distance": 0.3},   # Source row
    ]
    results = searcher._results(rows, rank_score=True)
    flagged, clean, source = (r.payload for r in results)
    assert flagged["pii_warning"] is True
    assert flagged["text"] == "Contact jane.doe@example.com"   # content returned, never blocked
    assert "pii_flag" not in flagged                           # surfaced as pii_warning only
    assert "pii_warning" not in clean and "pii_flag" not in clean
    assert "pii_warning" not in source


def test_pii_flag_is_selected_by_every_chunk_search():
    assert "pii_flag" in searcher._PAYLOAD
