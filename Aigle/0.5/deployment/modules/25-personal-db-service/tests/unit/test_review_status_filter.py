"""CM-5 / VIE01-408: _filters()/_source_filters() exclude 'flagged' (pending
review) and 'removed' (confirmed violation) content by default; 'cleared'
(reviewed, false positive) passes through same as never-flagged (null).
Admin override via SearchRequest.include_flagged.

Pure unit test (no ArcadeDB) -- same style as test_pii_flag.py.
"""
from __future__ import annotations

from app.models.search import SearchRequest
from app.services.searcher import _filters, _source_filters


def _req(**overrides) -> SearchRequest:
    return SearchRequest(query="q", **overrides)


def test_chunk_filter_excludes_flagged_and_removed_by_default():
    where = _filters(_req(), {})
    assert "(review_status IS NULL OR (review_status != 'flagged' AND review_status != 'removed'))" in where


def test_source_filter_excludes_flagged_and_removed_by_default():
    where = _source_filters(_req(), {})
    assert "(review_status IS NULL OR (review_status != 'flagged' AND review_status != 'removed'))" in where


def test_include_flagged_true_drops_the_exclusion_clause():
    where = _filters(_req(include_flagged=True), {})
    assert "review_status" not in where


def test_include_flagged_true_drops_the_exclusion_clause_on_source_too():
    where = _source_filters(_req(include_flagged=True), {})
    assert "review_status" not in where


def test_include_flagged_defaults_to_false():
    assert SearchRequest(query="q").include_flagged is False
