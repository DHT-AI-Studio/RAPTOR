"""
V05-10 / VIE01-364: the operator job list's query building and row shape.
The same routes run against a real PostgreSQL in tests/test_jobs_postgres.py.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_jobs.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.models.job import JobStatus, JobType
from app.routers.jobs import _COLUMNS, _filters, _item
from fastapi import HTTPException


def test_no_filters_means_no_where_clause():
    assert _filters(None, None, None, None, None, None) == ("", [])


def test_every_filter_is_parameterized():
    t = datetime(2026, 10, 1, tzinfo=timezone.utc)
    where, args = _filters(JobType.ERASURE, JobStatus.PARTIAL_FAILURE, "bob", t, t, None)
    assert where == ("WHERE type = $1 AND status = $2 AND user_id = $3 "
                     "AND requested_at >= $4 AND requested_at <= $5")
    assert args == ["erasure", "partial_failure", "bob", t, t]
    assert "bob" not in where


def test_stuck_hours_selects_open_jobs_older_than_the_threshold():
    where, args = _filters(None, None, None, None, None, 24)
    assert where == "WHERE status = ANY($1) AND requested_at < $2"
    assert args[0] == ["pending", "in_progress"]
    assert (datetime.now(timezone.utc) - args[1]).total_seconds() >= 24 * 3600 - 5


@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.IN_PROGRESS])
def test_stuck_hours_with_an_open_status_is_their_intersection(status):
    where, args = _filters(None, status, None, None, None, 24)
    assert where == "WHERE status = $1 AND requested_at < $2"
    assert args[0] == status.value


@pytest.mark.parametrize("status", [JobStatus.PARTIAL_FAILURE, JobStatus.COMPLETED, JobStatus.ALREADY_ERASED])
def test_stuck_hours_with_a_closed_status_is_rejected_not_silently_empty(status):
    with pytest.raises(HTTPException) as exc:
        _filters(None, status, None, None, None, 24)
    assert exc.value.status_code == 422 and status.value in exc.value.detail


def test_download_urls_and_dsar_result_are_never_selected():
    assert "download_urls" not in _COLUMNS and "result" not in _COLUMNS


def test_item_lists_failed_stores_and_parses_json_text():
    row = {"job_id": "7d5e", "type": "erasure", "user_id": "bob", "requested_by": "bob",
           "requested_at": None, "status": "partial_failure", "completed_at": None, "notes": None,
           "store_statuses": '{"module_04": "deleted", "module_35": "failed", "module_25": "failed"}'}
    item = _item(row)
    assert item["failed_stores"] == ["module_25", "module_35"]
    assert item["store_statuses"]["module_04"] == "deleted"
