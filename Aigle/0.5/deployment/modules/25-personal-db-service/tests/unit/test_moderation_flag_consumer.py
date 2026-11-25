"""CM-5 / VIE01-408: unit tests for moderation_flag_consumer._apply() -- keeps
Chunk/Source.review_status in sync with Module 23's moderation.flagged events.

Same FakeClient style as test_graph_search.py: no live ArcadeDB, routes by call.
"""
from __future__ import annotations

import pytest
from app.services.moderation_flag_consumer import _apply


class FakeClient:
    def __init__(self, db_exists: bool = True):
        self._db_exists = db_exists
        self.commands: list[tuple[str, dict]] = []

    async def database_exists(self, db):
        return self._db_exists

    async def command(self, db, sql, params=None, ignore_exists=False):
        self.commands.append((sql, params))
        return []


def _event(**overrides) -> dict:
    return {"branch_id": "b1", "status": "flagged", "chunk_id": "c1", **overrides}


@pytest.mark.asyncio
async def test_chunk_id_updates_chunk_by_chunk_id():
    client = FakeClient()
    await _apply(client, _event(chunk_id="c1", asset_path=None, moment_id=None))

    assert len(client.commands) == 1
    sql, params = client.commands[0]
    assert "UPDATE Chunk" in sql and "chunk_id = :c" in sql
    assert params == {"s": "flagged", "c": "c1"}


@pytest.mark.asyncio
async def test_moment_id_updates_chunk_too_same_as_chunk_id():
    """A video moment's moment_id IS its Chunk.chunk_id in this database
    (same hash Module 11 derives) -- no separate "Moment" type exists."""
    client = FakeClient()
    await _apply(client, _event(chunk_id=None, moment_id="abc123", asset_path=None))

    sql, params = client.commands[0]
    assert "UPDATE Chunk" in sql and params == {"s": "flagged", "c": "abc123"}


@pytest.mark.asyncio
async def test_asset_path_updates_source_by_asset_path_and_version_id():
    client = FakeClient()
    await _apply(client, _event(chunk_id=None, moment_id=None, asset_path="assets/a.jpg", version_id="v1"))

    sql, params = client.commands[0]
    assert "UPDATE Source" in sql and "asset_path = :a" in sql and "version_id = :v" in sql
    assert params == {"s": "flagged", "a": "assets/a.jpg", "v": "v1"}


@pytest.mark.asyncio
async def test_status_transition_is_applied_the_same_way_as_creation():
    client = FakeClient()
    await _apply(client, _event(status="cleared"))

    assert client.commands[0][1]["s"] == "cleared"


@pytest.mark.parametrize("missing", ["branch_id", "status"])
@pytest.mark.asyncio
async def test_missing_branch_id_or_status_is_skipped(missing):
    client = FakeClient()
    event = _event()
    del event[missing]
    await _apply(client, event)

    assert client.commands == []


@pytest.mark.asyncio
async def test_uninitialized_database_is_skipped():
    client = FakeClient(db_exists=False)
    await _apply(client, _event())

    assert client.commands == []


@pytest.mark.asyncio
async def test_no_content_reference_is_skipped():
    client = FakeClient()
    await _apply(client, _event(chunk_id=None, moment_id=None, asset_path=None))

    assert client.commands == []


@pytest.mark.asyncio
async def test_asset_path_with_no_version_id_is_skipped():
    client = FakeClient()
    await _apply(client, _event(chunk_id=None, moment_id=None, asset_path="assets/a.jpg"))

    assert client.commands == []
