"""indexer.clone_version() -- the filename it gives the clone.

Found live: content-dedup clones a Source/Chunk onto a new asset_path, but
kept the SOURCE's filename verbatim when the new upload used a different one
-- every presign-url lookup for the clone then 404'd, since that filename
never existed at the new asset_path in LakeFS. target_filename fixes it.

Same FakeClient style as test_moderation_flag_consumer.py: no live ArcadeDB,
routes by SQL text. Only Source/Chunk queries return rows -- every other
query (edges, mentions, relations, facts, faces) returns [] so clone_version()
takes the "nothing to clone there" path and never reaches that logic, which
this test isn't about.

Run from the 25-personal-db-service/ directory:
    python -m pytest tests/unit/test_clone_version.py -v
"""
from __future__ import annotations

import pytest
from app.models.index import CloneVersionRequest
from app.services.indexer import clone_version


class FakeClient:
    def __init__(self, source_row: dict, chunk_rows: list[dict]):
        self._source_row = source_row
        self._chunk_rows = chunk_rows
        self.commands: list[tuple[str, dict]] = []

    async def database_exists(self, db):
        return True

    async def query(self, db, sql, params=None):
        if "FROM Source WHERE version_id" in sql and params["v"] == self._source_row["version_id"]:
            return [self._source_row]
        if "FROM Chunk WHERE version_id" in sql and params["v"] == self._source_row["version_id"]:
            return self._chunk_rows
        return []  # edges / mentions / relations / facts / faces: none in this test

    async def command(self, db, sql, params=None, ignore_exists=False):
        self.commands.append((sql, params))
        return []


def _source(**overrides) -> dict:
    return {"version_id": "sv1", "asset_path": "video/mp4/old_name", "filename": "old_name.mp4",
            "status": "active", **overrides}


def _chunk(chunk_id: str, filename: str) -> dict:
    return {"chunk_id": chunk_id, "version_id": "sv1", "asset_path": "video/mp4/old_name",
            "filename": filename, "status": "active"}


def _req(**overrides) -> CloneVersionRequest:
    return CloneVersionRequest(
        source_version_id="sv1", source_asset_path="video/mp4/old_name",
        target_version_id="tv1", target_asset_path="video/mp4/new_name", **overrides)


def _updates(client: FakeClient, table: str) -> list[dict]:
    return [params for sql, params in client.commands if sql.startswith(f"UPDATE {table}")]


@pytest.mark.asyncio
async def test_target_filename_overrides_the_cloned_sources_filename():
    client = FakeClient(_source(), [])
    await clone_version(client, "b1", _req(target_filename="new_name.mp4"))

    [update] = _updates(client, "Source")
    assert update["filename"] == "new_name.mp4"
    assert update["version_id"] == "tv1" and update["asset_path"] == "video/mp4/new_name"


@pytest.mark.asyncio
async def test_a_chunk_carrying_the_assets_own_filename_is_remapped_too():
    client = FakeClient(_source(), [_chunk("c1", "old_name.mp4")])
    await clone_version(client, "b1", _req(target_filename="new_name.mp4"))

    [update] = _updates(client, "Chunk")
    assert update["filename"] == "new_name.mp4"


@pytest.mark.asyncio
async def test_a_chunk_carrying_some_other_filename_is_left_alone():
    """An associated-file chunk (a different filename from the asset's own)
    -- nothing here knows what that file was renamed to, if anything, so it
    isn't touched. Documented gap, not this test's bug."""
    client = FakeClient(_source(), [_chunk("c1", "attachment.pdf")])
    await clone_version(client, "b1", _req(target_filename="new_name.mp4"))

    [update] = _updates(client, "Chunk")
    assert update["filename"] == "attachment.pdf"


@pytest.mark.asyncio
async def test_without_target_filename_the_source_filename_is_kept_unchanged():
    """A caller that predates this field (there shouldn't be one in this
    codebase, but the field is Optional) gets the old, buggy-for-rename
    behavior back, not a crash."""
    client = FakeClient(_source(), [_chunk("c1", "old_name.mp4")])
    await clone_version(client, "b1", _req())

    assert _updates(client, "Source")[0]["filename"] == "old_name.mp4"
    assert _updates(client, "Chunk")[0]["filename"] == "old_name.mp4"
