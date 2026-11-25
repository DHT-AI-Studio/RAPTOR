"""GD-1 / VIE01-246: DELETE /internal/gdpr/user/{user_id} erases the user's
voice-clone speaker embeddings, and only theirs."""
from __future__ import annotations

import pytest
from app.services.voice.speaker_cache import SpeakerCache

from tests.unit.conftest import make_settings

TOKEN = "test-internal-token"


@pytest.fixture
async def gdpr_client(make_client):
    return await make_client(make_settings(gdpr_internal_token=TOKEN))


async def _seed(redis, user_id: str, *speaker_ids: str) -> SpeakerCache:
    cache = SpeakerCache(redis, user_id)
    for speaker_id in speaker_ids:
        await cache.put(speaker_id, b"embedding-" + speaker_id.encode())
    return cache


async def test_deletes_every_embedding_of_the_user_and_nothing_else(gdpr_client, redis):
    alice = await _seed(redis, "alice", "demo", "narrator", "boss")
    bob = await _seed(redis, "bob", "demo")

    resp = await gdpr_client.delete("/internal/gdpr/user/alice", headers={"X-Internal-Token": TOKEN})

    assert resp.status_code == 200
    assert resp.json() == {"user_id": "alice", "speaker_embeddings_deleted": 3}
    assert [k async for k in redis.scan_iter(match=alice.prefix + "*")] == []  # embeddings and LRU index
    assert await alice.list() == []
    assert [s["id"] for s in await bob.list()] == ["demo"]
    assert await bob.get("demo") == b"embedding-demo"


async def test_also_deletes_embeddings_the_index_lost_track_of(gdpr_client, redis):
    alice = await _seed(redis, "alice", "demo")
    await redis.zrem(alice.index, "demo")  # e.g. pruned from the index, key still alive

    resp = await gdpr_client.delete("/internal/gdpr/user/alice", headers={"X-Internal-Token": TOKEN})

    assert resp.json()["speaker_embeddings_deleted"] == 1
    assert await redis.exists(alice.key("demo")) == 0


async def test_user_without_embeddings_is_not_an_error(gdpr_client):
    resp = await gdpr_client.delete("/internal/gdpr/user/nobody", headers={"X-Internal-Token": TOKEN})
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "nobody", "speaker_embeddings_deleted": 0}


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Token": "wrong"}])
async def test_rejects_missing_or_wrong_token(gdpr_client, redis, headers):
    alice = await _seed(redis, "alice", "demo")
    resp = await gdpr_client.delete("/internal/gdpr/user/alice", headers=headers)
    assert resp.status_code == 401
    assert await alice.get("demo") is not None


async def test_rejects_every_call_when_no_token_is_configured(make_client, redis):
    client = await make_client(make_settings(gdpr_internal_token=""))
    alice = await _seed(redis, "alice", "demo")
    resp = await client.delete("/internal/gdpr/user/alice", headers={"X-Internal-Token": ""})
    assert resp.status_code == 401
    assert await alice.get("demo") is not None


# ── GD-2 / VIE01-247: count for the export manifest ─────────────────────────

async def test_summary_counts_only_the_users_embeddings_and_changes_nothing(gdpr_client, redis):
    alice = await _seed(redis, "alice", "demo", "narrator")
    await _seed(redis, "bob", "demo")
    await redis.zrem(alice.index, "narrator")  # index lost track of it, key still alive
    before = sorted([k async for k in redis.scan_iter()])

    resp = await gdpr_client.get("/internal/gdpr/user/alice/summary", headers={"X-Internal-Token": TOKEN})

    assert resp.status_code == 200
    assert resp.json() == {"user_id": "alice", "speaker_embeddings": 2}
    assert sorted([k async for k in redis.scan_iter()]) == before
    assert await redis.zrange(alice.index, 0, -1) == [b"demo"]  # not pruned or repaired


async def test_summary_for_user_without_embeddings(gdpr_client):
    resp = await gdpr_client.get("/internal/gdpr/user/nobody/summary", headers={"X-Internal-Token": TOKEN})
    assert resp.json() == {"user_id": "nobody", "speaker_embeddings": 0}


async def test_summary_requires_the_token(gdpr_client):
    assert (await gdpr_client.get("/internal/gdpr/user/alice/summary")).status_code == 401
