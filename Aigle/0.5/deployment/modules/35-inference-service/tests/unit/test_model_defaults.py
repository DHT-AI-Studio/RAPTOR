"""INF-3: per-task defaults in Redis, env fallback, per-request override, 422 on unknown models."""
from __future__ import annotations

from redis.exceptions import ConnectionError as RedisConnectionError

from .conftest import make_settings


class DownRedis:
    async def get(self, *_):
        raise RedisConnectionError("down")

    async def mget(self, *_):
        raise RedisConnectionError("down")

    async def set(self, *_):
        raise RedisConnectionError("down")


async def test_get_defaults_uses_env_when_redis_empty(client):
    resp = await client.get("/models/default")
    assert resp.status_code == 200
    assert resp.json() == {"text": "gemma-prod", "stt": "whisper", "tts": None, "clone": "xtts-v2", "diarize": None}


async def test_post_default_persists_to_redis(client, redis):
    resp = await client.post("/models/default", json={"task": "text", "model": "gemma3:4b"})
    assert resp.status_code == 204
    assert resp.content == b""
    assert await redis.get("inf:model:default:text") == b"gemma3:4b"
    assert (await client.get("/models/default")).json()["text"] == "gemma3:4b"


async def test_post_default_unknown_model_is_422_with_available(client, redis):
    resp = await client.post("/models/default", json={"task": "text", "model": "nope"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "model_not_found" and body["model"] == "nope"
    assert "gemma-prod" in body["available"] and "whisper" not in body["available"]
    assert await redis.get("inf:model:default:text") is None


async def test_post_default_wrong_type_is_422(client):
    resp = await client.post("/models/default", json={"task": "stt", "model": "gemma-prod"})
    assert resp.status_code == 422
    assert resp.json()["available"] == ["whisper"]


async def test_post_default_invalid_task_is_400(client):
    resp = await client.post("/models/default", json={"task": "video", "model": "gemma-prod"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_request"


async def test_request_model_overrides_without_persisting(client, upstream, redis):
    await redis.set("inf:model:default:text", "gemma3:4b")
    resp = await client.post("/inference/text/complete", json={"model": "gemma-prod", "input": "hi"})
    assert resp.status_code == 200
    assert upstream.infer_calls[-1]["model_name"] == "gemma-prod"
    assert await redis.get("inf:model:default:text") == b"gemma3:4b"

    # Without an override the Redis default (not the env default) is used.
    resp = await client.post("/inference/text/complete", json={"input": "hi"})
    assert resp.status_code == 200
    assert upstream.infer_calls[-1]["model_name"] == "gemma3:4b"
    assert upstream.infer_calls[-1]["engine"] == "ollama"


async def test_env_default_used_when_redis_has_none(client, upstream):
    resp = await client.post("/inference/text/complete", json={"input": "hi"})
    assert resp.status_code == 200
    call = upstream.infer_calls[-1]
    assert (call["model_name"], call["model_version"]) == ("gemma-prod", "1")
    assert "engine" not in call


async def test_inference_with_unknown_model_is_422(client, upstream):
    resp = await client.post("/inference/text/translate",
                             json={"model": "nope", "input": "x", "options": {"target_language": "en"}})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "model_not_found" and body["model"] == "nope"
    assert isinstance(body["available"], list) and "gemma-prod" in body["available"]
    assert upstream.infer_calls == []


async def test_inference_with_non_text_model_is_422(client):
    resp = await client.post("/inference/text/complete", json={"model": "whisper", "input": "x"})
    assert resp.status_code == 422


async def test_stale_redis_default_is_422(client, redis):
    await redis.set("inf:model:default:text", "removed-model")
    resp = await client.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 422
    assert resp.json()["model"] == "removed-model"


async def test_no_default_and_no_model_is_400(make_client):
    c = await make_client(make_settings(default_models={}))
    resp = await c.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "model_required"


async def test_redis_down_get_falls_back_to_env(make_client):
    c = await make_client(redis_client=DownRedis())
    resp = await c.get("/models/default")
    assert resp.status_code == 200 and resp.json()["text"] == "gemma-prod"
    resp = await c.post("/inference/text/complete", json={"input": "x"})
    assert resp.status_code == 200


async def test_redis_down_post_is_503(make_client):
    c = await make_client(redis_client=DownRedis())
    resp = await c.post("/models/default", json={"task": "text", "model": "gemma-prod"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "redis_unavailable"
