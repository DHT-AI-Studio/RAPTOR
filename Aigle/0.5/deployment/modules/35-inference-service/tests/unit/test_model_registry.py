"""INF-2: registry aggregation, dedup, version policy, status, fallback and TTL cache."""
from __future__ import annotations

import httpx
from app.services.model_registry import ModelRegistry, select_version
from app.services.module07_client import Module07Client

from .conftest import API, OLLAMA, mv


def _registry(upstream, ttl=60.0) -> ModelRegistry:
    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    return ModelRegistry(Module07Client(http, API, OLLAMA, 5, 5), ttl_s=ttl)


def _by_name(models):
    return {m.name: m for m in models}


def test_select_version_prefers_production_then_highest_staging():
    assert select_version([mv("1", "production"), mv("3", "staging"), mv("4", "none")])["version"] == "1"
    assert select_version([mv("1", "production"), mv("2", "production")])["version"] == "2"
    assert select_version([mv("2", "staging"), mv("10", "staging")])["version"] == "10"
    assert select_version([mv("5", "none"), mv("6", "archived")]) is None
    assert select_version([mv("1", "Production")])["version"] == "1"


async def test_merges_both_sources_with_required_fields(upstream):
    models = _by_name(await _registry(upstream).list_models())

    prod = models["gemma-prod"]
    assert (prod.source, prod.type, prod.version, prod.stage) == ("mlflow", "llm", "1", "production")
    assert prod.size_gb == 1.36
    assert prod.capabilities == ["translate", "summarize", "complete", "transform"]
    assert prod.runtime_model == "gemma4:12b"
    assert prod.engine is None

    gemma = models["gemma3:4b"]
    assert (gemma.source, gemma.type, gemma.status, gemma.engine) == ("ollama", "llm", "loaded", "ollama")
    assert gemma.size_gb == round(3_340_000_000 / 1024 ** 3, 2)

    public = prod.public()
    for key in ("name", "type", "size_gb", "capabilities", "source", "status"):
        assert key in public


async def test_status_loaded_staged_available(upstream):
    models = _by_name(await _registry(upstream).list_models())
    assert models["hf-staging"].status == "loaded"          # v4 resident in 07 (HF)
    assert models["hf-staging"].version == "4"
    assert models["gemma-prod"].status == "available"
    assert models["gemma4:12b"].status == "available"

    upstream.loaded = {"details": {"hf-transformers": []}}
    models = _by_name(await _registry(upstream).list_models())
    assert models["hf-staging"].status == "staged"

    upstream.ps = [{"name": "gemma4:12b"}]
    models = _by_name(await _registry(upstream).list_models())
    assert models["gemma-prod"].status == "loaded"            # its Ollama tag is resident


async def test_hf_loaded_only_counts_selected_version(upstream):
    upstream.loaded = {"details": {"hf-transformers": [
        {"model_name": "hf-staging", "model_version": "2", "pending_unload": False},
        {"model_name": "hf-staging", "model_version": "4", "pending_unload": True},
    ]}}
    models = _by_name(await _registry(upstream).list_models())
    assert models["hf-staging"].status == "staged"


async def test_dedup_by_name_mlflow_wins(upstream):
    models = await _registry(upstream).list_models()
    llama = [m for m in models if m.name == "llama3:8b"]
    assert len(llama) == 1 and llama[0].source == "mlflow"


async def test_filters_unpromoted_unsupported_and_embeddings(upstream):
    names = {m.name for m in await _registry(upstream).list_models()}
    assert "unpromoted" not in names                  # no production / staging version
    assert "vision" not in names                      # vlm is not a Module 35 model type
    assert "nomic-embed-text:latest" not in names
    assert "bge-m3:latest" not in names
    assert "whisper" in names


async def test_type_mapping_and_override_tag(upstream):
    upstream.registered.append(
        {"name": "voice-clone", "versions": [mv("1", "production", runtime="hf-transformers",
                                                  task_family="tts", inf_model_type="clone")]}
    )
    models = _by_name(await _registry(upstream).list_models())
    assert models["whisper"].type == "stt"
    assert models["whisper"].capabilities == ["transcribe"]
    assert models["voice-clone"].type == "clone"


async def test_module07_unreachable_falls_back_to_ollama(upstream, caplog):
    upstream.api_down = True
    reg = _registry(upstream)
    snap = await reg.snapshot()
    assert snap.sources == {"mlflow": "unavailable", "ollama": "ok"}
    assert {m.source for m in snap.models} == {"ollama"}
    assert "llama3:8b" in {m.name for m in snap.models}
    assert any("Module 07 registry unavailable" in r.message for r in caplog.records)


async def test_module07_timeout_falls_back_to_ollama(upstream):
    def handler(request):
        if str(request.url).startswith(API):
            raise httpx.ReadTimeout("slow", request=request)
        return upstream.handler(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reg = ModelRegistry(Module07Client(http, API, OLLAMA, 5, 5))
    models = await reg.list_models()
    assert models and all(m.source == "ollama" for m in models)


async def test_module07_http_error_falls_back(upstream):
    def handler(request):
        if request.url.path == "/models/registered_in_mlflow":
            return httpx.Response(500, json={"detail": "mlflow down"})
        return upstream.handler(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    snap = await ModelRegistry(Module07Client(http, API, OLLAMA, 5, 5)).snapshot()
    assert snap.sources["mlflow"] == "unavailable"


async def test_everything_down_returns_empty_list(upstream):
    upstream.api_down = upstream.ollama_down = True
    snap = await _registry(upstream).snapshot()
    assert snap.models == []
    assert snap.sources == {"mlflow": "unavailable", "ollama": "unavailable"}


async def test_malformed_registry_row_is_skipped(upstream):
    upstream.registered.append({"name": "broken", "versions": [{"version": "1", "stage": "production", "tags": None}]})
    upstream.registered.append({"versions": []})
    names = {m.name for m in await _registry(upstream).list_models()}
    assert "gemma-prod" in names


async def test_ttl_cache_and_invalidate(upstream):
    reg = _registry(upstream, ttl=60)
    await reg.list_models()
    await reg.list_models()
    assert upstream.list_calls == 1
    reg.invalidate()
    await reg.list_models()
    assert upstream.list_calls == 2

    reg0 = _registry(upstream, ttl=0)
    await reg0.list_models()
    await reg0.list_models()
    assert upstream.list_calls == 4


async def test_get_refreshes_once_on_miss_of_old_snapshot(upstream, monkeypatch):
    import app.services.model_registry as mr

    reg = _registry(upstream, ttl=60)
    assert await reg.get("gemma-prod") is not None
    assert await reg.get("does-not-exist") is None   # fresh snapshot: no refresh
    assert upstream.list_calls == 1

    monkeypatch.setattr(mr, "_MISS_REFRESH_MIN_AGE_S", 0.0)
    upstream.registered.append({"name": "new-model", "versions": [
        mv("1", "production", runtime="ollama", task_family="text-generation", ollama_model_name="n:1")]})
    assert (await reg.get("new-model")).version == "1"
    assert upstream.list_calls == 2


async def test_get_models_routes(client):
    resp = await client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == {"mlflow": "ok", "ollama": "ok"}
    assert body["count"] == len(body["models"])

    one = await client.get("/models/gemma4:12b")
    assert one.status_code == 200 and one.json()["source"] == "ollama"

    missing = await client.get("/models/nope")
    assert missing.status_code == 404
    assert missing.json() == {"error": "model_not_found", "model": "nope"}


async def test_get_models_never_5xx_when_upstreams_down(client, upstream):
    upstream.api_down = upstream.ollama_down = True
    resp = await client.get("/models")
    assert resp.status_code == 200
    assert resp.json()["models"] == []


async def test_model_name_with_slash(client, upstream):
    upstream.tags.append({"name": "frob/model:7b", "size": 1, "details": {}})
    resp = await client.get("/models/frob/model:7b")
    assert resp.status_code == 200 and resp.json()["name"] == "frob/model:7b"
