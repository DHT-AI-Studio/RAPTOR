"""Shared fixtures: a fake Module 07 + Ollama behind httpx.MockTransport and fakeredis.

Nothing here needs network, GPU, Module 07, Ollama or a real Redis.
"""
from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
from app.config import Settings
from app.main import build_services, create_app
from fakeredis import aioredis

API = "http://m07.test"
OLLAMA = "http://ollama.test"
M33 = "http://m33.test"
M34 = "http://m34.test"


def _silent_wav(seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(seconds * sample_rate)), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def mlflow_model(name: str, *versions: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "alises": {}, "versions": list(versions)}


def mv(version: str, stage: str, **tags: str) -> dict[str, Any]:
    return {"version": version, "stage": stage, "tags": {"stage": stage, **tags}}


class FakeUpstream:
    """Stateful stand-in for Module 07 and the Ollama daemon."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = [
            mlflow_model(
                "gemma-prod",
                mv("1", "production", runtime="ollama", task_family="text-generation",
                   ollama_model_name="gemma4:12b", ollama_model_size_gb="1.36GB", max_output_tokens="4096"),
                mv("2", "staging", runtime="ollama", task_family="text-generation",
                   ollama_model_name="gemma4:12b"),
                mv("3", "none", runtime="ollama", task_family="text-generation",
                   ollama_model_name="gemma4:12b"),
            ),
            mlflow_model(
                "hf-staging",
                mv("4", "staging", runtime="hf-transformers", task_family="text-generation",
                   est_weights_vram_gb="1.12", max_output_tokens="256"),
                mv("2", "staging", runtime="hf-transformers", task_family="text-generation"),
            ),
            mlflow_model("whisper", mv("1", "production", runtime="hf-transformers", task_family="asr")),
            mlflow_model("vision", mv("1", "production", runtime="hf-transformers", task_family="vlm")),
            mlflow_model("unpromoted", mv("1", "none", runtime="ollama", task_family="text-generation",
                                          ollama_model_name="x:1")),
            # Same name as an Ollama tag: the MLflow entry must win.
            mlflow_model("llama3:8b", mv("1", "production", runtime="ollama", task_family="text-generation",
                                         ollama_model_name="llama3:8b")),
        ]
        self.tags: list[dict[str, Any]] = [
            {"name": "gemma4:12b", "size": 1_460_000_000, "details": {"family": "gemma4", "families": ["gemma4"]}},
            {"name": "llama3:8b", "size": 4_700_000_000, "details": {"family": "llama", "families": ["llama"]}},
            {"name": "gemma3:4b", "size": 3_340_000_000, "details": {"family": "gemma3", "families": ["gemma3"]}},
            {"name": "nomic-embed-text:latest", "size": 274_000_000,
             "details": {"family": "nomic-bert", "families": ["nomic-bert"]}},
            {"name": "bge-m3:latest", "size": 1_200_000_000, "details": {"family": "bert", "families": ["bert"]}},
        ]
        self.ps: list[dict[str, Any]] = [{"name": "gemma3:4b", "model": "gemma3:4b"}]
        self.loaded: dict[str, Any] = {
            "details": {"hf-transformers": [{"model_name": "hf-staging", "model_version": "4", "pending_unload": False}]}
        }
        self.limits: dict[str, int | None] = {"gemma-prod": 4096, "hf-staging": 256, "llama3:8b": None,
                                                  "gemma3:4b": 2048}
        self.api_down = False
        self.ollama_down = False
        self.responses: list[str] = []           # queued infer outputs; default echoes "ok"
        self.infer_error: httpx.Response | None = None
        self.infer_exception: Exception | None = None
        self.infer_calls: list[dict[str, Any]] = []
        self.info_calls: list[dict[str, Any]] = []
        self.list_calls = 0
        # GD-11: Module 33's GET /consent. Granted by default, so tests that aren't
        # about consent can clone; consent_calls records (user, internal token).
        self.voice_clone_consent = True
        self.consent_status = 200
        self.consent_down = False
        self.consent_calls: list[tuple[str, str]] = []
        # VIE01-316/317: Module 34's one-shot STT/TTS.
        self.module34_down = False
        self.module34_status = 200
        self.module34_detail = "module 34 error"
        self.transcribe_text = "你好"
        self.transcribe_language = "zh"
        self.synthesize_wav = _silent_wav()
        self.module34_calls: list[tuple[str, str | None, str | None]] = []  # (path, token, x-user-id)

    # ----- transport -----
    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(M33):
            self.consent_calls.append((request.headers.get("x-user-id"), request.headers.get("x-internal-token")))
            if self.consent_down:
                raise httpx.ConnectError("module 33 down", request=request)
            if self.consent_status != 200:
                return httpx.Response(self.consent_status, json={"detail": "error"})
            return httpx.Response(200, json={"purposes": [
                {"id": "voice_recording", "granted": True},
                {"id": "voice_clone", "granted": self.voice_clone_consent}]})
        if url.startswith(M34):
            self.module34_calls.append((request.url.path, request.headers.get("x-internal-token"),
                                        request.headers.get("x-user-id")))
            if self.module34_down:
                raise httpx.ConnectError("module 34 down", request=request)
            if self.module34_status != 200:
                return httpx.Response(self.module34_status, json={"detail": self.module34_detail})
            if request.url.path == "/audio/transcribe":
                return httpx.Response(200, json={
                    "text": self.transcribe_text, "language": self.transcribe_language,
                    "language_probability": 0.99, "confidence": 0.9,
                })
            if request.url.path == "/audio/synthesize":
                return httpx.Response(200, content=self.synthesize_wav,
                                      headers={"content-type": "audio/wav"})
            return httpx.Response(404)
        if url.startswith(OLLAMA):
            if self.ollama_down:
                raise httpx.ConnectError("ollama down", request=request)
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": self.tags})
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": self.ps})
            return httpx.Response(404)

        if self.api_down:
            raise httpx.ConnectError("07 down", request=request)
        path = request.url.path
        if path == "/models/registered_in_mlflow":
            self.list_calls += 1
            return httpx.Response(200, json=self.registered)
        if path == "/inference/loaded-models":
            return httpx.Response(200, json=self.loaded)
        if path == "/inference/model-info":
            params = dict(request.url.params)
            self.info_calls.append(params)
            name = params["model_name"]
            return httpx.Response(200, json={
                "model_name": name, "model_version": params.get("model_version"),
                "runtime": "ollama", "task_family": "text-generation",
                "limits": {"max_output_tokens": self.limits.get(name), "context_window_tokens": None},
            })
        if path == "/inference/infer":
            body = json.loads(request.content)
            self.infer_calls.append(body)
            if self.infer_exception is not None:
                raise self.infer_exception
            if self.infer_error is not None:
                return self.infer_error
            text = self.responses.pop(0) if self.responses else "ok"
            return httpx.Response(200, json={
                "success": True,
                "result": {"response": text, "metadata": {}},
                "model_name": body["model_name"], "model_version": body.get("model_version"),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })
        return httpx.Response(404, json={"detail": "Not Found"})


def make_settings(**overrides: Any) -> Settings:
    base = Settings(
        module07_api_url=API,
        module07_ollama_host=OLLAMA,
        module33_url=M33,
        module34_api_url=M34,
        gdpr_internal_token="test-internal-token",
        redis_url="redis://unused",
        # clone: "xtts-v2" matches what Settings.from_env() auto-derives for a real
        # deployment with exactly one configured backend (the fixture's clone_backends_runtime
        # default) -- without this, every clone test would need to pass model= explicitly.
        default_models={"text": "gemma-prod", "stt": "whisper", "tts": None, "clone": "xtts-v2", "diarize": None},
    )
    return replace(base, **overrides)


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def redis():
    return aioredis.FakeRedis()


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
async def make_client(upstream: FakeUpstream, redis):
    """Factory: build the app with the given settings / redis and return an ASGI client."""
    clients: list[httpx.AsyncClient] = []

    async def _make(settings: Settings | None = None, redis_client=None) -> httpx.AsyncClient:
        app = create_app(use_lifespan=False)
        http = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
        build_services(app, settings or make_settings(), http, redis_client or redis)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m35")
        client.app = app  # type: ignore[attr-defined]
        clients.extend([client, http])
        return client

    yield _make
    for c in clients:
        await c.aclose()


@pytest.fixture
async def client(make_client, settings):
    return await make_client(settings)


def error_response(status: int, code: str, **extra: Any) -> httpx.Response:
    return httpx.Response(status, json={"detail": {"error_type": "X", "message": code, "code": code, **extra}})


Handler = Callable[[httpx.Request], httpx.Response]
