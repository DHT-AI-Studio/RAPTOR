"""Module 35 -- LLM inference service.

Task-level text APIs (translate / summarize / complete / transform) and model
discovery / per-task defaults, on top of Module 07 inference.  35 owns prompts,
task options, model resolution and the public response/error contract; model
management, runtimes and GPU scheduling stay in Module 07.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.errors import install_error_handlers
from app.routers import gdpr, models, text, voice
from app.schemas import HealthResponse
from app.services.guardrail import Guardrail
from app.services.model_defaults import ModelDefaults, ModelResolver
from app.services.model_registry import ModelRegistry
from app.services.module07_client import Module07Client
from app.services.module34_client import Module34Client
from app.services.text_inference import TextInference
from app.services.voice.clone import (
    Cloning,
    build_clone_backend_specs,
    gpu_available,
    vram_warning,
)
from app.services.voice.diarize import Diarization, PyannoteWhisperX
from app.services.voice.gpu_lock import GpuLock

MODULE_ID = 35


def build_services(app: FastAPI, settings: Settings, http: httpx.AsyncClient, redis: Redis) -> None:
    client = Module07Client(
        http,
        api_url=settings.module07_api_url,
        ollama_host=settings.module07_ollama_host,
        inference_timeout_s=settings.module07_timeout_s,
        registry_timeout_s=settings.registry_timeout_s,
    )
    app.state.gpu_lock = GpuLock()
    app.state.diarization = Diarization(app.state.gpu_lock)
    # VIE01-419 / INF-20: XTTS-v2 weights are Coqui Public Model License
    # (non-commercial) -- off by default even when voice_enabled, until
    # CLONE_ENABLED is explicitly set. See README.md / API.md.
    # CLONE_BACKENDS_RUNTIME (not CLONE_ENABLED) is what controls whether XTTS
    # specifically gets loaded at all, so a deployment that never lists
    # "xtts-v2" there never loads non-commercial weights regardless of this
    # policy gate.
    clone_disabled_reason = None if settings.clone_enabled else (
        "voice cloning is disabled on this deployment (set CLONE_ENABLED=true -- "
        "XTTS-v2 weights are Coqui Public Model License, non-commercial)"
    )
    specs = build_clone_backend_specs(settings.voice_model_dir, settings.clone_backends_runtime)
    app.state.cloning = Cloning(app.state.gpu_lock, specs, disabled_reason=clone_disabled_reason)

    registry = ModelRegistry(client, ttl_s=settings.model_cache_ttl_s,
                              static_entries=app.state.cloning.builtin_model_entries)
    resolver = ModelResolver(registry, ModelDefaults(redis, settings.default_models))
    app.state.http = http
    app.state.registry = registry
    app.state.resolver = resolver
    app.state.text_inference = TextInference(client, resolver, settings, Guardrail(http, settings))
    app.state.voice_redis = redis
    app.state.settings = settings
    # VIE01-316/317: Module 34 already has Whisper/TTS loaded for live audio sessions --
    # /inference/voice/stt and /tts forward there instead of loading a second copy here.
    app.state.module34 = Module34Client(
        http, settings.module34_api_url, settings.gdpr_internal_token, settings.module34_timeout_s,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    http = httpx.AsyncClient()
    redis = Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)
    build_services(app, settings, http, redis)
    if settings.voice_enabled:
        import asyncio
        try:
            app.state.diarization.backend = await asyncio.to_thread(
                PyannoteWhisperX, settings.voice_model_dir,
            )
        except Exception:
            logging.getLogger(__name__).warning("Diarization models unavailable", exc_info=True)
        if settings.clone_enabled:
            for name in settings.clone_backends_warm & set(app.state.cloning.specs):
                try:
                    await app.state.cloning.preload(name)
                except Exception:
                    logging.getLogger(__name__).warning("Clone backend '%s' unavailable", name, exc_info=True)
    try:
        yield
    finally:
        await http.aclose()
        await redis.aclose()


def create_app(use_lifespan: bool = True) -> FastAPI:
    app = FastAPI(
        title="Raptor Module 35 — LLM Inference Service",
        version="0.5.0",
        lifespan=lifespan if use_lifespan else None,
    )
    install_error_handlers(app)

    @app.get("/health", tags=["system"], response_model=HealthResponse)
    async def health():
        settings = getattr(app.state, "settings", get_settings())
        
        module07_reachable = False
        http: httpx.AsyncClient | None = getattr(app.state, "http", None)
        if http is not None:
            try:
                r = await http.get(
                    f"{settings.module07_api_url}/health",
                    timeout=3.0,
                )
                module07_reachable = r.status_code == 200
            except (httpx.TimeoutException, httpx.HTTPError, OSError):
                module07_reachable = False
        module34_reachable = False
        if http is not None:
            try:
                r = await http.get(f"{settings.module34_api_url}/health", timeout=3.0)
                module34_reachable = r.status_code == 200
            except (httpx.TimeoutException, httpx.HTTPError, OSError):
                module34_reachable = False
        gpu_name: str | None = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except (ImportError, AttributeError, RuntimeError):
            pass
        cloning = getattr(app.state, "cloning", None)
        loaded_voice_models: list[str] = []
        if getattr(getattr(app.state, "diarization", None), "backend", None):
            loaded_voice_models.append("pyannote-whisperx")
        if cloning is not None:
            loaded_voice_models.extend(cloning.loaded_backends())

        gpu_ok = gpu_available()
        # Text profile never has (or needs) a GPU -- only let its absence mark
        # this instance "degraded" when voice is actually enabled and expects one.
        healthy = gpu_ok if settings.voice_enabled else True

        return {
            "status": "ok" if healthy else "degraded",
            "module": MODULE_ID,
            "gpu_available": gpu_ok,
            "gpu_name": gpu_name,
            "module07_reachable": module07_reachable,
            "module34_reachable": module34_reachable,
            "loaded_voice_models": loaded_voice_models,
            "voice": {
                "diarize_loaded": bool(getattr(getattr(app.state, "diarization", None), "backend", None)),
                # "usable" (configured + policy-enabled), not "resident" -- clone
                # backends load lazily now, so nothing may be resident yet by design.
                "clone_loaded": bool(cloning and cloning.specs and cloning.disabled_reason is None),
                "clone_configured_backends": cloning.configured_backends() if cloning else [],
            },
            "warnings": vram_warning() if settings.voice_enabled else [],
        }

    app.include_router(models.router)
    app.include_router(text.router)
    app.include_router(voice.router)
    app.include_router(gdpr.router)
    return app


app = create_app()
# INF-17 / VIE01-376: not inside create_app() -- tests build throwaway apps
# via create_app() directly, and Instrumentator registers its metrics in
# prometheus_client's process-wide default registry, so instrumenting more
# than once per process raises a duplicate-timeseries error.
Instrumentator().instrument(app).expose(app, include_in_schema=False)
