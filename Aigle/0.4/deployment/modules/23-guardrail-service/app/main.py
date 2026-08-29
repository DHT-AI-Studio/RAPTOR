from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Response, status
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.db import connection as db_conn
from app.db import redis_conn
from app.db import repo as policy_repo
from app.engine.models import PolicyDetail
from app.engine.policy_store import ACTIVE_POLICY_CACHE_KEY, ACTIVE_POLICY_CACHE_TTL
from app.routers import (check_guard, check_policy, check_policy_llm, debug_policy_check_llm,
                         guardrail, guardrail_policies, proxy_ollama, system)
from app.routers.check_guard import close_client as close_guard_client
from app.services import state
from app.services.proxy_checker import close_checker_client

_TZ_TW = timezone(timedelta(hours=8))


def _configure_logging(log_level: str) -> None:
    class _TZFmt(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=_TZ_TW)
            return dt.strftime(datefmt) if datefmt else dt.isoformat()

    fmt = _TZFmt("%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logging.root.handlers = [handler]
    logging.root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


async def _check_model(ollama_url: str, model: str) -> None:
    log = logging.getLogger(__name__)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
        if model not in available:
            log.warning("model '%s' not found on Ollama. Available: %s", model, available)
        else:
            log.info("model '%s' verified.", model)
    except Exception as exc:
        log.warning("Cannot verify model '%s' at startup: %s", model, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configure_logging(settings.log_level)
    log = logging.getLogger(__name__)
    log.info(
        "23-guardrail-service starting — proxy_model=%s guard_models=%s ollama=%s",
        settings.proxy_model,
        settings.active_models,
        settings.ollama_url,
    )

    # Policy storage: Postgres pool + schema
    await db_conn.init_db(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
    )

    # Active-policy cache: Redis
    await redis_conn.init_redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        db=settings.redis_db,
    )

    # Global enable/disable switch: seed from GR_DEFAULT_ENABLED on a cold Redis
    await state.init_default_state()

    # Warm the Redis cache from Postgres so the first /policy/check/* call
    # after a restart doesn't have to fall back to a DB query.
    pool = await db_conn.get_pool()
    active_row = await policy_repo.get_active_policy(pool)
    if active_row is not None:
        redis = await redis_conn.get_redis()
        policy = PolicyDetail(
            id=active_row["id"], name=active_row["name"], version=active_row["version"],
            raw_content=active_row["raw_content"],
            content_llama_guard=active_row["content_llama_guard"],
            content_granite_guardian=active_row["content_granite_guardian"],
            content_gpt_oss_safeguard=active_row["content_gpt_oss_safeguard"],
            is_active=active_row["is_active"], created_at=active_row["created_at"],
        )
        await redis.set(ACTIVE_POLICY_CACHE_KEY, policy.model_dump_json(), ex=ACTIVE_POLICY_CACHE_TTL)
        log.info("warmed active-policy cache: %s v%s", policy.name, policy.version)

    # verify confidence-based classifier model
    await _check_model(settings.ollama_url, settings.proxy_model)

    # verify guard model(s) (Llama-Guard3 / Granite / GPT-OSS)
    for m in settings.active_models:
        await _check_model(settings.ollama_url, m)

    log.info("23-guardrail-service ready")
    yield

    await close_checker_client()
    await close_guard_client()
    await db_conn.close_db()
    await redis_conn.close_redis()
    log.info("23-guardrail-service shutdown complete")


app = FastAPI(
    title="Raptor Guardrail Service",
    description=(
        "Content moderation for the Raptor platform — confidence-based proxy, "
        "multi-model guard classification, and a deterministic policy engine.\n\n"
        "## Endpoints\n\n"
        "| Group | Prefix | Description |\n"
        "|---|---|---|\n"
        "| Proxy | `/api/generate` | Confidence-based guard |\n"
        "| Guard | `/guard/check/*` | Llama-Guard3 / Granite / GPT-OSS (S1-S14) |\n"
        "| Policy | `/policy/check/*` | Deterministic: signal extraction + Python rule matching |\n"
        "| Policy (LLM-judged) | `/policy/check/llm/*` | Same multi-model guard dispatch as `/guard/check/*`, "
        "with the active policy as the system prompt |\n"
        "| Policy Admin | `/guardrail/policies*` | Upload/list/activate/delete versioned policies (PostgreSQL + Redis) |\n"
    ),
    version="0.1",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

app.include_router(proxy_ollama.router)
app.include_router(check_guard.router, prefix="/guard")
app.include_router(check_policy.router, prefix="/policy")
app.include_router(check_policy_llm.router, prefix="/policy")
app.include_router(guardrail_policies.router, prefix="/guardrail")
app.include_router(guardrail.router, prefix="/guardrail")   # GB-4 policy check /guardrail/check/*
app.include_router(system.router, prefix="/guardrail")      # global enable/disable /guardrail/system/*
app.include_router(debug_policy_check_llm.router, prefix="/debug")  # debug /debug/policy/check/llm/*


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


async def _missing_guard_models(settings) -> tuple[Optional[list[str]], Optional[str]]:
    """Which configured guard models Ollama does not have.

    Returns (missing, error). `error` is set when Ollama could not be asked at
    all, which is a different failure from "asked, and the model is not there".
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            resp.raise_for_status()
            available = {m["name"] for m in resp.json().get("models", [])}
    except Exception as exc:
        return None, str(exc)
    return [m for m in settings.active_models if m not in available], None


@app.get("/health", tags=["Health"], summary="Service health check")
async def health(response: Response):
    """Reports unhealthy when a configured guard model is not on Ollama.

    A missing guard model does not fail requests: the check endpoint returns 502,
    Module 07's hook is fail-open by design, and inference carries on unguarded.
    That is the dangerous case — nothing is blocked and nothing looks wrong. So
    the condition is surfaced where it cannot be missed instead: 503 here makes
    the container healthcheck fail, so a deploy with an unpulled guard model
    never comes up green.

    Checked live rather than cached from startup, so pulling the model afterwards
    clears the condition without a restart, and deleting it is noticed.
    """
    settings = get_settings()
    try:
        guardrail_enabled = await state.is_enabled()
    except Exception:
        guardrail_enabled = None
    body = {
        "status":            "ok",
        "guardrail_enabled": guardrail_enabled,
        "service":           "23-guardrail-service",
        "proxy_model":        settings.proxy_model,
        "proxy_mode":         settings.proxy_mode,
        "guard_models":       settings.active_models,
        "guard_mode":         f"{len(settings.active_models)}-model" if len(settings.active_models) > 1 else "single",
        "ollama_url":         settings.ollama_url,
    }

    missing, error = await _missing_guard_models(settings)
    if error is not None:
        body["status"] = "degraded"
        body["reason"] = f"cannot reach Ollama to verify guard models: {error}"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif missing:
        body["status"] = "degraded"
        body["reason"] = (f"guard model(s) not on Ollama: {', '.join(missing)} — "
                          f"guardrail checks will fail and inference will run unguarded. "
                          f"Pull them (e.g. `ollama pull {missing[0]}`) or set GUARD_MODEL.")
        body["missing_guard_models"] = missing
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return body
