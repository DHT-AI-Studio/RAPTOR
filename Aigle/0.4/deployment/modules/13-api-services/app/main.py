"""FastAPI application entry point for the API gateway."""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from aiokafka import AIOKafkaProducer
from fastapi import FastAPI, Request, status
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from redis.asyncio import Redis  # type: ignore[import]
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import RedirectResponse
from fastapi import Depends

from app.routers.processing import router as processing_router
from app.routers.search import router as search_router
from app.routers.video_search import router as video_search_router
from app.routers.chat import router as chat_router
from app.core.config import Settings, get_settings
from app.middlewares.logging import RequestLoggingMiddleware
from app.middlewares.legacy_api_alias import LegacyApiAliasMiddleware
from app.middlewares.guardrail import GuardrailMiddleware
from app.routers.asset import router as asset_router
from app.routers.training import router as training_proxy_router
from app.routers.benchmark import router as benchmark_proxy_router
from app.routers.aiml_life_cycle import router as aiml_router
from app.routers.SSO import router as sso
from app.routers.agent_protocol import router as agent_protocol_router, well_known_router
from app.routers.media_sync import router as media_sync_router
from app.routers.memory import router as memory_router
from app.routers.personal_db import router as personal_db_router
from app.routers.mcp import router as mcp_router
from app.api.dependencies import get_current_user


def configure_logging(settings: Settings) -> None:
    """Configure application-wide logging with unified format and UTC+8 timestamps."""

    class TZFormatter(logging.Formatter):
        def __init__(self, *args, tz: timezone, **kwargs):
            super().__init__(*args, **kwargs)
            self.tz = tz

        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=self.tz)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.isoformat()

    class TZJSONFormatter(jsonlogger.JsonFormatter):
        def __init__(self, *args, tz: timezone, **kwargs):
            super().__init__(*args, **kwargs)
            self.tz = tz

        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=self.tz)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.isoformat()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.root.handlers = []

    tz_tw = timezone(timedelta(hours=8))

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    console_formatter = TZFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz=tz_tw,
    )

    json_formatter = TZJSONFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "log_level"},
        tz=tz_tw,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "gateway.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_formatter)

    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)
    logging.root.setLevel(log_level)

    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = [console_handler]
        logger.setLevel(log_level)
        logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown events."""
    settings = get_settings()
    configure_logging(settings)

    timeout = httpx.Timeout(settings.request_timeout)
    limits = httpx.Limits(
        max_connections=settings.max_connections,
        max_keepalive_connections=settings.max_keepalive_connections,
    )
    client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
    )

    kafka_producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
    )
    try:
        await asyncio.wait_for(kafka_producer.start(), timeout=10.0)
    except Exception as exc:
        logging.getLogger(__name__).warning("Kafka producer failed to start: %s", exc)

    # 構建 Redis 參數字典，只包含非 None 的值
    redis_kwargs = {
        "host": settings.redis_host,
        "port": settings.redis_port,
        "db": settings.redis_db,
        "decode_responses": True,
    }
    if settings.redis_password is not None:
        redis_kwargs["password"] = settings.redis_password

    redis_client = Redis(**redis_kwargs)

    app.state.settings = settings
    app.state.http_client = client
    app.state.kafka_producer = kafka_producer
    app.state.redis_client = redis_client

    _logger = logging.getLogger(__name__)
    _logger.info(
        "Gateway startup complete",
        extra={
            "max_connections": settings.max_connections,
            "max_keepalive_connections": settings.max_keepalive_connections,
            "batch_upload_concurrency": settings.batch_upload_concurrency,
        }
    )

    try:
        yield
    finally:
        await client.aclose()
        await kafka_producer.stop()
        await redis_client.close()
        _logger.info("Gateway shutdown complete")


_v = get_settings().api_version

app = FastAPI(title=f"Raptor FastAPI Gateway", lifespan=lifespan)
Instrumentator(
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
).instrument(app).expose(app, include_in_schema=False)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Starlette's add_middleware() makes the *last* call the *outermost* layer (it
# runs first on the way in) — so registration order below is innermost-first,
# the reverse of the actual request-processing order. Actual order (outer to
# inner): SlowAPI -> RequestLogging -> LegacyApiAlias -> Guardrail -> router.
# (A prior version of this file added them in request-processing order under
# the mistaken assumption that add_middleware() appends outermost-last; that
# put GuardrailMiddleware outside LegacyApiAliasMiddleware, so it only ever
# matched the canonical /api/{v}/* path and silently skipped every
# /api/0.3/{chat,a2a}/* request — confirmed via a live probe: /guard/check/input
# was called for /api/0.4/chat/completions but never for the identical request
# against /api/0.3/chat/completions.)
#
# Guardrail intercept (V04-10) — must end up innermost, after
# LegacyApiAliasMiddleware's rewrite, so it sees the canonical path and covers
# /api/0.3/{chat,a2a}/* for free. GR_ENABLED=false (default) short-circuits to
# a single boolean check. Added first so later add_middleware() calls wrap
# *outside* it.
app.add_middleware(GuardrailMiddleware)
# /api/0.3/{sso,chat,search,asset,processing,a2a}/* compatibility alias —
# rewrites to the canonical /api/{_v}/* path before routing, then tags the
# response Deprecation: true + Link: rel="successor-version". Must run before
# GuardrailMiddleware (added above) and after RequestLogging/SlowAPI (added
# below), so those still see/log the original legacy request path while
# Guardrail sees the rewritten one.
app.add_middleware(LegacyApiAliasMiddleware, canonical_prefix=f"/api/{_v}/")
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SlowAPIMiddleware)

# /.well-known/agent.json and /.well-known/agent-card — root-level (A2A spec)
app.include_router(well_known_router)

# SSO login — no auth required
app.include_router(sso, prefix=f"/api/{_v}/sso")

# Protected routers — JWT verified + permission checked via module 06 /auth/permission
# dependencies=[Depends(get_current_user)] is set at the router-mount level (not just
# per-route) so any future route added to these files is denied-by-default: auth runs
# before the handler body even for routes that forget to declare it themselves.
app.include_router(search_router, prefix=f"/api/{_v}/search", dependencies=[Depends(get_current_user)])
app.include_router(video_search_router, prefix=f"/api/{_v}/search/video_search", dependencies=[Depends(get_current_user)])
app.include_router(asset_router, prefix=f"/api/{_v}/asset", dependencies=[Depends(get_current_user)])
app.include_router(processing_router, prefix=f"/api/{_v}/processing", dependencies=[Depends(get_current_user)])
app.include_router(chat_router, prefix=f"/api/{_v}/chat", dependencies=[Depends(get_current_user)])
app.include_router(training_proxy_router, prefix=f"/api/{_v}/training", dependencies=[Depends(get_current_user)])
app.include_router(benchmark_proxy_router, prefix=f"/api/{_v}/benchmark", dependencies=[Depends(get_current_user)])
app.include_router(aiml_router, prefix=f"/api/{_v}/aiml", dependencies=[Depends(get_current_user)])
app.include_router(media_sync_router, prefix=f"/api/{_v}/sync", dependencies=[Depends(get_current_user)])
app.include_router(agent_protocol_router, prefix=f"/api/{_v}/a2a", dependencies=[Depends(get_current_user)])
app.include_router(memory_router, prefix=f"/api/{_v}/memory", dependencies=[Depends(get_current_user)])
app.include_router(personal_db_router, prefix="/api/0.4/personal-db",
                   dependencies=[Depends(get_current_user)])

# MCP proxy — no gateway-level JWT gate; the MCP server extracts and forwards
# the bearer token itself (see app/routers/mcp.py docstring)
app.include_router(mcp_router, prefix=f"/api/{_v}/mcp")


@app.get("/", tags=["root"])
def root():
    """Health check endpoint for liveness probes."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["health"])
@app.get(f"/api/{_v}/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Health check endpoint for liveness probes."""
    return {"version": _v, "status": "ok"}


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "rate_limit_exceeded", "detail": str(exc.detail)},
    )


@app.exception_handler(httpx.HTTPStatusError)
async def downstream_http_status_handler(_: Request, exc: httpx.HTTPStatusError):
    return JSONResponse(
        status_code=exc.response.status_code,
        content={"error": "downstream_error", "detail": exc.response.text},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred"},
    )
