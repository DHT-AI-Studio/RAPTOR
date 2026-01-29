"""FastAPI application entry point for the API gateway."""
from __future__ import annotations

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
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from redis.asyncio import Redis  # type: ignore[import]
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import RedirectResponse
from fastapi import Depends

# from app.routers.auth import router as auth_router
from app.routers.processing import router as processing_router
from app.routers.asset import router as upload_router
from app.routers.search import router as search_router
from app.routers.chat import router as chat_router
from app.core.config import ServiceRegistry, Settings, get_settings
from app.middlewares.logging import RequestLoggingMiddleware
from app.services.user_service import UserService
from app.routers.vision_analyze_Qwen import router as TRE_router
# ---------- import keycloak package ----------
from app.my_keycloak.app.routers import login as sso
from app.my_keycloak.app.routers import creat_user
from app.my_keycloak.app.dependencies.security import APIsecurity


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

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Console formatter (UTC+8)
    console_formatter = TZFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz=tz_tw,
    )

    # JSON formatter for file logging (structured logs)
    json_formatter = TZJSONFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "log_level"},
        tz=tz_tw,
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "gateway.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_formatter)

    # Apply handlers to root
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)
    logging.root.setLevel(log_level)

    # Align uvicorn/fastapi loggers with the same handlers and level
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

    # Initialize HTTP client with optimized connection pool
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

    # Initialize Kafka producer
    kafka_producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
    )
    # await kafka_producer.start()  # 0.2 code 因為是在本地端開發所以暫不使用 kafka 
 
    # Initialize Redis client
    redis_client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )

    # Initialize shared services
    user_service = UserService(
        settings.gateway_db_dsn,
        min_size=settings.gateway_db_pool_min_size,
        max_size=settings.gateway_db_pool_max_size,
    )
    await user_service.initialise()

    # Store shared resources in app state
    app.state.settings = settings
    # app.state.service_registry = registry
    app.state.http_client = client
    app.state.kafka_producer = kafka_producer
    app.state.redis_client = redis_client
    app.state.user_service = user_service

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
        # Clean up resources
        await client.aclose()
        await kafka_producer.stop()
        await redis_client.close()
        await user_service.close()
        _logger.info("Gateway shutdown complete")

version = 0.2   # raptor version

app = FastAPI(title=f"Raptor FastAPI Gateway", lifespan=lifespan)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Middlewares
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# API Routers (order matters for path matching)
app.include_router(sso.router, prefix=f"/api/{version}/SSO")
app.include_router(creat_user.router, tags=["Administration"], prefix=f"/api/{version}/admin")
# app.include_router(testpath.router, prefix=f"/api/{version}/test", tags=["security test"], dependencies=[Depends(APIsecurity.authenticate)])
# app.include_router(auth_router, prefix=f"/api/{version}/auth")
app.include_router(search_router, prefix=f"/api/{version}/search", dependencies=[Depends(APIsecurity.authenticate)])
app.include_router(upload_router, prefix=f"/api/{version}/asset", dependencies=[Depends(APIsecurity.authenticate)])
app.include_router(processing_router, prefix=f"/api/{version}/processing", dependencies=[Depends(APIsecurity.authenticate)])
app.include_router(chat_router, prefix=f"/api/{version}/chat", dependencies=[Depends(APIsecurity.authenticate)])
app.include_router(TRE_router, tags=["video analyze"], prefix=f"/api/{version}/analyze", dependencies=[Depends(APIsecurity.authenticate)])

@app.get("/", tags=["root"])
def root():
    """Health check endpoint for liveness probes."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Health check endpoint for liveness probes.

    **Usage:**
    ```python
    url = f"{self.base_url}/health"
    response = requests.get(url)
    response.raise_for_status()
    result = response.json()
    ```
    """
    return {"status": "ok"}


# Exception Handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "rate_limit_exceeded", "detail": str(exc.detail)},
    )


@app.exception_handler(httpx.HTTPStatusError)
async def downstream_http_status_handler(_: Request, exc: httpx.HTTPStatusError):
    """Handle errors from downstream services."""
    return JSONResponse(
        status_code=exc.response.status_code,
        content={"error": "downstream_error", "detail": exc.response.text},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    """Handle any other unhandled exceptions."""
    logging.getLogger(__name__).exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred"},
    )

