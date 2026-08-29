from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from redis.asyncio import Redis

from core.config import settings
from routers.compact import router as compact_router
from routers.management import router as management_router
from routers.multimedia import router as multimedia_router
from routers.search import router as search_router
from routers.sessions import router as sessions_router
from routers.users import router as users_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis_kwargs: dict = {
        "host": settings.redis_host,
        "port": settings.redis_port,
        "db": settings.redis_db,
        "decode_responses": True,
    }
    if settings.redis_password:
        redis_kwargs["password"] = settings.redis_password

    app.state.redis_client = Redis(**redis_kwargs)

    from services.memvid_store import _get_embedder
    await asyncio.to_thread(_get_embedder)
    logger.info("Memory Service started — Redis connected, embedder loaded")
    try:
        yield
    finally:
        await app.state.redis_client.aclose()
        logger.info("Memory Service shutdown — Redis closed")


app = FastAPI(title="Memory Service", version="0.4.0", lifespan=lifespan)

app.include_router(sessions_router)
app.include_router(users_router)
app.include_router(multimedia_router)
app.include_router(search_router)
app.include_router(management_router)
app.include_router(compact_router)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    from importlib.metadata import PackageNotFoundError, version
    try:
        memvid_version = version("memvid-sdk")
    except PackageNotFoundError:
        memvid_version = "unknown"
    return {"status": "ok", "memvid_version": memvid_version}
