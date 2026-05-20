"""Chat Service — FastAPI application entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from redis.asyncio import Redis

from .core.config import settings
from .routers.chat import router as chat_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Chat service starting up...")
    logger.info(f"LLM base URL: {settings.LLM_BASE_URL}")
    logger.info(f"Hybrid search URL: {settings.HYBRID_SEARCH_URL}")

    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.REQUEST_TIMEOUT),
        limits=httpx.Limits(
            max_connections=settings.MAX_CONNECTIONS,
            max_keepalive_connections=settings.MAX_KEEPALIVE_CONNECTIONS,
        ),
    )

    redis_kwargs: dict = {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "db": settings.REDIS_DB,
        "decode_responses": True,
    }
    if settings.REDIS_PASSWORD:
        redis_kwargs["password"] = settings.REDIS_PASSWORD

    app.state.redis_client = Redis(**redis_kwargs)

    logger.info("Chat service started successfully")
    try:
        yield
    finally:
        await app.state.http_client.aclose()
        await app.state.redis_client.close()
        logger.info("Chat service shutdown complete")


app = FastAPI(
    title="Chat Service",
    description="RAG chat service with LangGraph, hybrid search, and Redis memory",
    version="0.3.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {"service": "chat-service", "version": "0.3.0", "status": "running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "service": "chat-service"}
