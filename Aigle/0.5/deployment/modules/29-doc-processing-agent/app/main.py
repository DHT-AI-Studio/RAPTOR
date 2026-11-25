"""DA-09a — DocAgent FastAPI application entry point."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.adapters.doc12_adapter import Doc12Adapter
from app.agent.orchestrator import DocAgentOrchestrator
from app.core.config import get_settings
from app.core.doc_collection import ensure_doc_collection
from app.routers.docagent import router as docagent_router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources on startup; clean up on shutdown."""
    logger.info("DocAgent starting up (port=%d)", settings.port)

    # Shared async HTTP client (reused by adapter + LLM tools)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.llm_timeout),
        limits=httpx.Limits(max_connections=settings.max_concurrent_requests * 2),
    )

    # Module 12 adapter
    adapter = Doc12Adapter(settings=settings, http_client=http_client)

    # Agent orchestrator (passes the same http_client to all LLM tools)
    orchestrator = DocAgentOrchestrator(settings=settings, http_client=http_client)

    # Bootstrap Qdrant collection (idempotent — safe to call every startup)
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=settings.qdrant_url)
        ensure_doc_collection(qc, settings.qdrant_doc_collection, settings.qdrant_vector_size)
        logger.info("Qdrant collection %r ready", settings.qdrant_doc_collection)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant not reachable at startup (search will fail): %s", exc)

    app.state.settings    = settings
    app.state.http_client = http_client
    app.state.doc12_adapter = adapter
    app.state.orchestrator  = orchestrator

    logger.info("DocAgent startup complete")
    try:
        yield
    finally:
        await http_client.aclose()
        logger.info("DocAgent shutdown complete")


app = FastAPI(
    title="DocAgent",
    version="1.0.0",
    description="DA-09a — Document AI Agent: classify, extract, summarize, compare, qa, parse_table, search",
    lifespan=lifespan,
)

app.include_router(docagent_router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Liveness probe."""
    m12_ok = await Doc12Adapter(settings=settings).health()
    return {
        "status": "ok",
        "service": "doc-processing-agent",
        "version": "1.0.0",
        "dependencies": {
            "module12_sync_api": "ok" if m12_ok else "unavailable",
        },
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )
