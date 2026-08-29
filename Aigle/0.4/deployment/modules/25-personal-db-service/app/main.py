"""Raptor 0.4 — Personal DB Service (Module 25).

Per-user ArcadeDB databases: lifecycle (init / stats / delete) now;
indexing + hybrid/vector/BM25/graph search to follow (Sprint 2).
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.routers import database, index, graph_index, search, graph, publish
from app.services.arcadedb_client import ArcadeDBClient
from app.services.audit import close_pool
from app.services.kafka_consumer import drain_graph_extraction_tasks, run_consumer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("personal_db.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run the PA-8 Kafka consumer alongside the HTTP server (PD_KAFKA_ENABLED)."""
    task = None
    if settings.kafka_enabled:
        client = ArcadeDBClient()

        async def _runner():
            try:
                await run_consumer(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # broker down, etc. — log, don't crash the API
                logger.error("[main] Kafka consumer stopped: %s", exc)

        task = asyncio.create_task(_runner())
        logger.info("[main] Kafka consumer started (topic=%s)", settings.kafka_topic)
    else:
        logger.info("[main] Kafka consumer disabled (PD_KAFKA_ENABLED=0)")
    yield
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    # Video graph extraction (PA-7) runs as background tasks spawned outside
    # this lifespan's own task tracking (see kafka_consumer.py) -- cancel them
    # too, or a redeploy leaves them orphaned mid-run with no one to await them.
    await drain_graph_extraction_tasks()
    await close_pool()


app = FastAPI(
    title="Raptor 0.4 — Personal DB Service",
    version="0.4.0",
    description="Per-user ArcadeDB database lifecycle and search.",
    lifespan=lifespan,
)

app.include_router(database.router)
app.include_router(index.router)         # PA-4 — document/moment indexing
app.include_router(graph_index.router)   # PA-5 — entity/relationship/temporal-fact indexing
app.include_router(search.router)        # PA-6 — hybrid/vector/bm25 search
app.include_router(graph.search_router)  # PA-7 — graph/tkg/graphrag search
app.include_router(graph.graph_router)   # PA-7 — entity list/detail + raw query
app.include_router(publish.router)       # PA-8 — publish index-request to Kafka (test/demo)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["admin"])
async def health():
    return {"status": "ok"}
