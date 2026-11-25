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
from app.routers import (
    database,
    face_index,
    gdpr,
    graph,
    graph_index,
    identity,
    index,
    publish,
    search,
    temporal,
)
from app.services import embedder, reranker, source_expiry, temporal_jobs
from app.services.arcadedb_client import ArcadeDBClient
from app.services.audit import close_pool
from app.services.kafka_consumer import drain_graph_extraction_tasks, run_consumer
from app.services.moderation_flag_consumer import run_moderation_flag_consumer

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
            except Exception as exc:  # noqa: BLE001 -- broker down, etc. — log, don't crash the API
                logger.error("[main] Kafka consumer stopped: %s", exc)

        task = asyncio.create_task(_runner())
        logger.info("[main] Kafka consumer started (topic=%s)", settings.kafka_topic)
    else:
        logger.info("[main] Kafka consumer disabled (PD_KAFKA_ENABLED=0)")

    moderation_task = None
    if settings.moderation_flag_consumer_enabled:
        moderation_client = ArcadeDBClient()

        async def _moderation_runner():
            try:
                await run_moderation_flag_consumer(moderation_client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- broker down, etc. — log, don't crash the API
                logger.error("[main] moderation-flag consumer stopped: %s", exc)

        moderation_task = asyncio.create_task(_moderation_runner())
        logger.info("[main] moderation-flag consumer started (topic=%s)", settings.moderation_flag_topic)
    else:
        logger.info("[main] moderation-flag consumer disabled (PD_MODERATION_FLAG_CONSUMER_ENABLED=0)")

    # P12: the durable temporal job worker. Runs with or without the consumer -- it also resumes
    # jobs a previous process left unfinished.
    if settings.temporal_jobs_enabled:
        temporal_jobs.runner = temporal_jobs.TemporalJobRunner(ArcadeDBClient())
        await temporal_jobs.runner.start()
    # P21: archives inline sources whose 7 days have passed
    sweeper = source_expiry.ExpirySweeper(ArcadeDBClient()) if settings.temporal_expiry_sweep_seconds > 0 else None
    if sweeper:
        await sweeper.start()

    async def _warm(name: str, get_model):
        try:
            await get_model()
            logger.info("[main] %s model warmed", name)
        except Exception as exc:  # noqa: BLE001 -- download/load failure — log, don't crash the API
            logger.error("[main] %s model warm-up failed: %s", name, exc)

    asyncio.create_task(_warm("embedder", embedder._get_model))
    asyncio.create_task(_warm("reranker", reranker._get_model))
    yield
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if moderation_task:
        moderation_task.cancel()
        try:
            await moderation_task
        except asyncio.CancelledError:
            pass
    # Video graph extraction (PA-7) runs as background tasks spawned outside
    # this lifespan's own task tracking (see kafka_consumer.py) -- cancel them
    # too, or a redeploy leaves them orphaned mid-run with no one to await them.
    await drain_graph_extraction_tasks()
    if sweeper:
        await sweeper.stop()
    if temporal_jobs.runner:
        await temporal_jobs.runner.stop()       # an interrupted job stays running and is resumed at the next start
        temporal_jobs.runner = None
    await close_pool()


app = FastAPI(
    title="Raptor 0.4 — Personal DB Service",
    version="0.5.0",
    description="Per-user ArcadeDB database lifecycle and search.",
    lifespan=lifespan,
)

app.include_router(database.router)
app.include_router(index.router)         # PA-4 — document/moment indexing
app.include_router(face_index.router)    # people layer — face trace/group write (dev plan v1.4 §4.5)
app.include_router(identity.router)      # people layer — Identity API (dev plan v1.4 §4.5)
app.include_router(graph_index.router)   # PA-5 — entity/relationship/temporal-fact indexing
app.include_router(search.router)        # PA-6 — hybrid/vector/bm25 search
app.include_router(graph.search_router)  # PA-7 — graph/tkg/graphrag search
app.include_router(graph.graph_router)   # PA-7 — entity list/detail + raw query
app.include_router(publish.router)       # PA-8 — publish index-request to Kafka (test/demo)
app.include_router(gdpr.router)          # GD-1 — cross-store erasure, called only by Module 33
app.include_router(temporal.router)      # P12 — temporal job status


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["admin"])
async def health():
    return {"status": "ok"}
