from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Header, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from raptor_audit import AUDIT_EVENTS_DDL
from redis.asyncio import Redis

from app.core.config import settings
from app.routers.access import audit_router
from app.routers.access import router as access_router
from app.routers.admin import router as admin_router
from app.routers.consent import router as consent_router
from app.routers.erasure import router as erasure_router
from app.routers.erasure import start_erasure
from app.routers.export import router as export_router
from app.routers.jobs import router as jobs_router
from app.routers.records import router as records_router
from app.routers.reports import router as reports_router
from app.routers.retention import router as retention_router
from app.routers.rights import router as rights_router
from app.services.consent_store import ConsentStore
from app.services.dsar_assembler import DSARAssembler
from app.services.erasure_orchestrator import ErasureOrchestrator
from app.services.export_orchestrator import ExportOrchestrator
from app.services.kafka_topic_retention import apply_topic_retention
from app.services.legal_hold import LegalHoldService
from app.services.legal_hold_state import LEGAL_HOLDS_DDL
from app.services.retention_engine import RetentionEngine
from app.services.rights_service import RightsService


def require_internal_token(x_internal_token: str = Header(None, alias="X-Internal-Token")) -> None:
    if x_internal_token != settings.internal_token:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Internal-Token")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("privacy_compliance")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pool = await asyncpg.create_pool(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        database=settings.postgres_db, min_size=1, max_size=5,
    )
    # V05-10 / VIE01-364: this service's CF-4 audit_events table (idempotent).
    async with pool.acquire() as conn:
        await conn.execute(AUDIT_EVENTS_DDL)
        # PRIV-X4 / VIE01-358: every erasure checks gdpr_legal_holds (idempotent).
        await conn.execute(LEGAL_HOLDS_DDL)

    redis_kwargs: dict = {"host": settings.redis_host, "port": settings.redis_port, "decode_responses": True}
    if settings.redis_password:
        redis_kwargs["password"] = settings.redis_password
    redis = Redis(**redis_kwargs)

    kafka_producer: AIOKafkaProducer | None = None
    try:
        kafka_producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers.split(","))
        await kafka_producer.start()
    except KafkaError as exc:
        logger.warning("Kafka producer failed to start (tombstone publish will be skipped): %s", exc)
        kafka_producer = None

    app.state.orchestrator = ErasureOrchestrator(pool=pool, redis=redis, kafka_producer=kafka_producer)
    app.state.export_orchestrator = ExportOrchestrator(pool=pool)
    app.state.retention_engine = RetentionEngine(pool=pool)
    app.state.dsar_assembler = DSARAssembler(pool=pool)
    app.state.consent_store = ConsentStore(pool=pool, redis=redis, orchestrator=app.state.orchestrator)
    # PRIV-X3 / VIE01-357: rectification, restriction and objection.
    app.state.rights_service = RightsService(pool=pool, consent_store=app.state.consent_store)
    # PRIV-X4 / VIE01-358: legal holds; their expiry starts deferred erasures.
    app.state.legal_hold_service = LegalHoldService(
        pool=pool, consent_store=app.state.consent_store, orchestrator=app.state.orchestrator,
        start_erasure=lambda job_id: start_erasure(app, job_id),
    )
    app.state.background_tasks = set()
    # GD-7: the audit-log routes in routers/access.py query gdpr_access_log directly.
    app.state.pool = pool

    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(settings.export_cleanup_interval_seconds)
            try:
                await app.state.export_orchestrator.cleanup_expired()
            except Exception as exc:  # noqa: BLE001 -- asyncpg/S3 failures mixed; background sweep must keep running on any failure, already logged
                logger.error("export cleanup sweep failed: %s", exc)
            # GD-4 / VIE01-249: same hourly sweep clears expired DSAR job
            # results (the stored report copy, not the job row).
            try:
                await app.state.dsar_assembler.cleanup_expired()
            except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
                logger.error("dsar cleanup sweep failed: %s", exc)

    cleanup_task = asyncio.create_task(_cleanup_loop())

    # GD-3 / VIE01-248: ticket's own spec -- "Background scheduler runs
    # daily at 02:00 UTC". Same AsyncIOScheduler + CronTrigger pattern
    # Module 04's endpoints.py already uses for auto-archive/auto-destroy.
    scheduler = AsyncIOScheduler()
    scheduler.start()
    scheduler.add_job(
        app.state.retention_engine.enforce,
        CronTrigger(hour=settings.retention_hour_utc, minute=settings.retention_minute_utc, timezone="UTC"),
        id="gdpr_retention_enforce",
        name=f"Daily GDPR retention enforcement at {settings.retention_hour_utc:02d}:{settings.retention_minute_utc:02d} UTC",
    )
    # V05-10 / VIE01-364: CF-4 audit events, each kept its own retention_days.
    scheduler.add_job(
        app.state.retention_engine.purge_audit_events,
        CronTrigger(hour=settings.retention_hour_utc, minute=settings.retention_minute_utc, timezone="UTC"),
        id="audit_events_purge",
        name="Daily CF-4 audit_events purge (per-event retention_days)",
    )
    # PRIV-X4 / VIE01-358: the ticket's daily hold-expiry job.
    scheduler.add_job(
        app.state.legal_hold_service.release_expired,
        CronTrigger(hour=settings.retention_hour_utc, minute=settings.retention_minute_utc, timezone="UTC"),
        id="gdpr_legal_hold_expiry",
        name="Daily legal hold expiry: release ended holds, run the erasures they deferred",
    )
    app.state.scheduler = scheduler
    # Fire-and-forget: Kafka's own connection attempt can take up to
    # request_timeout_ms (default 40s) to fail when the broker is
    # unreachable -- must never delay this service becoming healthy.
    kafka_retention_task = asyncio.create_task(apply_topic_retention())

    logger.info("Privacy Compliance Service started")
    try:
        yield
    finally:
        kafka_retention_task.cancel()
        scheduler.shutdown()
        cleanup_task.cancel()
        await app.state.orchestrator.close()
        await app.state.export_orchestrator.close()
        await app.state.retention_engine.close()
        await app.state.dsar_assembler.close()
        await app.state.consent_store.close()
        if kafka_producer is not None:
            await kafka_producer.stop()
        await redis.aclose()
        await pool.close()
        logger.info("Privacy Compliance Service shutdown")


app = FastAPI(title="Privacy Compliance Service", version="0.5.0", lifespan=lifespan)

app.include_router(erasure_router, dependencies=[Depends(require_internal_token)])
app.include_router(export_router, dependencies=[Depends(require_internal_token)])
app.include_router(retention_router, dependencies=[Depends(require_internal_token)])
app.include_router(access_router, dependencies=[Depends(require_internal_token)])
app.include_router(consent_router, dependencies=[Depends(require_internal_token)])
app.include_router(audit_router, dependencies=[Depends(require_internal_token)])
app.include_router(rights_router, dependencies=[Depends(require_internal_token)])
app.include_router(records_router, dependencies=[Depends(require_internal_token)])
# V05-10 / VIE01-364: operator job list.
app.include_router(jobs_router, dependencies=[Depends(require_internal_token)])
# PRIV-X4 / VIE01-358: legal holds.
app.include_router(admin_router, dependencies=[Depends(require_internal_token)])
# CM-7 / VIE01-410: compliance audit report.
app.include_router(reports_router, dependencies=[Depends(require_internal_token)])

Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health() -> dict:
    """GD-9: extended health check — includes scheduler state and last retention run."""
    scheduler = getattr(app.state, "scheduler", None)
    scheduler_state = "running" if (scheduler is not None and scheduler.running) else "stopped"

    retention_engine = getattr(app.state, "retention_engine", None)
    last_run = None
    if retention_engine is not None:
        raw = getattr(retention_engine, "last_run", None)
        if raw is not None:
            # RetentionEngine.last_run is a datetime; serialise to ISO 8601
            last_run = raw.isoformat() if hasattr(raw, "isoformat") else str(raw)

    return {
        "status": "ok",
        "scheduler": scheduler_state,
        "retention_last_run": last_run,
    }

