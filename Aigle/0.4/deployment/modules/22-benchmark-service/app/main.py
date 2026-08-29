"""Benchmark Service — Module 22.

User-defined marking schemas, pipeline execution, multi-method scoring,
LLM-as-judge, and comparable run history. FastAPI entry point.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.core.db import db
from app.routers.optimize import router as optimize_router
from app.routers.runs import router as runs_router
from app.routers.schemas import router as schemas_router
from app.services.autotune import orchestrator

logger = logging.getLogger(__name__)


def _configure_logging(log_level: str) -> None:
    tz_tw = timezone(timedelta(hours=8))

    class _TZFmt(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=tz_tw)
            return dt.strftime(datefmt) if datefmt else dt.isoformat()

    fmt = _TZFmt("%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logging.root.handlers = [handler]
    logging.root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configure_logging(settings.log_level)
    await db.connect()
    # Resume any experiments left 'running' by a crash (AUTOTUNE §B7).
    resumed = await orchestrator.resume_running()
    if resumed:
        logger.info("Resumed %d interrupted experiment(s)", resumed)
    logger.info("Benchmark Service started")
    yield
    await db.disconnect()
    logger.info("Benchmark Service stopped")


_TAGS = [
    {"name": "Schemas", "description": "上傳、查詢、刪除評分 schema（marking schema）"},
    {"name": "Runs", "description": "提交 benchmark run、查詢狀態、歷史、比較兩次 run"},
    {"name": "Auto-Tune", "description": "自然語言驅動的自動微調實驗（訓練→評分迴圈）"},
    {"name": "Health", "description": "服務健康狀態"},
]

app = FastAPI(title="Benchmark Service", version="0.3.0", lifespan=lifespan, openapi_tags=_TAGS)
Instrumentator().instrument(app).expose(app)

app.include_router(schemas_router, prefix="/api/v1")
app.include_router(runs_router, prefix="/api/v1")
app.include_router(optimize_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok"}
