import traceback
import logging
import asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.api.v1 import ingest, search
from app.core.dependencies import (
    get_embedding_service,
    get_opensearch_service,
    get_qdrant_service,
    get_reranker_service
)
from app.core.config import settings
from app.core.exceptions import AppBaseException


# 配置 Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API starting up...")

    # 1. 載入模型 (使用 to_thread 避免阻塞主執行緒)
    def _log_task_error(task: asyncio.Task):
        if not task.cancelled() and task.exception():
            logger.error(f"Background model loading failed: {task.exception()}")

    embedder = get_embedding_service()
    t1 = asyncio.create_task(asyncio.to_thread(embedder.load_model))
    t1.add_done_callback(_log_task_error)

    reranker = get_reranker_service()
    t2 = asyncio.create_task(asyncio.to_thread(reranker.load_model))
    t2.add_done_callback(_log_task_error)
    
    # 2. 初始化資源
    try:
        await asyncio.gather(
            get_opensearch_service().initialize(),
            get_qdrant_service().initialize()
        )
        logger.info("✅ OpenSearch and Qdrant initialized.")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")

    yield
    
    # 3. 關閉資源
    logger.info("🛑 Shutting down...")
    await get_opensearch_service().client.close()
    await get_qdrant_service().client.close()

app = FastAPI(
    title="Hybrid Search API",
    description="OpenSearch + Qdrant Hybrid Search API",
    version="1.0.0",
    lifespan=lifespan
)
Instrumentator().instrument(app).expose(app)

@app.exception_handler(AppBaseException)
async def app_exception_handler(request, exc: AppBaseException):
    logger.error(f"App error: {exc.detail}", exc_info=True)

    response_content = {"error": exc.detail}

    if settings.DEBUG:
        response_content["traceback"] = traceback.format_exc()

    return JSONResponse(
        status_code=exc.status_code,
        content=response_content
    )

# 註冊路由
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])

@app.get("/api/v1/health/live", tags=["System"])
async def liveness():
    return {"status": "alive"}

@app.get("/api/v1/health/ready", tags=["System"])
async def readiness():
    embedder = get_embedding_service()
    reranker = get_reranker_service()
    os_service = get_opensearch_service()
    qdrant_service = get_qdrant_service()

    os_ok = await os_service.client.ping()
    qdrant_ok = await qdrant_service.readiness()

    ready = all([
        embedder.is_ready,
        reranker.is_ready,
        os_ok,
        qdrant_ok,
    ])

    return {
        "status": "ready" if ready else "not_ready",
        "embedding_model_ready": embedder.is_ready,
        "reranker_model_ready": reranker.is_ready,
        "opensearch_connected": os_ok,
        "qdrant_connected": qdrant_ok,
    }
