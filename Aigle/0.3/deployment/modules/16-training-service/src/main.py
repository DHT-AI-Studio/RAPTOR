"""Main FastAPI application for Training Service.

This is the entry point for the RAPTOR Training Service, which provides
REST API endpoints for submitting and managing AI model training jobs.
"""

import logging
import multiprocessing
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Response

from .api.training_api import router as training_router
from .core.gpu_manager import get_gpu_manager


# Configure multiprocessing start method for CUDA safety
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method('forkserver', force=True)
    logging.info("Multiprocessing start method set to 'forkserver' for CUDA safety")

def setup_logging(level: str = "INFO"):
    """統一應用程式日誌配置"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 移除所有現有的 handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 配置根記錄器
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # 減少第三方庫的日誌噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.INFO)

    return root_logger

# 在應用啟動時呼叫
logger = setup_logging("INFO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    gpu_manager = get_gpu_manager()
    if gpu_manager.is_gpu_available():
        logger.info(f"RAPTOR Training Service started. {gpu_manager.get_device_count()} GPU(s) available.")
    else:
        logger.warning("No CUDA devices found. Training service will run in CPU mode.")

    yield

# Create FastAPI application
app = FastAPI(
    title="RAPTOR Training Service",
    description="AI Model Training and Fine-tuning Service for Aigle Platform. "
                "Provides API endpoints for submitting, tracking, and managing model training jobs.",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Register routes
app.include_router(training_router, prefix="/api/v1")


@app.get("/", tags=["System Status"])
def read_root():
    """Root endpoint returning service information."""
    return {
        "service": "RAPTOR Training Service",
        "version": "0.2.0",
        "status": "running",
        "description": "AI Model Training and Fine-tuning Service for Aigle Platform",
        "api_endpoints": {
            "training": "/api/v1/training"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json"
        }
    }


@app.get("/health", tags=["System Status"])
async def health_check(response: Response):
    """Health check endpoint."""
    try:
        from datetime import datetime
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "gpu_manager": False,
                "training_job_manager": False,
            },
            "api_routes": {
                "training_api": True,
            }
        }
        
        # 檢查 GPU manager
        try:
            health_status["services"]["gpu_manager"] = get_gpu_manager().is_gpu_available()
        except Exception as e:
            logger.warning(f"GPU manager health check failed: {e}")
        
        # 檢查 training job manager
        try:
            from .core.training_job_manager import get_job_manager
            job_manager = get_job_manager()
            health_status["services"]["training_job_manager"] = job_manager is not None
        except Exception as e:
            logger.warning(f"Job manager health check failed: {e}")
        
        # 確定整體健康狀態
        all_services_healthy = all(health_status["services"].values())
        if not all_services_healthy:
            health_status["status"] = "degraded"
            response.status_code = 200  # degraded 仍返回 200
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        response.status_code = 503
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Health check failed"
        }



@app.get("/api/info", tags=["System Status"])
def get_api_info():
    """Get detailed API information and endpoint list."""
    return {
        "api_version": "0.2.0",
        "service_name": "RAPTOR Training Service",
        "available_endpoints": {
            "training": {
                "prefix": "/api/v1/training",
                "description": "Model training job submission, status tracking, cancellation and deletion",
                "key_endpoints": [
                    "POST /api/v1/training/submit - Submit training job",
                    "GET /api/v1/training/status/{job_id} - Query job status",
                    "GET /api/v1/training/list - List all jobs",
                    "POST /api/v1/training/cancel/{job_id} - Cancel job",
                    "DELETE /api/v1/training/delete/{job_id} - Delete job records"
                ]
            }
        },
        "features": [
            "Training job submission with LoRA fine-tuning",
            "Real-time training progress tracking",
            "GPU resource scheduling and management",
            "MLflow experiment tracking integration",
            "Redis-based job state persistence",
            "Graceful job cancellation",
            "Multi-GPU distributed training support"
        ]
    }
