# src/main.py
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from .api import models_api, datasets_api, gpu_api, config_api, inference_api, openai_api
from .core.gpu_manager import gpu_manager
from .core.model_manager import model_manager  # 確保 model_manager 被初始化

app = FastAPI(
    title="AI 模型生命週期管理平台",
    description="一個用於管理 AI 模型從訓練到部署的綜合性 MLOps 平台，提供完整的模型、數據集、推理和資源管理功能。",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
Instrumentator(
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
).instrument(app).expose(app)

# 引入所有 API 路由
app.include_router(models_api.router)
app.include_router(datasets_api.router)
app.include_router(inference_api.router)
app.include_router(openai_api.router)  # OpenAI 相容層（/v1/*，xinference 式）
app.include_router(gpu_api.router)
app.include_router(config_api.router)

@app.get("/", tags=["系統狀態 (System Status)"])
def read_root():
    """系統根端點，返回平台基本信息和狀態"""
    try:
        return {
            "message": "歡迎使用 AI 模型生命週期管理 API v3.0",
            "platform": "AI Model Lifecycle Management Platform",
            "version": "3.0.0",
            "status": "running",
            "gpu_available": gpu_manager.is_gpu_available(),
            "gpu_count": gpu_manager.get_device_count(),
            "api_endpoints": {
                "models": "/models",
                "datasets": "/datasets",
                "inference": "/inference",
                "gpu": "/gpu",
                "config": "/config",
            },
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json"
            }
        }
    except Exception as e:
        return {
            "message": "AI 模型生命週期管理 API",
            "status": "running_with_warnings",
            "error": str(e),
            "version": "3.0.0"
        }

@app.get("/health", tags=["系統狀態 (System Status)"])
def health_check():
    """系統健康檢查端點"""
    try:
        from datetime import datetime
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "gpu_manager": False,
                "model_manager": False,
                "dataset_manager": False,
                "inference_service": False,
                "config_manager": False
            },
            "api_routes": {
                "models_api": True,
                "datasets_api": True,
                "inference_api": True,
                "gpu_api": True,
                "config_api": True,
            }
        }

        # 檢查 GPU 管理器
        try:
            health_status["services"]["gpu_manager"] = gpu_manager.is_gpu_available() is not None
        except Exception:
            pass

        # 檢查模型管理器
        try:
            from .core.model_manager import model_manager
            health_status["services"]["model_manager"] = model_manager is not None
        except Exception:
            pass

        # 檢查數據集管理器
        try:
            from .core.dataset_manager import dataset_manager
            health_status["services"]["dataset_manager"] = dataset_manager is not None
        except Exception:
            pass

        # 檢查推理服務
        try:
            from .inference.service import inference_service
            health_status["services"]["inference_service"] = inference_service is not None
        except Exception:
            pass

        # 檢查配置管理器
        try:
            from .core.config import config
            health_status["services"]["config_manager"] = config is not None
        except Exception:
            pass

        # 確定總體健康狀態
        all_services_healthy = all(health_status["services"].values())
        if not all_services_healthy:
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "健康檢查失敗"
        }

@app.get("/api/info", tags=["系統狀態 (System Status)"])
def get_api_info():
    """獲取 API 詳細信息和端點列表"""
    return {
        "api_version": "3.0.0",
        "platform_name": "AI Model Lifecycle Management Platform",
        "available_endpoints": {
            "models": {
                "prefix": "/models",
                "description": "模型下載、上傳、註冊、版本管理",
                "key_endpoints": [
                    "POST /models/download - 下載模型",
                    "POST /models/upload_to_lakefs - 上傳到 lakeFS",
                    "POST /models/register_from_lakefs - 從 lakeFS 註冊到 MLflow",
                    "POST /models/register_ollama - 註冊本地 Ollama 模型到 MLflow",
                    "GET /models/registered_in_mlflow - 列出已註冊模型",
                    "POST /models/transition_stage - 切換階段",
                    "POST /models/batch_download - 批量下載",
                    "GET /models/stats - 模型統計",
                    "GET /models/health - 服務健康檢查",
                ],
            },
            "datasets": {
                "prefix": "/datasets",
                "description": "數據集上傳、下載、註冊、版本管理",
                "key_endpoints": [
                    "GET /datasets/remote - 列出遠程數據集",
                    "GET /datasets/local - 列出本地數據集",
                    "POST /datasets/upload - 上傳數據集",
                    "POST /datasets/download - 下載數據集",
                    "POST /datasets/register - 註冊到 MLflow",
                    "POST /datasets/batch_upload - 批量上傳",
                    "GET /datasets/stats - 數據集統計",
                ],
            },
            "openai_compat": {
                "prefix": "/v1",
                "description": "OpenAI 相容 API（xinference 式；任何 OpenAI SDK 可直接對接）",
                "key_endpoints": [
                    "GET /v1/models - 列出可用模型（OpenAI 格式）",
                    "POST /v1/chat/completions - Chat（LLM / VLM 通用）",
                    "POST /v1/completions - 傳統 prompt completion",
                    "POST /v1/audio/transcriptions - 語音轉文字（Whisper 式）",
                ],
            },
            "inference": {
                "prefix": "/inference",
                "description": "統一推理 API（spec-driven）",
                "key_endpoints": [
                    "POST /inference/infer - 統一推理入口",
                    "GET /inference/health - 推理服務健康狀態",
                    "GET /inference/stats - 推理統計",
                    "GET /inference/supported-tasks - 列出支援任務",
                    "GET /inference/loaded-models - 已載入模型",
                    "POST /inference/cache/clear - 清空所有 adapter cache",
                    "POST /inference/unload-model - 卸載指定模型",
                    "POST /inference/unload-all-models - 卸載全部模型",
                    "POST /inference/tts - TTS 合成（legacy，僅供 module 09）",
                ],
            },
            "gpu": {
                "prefix": "/gpu",
                "description": "GPU 資源管理、監控、分配",
                "key_endpoints": [
                    "GET /gpu/status - GPU 狀態",
                    "GET /gpu/devices - 列出 GPU 設備",
                    "POST /gpu/allocate - 智能分配 GPU",
                    "POST /gpu/memory/clear - 清理 GPU 快取",
                    "GET /gpu/benchmark - 性能測試",
                    "GET /gpu/health - GPU 健康檢查",
                ],
            },
            "config": {
                "prefix": "/config",
                "description": "系統配置管理和查詢",
                "key_endpoints": [
                    "GET /config/ - 獲取完整配置",
                    "GET /config/sections - 列出配置區段",
                    "GET /config/get/{section} - 獲取特定區段",
                ],
            },
        },
        "features": [
            "模型生命週期管理（MLflow Registry + LakeFS commit）",
            "Spec-driven 推理（runtime / model_class / handler 由 MLflow tag 決定）",
            "統一推理 API（所有 task 共用 /inference/infer；task 可省略）",
            "OpenAI 相容 API（/v1/*，任何 OpenAI SDK 直接對接已註冊模型）",
            "閒置自動卸載（ollama 式 keep_alive；idle 模型自動釋放 GPU）",
            "GPU 資源管理與 VRAM 估算",
            "資料集版本控制",
            "Adapter LRU 模型快取",
        ],
    }
