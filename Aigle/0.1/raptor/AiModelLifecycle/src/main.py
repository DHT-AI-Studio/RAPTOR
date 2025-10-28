# src/main.py
from fastapi import FastAPI
from .api import models_api, datasets_api, gpu_api, config_api, inference_api
from .core.gpu_manager import gpu_manager
from .core.model_manager import model_manager  # 確保 model_manager 被初始化

app = FastAPI(
    title="AI 模型生命週期管理平台",
    description="一個用於管理 AI 模型從訓練到部署的綜合性 MLOps 平台，提供完整的模型、數據集、推理和資源管理功能。",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 引入所有 API 路由
app.include_router(models_api.router)
app.include_router(datasets_api.router)
app.include_router(inference_api.router)
app.include_router(gpu_api.router)
app.include_router(config_api.router)

@app.get("/", tags=["系統狀態 (System Status)"])
def read_root():
    """系統根端點，返回平台基本信息和狀態"""
    try:
        return {
            "message": "歡迎使用 AI 模型生命週期管理 API v2.0",
            "platform": "AI Model Lifecycle Management Platform",
            "version": "2.0.0",
            "status": "running",
            "gpu_available": gpu_manager.is_gpu_available(),
            "gpu_count": gpu_manager.get_device_count(),
            "api_endpoints": {
                "models": "/models",
                "datasets": "/datasets", 
                "inference": "/inference",
                "gpu": "/gpu",
                "config": "/config",
                "inference_engine": "/inference_engine"
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
            "version": "2.0.0"
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
                "inference_manager": False,
                "config_manager": False
            },
            "api_routes": {
                "models_api": True,
                "datasets_api": True,
                "inference_api": True,
                "gpu_api": True,
                "config_api": True,
                "inference_engine_api": True
            }
        }
        
        # 檢查 GPU 管理器
        try:
            health_status["services"]["gpu_manager"] = gpu_manager.is_gpu_available() is not None
        except:
            pass
        
        # 檢查模型管理器
        try:
            from .core.model_manager import model_manager
            health_status["services"]["model_manager"] = model_manager is not None
        except:
            pass
        
        # 檢查數據集管理器
        try:
            from .core.dataset_manager import dataset_manager
            health_status["services"]["dataset_manager"] = dataset_manager is not None
        except:
            pass
        
        # 檢查推理管理器
        try:
            from .inference.manager import inference_manager as local_model_manager
            health_status["services"]["inference_manager"] = local_model_manager is not None
        except:
            pass
        
        # 檢查配置管理器
        try:
            from .core.config import config
            health_status["services"]["config_manager"] = config is not None
        except:
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
    """獲取API詳細信息和端點列表"""
    return {
        "api_version": "2.0.0",
        "platform_name": "AI Model Lifecycle Management Platform",
        "available_endpoints": {
            "models": {
                "prefix": "/models",
                "description": "模型下載、上傳、註冊、版本管理",
                "key_endpoints": [
                    "POST /models/download - 下載模型",
                    "POST /models/upload_to_lakefs - 上傳到lakeFS", 
                    "POST /models/register_from_lakefs - 從lakeFS註冊到MLflow",
                    "GET /models/registered_in_mlflow - 列出已註冊模型",
                    "POST /models/batch_download - 批量下載",
                    "GET /models/stats - 模型統計",
                    "GET /models/health - 服務健康檢查"
                ]
            },
            "datasets": {
                "prefix": "/datasets",
                "description": "數據集上傳、下載、註冊、版本管理",
                "key_endpoints": [
                    "GET /datasets/remote - 列出遠程數據集",
                    "GET /datasets/local - 列出本地數據集",
                    "POST /datasets/upload - 上傳數據集",
                    "POST /datasets/download - 下載數據集",
                    "POST /datasets/register - 註冊到MLflow",
                    "POST /datasets/batch_upload - 批量上傳",
                    "GET /datasets/stats - 數據集統計"
                ]
            },
            "inference": {
                "prefix": "/inference", 
                "description": "模型推理、選擇、VRAM管理",
                "key_endpoints": [
                    "POST /inference/infer - 執行推理",
                    "POST /inference/batch_infer - 批量推理",
                    "POST /inference/select_model - 選擇最佳模型",
                    "GET /inference/available_models/{task} - 獲取可用模型",
                    "POST /inference/estimate_vram - VRAM估算",
                    "GET /inference/gpu_status - GPU狀態",
                    "GET /inference/supported_tasks - 支持的任務"
                ]
            },
            "gpu": {
                "prefix": "/gpu",
                "description": "GPU資源管理、監控、分配",
                "key_endpoints": [
                    "GET /gpu/status - GPU狀態",
                    "GET /gpu/devices - 列出GPU設備",
                    "POST /gpu/allocate - 智能分配GPU",
                    "POST /gpu/memory/clear - 清理GPU快取",
                    "GET /gpu/benchmark - 性能測試",
                    "GET /gpu/health - GPU健康檢查"
                ]
            },
            "config": {
                "prefix": "/config",
                "description": "系統配置管理和查詢",
                "key_endpoints": [
                    "GET /config/ - 獲取完整配置",
                    "GET /config/inference - 推理配置",
                    "GET /config/storage - 存儲配置",
                    "GET /config/services - 服務配置",
                    "GET /config/validate - 配置驗證",
                    "GET /config/stats - 配置統計"
                ]
            },
            "inference_engine": {
                "prefix": "/inference_engine",
                "description": "推理引擎直接管理和測試",
                "key_endpoints": [
                    "GET /inference_engine/available_engines - 可用引擎",
                    "POST /inference_engine/create_engine - 創建引擎實例",
                    "POST /inference_engine/test_engine - 測試引擎",
                    "POST /inference_engine/estimate_vram - VRAM估算",
                    "GET /inference_engine/engine_configs/{type} - 引擎配置"
                ]
            }
        },
        "features": [
            "模型生命週期管理",
            "數據集版本控制", 
            "智能推理調度",
            "GPU資源優化",
            "配置中心化管理",
            "多引擎推理支持",
            "批量操作支持",
            "健康監控",
            "性能統計"
        ]
    }

