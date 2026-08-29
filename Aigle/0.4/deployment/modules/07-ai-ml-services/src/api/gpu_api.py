# src/api/gpu_api.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ..core.gpu_manager import gpu_manager

router = APIRouter(
    prefix="/gpu",
    tags=["GPU管理 (GPU Management)"]
)

# ===== Pydantic 模型定義 =====

class GPUAllocationRequest(BaseModel):
    strategy: str = Field("least_used", description="分配策略 (least_used, round_robin, first_fit)")
    required_memory_mb: Optional[int] = Field(None, description="所需最小記憶體 (MB)")

class GPUReservationRequest(BaseModel):
    device_id: int = Field(..., description="GPU 設備 ID")
    duration_seconds: Optional[int] = Field(3600, description="保留時間（秒）")
    purpose: Optional[str] = Field(None, description="保留目的")

# ===== GPU 狀態查詢 API =====

@router.get("/status", summary="獲取 GPU 總體狀態")
def get_gpu_status():
    """獲取所有 GPU 設備的基本狀態 (使用 gpu_manager.get_gpu_info)。"""
    try:
        info_list = gpu_manager.get_gpu_info()
        # 統計聚合
        total_memory_gb = sum(i.get("total_memory_gb", 0) or 0 for i in info_list)
        used_memory_gb = sum(i.get("used_memory_gb", 0) or 0 for i in info_list)
        return {
            "success": True,
            "gpu_available": gpu_manager.is_gpu_available(),
            "device_count": gpu_manager.get_device_count(),
            "total_memory_gb": round(total_memory_gb, 2),
            "total_used_memory_gb": round(used_memory_gb, 2),
            "gpu_info": info_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取 GPU 狀態失敗: {e}")

@router.get("/devices", summary="列出所有 GPU 設備")
def list_gpu_devices():
    """列出系統中所有可用的 GPU 設備 (使用單一 get_gpu_info)。"""
    try:
        info_list = gpu_manager.get_gpu_info()
        return {
            "success": True,
            "devices": info_list,
            "total_count": len(info_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取設備列表失敗: {e}")

@router.get("/devices/{device_id}", summary="獲取特定 GPU 設備詳情")
def get_device_details(device_id: int):
    """獲取指定 GPU 設備的詳細信息 (從 get_gpu_info 過濾)。"""
    try:
        info_list = gpu_manager.get_gpu_info()
        device = next((d for d in info_list if d.get("id") == device_id), None)
        if device is None:
            raise HTTPException(status_code=404, detail=f"GPU 設備 {device_id} 不存在")
        return {"success": True, "device": device}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取設備詳情失敗: {e}")

# ===== GPU 記憶體管理 API =====

@router.get("/memory", summary="獲取所有 GPU 記憶體狀態")
def get_memory_status():
    """從 get_gpu_info 擷取記憶體資訊。"""
    try:
        info_list = gpu_manager.get_gpu_info()
        memory_status = [
            {
                "device_id": d.get("id"),
                "total_memory_gb": d.get("total_memory_gb"),
                "used_memory_gb": d.get("used_memory_gb"),
                "free_memory_gb": d.get("free_memory_gb")
            } for d in info_list
        ]
        return {"success": True, "memory_status": memory_status, "total_devices": len(info_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取記憶體狀態失敗: {e}")

@router.post("/memory/clear", summary="清理 GPU 快取")
def clear_gpu_cache():
    """清理所有 GPU 的快取 (Torch CUDA cache)。"""
    try:
        gpu_manager.clear_cache()
        return {"success": True, "message": "GPU 快取已清理"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理快取失敗: {e}")

# ===== GPU 分配和保留 API =====

@router.post("/allocate", summary="智能分配 GPU 設備")
def allocate_gpu_device(request: GPUAllocationRequest):
    """根據策略 (least_used / round_robin / first_fit) 選擇一個 GPU。"""
    try:
        strategy_map = {"least_used": "least_used", "round_robin": "round_robin", "first_fit": "first_fit"}
        strategy = strategy_map.get(request.strategy, "least_used")
        device_id = gpu_manager.select_device(strategy=strategy)
        if device_id is None:
            raise HTTPException(status_code=503, detail="當前沒有可用的 GPU 設備")
        info_list = gpu_manager.get_gpu_info()
        device_info = next((d for d in info_list if d.get("id") == device_id), {})
        return {"success": True, "allocated_device_id": device_id, "strategy": strategy, "device_info": device_info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPU 分配失敗: {e}")

@router.post("/reserve/{device_id}", summary="保留特定 GPU 設備")
def reserve_gpu_device(device_id: int, request: GPUReservationRequest):
    """示意接口: 當前 GPUManager 未實作保留，僅回傳請求資訊。"""
    try:
        if device_id >= gpu_manager.get_device_count():
            raise HTTPException(status_code=404, detail=f"GPU 設備 {device_id} 不存在")
        return {"success": True, "message": f"GPU {device_id} 保留 (模擬) {request.duration_seconds}s", "device_id": device_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設備保留失敗: {e}")

# ===== GPU 監控 API =====

@router.get("/monitoring/start", summary="啟動 GPU 監控")
def start_gpu_monitoring():
    """啟動 GPU 後台監控"""
    try:
        gpu_manager.start_monitoring()
        return {
            "success": True,
            "message": "GPU 監控已啟動"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"啟動監控失敗: {str(e)}"
        )

@router.get("/monitoring/stop", summary="停止 GPU 監控")
def stop_gpu_monitoring():
    """停止 GPU 後台監控"""
    try:
        gpu_manager.stop_monitoring()
        return {
            "success": True,
            "message": "GPU 監控已停止"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止監控失敗: {str(e)}"
        )

@router.get("/monitoring/status", summary="獲取監控狀態")
def get_monitoring_status():
    """獲取 GPU 監控服務狀態"""
    try:
        is_monitoring = gpu_manager.is_monitoring_active()
        return {
            "success": True,
            "monitoring_active": is_monitoring,
            "monitoring_interval": getattr(gpu_manager, '_monitoring_interval', 'unknown')
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取監控狀態失敗: {str(e)}"
        )

# ===== GPU 性能和診斷 API =====

@router.get("/benchmark", summary="GPU 性能基準 (簡化)")
def gpu_benchmark():
    """使用記憶體大小做為簡化性能評分。"""
    try:
        info_list = gpu_manager.get_gpu_info()
        results = []
        for d in info_list:
            total = d.get("total_memory_gb") or 0
            results.append({
                "device_id": d.get("id"),
                "device_name": d.get("name"),
                "memory_gb": total,
                "performance_score": total,
            })
        return {"success": True, "benchmark_results": results, "test_type": "memory_based"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"性能測試失敗: {e}")

@router.get("/health", summary="GPU 健康檢查")
def gpu_health_check():
    """簡化健康檢查：僅回傳是否可用以及列出設備。"""
    try:
        info_list = gpu_manager.get_gpu_info()
        return {"success": True, "gpu_available": gpu_manager.is_gpu_available(), "device_count": len(info_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康檢查失敗: {e}")