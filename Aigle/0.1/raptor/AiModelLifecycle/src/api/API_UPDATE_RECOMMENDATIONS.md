# Inference API 更新建議

**日期**: 2025-10-13  
**目標**: 根據重構後的 Inference 模組更新 API

---

## 執行摘要

✅ **好消息**: 當前 API 已經正確整合了重構後的 inference_manager，核心功能完整。

⚠️ **改進空間**: 建議添加一些進階功能以提升 API 的完整性和可用性。

---

## 快速檢查結果

### ✅ 已正確實現的功能

1. **統一推理端點** (`POST /inference/infer`)
   - ✅ 正確調用 `inference_manager.infer()`
   - ✅ 支持所有任務類型
   - ✅ 完整的參數驗證
   - ✅ 詳細的文檔

2. **健康檢查** (`GET /inference/health`)
   - ✅ 調用 `inference_manager.health_check()`
   - ✅ 返回組件狀態

3. **任務列表** (`GET /inference/supported-tasks`)
   - ✅ 調用 `inference_manager.get_supported_tasks()`

4. **統計信息** (`GET /inference/stats`)
   - ✅ 調用 `inference_manager.get_stats()`

5. **清理緩存** (`POST /inference/clear-cache`)
   - ✅ 調用 `inference_manager.clear_cache()`

6. **向下兼容** (`/infer_fixed`, `/infer_multimodal`)
   - ✅ 提供舊版端點兼容

7. **使用示例** (`GET /inference/examples`)
   - ✅ 提供完整示例

### ⚠️ 建議添加的功能

#### 高優先級

1. **詳細緩存管理**
   ```
   GET    /inference/cache/stats          - 詳細緩存統計
   GET    /inference/cache/models         - 緩存模型列表
   DELETE /inference/cache/models/{key}   - 移除特定模型
   PUT    /inference/cache/config         - 更新緩存配置
   ```

2. **批次推理**
   ```
   POST /inference/batch - 批次處理多個請求
   ```

3. **改進的錯誤處理**
   - 統一錯誤響應格式
   - 自定義異常處理器

#### 中優先級

4. **異步推理**
   ```
   POST /inference/async        - 提交異步任務
   GET  /inference/async/{id}   - 查詢任務狀態
   ```

5. **模型預加載**
   ```
   POST /inference/models/preload  - 預加載模型到緩存
   ```

#### 低優先級

6. **流式推理** (WebSocket)
7. **監控儀表板**

---

## 詳細建議

### 建議 1: 添加緩存管理端點 🔴 高優先級

**原因**: `ModelCache` 類提供了豐富的方法，但 API 僅暴露了基本的 `clear_cache()`

**可用方法**:
- `cache.get_stats()` - 詳細統計
- `cache.get_cached_models()` - 模型列表
- `cache.remove(model_key)` - 移除特定模型
- `cache.resize_cache(size)` - 調整緩存大小
- `cache.set_memory_limit(gb)` - 設置內存限制

**建議實現**:

```python
@router.get("/cache/stats", summary="獲取詳細緩存統計")
def get_detailed_cache_stats():
    """
    獲取詳細的緩存統計信息
    
    返回:
    - 緩存命中率
    - 內存使用情況
    - 模型訪問統計
    - 驅逐統計
    """
    try:
        cache_stats = inference_manager.cache.get_stats()
        return {
            "success": True,
            "stats": cache_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"獲取緩存統計失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"獲取緩存統計失敗: {str(e)}"
        )

@router.get("/cache/models", summary="獲取緩存模型列表")
def get_cached_model_list():
    """
    獲取當前緩存中的所有模型
    
    包含每個模型的:
    - 模型鍵值
    - 加載時間
    - 訪問次數
    - 最後訪問時間
    - 內存占用
    """
    try:
        cached_models = inference_manager.cache.get_cached_models()
        return {
            "success": True,
            "models": cached_models,
            "count": len(cached_models),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"獲取緩存模型列表失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"獲取緩存模型列表失敗: {str(e)}"
        )

@router.delete("/cache/models/{model_key}", summary="移除特定緩存模型")
def remove_specific_cached_model(model_key: str):
    """
    從緩存中移除特定模型
    
    Args:
        model_key: 模型鍵值（格式：engine:model_name）
        
    示例:
        DELETE /inference/cache/models/ollama:llama2-7b
    """
    try:
        success = inference_manager.cache.remove(model_key)
        if success:
            return {
                "success": True,
                "message": f"模型 '{model_key}' 已從緩存移除",
                "timestamp": time.time()
            }
        else:
            return {
                "success": False,
                "message": f"模型 '{model_key}' 不在緩存中",
                "timestamp": time.time()
            }
    except Exception as e:
        logger.error(f"移除緩存模型失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"移除緩存模型失敗: {str(e)}"
        )

@router.put("/cache/config", summary="更新緩存配置")
def update_cache_configuration(
    max_cache_size: Optional[int] = Query(None, ge=1, le=20, description="最大緩存模型數量"),
    max_memory_gb: Optional[float] = Query(None, ge=1.0, le=100.0, description="最大內存使用（GB）")
):
    """
    動態更新緩存配置
    
    可以調整:
    - max_cache_size: 緩存中保存的最大模型數量
    - max_memory_gb: 緩存可使用的最大內存
    
    注意:
    - 減小緩存大小可能觸發模型驅逐
    - 設置會立即生效
    """
    try:
        updates = []
        
        if max_cache_size is not None:
            inference_manager.cache.resize_cache(max_cache_size)
            updates.append(f"max_cache_size 更新為 {max_cache_size}")
        
        if max_memory_gb is not None:
            inference_manager.cache.set_memory_limit(max_memory_gb)
            updates.append(f"max_memory_gb 更新為 {max_memory_gb}")
        
        if not updates:
            raise ValueError("至少需要提供一個配置參數")
        
        return {
            "success": True,
            "message": "緩存配置已更新",
            "updates": updates,
            "current_config": {
                "max_cache_size": inference_manager.cache.max_cache_size,
                "max_memory_gb": inference_manager.cache.max_memory_bytes / (1024**3)
            },
            "timestamp": time.time()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新緩存配置失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新緩存配置失敗: {str(e)}"
        )
```

**影響**: 
- ✅ 提供更精細的緩存管理
- ✅ 有助於性能調優
- ✅ 不影響現有功能

---

### 建議 2: 添加批次推理端點 🔴 高優先級

**原因**: 提高吞吐量，減少網絡開銷

**建議實現**:

```python
from typing import List
from pydantic import Field, validator

class BatchInferenceRequest(BaseModel):
    """批次推理請求模型"""
    requests: List[InferenceRequest] = Field(
        ...,
        description="推理請求列表",
        min_items=1,
        max_items=100
    )
    parallel: bool = Field(
        default=False,
        description="是否並行執行（需要足夠資源）"
    )
    
    @validator('requests')
    def validate_requests(cls, v):
        if len(v) > 100:
            raise ValueError("批次大小不能超過 100")
        return v

@router.post("/batch", summary="批次推理")
def batch_inference_endpoint(batch_request: BatchInferenceRequest):
    """
    批次推理接口
    
    一次性提交多個推理請求，可選擇順序或並行執行。
    
    **限制**:
    - 最多 100 個請求/批次
    - 並行模式需要足夠的資源
    
    **建議**:
    - 相同模型的請求分組可提高緩存效率
    - 大批次建議使用順序模式避免資源耗盡
    
    **示例**:
    ```json
    {
      "requests": [
        {
          "task": "text-generation",
          "engine": "ollama",
          "model_name": "llama2-7b",
          "data": {"inputs": "Hello"},
          "options": {}
        },
        {
          "task": "text-generation",
          "engine": "ollama",
          "model_name": "llama2-7b",
          "data": {"inputs": "World"},
          "options": {}
        }
      ],
      "parallel": false
    }
    ```
    """
    try:
        start_time = time.time()
        results = []
        
        # 順序執行（安全且可預測）
        for idx, req in enumerate(batch_request.requests):
            try:
                result = inference_manager.infer(
                    task=req.task,
                    engine=req.engine,
                    model_name=req.model_name,
                    data=req.data,
                    options=req.options or {}
                )
                results.append({
                    "index": idx,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"批次請求 {idx} 失敗: {e}")
                results.append({
                    "index": idx,
                    "success": False,
                    "error": str(e),
                    "task": req.task,
                    "model_name": req.model_name
                })
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.get('success', False))
        failed_count = len(results) - success_count
        
        return {
            "success": True,
            "summary": {
                "total_requests": len(batch_request.requests),
                "successful": success_count,
                "failed": failed_count,
                "success_rate": success_count / len(batch_request.requests) if results else 0
            },
            "results": results,
            "total_processing_time": total_time,
            "average_time_per_request": total_time / len(results) if results else 0,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"批次推理執行失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"批次推理執行失敗: {str(e)}"
        )
```

**影響**:
- ✅ 提高批量處理效率
- ✅ 減少網絡往返次數
- ✅ 更好的緩存利用

---

### 建議 3: 改進錯誤處理 🟡 中優先級

**原因**: 提供更友好和結構化的錯誤信息

**建議實現**:

```python
from fastapi.responses import JSONResponse
from ..inference.manager import (
    InferenceError, 
    ModelNotFoundError, 
    UnsupportedTaskError
)

# 定義錯誤碼
class ErrorCode:
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    TIMEOUT = "TIMEOUT"

# 統一錯誤響應模型
class ErrorResponse(BaseModel):
    """統一錯誤響應"""
    success: bool = False
    error: Dict[str, Any] = Field(
        description="錯誤詳情",
        example={
            "type": "ModelNotFoundError",
            "message": "模型未找到",
            "code": "MODEL_NOT_FOUND",
            "details": {}
        }
    )
    timestamp: float

# 添加異常處理器
@app.exception_handler(UnsupportedTaskError)
async def unsupported_task_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "type": "UnsupportedTaskError",
                "message": str(exc),
                "code": ErrorCode.UNSUPPORTED_TASK,
                "details": {
                    "hint": "請檢查任務類型和引擎組合是否支持"
                }
            },
            "timestamp": time.time()
        }
    )

@app.exception_handler(ModelNotFoundError)
async def model_not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": {
                "type": "ModelNotFoundError",
                "message": str(exc),
                "code": ErrorCode.MODEL_NOT_FOUND,
                "details": {
                    "hint": "請確認模型已在 MLflow 中註冊"
                }
            },
            "timestamp": time.time()
        }
    )

# 在 unified_inference 中改進錯誤處理
@router.post("/infer", summary="統一推理接口")
def unified_inference(request: InferenceRequest):
    try:
        # ... 現有代碼 ...
        
    except UnsupportedTaskError as e:
        logger.error(f"不支持的任務: {e}")
        raise
    except ModelNotFoundError as e:
        logger.error(f"模型未找到: {e}")
        raise
    except ValueError as e:
        logger.error(f"參數錯誤: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "type": "ValueError",
                "message": str(e),
                "code": ErrorCode.INVALID_INPUT
            }
        )
    except Exception as e:
        logger.error(f"推理執行失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": type(e).__name__,
                "message": str(e),
                "code": ErrorCode.INFERENCE_FAILED
            }
        )
```

**影響**:
- ✅ 更清晰的錯誤信息
- ✅ 更好的調試體驗
- ✅ 標準化的錯誤格式

---

## 實施計劃

### Phase 1: 立即實施（本周）

```
□ 添加緩存管理端點
  □ GET /cache/stats
  □ GET /cache/models
  □ DELETE /cache/models/{key}
  □ PUT /cache/config

□ 添加批次推理端點
  □ POST /batch

□ 測試新端點
```

### Phase 2: 短期實施（本月）

```
□ 改進錯誤處理
  □ 定義錯誤碼
  □ 實現異常處理器
  □ 更新現有端點

□ 完善文檔
  □ 更新 API 文檔
  □ 添加示例
```

### Phase 3: 長期規劃（下季度）

```
□ 異步推理支持
□ 模型預加載
□ WebSocket 流式推理
□ 監控儀表板
```

---

## 測試建議

### 測試緩存管理端點

```bash
# 1. 獲取緩存統計
curl http://localhost:8009/inference/cache/stats

# 2. 查看緩存模型
curl http://localhost:8009/inference/cache/models

# 3. 移除特定模型
curl -X DELETE http://localhost:8009/inference/cache/models/ollama:llama2-7b

# 4. 更新緩存配置
curl -X PUT "http://localhost:8009/inference/cache/config?max_cache_size=10&max_memory_gb=16"
```

### 測試批次推理

```bash
curl -X POST http://localhost:8009/inference/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "llama2-7b",
        "data": {"inputs": "Hello"},
        "options": {}
      },
      {
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "llama2-7b",
        "data": {"inputs": "World"},
        "options": {}
      }
    ],
    "parallel": false
  }'
```

---

## 結論

**當前狀態**: ✅ API 與 Inference 模組對齊良好，核心功能完整

**建議行動**: 
1. 🔴 添加緩存管理端點（高優先級）
2. 🔴 添加批次推理（高優先級）
3. 🟡 改進錯誤處理（中優先級）

**預期收益**:
- 更強大的緩存管理能力
- 更高的推理吞吐量
- 更好的開發者體驗

**風險評估**: ⚠️ 低風險
- 所有建議都是新增功能
- 不影響現有 API 的穩定性
- 向下兼容

---

**審查者**: _____________  
**批准日期**: _____________  
**實施負責人**: _____________
