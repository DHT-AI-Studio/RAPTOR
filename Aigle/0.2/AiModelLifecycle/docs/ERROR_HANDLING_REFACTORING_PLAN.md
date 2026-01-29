# AiModelLifecycle 錯誤處理重構方案

## 問題分析

### Code Review 發現的主要問題

**位置**: `src/inference/manager.py:190-202`

**問題描述**:
```python
except Exception as e:
    self.stats['failed_inferences'] += 1
    logger.error(f"推理失敗: {e}")
    
    return {
        'success': False,
        'error': str(e),
        ...
    }
```

**存在的問題**:
1. ❌ 捕獲所有異常並返回錯誤字典而不是拋出異常
2. ❌ API 層無法通過檢查字典來區分成功響應和錯誤響應
3. ❌ 違反了 "錯誤應該是例外" 原則
4. ❌ 導致 HTTP 狀態碼處理不一致

### 當前架構分析

**調用鏈**:
```
API Layer (inference_api.py)
    ↓
Manager Layer (manager.py) 
    ↓
Router Layer (router.py)
    ↓
Executor Layer (executor.py)
    ↓
Engine Layer (ollama.py / transformers.py)
```

**當前錯誤處理方式**:
- ✅ API 層: 正確使用 HTTPException
- ❌ Manager 層: 返回 success:False 字典
- ✅ Router/Executor 層: 拋出異常
- ✅ Engine 層: 拋出異常

---

## 修改方案

### 1. 統一異常體系設計

創建分層的自定義異常類型，便於 API 層區分和處理：

```python
# src/inference/exceptions.py (新建文件)

class InferenceError(Exception):
    """推理相關異常基類"""
    pass

# 4xx 客戶端錯誤
class ValidationError(InferenceError):
    """輸入驗證錯誤 - 對應 HTTP 400"""
    pass

class UnsupportedTaskError(ValidationError):
    """不支持的任務類型 - 對應 HTTP 400"""
    pass

class ModelNotFoundError(InferenceError):
    """模型未找到錯誤 - 對應 HTTP 404"""
    pass

class ResourceNotFoundError(InferenceError):
    """資源未找到（通用）- 對應 HTTP 404"""
    pass

# 5xx 服務器錯誤
class ModelLoadError(InferenceError):
    """模型加載失敗 - 對應 HTTP 500"""
    pass

class InferenceExecutionError(InferenceError):
    """推理執行失敗 - 對應 HTTP 500"""
    pass

class EngineError(InferenceError):
    """引擎相關錯誤 - 對應 HTTP 500"""
    pass

# 503 服務不可用
class ResourceExhaustedError(InferenceError):
    """資源耗盡（GPU/內存）- 對應 HTTP 503"""
    pass
```

### 2. Manager 層修改

#### 修改前 (manager.py:140-202):
```python
def infer(self, ...):
    start_time = time.time()
    self.stats['total_inferences'] += 1
    
    try:
        # ... 推理邏輯 ...
        self.stats['successful_inferences'] += 1
        
        return {
            'success': True,
            'result': result,
            ...
        }
        
    except Exception as e:
        self.stats['failed_inferences'] += 1
        logger.error(f"推理失敗: {e}")
        
        return {
            'success': False,
            'error': str(e),
            ...
        }
```

#### 修改後:
```python
def infer(self, ...):
    """統一推理接口
    
    Raises:
        ValidationError: 參數驗證失敗
        UnsupportedTaskError: 不支持的任務類型
        ModelNotFoundError: 模型未找到
        InferenceExecutionError: 推理執行失敗
        ResourceExhaustedError: 資源耗盡
    """
    start_time = time.time()
    
    # 執行緒安全地更新統計計數
    with self._stats_lock:
        self.stats['total_inferences'] += 1
    
    try:
        logger.info(f"開始推理 - 任務: {task}, 引擎: {engine}, 模型: {model_name}")
        
        # 參數驗證 - 拋出 ValidationError 或 UnsupportedTaskError
        self._validate_parameters(task, engine, model_name, data)
        
        # 任務路由 - 可能拋出 UnsupportedTaskError
        executor = self.router.route(task, engine, model_name)
        
        # 執行推理 - 可能拋出 ModelNotFoundError, InferenceExecutionError 等
        result = executor.execute(model_name, data, options or {})
        
        # 執行緒安全地更新成功統計
        with self._stats_lock:
            self.stats['successful_inferences'] += 1
        
        processing_time = time.time() - start_time
        logger.info(f"推理完成 - 用時: {processing_time:.2f}秒")
        
        # 只返回成功結果，不返回 success 標誌
        return {
            'result': result,
            'task': task,
            'engine': engine,
            'model_name': model_name,
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
    except (ValidationError, UnsupportedTaskError, ModelNotFoundError, 
            InferenceError) as e:
        # 已知的推理異常，直接向上傳播
        # 執行緒安全地更新失敗統計
        with self._stats_lock:
            self.stats['failed_inferences'] += 1
        
        logger.error(f"推理失敗: {type(e).__name__}: {e}")
        raise
        
    except Exception as e:
        # 未預期的異常，包裝為 InferenceExecutionError
        with self._stats_lock:
            self.stats['failed_inferences'] += 1
        
        logger.error(f"推理執行時發生未預期錯誤: {e}", exc_info=True)
        raise InferenceExecutionError(f"推理執行失敗: {str(e)}") from e
```

### 3. _validate_parameters 修改

```python
def _validate_parameters(self, task: str, engine: str, model_name: str, data: Dict) -> None:
    """驗證推理參數
    
    Raises:
        ValidationError: 參數無效
        UnsupportedTaskError: 不支持的任務或引擎組合
    """
    # 檢查參數完整性
    if not all([task, engine, model_name]):
        raise ValidationError("task, engine, model_name 不能為空")
    
    # 檢查任務類型
    supported_tasks = {...}
    if task not in supported_tasks:
        raise UnsupportedTaskError(
            f"不支持的任務類型: {task}，支持的任務: {supported_tasks}"
        )
    
    # 檢查引擎類型
    supported_engines = {'ollama', 'transformers'}
    if engine not in supported_engines:
        raise UnsupportedTaskError(
            f"不支持的引擎類型: {engine}，支持的引擎: {supported_engines}"
        )
    
    # 檢查引擎與任務的兼容性
    ollama_tasks = {'text-generation', 'text-generation-ollama'}
    if engine == 'ollama' and task not in ollama_tasks:
        raise UnsupportedTaskError(
            f"Ollama 引擎僅支持 {ollama_tasks} 任務，當前任務: {task}"
        )
    
    # 檢查輸入數據
    if not isinstance(data, dict) or not data:
        raise ValidationError("data 必須是非空字典")
    
    # 根據任務類型檢查必需的數據字段
    required_fields = {...}
    
    if task in required_fields:
        missing_fields = [field for field in required_fields[task] if field not in data]
        if missing_fields:
            raise ValidationError(
                f"任務 {task} 缺少必需的數據字段: {missing_fields}"
            )
```

### 4. API 層修改

#### 修改前 (inference_api.py:77-112):
```python
@router.post("/infer")
def unified_inference(request: InferenceRequest):
    try:
        result = inference_manager.infer(...)
        result['api_processing_time'] = ...
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"參數錯誤: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理執行失敗: {str(e)}")
```

#### 修改後:
```python
from ..inference.exceptions import (
    ValidationError, UnsupportedTaskError, ModelNotFoundError,
    InferenceError, InferenceExecutionError, ResourceExhaustedError
)

@router.post("/infer")
def unified_inference(request: InferenceRequest):
    """統一推理接口
    
    Raises:
        HTTPException: 各種錯誤情況，包含適當的 HTTP 狀態碼
    """
    try:
        start_time = time.time()
        logger.info(f"收到推理請求: {request.task} - {request.engine} - {request.model_name}")
        
        # 執行推理 - 可能拋出各種自定義異常
        result = inference_manager.infer(
            task=request.task,
            engine=request.engine,
            model_name=request.model_name,
            data=request.data,
            options=request.options
        )
        
        # 添加API層的元數據
        result['api_processing_time'] = time.time() - start_time
        result['request_id'] = id(request)
        result['success'] = True  # API 層添加成功標誌
        
        logger.info(f"推理完成: {request.task} - 用時 {result['processing_time']:.2f}秒")
        return result
    
    # 4xx 客戶端錯誤
    except (ValidationError, UnsupportedTaskError) as e:
        logger.warning(f"請求驗證失敗: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_type": type(e).__name__,
                "message": str(e),
                "task": request.task if request else None,
                "engine": request.engine if request else None
            }
        )
    
    except ModelNotFoundError as e:
        logger.warning(f"模型未找到: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_type": "ModelNotFoundError",
                "message": str(e),
                "model_name": request.model_name if request else None
            }
        )
    
    # 503 服務不可用
    except ResourceExhaustedError as e:
        logger.error(f"資源耗盡: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "ResourceExhaustedError",
                "message": str(e),
                "retry_after": 60  # 建議60秒後重試
            }
        )
    
    # 5xx 服務器錯誤
    except InferenceError as e:
        logger.error(f"推理錯誤: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": type(e).__name__,
                "message": str(e)
            }
        )
    
    except Exception as e:
        logger.error(f"未預期的錯誤: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "UnexpectedError",
                "message": "推理服務發生未預期錯誤，請聯繫管理員"
            }
        )
```

### 5. Router 層修改

```python
# src/inference/router.py

from .exceptions import UnsupportedTaskError, ValidationError

def route(self, task: str, engine: str, model_name: str) -> ModelExecutor:
    """路由到相應的執行器
    
    Raises:
        UnsupportedTaskError: 不支持的任務或引擎組合
        ValidationError: 參數無效
    """
    try:
        logger.debug(f"路由任務: {task} -> {engine} -> {model_name}")
        
        # 檢查任務類型是否支持
        if task not in self._task_engine_mapping:
            raise UnsupportedTaskError(f"不支持的任務類型: {task}")
        
        # 檢查引擎類型是否支持該任務
        task_engines = self._task_engine_mapping[task]
        if engine not in task_engines:
            supported_engines = list(task_engines.keys())
            raise UnsupportedTaskError(
                f"任務 '{task}' 不支持引擎 '{engine}'，支持的引擎: {supported_engines}"
            )
        
        # ... 其餘邏輯保持不變 ...
        
    except (UnsupportedTaskError, ValidationError):
        # 已知異常，直接向上傳播
        raise
    except Exception as e:
        logger.error(f"路由失敗: {e}")
        # 將未知異常包裝為推理錯誤
        raise InferenceError(f"任務路由失敗: {str(e)}") from e
```

### 6. Executor 層修改

```python
# src/inference/executor.py

from .exceptions import ModelLoadError, InferenceExecutionError

def execute(self, model_name: str, data: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """執行推理
    
    Raises:
        ModelLoadError: 模型加載失敗
        InferenceExecutionError: 推理執行失敗
    """
    try:
        start_time = time.time()
        logger.debug(f"開始執行推理: {model_name}")
        
        # 步驟 1: 確保模型已加載
        model = self._get_or_load_model(model_name)
        
        # 步驟 2-4: 預處理、推理、後處理
        # ... 保持不變 ...
        
        return final_result
        
    except ModelLoadError:
        # 模型加載錯誤，直接向上傳播
        raise
    except Exception as e:
        logger.error(f"推理執行失敗: {e}")
        raise InferenceExecutionError(f"推理執行失敗: {str(e)}") from e

def _get_or_load_model(self, model_name: str) -> Any:
    """獲取或加載模型
    
    Raises:
        ModelLoadError: 模型加載失敗
    """
    if model_name not in self._loaded_models:
        logger.info(f"加載模型: {model_name}")
        
        try:
            model = self.engine.load_model(model_name)
            self._loaded_models[model_name] = model
            logger.info(f"模型加載成功: {model_name}")
            
        except Exception as e:
            logger.error(f"模型加載失敗: {model_name}, 錯誤: {e}")
            raise ModelLoadError(f"模型加載失敗: {model_name}") from e
    else:
        logger.debug(f"重用已加載的模型: {model_name}")
        
    return self._loaded_models[model_name]
```

---

## 修改優點

### 1. 符合 RESTful 原則
- ✅ 使用標準 HTTP 狀態碼表示錯誤
- ✅ 錯誤響應格式統一且信息豐富

### 2. 提高可維護性
- ✅ 異常層次清晰，便於理解和擴展
- ✅ 錯誤處理邏輯集中在 API 層
- ✅ 各層職責明確

### 3. 改善用戶體驗
- ✅ 客戶端可以根據狀態碼判斷錯誤類型
- ✅ 錯誤消息詳細且有助於調試
- ✅ 可以針對不同錯誤實施不同的重試策略

### 4. 便於監控和調試
- ✅ 異常類型明確，便於日誌分析
- ✅ 錯誤統計更準確
- ✅ 便於設置監控告警

---

## 修改影響評估

### 受影響的文件

1. **新建文件**:
   - `src/inference/exceptions.py` - 自定義異常定義

2. **需要修改**:
   - `src/inference/manager.py` - 移除錯誤字典返回
   - `src/inference/router.py` - 使用自定義異常
   - `src/inference/executor.py` - 使用自定義異常
   - `src/api/inference_api.py` - 完善異常處理

3. **可能需要更新**:
   - `src/inference/engines/ollama.py` - 確保正確拋出異常
   - `src/inference/engines/transformers.py` - 確保正確拋出異常
   - 各種模型處理器 - 確保正確拋出異常

### 向後兼容性

❌ **不兼容**: 此修改會破壞向後兼容性

**原因**:
- API 返回格式變化（移除頂層 `success` 字段）
- 錯誤響應從 200 OK + success:false 變為適當的 4xx/5xx 狀態碼

**遷移建議**:
1. 如果有外部客戶端依賴當前 API，考慮：
   - 新增 v2 API 端點
   - 保留舊端點但標記為 deprecated
   - 提供遷移文檔和示例

2. 內部使用：
   - 更新所有調用代碼
   - 修改測試用例

---

## 測試計劃

### 1. 單元測試
- [ ] 測試各種異常類型的創建和傳播
- [ ] 測試 manager 層異常處理
- [ ] 測試 API 層異常轉換

### 2. 集成測試
- [ ] 測試完整的推理流程（成功案例）
- [ ] 測試各種錯誤場景（400, 404, 500, 503）
- [ ] 測試異常信息的完整性

### 3. 負載測試
- [ ] 驗證並發情況下異常處理的正確性
- [ ] 驗證統計數據的準確性

---

## 實施步驟

1. ✅ 創建此修改方案文檔
2. ⏳ 創建 `exceptions.py` 文件
3. ⏳ 修改 `manager.py`
4. ⏳ 修改 `router.py` 和 `executor.py`
5. ⏳ 修改 `inference_api.py`
6. ⏳ 更新引擎和模型處理器（如需要）
7. ⏳ 編寫/更新測試用例
8. ⏳ 測試驗證
9. ⏳ 更新文檔

---

## 總結

此修改方案遵循以下原則：
1. **異常應該是例外的** - 錯誤通過異常而非返回值傳播
2. **單一職責** - API 層負責 HTTP 響應，業務層負責業務邏輯
3. **清晰的錯誤層次** - 不同類型的錯誤有不同的異常類
4. **符合 RESTful 標準** - 使用標準 HTTP 狀態碼

通過這些改進，代碼將更加健壯、可維護，並且提供更好的用戶體驗。
