# Ollama MLflow 模型名稱映射修復說明

## 問題描述

在使用 Ollama 引擎進行推理時，如果 MLflow 中註冊的模型名稱與本地 Ollama 服務中的實際模型名稱不同，會導致推理失敗。

### 問題根源

1. **模型名稱轉換正確**：`ollama.py` 的 `load_model()` 方法能正確從 MLflow 獲取 `ollama_model_name` tag，並返回本地 Ollama 模型名稱
2. **緩存機制問題**：`router.py` 每次調用都創建新的 `ModelExecutor` 實例，導致模型緩存失效
3. **結果**：即使第一次正確轉換了模型名稱，第二次推理時又會創建新的 executor，模型需要重新加載

## 修復方案

### 1. Router 層增加 Executor 緩存

**修改文件**: `src/inference/router.py`

#### 變更 1: 添加 executor 緩存字典

```python
def __init__(self):
    """初始化任務路由器"""
    # 引擎實例緩存
    self._engines: Dict[str, Any] = {}
    
    # 執行器緩存: {(task, engine): executor}  # 新增
    self._executors: Dict[tuple, ModelExecutor] = {}  # 新增
    
    # 任務與引擎的映射關係
    self._task_engine_mapping = {
        ...
    }
```

#### 變更 2: 修改 route() 方法以重用 executor

```python
def route(self, task: str, engine: str, model_name: str) -> ModelExecutor:
    """路由到相應的執行器"""
    try:
        logger.debug(f"路由任務: {task} -> {engine} -> {model_name}")
        
        # ... 參數驗證 ...
        
        # 檢查是否已有緩存的執行器（新增）
        executor_key = (task, engine)
        if executor_key in self._executors:
            logger.debug(f"重用已緩存的執行器: {task} -> {engine}")
            return self._executors[executor_key]
        
        # 獲取或創建引擎實例
        engine_instance = self._get_or_create_engine(engine, task_engines[engine])
        
        # 獲取模型處理器
        model_handler = get_model_handler(task, model_name)
        
        # 創建並緩存執行器（修改）
        executor = ModelExecutor(engine_instance, model_handler)
        self._executors[executor_key] = executor  # 新增緩存
        
        logger.debug(f"成功創建並緩存執行器: {task} -> {engine}")
        return executor
        
    except Exception as e:
        logger.error(f"路由失敗: {e}")
        raise
```

#### 變更 3: 更新 clear_engines() 方法

```python
def clear_engines(self):
    """清理引擎和執行器緩存"""
    # 清理執行器中的模型（新增）
    for executor in self._executors.values():
        executor.clear_models()
    
    self._executors.clear()  # 新增
    self._engines.clear()
    logger.info("引擎和執行器緩存已清理")
```

## 如何在 MLflow 中正確註冊 Ollama 模型

### 方法 1: 使用 MLflow API 註冊

```python
import mlflow

# 註冊模型時添加 ollama_model_name tag
mlflow.register_model(
    model_uri="models:/my-model/1",
    name="llama2-7b-chat",  # MLflow 中的註冊名稱
    tags={
        "ollama_model_name": "llama2:7b",  # 本地 Ollama 實際模型名稱
        "engine": "ollama",
        "task": "text-generation"
    }
)
```

### 方法 2: 更新現有模型的 tags

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 為現有模型版本添加 tag
client.set_model_version_tag(
    name="llama2-7b-chat",
    version="1",
    key="ollama_model_name",
    value="llama2:7b"
)
```

### 方法 3: 通過 UI 手動添加

1. 打開 MLflow UI（通常是 http://localhost:5000）
2. 進入 Models 頁面
3. 選擇對應的模型版本
4. 在 Tags 區域添加：
   - Key: `ollama_model_name`
   - Value: `llama2:7b`（本地 Ollama 模型實際名稱）

## 推理流程

### 修復後的完整流程

```
1. API 收到請求（使用 MLflow 模型名稱）
   ↓
2. InferenceManager.infer(model_name="llama2-7b-chat")
   ↓
3. Router.route() 
   ├─ 檢查緩存的 executor (task, engine)
   └─ 如有緩存 → 重用；無緩存 → 創建新的
   ↓
4. Executor.execute(model_name="llama2-7b-chat")
   ↓
5. Executor._get_or_load_model("llama2-7b-chat")
   ├─ 檢查 _loaded_models 緩存
   └─ 如無緩存 → 調用 engine.load_model()
   ↓
6. OllamaEngine.load_model("llama2-7b-chat")
   ├─ 從 MLflow 獲取模型信息
   ├─ 提取 tags['ollama_model_name'] = "llama2:7b"
   ├─ 檢查本地 Ollama 是否有 "llama2:7b"
   └─ 返回 "llama2:7b"
   ↓
7. Executor 緩存映射: _loaded_models["llama2-7b-chat"] = "llama2:7b"
   ↓
8. Engine.infer(model="llama2:7b", ...)  # 使用本地 Ollama 名稱
   ↓
9. 返回推理結果
```

### 第二次調用的流程

```
1. API 收到請求（相同的 model_name）
   ↓
2. Router.route()
   └─ 返回緩存的 executor ✅
   ↓
3. Executor.execute()
   └─ 從 _loaded_models 緩存獲取 "llama2:7b" ✅
   ↓
4. 直接使用 "llama2:7b" 進行推理 ✅
```

## 測試

### 運行測試

```bash
cd /opt/home/george/george-test/AiModelLifecycle/VIE01/AiModelLifecycle
python test/test_ollama_mlflow_mapping.py
```

### 測試覆蓋

1. **MLflow 名稱映射測試**
   - 使用 MLflow 註冊的模型名稱
   - 驗證自動轉換為本地 Ollama 模型名稱
   - 驗證緩存機制正常工作
   
2. **直接 Ollama 名稱測試**
   - 直接使用本地 Ollama 模型名稱
   - 驗證不通過 MLflow 也能正常工作

### 預期結果

```
✅ 第一次推理成功 - 從 MLflow 獲取映射
✅ 第二次推理成功 - 使用緩存的映射
✅ 模型名稱映射保持一致
✅ Executor 被正確重用
✅ 推理性能提升（無需重複加載）
```

## API 使用示例

### 使用 MLflow 註冊的模型名稱

```python
import requests

response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "llama2-7b-chat",  # MLflow 註冊名稱
        "data": {"inputs": "你好，請介紹一下自己"},
        "options": {"max_length": 100, "temperature": 0.7}
    }
)

print(response.json())
```

### 使用本地 Ollama 模型名稱

```python
response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "llama2:7b",  # 本地 Ollama 名稱
        "data": {"inputs": "你好，請介紹一下自己"},
        "options": {"max_length": 100, "temperature": 0.7}
    }
)

print(response.json())
```

## 注意事項

1. **必須在 MLflow 中註冊 Ollama 模型**：即使使用本地名稱，也建議在 MLflow 中註冊以便追蹤
2. **Tag 命名規範**：使用 `ollama_model_name` 作為 tag key
3. **模型名稱格式**：Ollama 本地模型名稱通常格式為 `model:tag`，如 `llama2:7b`
4. **緩存清理**：如需清理緩存，調用 `inference_manager.clear_cache()`

## 相關文件

- `src/inference/engines/ollama.py` - Ollama 引擎實現
- `src/inference/router.py` - 任務路由器（已修復）
- `src/inference/executor.py` - 模型執行器
- `src/inference/manager.py` - 推理管理器
- `test/test_ollama_mlflow_mapping.py` - 測試文件

## 版本歷史

- **v2.0.1** (2025-10-13): 修復 Ollama MLflow 模型名稱映射問題
  - 添加 Router 層 executor 緩存
  - 確保模型名稱映射持久化
  - 提升推理性能（減少重複加載）
