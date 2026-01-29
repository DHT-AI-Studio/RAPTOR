# Ollama 推理模型名稱映射問題修復總結

## 📋 問題描述

在使用 Ollama 引擎進行推理時，發現了模型名稱映射的問題：

- **現象**: API 通過 MLflow 註冊的模型名稱調用時，推理失敗
- **原因**: MLflow 中註冊的模型名稱（如 `llama2-7b-chat`）與本地 Ollama 服務中的實際模型名稱（如 `llama2:7b`）不同
- **根源**: Router 每次調用都創建新的 Executor 實例，導致模型名稱映射緩存失效

## 🔍 問題分析

### 原有流程

```
API 請求 (MLflow 名稱: "llama2-7b-chat")
    ↓
Router.route() → 創建新的 Executor ❌
    ↓
Executor._get_or_load_model() → 新實例，緩存為空 ❌
    ↓
Engine.load_model("llama2-7b-chat")
    ├─ 從 MLflow 獲取 tag: ollama_model_name = "llama2:7b" ✅
    └─ 返回 "llama2:7b" ✅
    ↓
Executor 緩存: _loaded_models["llama2-7b-chat"] = "llama2:7b" ✅
    ↓
第二次請求 → Router 又創建新 Executor ❌
    └─ 新實例緩存為空，又要重新加載 ❌
```

### 問題關鍵

1. ✅ **Ollama.load_model()**: 正確處理了 MLflow → Ollama 名稱轉換
2. ✅ **Executor 緩存邏輯**: 正確緩存了模型映射
3. ❌ **Router.route()**: 每次都創建新 Executor，導致緩存失效

## ✅ 修復方案

### 修改的文件

**`src/inference/router.py`**

### 具體變更

#### 1. 添加 Executor 緩存

```python
def __init__(self):
    """初始化任務路由器"""
    self._engines: Dict[str, Any] = {}
    self._executors: Dict[tuple, ModelExecutor] = {}  # 新增
    self._task_engine_mapping = {...}
```

#### 2. 修改 route() 方法

```python
def route(self, task: str, engine: str, model_name: str) -> ModelExecutor:
    # 檢查是否已有緩存的執行器
    executor_key = (task, engine)
    if executor_key in self._executors:
        logger.debug(f"重用已緩存的執行器: {task} -> {engine}")
        return self._executors[executor_key]  # 直接返回緩存
    
    # 創建新的執行器並緩存
    executor = ModelExecutor(engine_instance, model_handler)
    self._executors[executor_key] = executor  # 緩存
    return executor
```

#### 3. 更新 clear_engines() 方法

```python
def clear_engines(self):
    """清理引擎和執行器緩存"""
    for executor in self._executors.values():
        executor.clear_models()  # 清理模型
    
    self._executors.clear()  # 清理 executor 緩存
    self._engines.clear()
```

## 🎯 修復後的流程

```
第一次請求 (MLflow 名稱: "llama2-7b-chat")
    ↓
Router.route() → 創建新 Executor，緩存 key=(task, engine) ✅
    ↓
Executor.load_model("llama2-7b-chat")
    ├─ 從 MLflow 獲取 ollama_model_name = "llama2:7b" ✅
    ├─ 緩存: _loaded_models["llama2-7b-chat"] = "llama2:7b" ✅
    └─ 返回 "llama2:7b"
    ↓
Engine.infer(model="llama2:7b") ✅
    ↓
第二次請求 (相同任務和引擎)
    ↓
Router.route() → 從緩存返回相同 Executor ✅
    ↓
Executor.load_model("llama2-7b-chat")
    └─ 從 _loaded_models 緩存直接返回 "llama2:7b" ✅
    ↓
Engine.infer(model="llama2:7b") ✅
```

## 📊 效果對比

### 修復前

| 指標 | 第一次請求 | 第二次請求 |
|------|-----------|-----------|
| Executor 創建 | ✅ 新建 | ❌ 又新建 |
| 模型映射查詢 | ✅ 從 MLflow 獲取 | ❌ 又從 MLflow 獲取 |
| 緩存使用 | ❌ 無緩存 | ❌ 無緩存 |
| 性能 | 慢 | 慢 |

### 修復後

| 指標 | 第一次請求 | 第二次請求 |
|------|-----------|-----------|
| Executor 創建 | ✅ 新建 | ✅ 重用緩存 |
| 模型映射查詢 | ✅ 從 MLflow 獲取 | ✅ 使用 Executor 緩存 |
| 緩存使用 | ✅ 緩存映射 | ✅ 直接使用緩存 |
| 性能 | 慢（首次） | 快（緩存） |

## 📝 使用說明

### 1. MLflow 模型註冊

在 MLflow 註冊 Ollama 模型時，必須添加 `ollama_model_name` tag：

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_model_version_tag(
    name="llama2-7b-chat",      # MLflow 註冊名稱
    version="1",
    key="ollama_model_name",
    value="llama2:7b"            # 本地 Ollama 實際名稱
)
```

### 2. API 調用

可以使用兩種方式：

```python
# 方式 1: 使用 MLflow 註冊名稱（推薦）
{
    "task": "text-generation",
    "engine": "ollama",
    "model_name": "llama2-7b-chat",  # MLflow 名稱
    "data": {"inputs": "你好"},
    "options": {"max_length": 100}
}

# 方式 2: 直接使用 Ollama 本地名稱
{
    "task": "text-generation",
    "engine": "ollama",
    "model_name": "llama2:7b",       # Ollama 名稱
    "data": {"inputs": "你好"},
    "options": {"max_length": 100}
}
```

## 🧪 測試

### 運行測試

```bash
cd /opt/home/george/george-test/AiModelLifecycle/VIE01/AiModelLifecycle
python test/test_ollama_mlflow_mapping.py
```

### 測試覆蓋

- ✅ MLflow 名稱 → Ollama 名稱映射
- ✅ Executor 緩存重用
- ✅ 模型名稱映射持久化
- ✅ 直接使用 Ollama 名稱
- ✅ 多次推理性能驗證

## 📁 修改的文件

1. **src/inference/router.py** ⭐ 主要修復
   - 添加 `_executors` 緩存字典
   - 修改 `route()` 方法實現 executor 重用
   - 更新 `clear_engines()` 方法

2. **test/test_ollama_mlflow_mapping.py** 🆕 新增測試
   - MLflow 名稱映射測試
   - 直接 Ollama 名稱測試
   - 緩存機制驗證

3. **docs/OLLAMA_MLFLOW_MAPPING_FIX.md** 🆕 技術文檔
   - 問題分析和修復方案
   - 完整流程說明
   - API 使用示例

4. **docs/OLLAMA_MLFLOW_QUICKSTART.md** 🆕 快速指南
   - MLflow 模型註冊步驟
   - 完整示例腳本
   - 常見問題解答

## ⚠️ 注意事項

1. **必須添加 tag**: 所有 Ollama 模型在 MLflow 中註冊時必須添加 `ollama_model_name` tag
2. **命名格式**: Ollama 模型名稱通常為 `model:tag` 格式，如 `llama2:7b`
3. **緩存清理**: 需要清理緩存時調用 `inference_manager.clear_cache()`
4. **向下兼容**: 同時支持 MLflow 名稱和直接 Ollama 名稱

## 🚀 版本信息

- **版本**: v2.0.1
- **修復日期**: 2025-10-13
- **修復者**: George
- **影響範圍**: Ollama 引擎推理

## 📚 相關文檔

- [OLLAMA_MLFLOW_MAPPING_FIX.md](./OLLAMA_MLFLOW_MAPPING_FIX.md) - 詳細技術文檔
- [OLLAMA_MLFLOW_QUICKSTART.md](./OLLAMA_MLFLOW_QUICKSTART.md) - 快速開始指南
- [test_ollama_mlflow_mapping.py](../test/test_ollama_mlflow_mapping.py) - 測試代碼

## ✅ 驗證清單

- [x] 修復 Router 的 Executor 緩存問題
- [x] 驗證 MLflow → Ollama 名稱映射正常工作
- [x] 驗證 Executor 和模型緩存重用
- [x] 編寫完整的測試用例
- [x] 編寫技術文檔和使用指南
- [x] 保持向下兼容性

## 🎉 總結

通過在 Router 層添加 Executor 緩存機制，成功解決了 Ollama 模型名稱映射失效的問題。現在系統能夠：

1. ✅ 正確處理 MLflow 註冊名稱與本地 Ollama 名稱的映射
2. ✅ 重用 Executor 實例，避免重複創建
3. ✅ 利用緩存提升推理性能
4. ✅ 支持兩種模型名稱使用方式
5. ✅ 保持良好的向下兼容性
