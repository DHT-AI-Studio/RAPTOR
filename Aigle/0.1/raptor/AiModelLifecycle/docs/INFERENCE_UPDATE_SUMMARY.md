# Inference 模組配置更新總結

## 📊 更新內容

### 1. Ollama 模型名稱映射修復 ✅
- **問題**: MLflow 註冊名稱與本地 Ollama 名稱不同導致推理失敗
- **修復**: Router 層添加 Executor 緩存機制
- **文檔**: [OLLAMA_FIX_SUMMARY.md](./OLLAMA_FIX_SUMMARY.md)

### 2. 任務類型細分（與 Core 配置對齊）✅
- **變更**: 支持 `text-generation-ollama`, `text-generation-hf` 等細分任務
- **向下兼容**: 保留通用任務類型（如 `text-generation`）
- **文檔**: [INFERENCE_TASK_TYPES.md](./INFERENCE_TASK_TYPES.md)

## 🎯 核心改進

### 任務類型映射

| 原任務類型 | 新任務類型 | 引擎 | 狀態 |
|-----------|-----------|------|------|
| `text-generation` | `text-generation-ollama` | ollama | ⭐ 推薦 |
| `text-generation` | `text-generation-hf` | transformers | ⭐ 推薦 |
| `text-generation` | `text-generation` | both | ✅ 兼容 |
| `asr` | `asr-hf` | transformers | ⭐ 推薦 |
| `asr` | `asr` | transformers | ✅ 兼容 |
| `ocr` | `ocr-hf` | transformers | ⭐ 推薦 |
| `ocr` | `ocr` | transformers | ✅ 兼容 |
| - | `vad-hf` | transformers | 🆕 新增 |
| - | `scene-detection` | transformers | 🆕 新增 |
| - | `image-captioning` | transformers | 🆕 新增 |
| - | `video-summary` | transformers | 🆕 新增 |
| - | `audio-transcription` | transformers | 🆕 新增 |

## 📁 修改的文件

### 核心修復
1. **src/inference/router.py** ⭐
   - 添加 `_executors` 緩存字典
   - 支持細分任務類型
   - 更新任務描述

2. **src/inference/manager.py** ⭐
   - 更新參數驗證邏輯
   - 擴展支持的任務列表
   - 更新 `get_supported_tasks()` 方法

3. **src/api/inference_api.py**
   - 更新 API 文檔字符串
   - 列出所有支持的任務類型

### 文檔
4. **docs/OLLAMA_FIX_SUMMARY.md** 🆕
   - Ollama 問題分析和修復總結
   
5. **docs/OLLAMA_MLFLOW_MAPPING_FIX.md** 🆕
   - 技術細節和實現流程
   
6. **docs/OLLAMA_MLFLOW_QUICKSTART.md** 🆕
   - MLflow 模型註冊快速指南
   
7. **docs/OLLAMA_FIX_README.md** 🆕
   - Ollama 修復快速參考
   
8. **docs/INFERENCE_TASK_TYPES.md** 🆕
   - 任務類型完整配置說明
   
9. **docs/INFERENCE_TASK_QUICK_REF.md** 🆕
   - 任務類型快速參考

### 測試
10. **test/test_ollama_mlflow_mapping.py** 🆕
    - Ollama MLflow 映射測試套件

## 🚀 使用指南

### 方式 1: 使用細分任務類型（推薦）

```python
# Ollama 推理
{
    "task": "text-generation-ollama",
    "engine": "ollama",
    "model_name": "llama2-7b-chat",
    "data": {"inputs": "你好"},
    "options": {"max_length": 100}
}

# HuggingFace 推理
{
    "task": "text-generation-hf",
    "engine": "transformers",
    "model_name": "gpt2",
    "data": {"inputs": "AI"},
    "options": {"max_length": 50}
}
```

### 方式 2: 使用通用任務類型（向下兼容）

```python
{
    "task": "text-generation",
    "engine": "ollama",  # 通過 engine 區分
    "model_name": "llama2-7b-chat",
    "data": {"inputs": "你好"},
    "options": {"max_length": 100}
}
```

## ⚙️ MLflow 配置要求

### 1. Ollama 模型

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 必須添加兩個 tags
client.set_model_version_tag(
    name="llama2-7b-chat",
    version="1",
    key="inference_task",
    value="text-generation-ollama"  # 對應任務類型
)

client.set_model_version_tag(
    name="llama2-7b-chat",
    version="1",
    key="ollama_model_name",
    value="llama2:7b"  # 本地 Ollama 名稱
)
```

### 2. HuggingFace 模型

```python
client.set_model_version_tag(
    name="gpt2-chinese",
    version="1",
    key="inference_task",
    value="text-generation-hf"  # 對應任務類型
)
```

## 📋 與 inference.yaml 對應

```yaml
# src/core/configs/inference.yaml
task_to_models:
  text-generation-ollama:
    strategy: "priority"
    discovery:
      source: "mlflow"
      filter: "tags.inference_task = 'text-generation-ollama'"
  
  text-generation-hf:
    strategy: "priority"
    discovery:
      source: "mlflow"
      filter: "tags.inference_task = 'text-generation-hf'"
```

## 🧪 測試

```bash
# Ollama 映射測試
python test/test_ollama_mlflow_mapping.py

# API 測試
curl -X POST http://localhost:8000/inference/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation-ollama",
    "engine": "ollama",
    "model_name": "llama2-7b-chat",
    "data": {"inputs": "測試"},
    "options": {}
  }'

# 查看支持的任務
curl http://localhost:8000/inference/supported-tasks
```

## ✅ 檢查清單

配置 Ollama 模型時：
- [ ] 在 MLflow 註冊模型
- [ ] 添加 `inference_task` tag（值為 `text-generation-ollama`）
- [ ] 添加 `ollama_model_name` tag（值為本地 Ollama 名稱）
- [ ] 在 `inference.yaml` 中配置對應的 filter
- [ ] 測試推理功能

配置 HuggingFace 模型時：
- [ ] 在 MLflow 註冊模型
- [ ] 添加 `inference_task` tag（值為對應的任務類型，如 `text-generation-hf`）
- [ ] 在 `inference.yaml` 中配置對應的 filter
- [ ] 測試推理功能

## 📚 文檔導航

### 快速開始
- [OLLAMA_FIX_README.md](./OLLAMA_FIX_README.md) - Ollama 快速參考
- [INFERENCE_TASK_QUICK_REF.md](./INFERENCE_TASK_QUICK_REF.md) - 任務類型快速參考

### 詳細說明
- [OLLAMA_FIX_SUMMARY.md](./OLLAMA_FIX_SUMMARY.md) - Ollama 修復完整說明
- [INFERENCE_TASK_TYPES.md](./INFERENCE_TASK_TYPES.md) - 任務類型完整配置

### 操作指南
- [OLLAMA_MLFLOW_QUICKSTART.md](./OLLAMA_MLFLOW_QUICKSTART.md) - MLflow 註冊指南
- [OLLAMA_MLFLOW_MAPPING_FIX.md](./OLLAMA_MLFLOW_MAPPING_FIX.md) - 技術實現細節

## 🔄 版本信息

- **版本**: v2.1.0
- **更新日期**: 2025-10-13
- **主要變更**:
  1. 修復 Ollama 模型名稱映射問題（v2.0.1）
  2. 支持細分任務類型，與 Core 配置對齊（v2.1.0）
- **向下兼容**: 是 ✅
- **重大變更**: 否

## ⚠️ 注意事項

1. **命名規則**
   - `-ollama` 後綴：Ollama 引擎專用
   - `-hf` 後綴：HuggingFace 引擎專用
   - 無後綴：通用（向下兼容）

2. **MLflow Tags**
   - `inference_task`: 必須與任務類型名稱一致
   - `ollama_model_name`: Ollama 模型必需

3. **向下兼容**
   - 所有舊 API 調用仍然有效
   - 建議新代碼使用細分任務類型

4. **性能優化**
   - Router 層 Executor 緩存
   - 避免重複查詢 MLflow
   - 模型名稱映射持久化

## 🎉 總結

本次更新實現了兩個主要目標：

1. ✅ **修復 Ollama 模型名稱映射問題**
   - 添加 Router 層 Executor 緩存
   - 確保模型名稱映射持久化
   - 提升推理性能

2. ✅ **任務類型與 Core 配置對齊**
   - 支持細分任務類型（`-ollama`, `-hf` 後綴）
   - 保持向下兼容性
   - 擴展新任務類型支持

系統現在能夠：
- 正確處理 MLflow 和本地 Ollama 模型名稱映射
- 支持與 `inference.yaml` 一致的細分任務類型
- 保持良好的向下兼容性
- 提供更靈活的推理配置選項
