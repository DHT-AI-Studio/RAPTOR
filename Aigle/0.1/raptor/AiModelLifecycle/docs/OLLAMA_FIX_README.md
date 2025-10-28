# Ollama 模型名稱映射修復 - 快速參考

## 🎯 核心問題

MLflow 註冊的 Ollama 模型名稱與本地 Ollama 服務中的實際名稱不同時，推理會失敗。

**例如:**
- MLflow 名稱: `llama2-7b-chat` (友好名稱)
- Ollama 名稱: `llama2:7b` (實際名稱)

## ✅ 解決方案

### 1️⃣ 註冊模型時添加 Tag

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_model_version_tag(
    name="llama2-7b-chat",
    version="1",
    key="ollama_model_name",      # 👈 關鍵！
    value="llama2:7b"              # 👈 本地 Ollama 名稱
)
```

### 2️⃣ 使用 API 推理

```python
# 使用 MLflow 註冊名稱
response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "llama2-7b-chat",  # MLflow 名稱
        "data": {"inputs": "你好"},
        "options": {"max_length": 100}
    }
)
```

## 📋 快速檢查清單

- [ ] 在 MLflow 中為 Ollama 模型添加 `ollama_model_name` tag
- [ ] Tag 值為本地 Ollama 的實際模型名稱（格式: `model:tag`）
- [ ] 運行測試驗證: `python test/test_ollama_mlflow_mapping.py`

## 📚 詳細文檔

- **[OLLAMA_FIX_SUMMARY.md](./OLLAMA_FIX_SUMMARY.md)** - 問題分析和修復總結
- **[OLLAMA_MLFLOW_MAPPING_FIX.md](./OLLAMA_MLFLOW_MAPPING_FIX.md)** - 技術細節
- **[OLLAMA_MLFLOW_QUICKSTART.md](./OLLAMA_MLFLOW_QUICKSTART.md)** - 操作指南

## 🔧 修復內容

修改了 `src/inference/router.py`，添加了 Executor 緩存機制：
- ✅ Router 現在會緩存 Executor 實例
- ✅ 模型名稱映射在整個生命週期內保持一致
- ✅ 性能提升（避免重複創建和查詢）

## 🧪 測試

```bash
cd /opt/home/george/george-test/AiModelLifecycle/VIE01/AiModelLifecycle
python test/test_ollama_mlflow_mapping.py
```

## 💡 提示

支持兩種使用方式：
1. 使用 MLflow 註冊名稱（推薦，便於管理）
2. 直接使用 Ollama 本地名稱（簡單快速）
