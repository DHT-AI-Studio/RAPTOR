# Inference 模組任務類型配置說明

## 📋 概述

為了與 `src/core/configs/inference.yaml` 中的配置保持一致，inference 模組現已支持細分的任務類型。

## 🔄 變更內容

### 原有設計
```
text-generation → 支持 ollama 和 transformers
asr → 僅支持 transformers
ocr → 僅支持 transformers
```

### 新設計（與 core 配置一致）
```
text-generation-ollama → 僅支持 ollama
text-generation-hf → 僅支持 transformers
text-generation → 通用（向下兼容）

asr-hf → 僅支持 transformers
asr → 通用（向下兼容）

ocr-hf → 僅支持 transformers  
ocr → 通用（向下兼容）

新增: vad-hf, scene-detection, image-captioning, video-summary, audio-transcription
```

## 📊 完整任務類型列表

| 任務類型 | 支持的引擎 | MLflow Tag | 描述 |
|---------|-----------|-----------|------|
| `text-generation-ollama` | ollama | `inference_task = 'text-generation-ollama'` | Ollama 文本生成 |
| `text-generation-hf` | transformers | `inference_task = 'text-generation-hf'` | HuggingFace 文本生成 |
| `text-generation` | ollama, transformers | - | 通用文本生成（向下兼容） |
| `vlm` | transformers | `inference_task = 'vlm'` | 視覺語言模型 |
| `asr-hf` | transformers | `inference_task = 'asr-hf'` | HuggingFace 語音識別 |
| `asr` | transformers | `inference_task = 'asr'` | 通用語音識別 |
| `vad-hf` | transformers | `inference_task = 'vad-hf'` | 語音活動檢測 |
| `ocr-hf` | transformers | `inference_task = 'ocr-hf'` | HuggingFace OCR |
| `ocr` | transformers | `inference_task = 'ocr'` | 通用 OCR |
| `audio-classification` | transformers | `inference_task = 'audio_classification'` | 音頻分類 |
| `video-analysis` | transformers | - | 視頻分析 |
| `scene-detection` | transformers | `inference_task = 'scene_detection'` | 場景檢測 |
| `document-analysis` | transformers | `inference_task = 'document_analysis'` | 文檔分析 |
| `image-captioning` | transformers | `inference_task = 'image_captioning'` | 圖像標題生成 |
| `video-summary` | transformers | `inference_task = 'video_summary'` | 視頻摘要 |
| `audio-transcription` | transformers | `inference_task = 'audio_transcription'` | 音頻轉錄 |

## 🎯 使用示例

### 1. 使用 Ollama 引擎（推薦使用細分任務類型）

```python
# 新方式（推薦）
response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "text-generation-ollama",  # 明確指定 Ollama 任務
        "engine": "ollama",
        "model_name": "llama2-7b-chat",
        "data": {"inputs": "你好"},
        "options": {"max_length": 100}
    }
)

# 舊方式（向下兼容）
response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "text-generation",  # 通用任務類型
        "engine": "ollama",
        "model_name": "llama2-7b-chat",
        "data": {"inputs": "你好"},
        "options": {"max_length": 100}
    }
)
```

### 2. 使用 HuggingFace 引擎

```python
# 新方式（推薦）
response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "text-generation-hf",  # 明確指定 HF 任務
        "engine": "transformers",
        "model_name": "gpt2",
        "data": {"inputs": "人工智能"},
        "options": {"max_length": 50}
    }
)

# ASR 任務
response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "asr-hf",  # 明確指定 HF ASR
        "engine": "transformers",
        "model_name": "whisper-large",
        "data": {"audio": "/path/to/audio.wav"},
        "options": {}
    }
)
```

## 🔧 配置文件對應

### inference.yaml 配置
```yaml
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

### MLflow 模型註冊
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Ollama 模型
client.set_model_version_tag(
    name="llama2-7b-chat",
    version="1",
    key="inference_task",
    value="text-generation-ollama"  # 對應配置文件
)

client.set_model_version_tag(
    name="llama2-7b-chat",
    version="1",
    key="ollama_model_name",
    value="llama2:7b"  # Ollama 本地名稱
)

# HuggingFace 模型
client.set_model_version_tag(
    name="gpt2-chinese",
    version="1",
    key="inference_task",
    value="text-generation-hf"  # 對應配置文件
)
```

## 📝 遷移指南

### 從舊 API 遷移到新 API

#### 文本生成任務

**舊方式:**
```python
{
    "task": "text-generation",
    "engine": "ollama"  # 需要明確指定引擎
}
```

**新方式（推薦）:**
```python
{
    "task": "text-generation-ollama",  # 任務類型已包含引擎信息
    "engine": "ollama"
}
```

**向下兼容:**
- 舊的 `text-generation` 任務類型仍然支持
- 系統會根據 `engine` 參數自動路由

#### 語音識別任務

**舊方式:**
```python
{
    "task": "asr",
    "engine": "transformers"
}
```

**新方式（推薦）:**
```python
{
    "task": "asr-hf",  # 明確標註為 HuggingFace
    "engine": "transformers"
}
```

## ⚠️ 注意事項

1. **任務類型命名規則**
   - `-ollama` 後綴：專用於 Ollama 引擎
   - `-hf` 後綴：專用於 HuggingFace/Transformers 引擎
   - 無後綴：通用類型（向下兼容）

2. **MLflow Tag 對應**
   - 模型在 MLflow 中的 `inference_task` tag 應與任務類型名稱一致
   - Ollama 模型需額外添加 `ollama_model_name` tag

3. **向下兼容性**
   - 所有舊的任務類型名稱仍然支持
   - 建議新代碼使用細分的任務類型

4. **引擎驗證**
   - 系統會驗證任務類型與引擎的兼容性
   - 例如：`text-generation-ollama` 只能使用 `ollama` 引擎

## 🧪 測試

### 測試腳本

```bash
# 測試 Ollama 推理
python test/test_ollama_mlflow_mapping.py

# 測試不同任務類型
python test/test_task_types.py  # 需要創建
```

### API 測試

```python
import requests

# 測試所有支持的任務類型
tasks = [
    ("text-generation-ollama", "ollama", "llama2-7b"),
    ("text-generation-hf", "transformers", "gpt2"),
    ("asr-hf", "transformers", "whisper-large"),
    ("ocr-hf", "transformers", "trocr-base"),
]

for task, engine, model in tasks:
    response = requests.post(
        "http://localhost:8000/inference/infer",
        json={
            "task": task,
            "engine": engine,
            "model_name": model,
            "data": {"inputs": "測試"},
            "options": {}
        }
    )
    print(f"{task}: {response.status_code}")
```

## 📚 相關文檔

- [inference.yaml](../src/core/configs/inference.yaml) - Core 配置文件
- [OLLAMA_FIX_SUMMARY.md](./OLLAMA_FIX_SUMMARY.md) - Ollama 修復說明
- [router.py](../src/inference/router.py) - 任務路由實現
- [manager.py](../src/inference/manager.py) - 推理管理器

## 🔄 版本信息

- **版本**: v2.1.0
- **更新日期**: 2025-10-13
- **變更類型**: 功能增強，保持向下兼容
- **影響範圍**: 任務類型定義、路由邏輯、API 文檔
