# Inference 任務類型快速參考

## 🎯 核心變更

為了與 `src/core/configs/inference.yaml` 保持一致，任務類型現已細分：

### 文本生成
- `text-generation-ollama` → Ollama 引擎專用 ⭐ 推薦
- `text-generation-hf` → HuggingFace 引擎專用 ⭐ 推薦
- `text-generation` → 通用（向下兼容）

### 語音識別
- `asr-hf` → HuggingFace 專用 ⭐ 推薦
- `asr` → 通用（向下兼容）

### OCR
- `ocr-hf` → HuggingFace 專用 ⭐ 推薦
- `ocr` → 通用（向下兼容）

### 新增任務
- `vad-hf` → 語音活動檢測
- `scene-detection` → 場景檢測
- `image-captioning` → 圖像標題生成
- `video-summary` → 視頻摘要
- `audio-transcription` → 音頻轉錄

## 📝 使用範例

### Ollama 推理（推薦用法）

```python
{
    "task": "text-generation-ollama",  # 新：明確指定
    "engine": "ollama",
    "model_name": "llama2-7b-chat",
    "data": {"inputs": "你好"},
    "options": {"max_length": 100}
}
```

### HuggingFace 推理（推薦用法）

```python
# 文本生成
{
    "task": "text-generation-hf",  # 新：明確指定
    "engine": "transformers",
    "model_name": "gpt2",
    "data": {"inputs": "人工智能"},
    "options": {"max_length": 50}
}

# 語音識別
{
    "task": "asr-hf",  # 新：明確指定
    "engine": "transformers",
    "model_name": "whisper-large",
    "data": {"audio": "/path/to/audio.wav"},
    "options": {}
}
```

## 🔄 向下兼容

舊的 API 調用仍然有效：

```python
# 仍然支持
{
    "task": "text-generation",  # 舊：通用類型
    "engine": "ollama",
    "model_name": "llama2-7b-chat",
    "data": {"inputs": "你好"},
    "options": {"max_length": 100}
}
```

## 📋 MLflow 配置檢查清單

- [ ] 添加 `inference_task` tag（如 `text-generation-ollama`）
- [ ] Ollama 模型添加 `ollama_model_name` tag（如 `llama2:7b`）
- [ ] 確保 tag 值與 `inference.yaml` 中的 filter 一致

## 🔗 相關文檔

- **[INFERENCE_TASK_TYPES.md](./INFERENCE_TASK_TYPES.md)** - 完整說明
- **[OLLAMA_FIX_SUMMARY.md](./OLLAMA_FIX_SUMMARY.md)** - Ollama 修復
- **[inference.yaml](../src/core/configs/inference.yaml)** - Core 配置

## ⚡ 快速測試

```bash
# 測試 Ollama
curl -X POST http://localhost:8000/inference/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation-ollama",
    "engine": "ollama",
    "model_name": "llama2-7b-chat",
    "data": {"inputs": "你好"},
    "options": {"max_length": 50}
  }'

# 測試 HuggingFace
curl -X POST http://localhost:8000/inference/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation-hf",
    "engine": "transformers",
    "model_name": "gpt2",
    "data": {"inputs": "AI"},
    "options": {"max_length": 50}
  }'
```

## 💡 最佳實踐

1. ✅ **推薦**：使用細分的任務類型（如 `text-generation-ollama`）
2. ✅ **MLflow**: 確保 tags 與配置文件一致
3. ✅ **命名**: 遵循 `-ollama` 或 `-hf` 後綴規則
4. ⚠️ **避免**：混用引擎和任務類型（如 ollama 引擎用 `-hf` 任務）
