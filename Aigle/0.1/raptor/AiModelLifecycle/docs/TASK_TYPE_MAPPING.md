# 任務類型對照表

## 📊 Core 配置 vs Inference 模組映射

| inference.yaml 配置 | Inference 模組任務類型 | 引擎 | MLflow Tag |
|-------------------|---------------------|------|-----------|
| `text-generation-ollama` | `text-generation-ollama` | ollama | `inference_task = 'text-generation-ollama'` |
| `text-generation-hf` | `text-generation-hf` | transformers | `inference_task = 'text-generation-hf'` |
| - | `text-generation` | both | 通用（向下兼容） |
| `vlm` | `vlm` | transformers | `inference_task = 'vlm'` |
| `asr-hf` | `asr-hf` | transformers | `inference_task = 'asr-hf'` |
| `asr` | `asr` | transformers | `inference_task = 'asr'` |
| `vad-hf` | `vad-hf` | transformers | `inference_task = 'vad-hf'` |
| `ocr-hf` | `ocr-hf` | transformers | `inference_task = 'ocr-hf'` |
| - | `ocr` | transformers | 通用（向下兼容） |
| `audio_classification` | `audio-classification` | transformers | `inference_task = 'audio_classification'` |
| - | `video-analysis` | transformers | - |
| `scene_detection` | `scene-detection` | transformers | `inference_task = 'scene_detection'` |
| `document_analysis` | `document-analysis` | transformers | `inference_task = 'document_analysis'` |
| `image_captioning` | `image-captioning` | transformers | `inference_task = 'image_captioning'` |
| `video_summary` | `video-summary` | transformers | `inference_task = 'video_summary'` |
| `audio_transcription` | `audio-transcription` | transformers | `inference_task = 'audio_transcription'` |

## 📝 命名規則差異

### inference.yaml (Core 配置)
- 使用 **下劃線** `_` 作為分隔符
- 例如：`audio_classification`, `scene_detection`

### Inference 模組
- 使用 **連字符** `-` 作為分隔符
- 例如：`audio-classification`, `scene-detection`

### MLflow Tags
- 使用 **下劃線** `_` 作為分隔符（與 Core 配置一致）
- 例如：`audio_classification`, `scene_detection`

## 🔄 自動轉換（如需要）

如果需要在兩種命名風格之間轉換：

```python
# 連字符 → 下劃線（Inference → MLflow/Core）
task_for_mlflow = "audio-classification".replace("-", "_")
# 結果: "audio_classification"

# 下劃線 → 連字符（MLflow/Core → Inference）
task_for_inference = "audio_classification".replace("_", "-")
# 結果: "audio-classification"
```

## ⚠️ 特殊情況

### 後綴命名
無論使用哪種分隔符，後綴保持一致：
- `-ollama` / `_ollama`
- `-hf` / `_hf`

### 示例
- Core: `text-generation-ollama` ✅
- MLflow: `text-generation-ollama` ✅
- Inference: `text-generation-ollama` ✅

## 💡 建議

1. **推薦做法**：
   - Inference 模組：使用 `-` 連字符
   - MLflow tags：使用 `_` 下劃線（與 Core 配置一致）

2. **一致性**：
   - 在同一系統內保持命名風格一致
   - 文檔中明確說明使用的命名風格

3. **配置同步**：
   - 確保 `inference.yaml` 中的 task 名稱與 MLflow tag 值一致
   - Inference 模組的 task 名稱應與 API 調用保持一致
