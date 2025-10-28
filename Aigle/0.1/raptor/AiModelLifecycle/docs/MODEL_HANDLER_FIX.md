# 模型處理器註冊修復說明

## 🐛 問題描述

使用細分任務類型（如 `text-generation-ollama`）進行 API 調用時，出現以下錯誤：

```json
{
  "success": false,
  "error": "找不到任務 'text-generation-ollama' 和模型 'xxx' 的處理器。可用的處理器: [...]"
}
```

## 🔍 根本原因

雖然 `router.py` 和 `manager.py` 已經支持細分的任務類型，但是**模型處理器（Model Handler）**沒有為這些新任務類型註冊。

**問題流程**：
```
API 請求 (task="text-generation-ollama")
  ↓
Manager 驗證 ✅ (支持該任務)
  ↓
Router 路由 ✅ (找到對應引擎)
  ↓
獲取模型處理器 ❌ (未註冊該任務的處理器)
```

## ✅ 解決方案

修改 `src/inference/models/__init__.py`，為所有細分任務類型註冊處理器。

### 修改內容

```python
# 原有（只註冊通用類型）
model_registry.register_handler_manually('text-generation', 'default', TextGenerationHandler)
model_registry.register_handler_manually('asr', 'default', ASRHandler)
model_registry.register_handler_manually('ocr', 'default', OCRHandler)

# 修改後（註冊所有細分類型）
# 文本生成任務（包括細分類型）
model_registry.register_handler_manually('text-generation', 'default', TextGenerationHandler)
model_registry.register_handler_manually('text-generation-ollama', 'default', TextGenerationHandler)
model_registry.register_handler_manually('text-generation-hf', 'default', TextGenerationHandler)

# 語音識別任務（包括細分類型）
model_registry.register_handler_manually('asr', 'default', ASRHandler)
model_registry.register_handler_manually('asr-hf', 'default', ASRHandler)
model_registry.register_handler_manually('vad-hf', 'default', ASRHandler)

# OCR 任務（包括細分類型）
model_registry.register_handler_manually('ocr', 'default', OCRHandler)
model_registry.register_handler_manually('ocr-hf', 'default', OCRHandler)

# ... 以及其他新任務類型
```

## 📋 完整的處理器註冊列表

| 任務類型 | 處理器類 | 說明 |
|---------|---------|------|
| `text-generation` | TextGenerationHandler | 通用文本生成 |
| `text-generation-ollama` | TextGenerationHandler | Ollama 文本生成 |
| `text-generation-hf` | TextGenerationHandler | HuggingFace 文本生成 |
| `vlm` | VLMHandler | 視覺語言模型 |
| `asr` | ASRHandler | 通用語音識別 |
| `asr-hf` | ASRHandler | HuggingFace 語音識別 |
| `vad-hf` | ASRHandler | 語音活動檢測 |
| `ocr` | OCRHandler | 通用 OCR |
| `ocr-hf` | OCRHandler | HuggingFace OCR |
| `audio-classification` | AudioClassificationHandler | 音頻分類 |
| `audio-transcription` | ASRHandler | 音頻轉錄 |
| `video-analysis` | VideoAnalysisHandler | 視頻分析 |
| `scene-detection` | VideoAnalysisHandler | 場景檢測 |
| `video-summary` | VideoAnalysisHandler | 視頻摘要 |
| `document-analysis` | DocumentAnalysisHandler | 文檔分析 |
| `image-captioning` | VLMHandler | 圖像標題生成 |

## 🧪 驗證方法

### 1. 檢查處理器註冊

```bash
python test/test_handler_registry.py
```

預期輸出：
```
✅ text-generation-ollama         -> TextGenerationHandler
✅ text-generation-hf             -> TextGenerationHandler
✅ asr-hf                         -> ASRHandler
✅ ocr-hf                         -> OCRHandler
...
```

### 2. 測試 API 調用

```bash
# 使用 Python 腳本測試
python test/test_api_inference.py

# 或使用 curl 測試
curl -X POST http://localhost:8009/inference/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation-ollama",
    "engine": "ollama",
    "model_name": "qwen3:1.7b-ollama-1",
    "data": {"inputs": "你好"}
  }'
```

預期結果：
```json
{
  "success": true,
  "result": {
    "response": "你好！我是...",
    ...
  },
  "task": "text-generation-ollama",
  "engine": "ollama",
  ...
}
```

## 📝 為什麼需要註冊處理器？

### 處理器的作用

模型處理器負責：
1. **數據預處理**：將輸入數據轉換為模型可接受的格式
2. **結果後處理**：將模型輸出轉換為統一的響應格式
3. **任務特定邏輯**：處理不同任務類型的特殊需求

### 設計模式

使用**註冊表模式（Registry Pattern）**：
```
任務類型 → 處理器類 → 處理器實例
     ↓
text-generation-ollama → TextGenerationHandler → handler_instance
```

### 為什麼細分任務需要單獨註冊？

即使 `text-generation-ollama` 和 `text-generation` 使用相同的處理器類，也需要分別註冊，因為：

1. **精確匹配**：Registry 使用 `(task, model_type)` 元組作為鍵進行精確查找
2. **靈活擴展**：未來可能為特定任務類型使用不同的處理器
3. **清晰明確**：顯式註冊使代碼更容易理解和維護

## 🔄 完整流程（修復後）

```
API 請求 (task="text-generation-ollama")
  ↓
Manager 驗證 ✅
  └─ 任務類型在支持列表中
  ↓
Router 路由 ✅
  └─ 找到 (text-generation-ollama, ollama) → OllamaEngine
  ↓
獲取模型處理器 ✅
  └─ Registry 查找 text-generation-ollama → TextGenerationHandler
  ↓
創建 Executor
  └─ Engine + Handler
  ↓
執行推理 ✅
  ├─ 加載模型（從 MLflow 獲取映射）
  ├─ 預處理數據
  ├─ 執行推理
  └─ 後處理結果
  ↓
返回成功響應 ✅
```

## 📚 相關文件

- **src/inference/models/__init__.py** ⭐ 主要修復文件
- **src/inference/registry.py** - 註冊表實現
- **src/inference/router.py** - 任務路由
- **src/inference/manager.py** - 推理管理器
- **test/test_handler_registry.py** - 處理器註冊測試
- **test/test_api_inference.py** - API 推理測試

## ⚠️ 注意事項

1. **添加新任務類型時**：
   - 在 `router.py` 中添加任務映射 ✅
   - 在 `manager.py` 中添加參數驗證 ✅
   - **在 `models/__init__.py` 中註冊處理器** ⭐ 關鍵步驟

2. **處理器重用**：
   - 多個相關任務可以共用同一個處理器類
   - 例如：`asr`, `asr-hf`, `vad-hf` 都使用 `ASRHandler`

3. **測試驗證**：
   - 添加新任務後，運行 `test_handler_registry.py` 驗證
   - 確保 API 調用能正常工作

## 🎉 總結

通過在 `models/__init__.py` 中為所有細分任務類型註冊對應的處理器，解決了 API 調用時找不到處理器的問題。

**關鍵要點**：
- ✅ Router 支持任務類型
- ✅ Manager 驗證任務類型
- ✅ **Models 註冊處理器** ← 之前遺漏的關鍵步驟
- ✅ 完整的推理流程正常工作

現在所有細分任務類型（如 `text-generation-ollama`）都可以正常通過 API 調用了！
