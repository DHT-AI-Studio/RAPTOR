# 添加新任務類型檢查清單

## 🎯 當你需要添加新的任務類型時

### 必須完成的 3 個步驟

#### ✅ 1. Router - 添加任務映射
**文件**: `src/inference/router.py`

```python
self._task_engine_mapping = {
    'your-new-task': {
        'transformers': TransformersEngine  # 或 'ollama': OllamaEngine
    }
}
```

#### ✅ 2. Manager - 添加參數驗證
**文件**: `src/inference/manager.py`

```python
# 添加到支持的任務列表
supported_tasks = {
    'text-generation', 'your-new-task', ...
}

# 添加必需字段檢查
required_fields = {
    'your-new-task': ['required_field1', 'required_field2']
}

# 添加到 get_supported_tasks()
def get_supported_tasks(self):
    return {
        'your-new-task': {
            'engines': ['transformers'],
            'description': '新任務描述',
            'input_format': {...},
            'examples': [...]
        }
    }
```

#### ✅ 3. Models - 註冊處理器 ⭐ **關鍵！**
**文件**: `src/inference/models/__init__.py`

```python
# 選擇合適的處理器類並註冊
model_registry.register_handler_manually(
    'your-new-task',      # 任務名稱
    'default',            # 模型類型
    YourHandlerClass      # 處理器類
)
```

### 常見錯誤

❌ **只完成步驟 1 和 2，忘記步驟 3**
```
錯誤: 找不到任務 'your-new-task' 的處理器
```

### 驗證步驟

```bash
# 1. 檢查處理器註冊
python test/test_handler_registry.py

# 2. 測試 API 調用
curl -X POST http://localhost:8009/inference/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "your-new-task",
    "engine": "transformers",
    "model_name": "test-model",
    "data": {...}
  }'
```

## 🔄 可重用的處理器類

| 處理器類 | 適用任務 |
|---------|---------|
| `TextGenerationHandler` | 所有文本生成相關 |
| `VLMHandler` | 視覺語言、圖像標題 |
| `ASRHandler` | 語音識別、轉錄、VAD |
| `OCRHandler` | 光學字符識別 |
| `AudioClassificationHandler` | 音頻分類 |
| `VideoAnalysisHandler` | 視頻分析、場景檢測、摘要 |
| `DocumentAnalysisHandler` | 文檔分析 |

## 💡 示例：添加 "sentiment-analysis" 任務

### 1. Router
```python
'sentiment-analysis': {
    'transformers': TransformersEngine
}
```

### 2. Manager
```python
supported_tasks = {
    ..., 'sentiment-analysis'
}

required_fields = {
    'sentiment-analysis': ['text']
}
```

### 3. Models ⭐
```python
model_registry.register_handler_manually(
    'sentiment-analysis',
    'default',
    TextGenerationHandler  # 重用現有處理器
)
```

### 4. 測試
```bash
curl -X POST http://localhost:8009/inference/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "sentiment-analysis",
    "engine": "transformers",
    "model_name": "sentiment-bert",
    "data": {"text": "這個產品很棒！"}
  }'
```

## 📚 相關文檔

- [MODEL_HANDLER_FIX.md](./MODEL_HANDLER_FIX.md) - 處理器修復詳情
- [INFERENCE_TASK_TYPES.md](./INFERENCE_TASK_TYPES.md) - 任務類型配置
- [INFERENCE_UPDATE_SUMMARY.md](./INFERENCE_UPDATE_SUMMARY.md) - 完整更新總結
