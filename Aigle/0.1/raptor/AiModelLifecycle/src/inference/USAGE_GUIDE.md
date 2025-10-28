# Inference 模組 - 使用指南

> 重構版本 v2.0.0 - 簡化、統一、可擴展

## 概述

重構後的 Inference 模組提供統一的 AI 模型推理接口，支持多種任務類型和引擎。

### 核心特點

- **統一 API**：單一 `/infer` 端點支持所有任務
- **簡化架構**：清晰的 3 層結構（Engine → Manager → API）
- **MLflow 整合**：自動從模型註冊中心獲取模型
- **可擴展性**：插件式架構，易於添加新模型
- **類型安全**：完整的類型提示和註解

## 架構設計

```
┌─────────────────────────────────────┐
│      API Layer (inference_api.py)  │  統一的推理接口
├─────────────────────────────────────┤
│   Manager Layer (manager.py)       │  推理管理與路由
├──────────────┬──────────────────────┤
│  TaskRouter  │  ModelExecutor      │  任務路由與執行
├──────────────┼──────────────────────┤
│  Engine Layer                       │  引擎實現
│  ├── OllamaEngine (text-gen)       │  
│  └── TransformersEngine (all)      │  
├─────────────────────────────────────┤
│  Model Handlers (preprocess/post)  │  任務處理器
│  ├── TextGenerationHandler         │
│  ├── VLMHandler                     │
│  ├── ASRHandler                     │
│  └── ...                            │
└─────────────────────────────────────┘
```

## 快速開始

### 1. 基本使用

```python
from src.inference import inference_manager

# 文本生成（Ollama）
result = inference_manager.infer(
    task="text-generation",
    engine="ollama",
    model_name="llama2-7b",
    data={"inputs": "請解釋什麼是深度學習"},
    options={"max_length": 200, "temperature": 0.7}
)

print(result['result']['generated_text'])
```

### 2. 視覺語言模型（VLM）

```python
# VLM 推理
result = inference_manager.infer(
    task="vlm",
    engine="transformers",
    model_name="llava-1.5-7b",
    data={
        "image": "/path/to/image.jpg",
        "prompt": "描述這張圖片中的內容"
    },
    options={"max_length": 256}
)
```

### 3. 語音識別（ASR）

```python
# ASR 推理
result = inference_manager.infer(
    task="asr",
    engine="transformers",
    model_name="whisper-large-v3",
    data={"audio": "/path/to/audio.wav"},
    options={}
)
```

## API 規格

### 統一推理端點

**POST** `/inference/infer`

#### 請求格式

```json
{
  "task": "text-generation",          // 必需：任務類型
  "engine": "ollama",                 // 必需：引擎類型
  "model_name": "llama2-7b",          // 必需：模型名稱（MLflow 註冊名）
  "data": {                           // 必需：輸入數據
    "inputs": "輸入文本"
  },
  "options": {                        // 可選：推理選項
    "max_length": 200,
    "temperature": 0.7
  }
}
```

#### 支持的任務類型

| 任務類型 | 引擎支持 | 說明 | 輸入格式 |
|---------|---------|------|---------|
| `text-generation` | ollama, transformers | 文本生成 | `{"inputs": str}` |
| `vlm` | transformers | 視覺語言模型 | `{"image": str/PIL.Image, "prompt": str}` |
| `asr` | transformers | 語音識別 | `{"audio": str}` |
| `ocr` | transformers | 文字識別 | `{"image": str}` |
| `audio-classification` | transformers | 音頻分類 | `{"audio": str}` |
| `video-analysis` | transformers | 視頻分析 | `{"video": str}` |
| `document-analysis` | transformers | 文檔分析 | `{"document": str}` |

#### 響應格式

```json
{
  "success": true,
  "result": {
    "generated_text": "生成的文本...",
    "metadata": {
      // 任務特定的元數據
    }
  },
  "task": "text-generation",
  "engine": "ollama",
  "model_name": "llama2-7b",
  "processing_time": 1.234,
  "timestamp": 1234567890.123
}
```

## 引擎配置

### Ollama 引擎

```yaml
# config.yaml
ollama:
  api_base: "http://localhost:11434"
  timeout: 300
  auto_pull: true  # 自動拉取不存在的模型
```

**特點**：
- 僅支持 `text-generation` 任務
- 模型由 Ollama 服務端管理
- 統一的推理流程

### Transformers 引擎

```yaml
# config.yaml
transformers:
  device: "auto"            # auto, cuda, cpu
  torch_dtype: "auto"       # auto, fp16, fp32
  trust_remote_code: true
```

**特點**：
- 支持所有任務類型
- 與 MLflow 深度整合
- 自動設備管理

## 模型註冊與使用

### 1. 在 MLflow 中註冊模型

```python
from src.core.model_manager import model_manager

# 註冊 HuggingFace 模型
model_manager.register_model_to_mlflow(
    model_name="llama2-7b",
    model_source="huggingface",
    model_path="/path/to/model",
    tags={
        "task": "text-generation",
        "engine": "transformers",
        "repo_id": "meta-llama/Llama-2-7b-hf"
    }
)

# 註冊 Ollama 模型
model_manager.register_model_to_mlflow(
    model_name="llama2-7b-ollama",
    model_source="ollama",
    tags={
        "task": "text-generation",
        "engine": "ollama",
        "ollama_model_name": "llama2:7b"
    }
)
```

### 2. 使用已註冊的模型

```python
# 模型會自動從 MLflow 獲取路徑
result = inference_manager.infer(
    task="text-generation",
    engine="transformers",
    model_name="llama2-7b",  # MLflow 註冊名稱
    data={"inputs": "Hello"},
    options={}
)
```

## 擴展指南

### 添加自定義 ModelHandler

```python
from src.inference.models.base import BaseModelHandler
from src.inference.registry import model_registry

class CustomTaskHandler(BaseModelHandler):
    """自定義任務處理器"""
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        預處理輸入數據
        
        Args:
            data: 原始輸入數據
            options: 處理選項
            
        Returns:
            預處理後的數據
        """
        # 實現預處理邏輯
        self.validate_input(data, ['custom_input'])
        processed = {
            'inputs': self._process_custom_input(data['custom_input'])
        }
        return processed
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        後處理推理結果
        
        Args:
            result: 原始推理結果
            options: 處理選項
            
        Returns:
            格式化的結果
        """
        # 實現後處理邏輯
        return {
            'custom_output': self._format_output(result),
            'metadata': {}
        }
    
    def _process_custom_input(self, custom_input):
        """自定義輸入處理"""
        # 實現邏輯
        return processed_input
    
    def _format_output(self, result):
        """自定義輸出格式化"""
        # 實現邏輯
        return formatted_output

# 註冊處理器
model_registry.register_handler_manually(
    'custom-task',
    'default',
    CustomTaskHandler
)
```

### 添加自定義引擎（進階）

```python
from src.inference.engines.base import BaseEngine

class CustomEngine(BaseEngine):
    """自定義推理引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_tasks = ['custom-task']
        self.is_initialized = True
    
    def load_model(self, model_name: str, **kwargs) -> Any:
        """加載模型"""
        # 實現模型加載邏輯
        model = self._load_custom_model(model_name)
        return model
    
    def infer(self, model: Any, inputs: Dict[str, Any], options: Dict[str, Any]) -> Any:
        """執行推理"""
        # 實現推理邏輯
        result = model.predict(inputs)
        return {'response': result, 'metadata': {}}
    
    def unload_model(self, model: Any) -> bool:
        """卸載模型"""
        # 實現卸載邏輯
        return True
```

## 配置選項

### 通用推理選項

```python
options = {
    # 生成長度
    "max_length": 512,           # 最大總長度
    "max_new_tokens": 256,       # 最大新生成 tokens
    
    # 採樣策略
    "temperature": 0.7,          # 溫度 (0.0-1.0)
    "top_p": 0.9,                # Nucleus sampling
    "top_k": 50,                 # Top-K sampling
    "do_sample": True,           # 是否採樣
    
    # Beam Search
    "num_beams": 1,              # Beam 數量
    "early_stopping": False,     # 提前停止
    
    # 懲罰參數
    "repetition_penalty": 1.0,   # 重複懲罰
    "length_penalty": 1.0,       # 長度懲罰
}
```

### Ollama 特定選項

```python
ollama_options = {
    "num_predict": 128,          # 預測 token 數
    "repeat_penalty": 1.1,       # 重複懲罰
    "mirostat": 0,               # Mirostat 模式
    "mirostat_tau": 5.0,         # Mirostat tau
    "num_ctx": 2048,             # 上下文窗口
}
```

## 監控與管理

### 健康檢查

```python
# 檢查系統健康狀態
health = inference_manager.health_check()
print(f"狀態: {health['status']}")
print(f"組件: {health['components']}")
```

### 獲取統計信息

```python
# 獲取推理統計
stats = inference_manager.get_stats()
print(f"總推理次數: {stats['total_inferences']}")
print(f"成功率: {stats['success_rate']:.2%}")
print(f"緩存命中率: {stats['cache_hit_rate']:.2%}")
```

### 緩存管理

```python
# 獲取緩存的模型
cached = inference_manager.get_cached_models()

# 清理緩存
inference_manager.clear_cache()
```

## 最佳實踐

### 1. 模型選擇

- **Ollama**：適用於文本生成，資源占用少，部署簡單
- **Transformers**：適用於多模態任務，功能完整

### 2. 性能優化

- 重用模型實例（利用緩存）
- 合理設置 `max_length` 避免過長生成
- 使用 GPU 加速（Transformers）
- 批處理請求（如果支持）

### 3. 錯誤處理

```python
try:
    result = inference_manager.infer(...)
    if result['success']:
        output = result['result']
    else:
        print(f"推理失敗: {result.get('error')}")
except Exception as e:
    print(f"異常: {e}")
```

### 4. 生產環境建議

- 設置適當的 `timeout`
- 實現重試機制
- 監控系統資源使用
- 記錄詳細日誌
- 定期清理緩存

## 故障排除

### 常見問題

**Q: Ollama 連接失敗**
```
A: 檢查 Ollama 服務是否運行：
   curl http://localhost:11434/api/tags
   
   修改配置文件中的 api_base 地址
```

**Q: 模型加載失敗**
```
A: 檢查模型是否在 MLflow 中註冊：
   - 訪問 MLflow UI
   - 確認模型路徑存在
   - 檢查 physical_path 標籤
```

**Q: GPU 內存不足**
```
A: 解決方案：
   - 使用較小的模型
   - 設置 device='cpu'
   - 減小 batch_size
   - 清理緩存
```

**Q: 推理速度慢**
```
A: 優化建議：
   - 使用 GPU 加速
   - 減小 max_length
   - 使用量化模型
   - 啟用模型緩存
```

## 版本歷史

### v2.0.0 (當前版本)
- ✅ 重構 Engine 架構，簡化接口
- ✅ 與 MLflow 深度整合
- ✅ 統一 API 設計
- ✅ 完整的類型提示和註解
- ✅ 可擴展的插件架構

### v1.0.0 (舊版本)
- 基礎推理功能
- 多引擎支持

## 相關資源

- [MLflow 文檔](https://mlflow.org/docs/latest/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ollama 文檔](https://github.com/ollama/ollama)
- [專案 README](./README.md)

## 聯繫與支持

如有問題或建議，請聯繫開發團隊。
