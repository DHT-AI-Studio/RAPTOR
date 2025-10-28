# Inference 模組

> AI 模型統一推理框架 - 簡化、統一、可擴展

版本: **v2.0.0**

## 目錄

- [概述](#概述)
- [核心特性](#核心特性)
- [架構設計](#架構設計)
- [快速開始](#快速開始)
- [API 文檔](#api-文檔)
- [支援任務](#支援任務)
- [配置說明](#配置說明)
- [擴展開發](#擴展開發)
- [故障排除](#故障排除)

---

## 概述

Inference 模組是一個重構後的統一 AI 模型推理框架，提供簡化的接口來執行多種 AI 任務。本模組整合了 Ollama 和 Transformers 引擎，支持從文本生成到多模態分析的各種任務。

### 重構亮點

與舊版本相比，v2.0 帶來了革命性的改進：

| 特性 | v1.0 (舊版) | v2.0 (新版) | 改進 |
|-----|-----------|-----------|------|
| 架構層級 | 5層（複雜） | 3層（清晰） | **-40%** |
| API 端點 | 5個分散端點 | 1個統一端點 | **統一化** |
| 函數註解 | ~30% | 100% | **+233%** |
| MLflow 整合 | 部分支持 | 完全整合 | **全面** |
| 代碼可讀性 | 中等 | 優秀 | **顯著提升** |

---

## 核心特性

### 主要功能

- **統一 API**: 單一 `/infer` 端點支持所有任務類型
- **簡化架構**: 清晰的 3 層結構（Engine → Manager → API）
- **智能路由**: 自動選擇最佳引擎和處理器
- **智能緩存**: LRU 緩存機制，自動資源管理
- **MLflow 整合**: 無縫對接模型註冊中心
- **高性能**: 支持 GPU 加速和並發推理
- **可監控**: 內置健康檢查和統計功能
- **可擴展**: 插件式架構，輕鬆添加新模型

### 支援引擎

| 引擎 | 支援任務 | 特點 |
|-----|---------|------|
| **Ollama** | text-generation | 輕量級、易部署、資源占用少 |
| **Transformers** | 所有任務 | 功能完整、支持多模態、GPU 加速 |

---

## 架構設計

### 整體架構

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer                            │
│              /inference/infer                           │
│  統一推理端點，處理所有類型的推理請求                      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 Manager Layer                           │
│           InferenceManager (Singleton)                  │
│  • 請求驗證與預處理                                       │
│  • 任務路由與執行協調                                     │
│  • 資源管理與緩存控制                                     │
│  • 統計與監控                                            │
└──────────┬─────────────────────────┬────────────────────┘
           │                         │
┌──────────▼──────────┐    ┌────────▼────────────────────┐
│   TaskRouter        │    │   ModelExecutor             │
│  • 任務類型識別      │    │  • 模型加載與緩存            │
│  • 引擎選擇         │    │  • 預處理/後處理協調         │
│  • 處理器映射       │    │  • 推理執行                 │
└──────────┬──────────┘    └────────┬────────────────────┘
           │                         │
┌──────────▼─────────────────────────▼────────────────────┐
│                   Engine Layer                          │
├──────────────────────┬──────────────────────────────────┤
│   OllamaEngine       │   TransformersEngine             │
│  • Ollama API 調用   │  • HuggingFace Pipeline          │
│  • 模型管理          │  • 多模態支持                     │
│  • 自動拉取          │  • GPU/CPU 自動選擇              │
└──────────────────────┴──────────────────────────────────┘
           │                         │
┌──────────▼─────────────────────────▼────────────────────┐
│              Model Handlers Layer                       │
│  TextGeneration │ VLM │ ASR │ OCR │ Audio │ Video │ Doc │
│  • 數據預處理                                            │
│  • 格式轉換                                              │
│  • 結果後處理                                            │
└─────────────────────────────────────────────────────────┘
```

### 核心組件

#### 1. **InferenceManager** - 推理管理器

統一的推理入口，負責：
- 請求路由與分發
- 資源管理與優化
- 緩存控制與清理
- 統計數據收集

```python
# 單例模式，全局唯一實例
from src.inference import inference_manager

result = inference_manager.infer(
    task="text-generation",
    engine="ollama",
    model_name="llama2-7b",
    data={"inputs": "Hello"},
    options={}
)
```

#### 2. **TaskRouter** - 任務路由器

智能任務分發，負責：
- 任務類型驗證
- 引擎選擇與實例化
- 處理器映射
- 執行器創建

#### 3. **ModelExecutor** - 模型執行器

模型執行協調，負責：
- 模型加載與管理
- 預處理/後處理協調
- 推理執行
- 錯誤處理

#### 4. **ModelCache** - 模型緩存

LRU 緩存系統，負責：
- 模型實例緩存
- 內存管理
- 自動清理
- 緩存統計

#### 5. **Engines** - 推理引擎

實際的推理執行層：
- **OllamaEngine**: Ollama 服務調用
- **TransformersEngine**: HuggingFace 模型推理

#### 6. **Model Handlers** - 模型處理器

任務特定的處理邏輯：
- 數據預處理
- 格式驗證
- 結果後處理
- 錯誤處理

---

## 快速開始

### 安裝

```bash
# 克隆專案
git clone <repository-url>
cd AiModelLifecycle

# 安裝依賴
pip install -r requirements.txt
```

### 基本使用

#### 1. 文本生成 (Ollama)

```python
from src.inference import inference_manager

# 使用 Ollama 進行文本生成
result = inference_manager.infer(
    task="text-generation",
    engine="ollama",
    model_name="llama2-7b",
    data={"inputs": "請解釋什麼是深度學習"},
    options={
        "max_length": 200,
        "temperature": 0.7,
        "top_p": 0.9
    }
)

if result['success']:
    print(result['result']['generated_text'])
else:
    print(f"錯誤: {result.get('error')}")
```

#### 2. 視覺語言模型 (VLM)

```python
from PIL import Image

# VLM 推理
result = inference_manager.infer(
    task="vlm",
    engine="transformers",
    model_name="llava-1.5-7b",
    data={
        "image": "/path/to/image.jpg",  # 或 PIL.Image 對象
        "prompt": "詳細描述這張圖片中的內容"
    },
    options={"max_length": 256}
)

print(result['result']['description'])
```

#### 3. 語音識別 (ASR)

```python
# 自動語音識別
result = inference_manager.infer(
    task="asr",
    engine="transformers",
    model_name="whisper-large-v3",
    data={"audio": "/path/to/audio.wav"},
    options={"language": "zh"}
)

print(result['result']['transcription'])
```

#### 4. 使用 API 調用

```bash
# 文本生成
curl -X POST "http://localhost:8009/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation",
    "engine": "ollama",
    "model_name": "llama2-7b",
    "data": {"inputs": "Hello, how are you?"},
    "options": {"max_length": 100, "temperature": 0.7}
  }'

# VLM 推理
curl -X POST "http://localhost:8009/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "vlm",
    "engine": "transformers",
    "model_name": "llava-1.5-7b",
    "data": {
      "image": "/path/to/image.jpg",
      "prompt": "What is in this image?"
    },
    "options": {"max_length": 256}
  }'
```

---

## API 文檔

### 統一推理端點

**POST** `/inference/infer`

執行統一的模型推理任務。

#### 請求格式

```json
{
  "task": "text-generation",          // 必需：任務類型
  "engine": "ollama",                 // 必需：引擎類型
  "model_name": "llama2-7b",          // 必需：模型名稱（MLflow 註冊名）
  "data": {                           // 必需：輸入數據
    "inputs": "你的輸入文本"
  },
  "options": {                        // 可選：推理選項
    "max_length": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

#### 請求參數說明

| 參數 | 類型 | 必需 | 說明 |
|-----|------|-----|------|
| `task` | string | ✅ | 任務類型，見[支援任務](#支援任務) |
| `engine` | string | ✅ | 引擎類型：`ollama` 或 `transformers` |
| `model_name` | string | ✅ | 在 MLflow 中註冊的模型名稱 |
| `data` | object | ✅ | 輸入數據，格式取決於任務類型 |
| `options` | object | ❌ | 推理選項，見[配置說明](#配置說明) |

#### 響應格式

**成功響應** (200 OK):

```json
{
  "success": true,
  "result": {
    "generated_text": "生成的文本內容...",
    "metadata": {
      "tokens": 150,
      "finish_reason": "stop"
    }
  },
  "task": "text-generation",
  "engine": "ollama",
  "model_name": "llama2-7b",
  "processing_time": 1.234,
  "timestamp": 1697145600.123
}
```

**失敗響應** (400/500):

```json
{
  "success": false,
  "error": "錯誤描述",
  "error_type": "ModelNotFoundError",
  "task": "text-generation",
  "engine": "ollama",
  "model_name": "llama2-7b",
  "timestamp": 1697145600.123
}
```

### 管理端點

#### 健康檢查

**GET** `/inference/health`

```json
{
  "status": "healthy",
  "components": {
    "manager": true,
    "router": true,
    "cache": true
  },
  "timestamp": 1697145600.123
}
```

#### 獲取統計信息

**GET** `/inference/stats`

```json
{
  "total_inferences": 1000,
  "successful_inferences": 980,
  "failed_inferences": 20,
  "success_rate": 0.98,
  "cache_hit_rate": 0.75,
  "cached_models": 5,
  "uptime": 86400.0
}
```

#### 清理緩存

**POST** `/inference/cache/clear`

```json
{
  "success": true,
  "message": "緩存已清理",
  "cleared_models": 5
}
```

---

## 支援任務

### 任務類型列表

| 任務類型 | 引擎支持 | 說明 | 輸入格式 | 輸出格式 |
|---------|---------|------|---------|---------|
| `text-generation` | ollama, transformers | 文本生成 | `{"inputs": str}` | `{"generated_text": str}` |
| `vlm` | transformers | 視覺語言模型 | `{"image": str/PIL.Image, "prompt": str}` | `{"description": str}` |
| `asr` | transformers | 自動語音識別 | `{"audio": str}` | `{"transcription": str}` |
| `ocr` | transformers | 光學字符識別 | `{"image": str}` | `{"text": str}` |
| `audio-classification` | transformers | 音頻分類 | `{"audio": str}` | `{"labels": list, "scores": list}` |
| `video-analysis` | transformers | 視頻分析 | `{"video": str, "prompt": str}` | `{"analysis": str}` |
| `document-analysis` | transformers | 文檔分析 | `{"document": str, "query": str}` | `{"answer": str}` |

### 任務詳細說明

#### 1. Text Generation (文本生成)

**支援引擎**: Ollama, Transformers

用於生成連續的文本內容，包括對話、寫作、翻譯等。

```python
result = inference_manager.infer(
    task="text-generation",
    engine="ollama",  # 或 "transformers"
    model_name="llama2-7b",
    data={
        "inputs": "寫一首關於春天的詩"
    },
    options={
        "max_length": 200,
        "temperature": 0.8,
        "do_sample": True
    }
)
```

#### 2. VLM (視覺語言模型)

**支援引擎**: Transformers

結合視覺和語言理解，用於圖像描述、視覺問答等。

```python
result = inference_manager.infer(
    task="vlm",
    engine="transformers",
    model_name="llava-1.5-7b",
    data={
        "image": "/path/to/image.jpg",
        "prompt": "圖片中有什麼物體？"
    },
    options={"max_length": 256}
)
```

#### 3. ASR (自動語音識別)

**支援引擎**: Transformers

將語音轉換為文本。

```python
result = inference_manager.infer(
    task="asr",
    engine="transformers",
    model_name="whisper-large-v3",
    data={"audio": "/path/to/audio.wav"},
    options={"language": "zh", "task": "transcribe"}
)
```

#### 4. OCR (光學字符識別)

**支援引擎**: Transformers

從圖像中提取文本。

```python
result = inference_manager.infer(
    task="ocr",
    engine="transformers",
    model_name="trocr-base",
    data={"image": "/path/to/document.jpg"},
    options={}
)
```

#### 5. Audio Classification (音頻分類)

**支援引擎**: Transformers

對音頻進行分類，如情感識別、事件檢測等。

```python
result = inference_manager.infer(
    task="audio-classification",
    engine="transformers",
    model_name="wav2vec2-audio-classifier",
    data={"audio": "/path/to/audio.wav"},
    options={}
)
```

#### 6. Video Analysis (視頻分析)

**支援引擎**: Transformers

分析視頻內容，提取關鍵信息。

```python
result = inference_manager.infer(
    task="video-analysis",
    engine="transformers",
    model_name="video-llama",
    data={
        "video": "/path/to/video.mp4",
        "prompt": "描述視頻中發生了什麼"
    },
    options={}
)
```

#### 7. Document Analysis (文檔分析)

**支援引擎**: Transformers

分析文檔並回答相關問題。

```python
result = inference_manager.infer(
    task="document-analysis",
    engine="transformers",
    model_name="layoutlm-v3",
    data={
        "document": "/path/to/document.pdf",
        "query": "合同的有效期是多久？"
    },
    options={}
)
```

---

## 配置說明

### 通用推理選項

適用於所有任務類型的選項：

```python
options = {
    # 生成控制
    "max_length": 512,           # 最大總長度
    "max_new_tokens": 256,       # 最大新生成的 tokens
    "min_length": 10,            # 最小長度
    
    # 採樣策略
    "do_sample": True,           # 是否使用採樣
    "temperature": 0.7,          # 溫度參數 (0.0-2.0)，越高越隨機
    "top_p": 0.9,                # Nucleus sampling 閾值
    "top_k": 50,                 # Top-K sampling 數量
    
    # Beam Search
    "num_beams": 1,              # Beam 數量（1 = 貪婪搜索）
    "early_stopping": False,     # 是否提前停止
    
    # 懲罰參數
    "repetition_penalty": 1.0,   # 重複懲罰 (1.0 = 無懲罰)
    "length_penalty": 1.0,       # 長度懲罰
    "no_repeat_ngram_size": 0,   # 禁止重複的 n-gram 大小
    
    # 其他
    "num_return_sequences": 1,   # 返回序列數量
    "pad_token_id": None,        # Padding token ID
    "eos_token_id": None,        # End-of-sequence token ID
}
```

### Ollama 特定選項

僅適用於 Ollama 引擎：

```python
ollama_options = {
    "num_predict": 128,          # 預測的 token 數量
    "repeat_penalty": 1.1,       # 重複懲罰
    "temperature": 0.8,          # 溫度
    "top_p": 0.9,                # Top-P
    "top_k": 40,                 # Top-K
    
    # Mirostat 採樣
    "mirostat": 0,               # Mirostat 模式 (0=關閉, 1, 2)
    "mirostat_tau": 5.0,         # Mirostat 目標熵
    "mirostat_eta": 0.1,         # Mirostat 學習率
    
    # 上下文
    "num_ctx": 2048,             # 上下文窗口大小
    "num_thread": None,          # 線程數
}
```

### Transformers 特定選項

僅適用於 Transformers 引擎：

```python
transformers_options = {
    # 設備配置
    "device": "auto",            # 'auto', 'cuda', 'cpu', 'cuda:0'
    "device_map": "auto",        # 設備映射策略
    "torch_dtype": "auto",       # 'auto', 'float16', 'float32', 'int8'
    
    # 信任設置
    "trust_remote_code": True,   # 是否信任遠程代碼
    
    # 批處理
    "batch_size": 1,             # 批處理大小
    
    # 特殊任務選項
    "language": "zh",            # ASR: 語言代碼
    "task": "transcribe",        # ASR: 'transcribe' 或 'translate'
    "return_timestamps": False,  # ASR: 是否返回時間戳
}
```

### 引擎配置文件

在 `engines_config.yaml` 中配置引擎：

```yaml
# Ollama 引擎配置
ollama:
  api_base: "http://localhost:11434"
  timeout: 300
  auto_pull: true              # 自動拉取不存在的模型
  verify_ssl: false            # SSL 驗證

# Transformers 引擎配置
transformers:
  device: "auto"               # 'auto', 'cuda', 'cpu'
  torch_dtype: "auto"          # 'auto', 'fp16', 'fp32'
  trust_remote_code: true
  cache_dir: "./cache/models"  # 模型緩存目錄
  
# 模型緩存配置
cache:
  max_size: 5                  # 最大緩存模型數
  ttl: 3600                    # 緩存存活時間（秒）
  strategy: "lru"              # 緩存策略：'lru', 'lfu'
```

---

## 擴展開發

### 添加自定義 Model Handler

如果需要支持新的任務類型，可以創建自定義的 Model Handler：

```python
# my_custom_handler.py
from src.inference.models.base import BaseModelHandler
from typing import Dict, Any

class CustomTaskHandler(BaseModelHandler):
    """
    自定義任務處理器
    
    用於處理特定的任務邏輯
    """
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        預處理輸入數據
        
        Args:
            data: 原始輸入數據
                - custom_input: 自定義輸入字段
            options: 處理選項
            
        Returns:
            預處理後的數據字典
            
        Raises:
            ValueError: 如果輸入數據無效
        """
        # 驗證必需字段
        self.validate_input(data, ['custom_input'])
        
        # 處理邏輯
        processed_input = self._transform_input(data['custom_input'])
        
        return {
            'inputs': processed_input,
            'options': options
        }
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        後處理推理結果
        
        Args:
            result: 原始推理結果
            options: 處理選項
            
        Returns:
            格式化的結果字典
        """
        # 格式化輸出
        formatted_output = self._format_result(result)
        
        return {
            'custom_output': formatted_output,
            'metadata': {
                'processing_method': 'custom',
                'version': '1.0'
            }
        }
    
    def _transform_input(self, custom_input):
        """自定義輸入轉換邏輯"""
        # 實現你的轉換邏輯
        return transformed_input
    
    def _format_result(self, result):
        """自定義結果格式化邏輯"""
        # 實現你的格式化邏輯
        return formatted_result

# 註冊處理器
from src.inference.registry import model_registry

model_registry.register_handler_manually(
    task='custom-task',
    model_pattern='default',
    handler_class=CustomTaskHandler
)
```

### 添加自定義引擎

如果需要支持新的推理引擎（如 vLLM、TensorRT）：

```python
# my_custom_engine.py
from src.inference.engines.base import BaseEngine
from typing import Any, Dict, Optional

class CustomEngine(BaseEngine):
    """
    自定義推理引擎
    
    用於集成特定的推理框架
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化引擎
        
        Args:
            config: 引擎配置
        """
        super().__init__(config)
        
        # 設置支持的任務
        self.supported_tasks = ['custom-task', 'text-generation']
        
        # 初始化引擎特定的組件
        self._init_custom_components()
        
        self.is_initialized = True
    
    def load_model(self, model_name: str, **kwargs) -> Any:
        """
        加載模型
        
        Args:
            model_name: 模型名稱
            **kwargs: 額外參數
                - model_path: 模型路徑
                - task: 任務類型
                
        Returns:
            加載的模型對象
            
        Raises:
            FileNotFoundError: 模型不存在
            RuntimeError: 加載失敗
        """
        # 獲取模型路徑（可以從 MLflow 獲取）
        model_path = kwargs.get('model_path') or self._get_model_path(model_name)
        
        # 加載模型
        model = self._load_custom_model(model_path)
        
        return model
    
    def infer(self, model: Any, inputs: Dict[str, Any], options: Dict[str, Any]) -> Any:
        """
        執行推理
        
        Args:
            model: 模型對象
            inputs: 輸入數據
            options: 推理選項
            
        Returns:
            推理結果
            
        Raises:
            RuntimeError: 推理失敗
        """
        # 執行推理
        result = model.predict(inputs, **options)
        
        return {
            'response': result,
            'metadata': {
                'engine': 'custom',
                'model_version': model.version
            }
        }
    
    def unload_model(self, model: Any) -> bool:
        """
        卸載模型
        
        Args:
            model: 要卸載的模型
            
        Returns:
            是否成功卸載
        """
        # 清理資源
        if hasattr(model, 'cleanup'):
            model.cleanup()
        
        return True
    
    def _init_custom_components(self):
        """初始化引擎特定組件"""
        pass
    
    def _get_model_path(self, model_name: str) -> str:
        """從 MLflow 獲取模型路徑"""
        from src.core.model_manager import model_manager
        model_info = model_manager.get_model_details_from_mlflow(model_name)
        return model_info['tags'].get('physical_path', model_name)
    
    def _load_custom_model(self, model_path: str):
        """加載自定義模型"""
        # 實現模型加載邏輯
        pass

# 在 router.py 中註冊引擎
# self._task_engine_mapping['custom-task'] = {
#     'custom': CustomEngine
# }
```

### 註冊流程

1. **自動註冊** (推薦):
   
   在 `src/inference/models/__init__.py` 中導入你的處理器：
   
   ```python
   from .my_custom_handler import CustomTaskHandler
   ```

2. **手動註冊**:
   
   ```python
   from src.inference.registry import model_registry
   from my_custom_handler import CustomTaskHandler
   
   model_registry.register_handler_manually(
       task='custom-task',
       model_pattern='default',
       handler_class=CustomTaskHandler
   )
   ```

3. **在 Router 中添加引擎映射**:
   
   編輯 `src/inference/router.py`：
   
   ```python
   self._task_engine_mapping['custom-task'] = {
       'custom': CustomEngine,
       'transformers': TransformersEngine
   }
   ```

---

### 測試覆蓋

模組包含以下測試：

```
✅ 模組導入測試
✅ 引擎初始化測試
✅ 管理器初始化測試
✅ 任務路由器測試
✅ Model Handlers 測試
✅ API 兼容性測試
✅ 緩存系統測試
✅ 錯誤處理測試
```

### 手動測試示例

```python
# test_inference.py
from src.inference import inference_manager

def test_text_generation():
    """測試文本生成"""
    result = inference_manager.infer(
        task="text-generation",
        engine="ollama",
        model_name="llama2-7b",
        data={"inputs": "Hello, world!"},
        options={"max_length": 50}
    )
    
    assert result['success'] == True
    assert 'generated_text' in result['result']
    print("✅ 文本生成測試通過")

def test_health_check():
    """測試健康檢查"""
    health = inference_manager.health_check()
    
    assert health['status'] == 'healthy'
    assert all(health['components'].values())
    print("✅ 健康檢查測試通過")

def test_stats():
    """測試統計功能"""
    stats = inference_manager.get_stats()
    
    assert 'total_inferences' in stats
    assert 'success_rate' in stats
    print("✅ 統計功能測試通過")

if __name__ == "__main__":
    test_text_generation()
    test_health_check()
    test_stats()
    print("\n🎉 所有測試通過！")
```

---

## 故障排除

### 常見問題

#### 1. Ollama 連接失敗

**錯誤**: `ConnectionError: Failed to connect to Ollama server`

**解決方案**:
```bash
# 檢查 Ollama 服務是否運行
curl http://localhost:11434/api/tags

# 啟動 Ollama 服務
ollama serve

# 修改配置文件中的 api_base
# engines_config.yaml
ollama:
  api_base: "http://your-server:11434"
```

#### 2. 模型未找到

**錯誤**: `ModelNotFoundError: Model 'model-name' not found in MLflow`

**解決方案**:
```python
# 檢查模型是否已註冊
from src.core.model_manager import model_manager

# 列出所有模型
models = model_manager.list_models_from_mlflow()
print(models)

# 註冊模型
model_manager.register_model_to_mlflow(
    model_name="your-model",
    model_source="huggingface",
    model_path="/path/to/model",
    tags={"task": "text-generation", "engine": "transformers"}
)
```

#### 3. GPU 內存不足

**錯誤**: `RuntimeError: CUDA out of memory`

**解決方案**:
```python
# 方案 1: 使用 CPU
result = inference_manager.infer(
    task="text-generation",
    engine="transformers",
    model_name="model",
    data={"inputs": "..."},
    options={"device": "cpu"}
)

# 方案 2: 清理緩存
inference_manager.clear_cache()

# 方案 3: 使用較小的模型
# 選擇 7B 而非 13B/70B 模型

# 方案 4: 使用量化模型
# 選擇 int8 或 int4 量化版本
```

#### 4. 推理速度慢

**問題**: 推理時間過長

**優化方案**:
```python
# 1. 使用 GPU
options = {"device": "cuda"}

# 2. 減小生成長度
options = {"max_length": 128}  # 而非 512

# 3. 使用緩存
# 重複使用同一模型，利用模型緩存

# 4. 調整 batch_size（如果支持）
options = {"batch_size": 4}

# 5. 使用量化模型
options = {"torch_dtype": "float16"}
```

#### 5. 函數參數錯誤

**錯誤**: `ValueError: Invalid input format`

**解決方案**:
```python
# 檢查任務的輸入格式要求
# text-generation
data = {"inputs": "your text"}

# vlm
data = {
    "image": "/path/to/image.jpg",  # 或 PIL.Image
    "prompt": "your prompt"
}

# asr
data = {"audio": "/path/to/audio.wav"}

# 確保必需字段都存在
```

### 調試技巧

#### 啟用詳細日誌

```python
import logging

# 設置日誌級別
logging.basicConfig(level=logging.DEBUG)

# 或針對特定模組
logging.getLogger('src.inference').setLevel(logging.DEBUG)
```

#### 檢查系統狀態

```python
from src.inference import inference_manager

# 健康檢查
health = inference_manager.health_check()
print("健康狀態:", health)

# 統計信息
stats = inference_manager.get_stats()
print("統計信息:", stats)

# 緩存狀態
cached = inference_manager.get_cached_models()
print("緩存的模型:", cached)
```

#### 驗證配置

```python
from src.inference.engines.ollama import OllamaEngine
from src.inference.engines.transformers import TransformersEngine

# 檢查引擎配置
ollama_engine = OllamaEngine()
print("Ollama 配置:", ollama_engine.config)

transformers_engine = TransformersEngine()
print("Transformers 配置:", transformers_engine.config)
```

---

## 最佳實踐

### 1. 模型選擇

- **Ollama**: 適用於文本生成，資源占用少，部署簡單
- **Transformers**: 適用於多模態任務，功能完整，GPU 加速

### 2. 性能優化

```python
# ✅ 好的做法
# 重用模型實例（利用緩存）
for text in texts:
    result = inference_manager.infer(
        task="text-generation",
        engine="ollama",
        model_name="llama2-7b",  # 同一模型
        data={"inputs": text},
        options={}
    )

# ❌ 避免
# 頻繁切換模型導致重複加載
for model_name in models:
    result = inference_manager.infer(
        task="text-generation",
        engine="ollama",
        model_name=model_name,  # 不同模型
        data={"inputs": "test"},
        options={}
    )
```

### 3. 錯誤處理

```python
# ✅ 推薦的錯誤處理
from src.inference import inference_manager, InferenceError

try:
    result = inference_manager.infer(...)
    
    if result['success']:
        output = result['result']
        # 處理成功結果
    else:
        # 記錄錯誤
        logger.error(f"推理失敗: {result.get('error')}")
        # 實現降級策略
        
except InferenceError as e:
    # 處理已知的推理錯誤
    logger.error(f"推理錯誤: {e}")
    
except Exception as e:
    # 處理未預期的錯誤
    logger.exception(f"未預期的錯誤: {e}")
```

### 4. 資源管理

```python
# 定期清理緩存
import schedule

def cleanup_cache():
    inference_manager.clear_cache()
    logger.info("緩存已清理")

# 每小時清理一次
schedule.every(1).hours.do(cleanup_cache)

# 在應用關閉時清理
import atexit
atexit.register(inference_manager.clear_cache)
```

### 5. 生產環境建議

```python
# 1. 配置適當的超時
options = {"timeout": 60}  # 60 秒超時

# 2. 實現重試機制
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def robust_infer(**kwargs):
    return inference_manager.infer(**kwargs)

# 3. 監控資源使用
import psutil

def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    if memory.percent > 90:
        logger.warning("內存使用率過高，清理緩存")
        inference_manager.clear_cache()

# 4. 記錄詳細日誌
logger.info(f"推理請求: task={task}, model={model_name}")
logger.info(f"推理結果: success={result['success']}, time={result['processing_time']}")
```

---

## 文件結構

```
src/inference/
├── __init__.py                 # 模組入口，導出主要組件
├── README.md                   # 本文檔
├── USAGE_GUIDE.md             # 詳細使用指南
├── REFACTORING_SUMMARY.md     # 重構總結
├── manager.py                  # 推理管理器（核心）
├── router.py                   # 任務路由器
├── executor.py                 # 模型執行器
├── registry.py                 # 模型註冊表
├── cache.py                    # 模型緩存管理
├── vram_estimator.py          # VRAM 估算工具
├── engines_config.yaml        # 引擎配置文件
│
├── engines/                    # 推理引擎實現
│   ├── __init__.py
│   ├── base.py                # 引擎基類（抽象接口）
│   ├── ollama.py             # Ollama 引擎實現
│   └── transformers.py       # Transformers 引擎實現
│
└── models/                     # 任務處理器實現
    ├── __init__.py
    ├── base.py                # 處理器基類
    ├── text_generation.py    # 文本生成處理器
    ├── vlm.py                 # 視覺語言模型處理器
    ├── asr.py                 # 語音識別處理器
    ├── ocr.py                 # 文字識別處理器
    ├── audio_classification.py  # 音頻分類處理器
    ├── video_analysis.py     # 視頻分析處理器
    └── document_analysis.py  # 文檔分析處理器
```

## 相關資源

### 內部文檔

- [詳細使用指南](./USAGE_GUIDE.md) - 完整的使用說明和範例
- [重構總結](./REFACTORING_SUMMARY.md) - 重構過程和成果
- [API 文檔](../api/inference_api.py) - FastAPI 路由實現

### 外部資源

- [MLflow 文檔](https://mlflow.org/docs/latest/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ollama 文檔](https://github.com/ollama/ollama)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)

### 相關模組

- `src/core/model_manager.py` - 模型管理器
- `src/core/gpu_manager.py` - GPU 資源管理
- `src/core/config.py` - 配置管理
- `src/api/inference_api.py` - API 路由
