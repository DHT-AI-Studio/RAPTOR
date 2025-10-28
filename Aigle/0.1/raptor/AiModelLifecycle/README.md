# AI 模型生命週期管理平台

這是一個全面的 AI 模型生命週期管理（MLOps）平台，旨在自動化和標準化從模型下載、版本控制、註冊、部署到線上推論的整個流程。此平台專為需要持續迭代和可靠維運 AI 模型的團隊而設計。

## 快速開始

### Ollama 模型 5 分鐘部署
如果您已經有本地 Ollama 模型，可以在 5 分鐘內完成註冊和推論：

```bash
# 1. 確認 Ollama 模型（兩種方式）
ollama list
# 或使用 API: 
curl "http://192.168.157.165:8009/models/local?model_source=ollama"

# 2. 啟動服務
docker compose up -d

# 3. 註冊模型（使用 Swagger UI 或 curl）
curl -X POST "http://192.168.157.165:8009/models/register_ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "local_model_name": "qwen3:1.7b",
    "task": "text-generation-ollama",
    "registered_name": "my-qwen-model",
    "stage": "production"
  }'

# 4. 開始推論
curl -X POST "http://192.168.157.165:8009/inference/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "text-generation-ollama",
    "engine": "ollama",
    "model_name": "my-qwen-model",
    "data": {"inputs": "你好，請介紹一下你自己。"}
  }'
```

詳細步驟請參考 [工作流 A：Ollama 模型快速註冊與推論](#工作流-aollama-模型快速註冊與推論)

---

## 目錄
1.  [專案概覽](#專案概覽)
2.  [核心功能](#核心功能)
3.  [目錄結構](#目錄結構)
4.  [工作流設計](#工作流設計)
5.  [部署與使用指南](#部署與使用指南)
    -   [前置準備](#前置準備)
    -   [步驟一：設定環境](#步驟一設定環境)
    -   [步驟二：啟動所有服務](#步驟二啟動所有服務)
    -   [步驟三：訪問服務介面](#步驟三訪問服務介面)
    -   [範例工作流](#範例工作流)
6.  [API 端點說明](#api-端點說明)
7.  [常見問題](#常見問題)
---
## 專案概覽
本專案提供了一個以 MLflow 為核心、LakeFS 為版本控制基礎的 MLOps 解決方案，其核心組件包括：
-   **MLflow Tracking Server**: 所有實驗的中心樞紐，用於記錄和查詢參數、指標、程式碼版本以及模型成品 (Artifacts)。
-   **MLflow Model Registry**: 集中管理模型的生命週期（透過標籤實現 Staging, Production, Archived），並提供版本化的模型註冊表。
-   **LakeFS 作為模型與數據儲存庫**: 所有模型和數據集都儲存在 LakeFS 中，利用其類 Git 的特性實現資料的原子化版本控制。
-   **推論閘道 (Inference Gateway)**: 一個 FastAPI 應用，能根據 MLflow 中的模型類型，將請求分發到不同的後端推論引擎（目前主要整合 Ollama）。
-   **統一管理 API**: 透過 FastAPI 提供一組 RESTful API，用以控制模型的下載、上傳、註冊和階段轉換。

### 核心功能
-   **模型來源無縫整合**: 支援從 HuggingFace 直接下載模型，或納管本地的 Ollama 模型。
-   **數據與模型溯源**: 每個在 MLflow 中註冊的模型版本，都可透過標籤精確追溯到其儲存在 LakeFS 中的 commit ID。
-   **Git-like 數據版本控制**: 底層使用 LakeFS 儲存，允許對數據和模型進行分支、提交、合併等原子操作，確保實驗的可重複性。
-   **統一推理系統**: 
    -   **單一 API 端點**: 所有任務類型通過統一的 `/infer` 端點處理
    -   **清晰架構**: 3 層結構（Engine → Manager → API）設計
    -   **智能路由**: 根據任務類型和引擎自動選擇最佳處理器
    -   **LRU 緩存**: 自動模型緩存管理，優化資源使用
-   **多引擎支持**: 原生整合 Ollama 和 Transformers 引擎，支持從文本生成到多模態分析的各種任務
-   **MLflow 深度整合**: 所有模型從 MLflow Model Registry 自動獲取，支持 Ollama 模型的無縫註冊與推理
-   **可擴展架構**: 插件式設計，輕鬆添加新的任務處理器和引擎
-   **基於標籤的階段管理**: 靈活地透過 API 將模型版本標記為 `production`, `staging`, 或 `archived`，以控制其生命週期狀態。

## 目錄結構
```bash
AiModelLifecycle/
├── src/
│   ├── api/                  # API 路由模組
│   │   ├── models_api.py       # 模型管理相關 API
│   │   └── inference_api.py    # 推論相關 API
│   ├── core/                 # 核心業務邏輯
│   │   ├── configs/
│   │   │   └── base.yaml       # 應用程式設定檔
│   │   ├── config.py           # 設定管理器  
│   │   ├── model_manager.py    # 實現模型下載、註冊、查詢等邏輯
│   │   └── gpu_manager.py      # GPU 資源管理
│   ├── inference/            # 推理核心模塊
│   │   ├── manager.py          # InferenceManager: 統一推理管理器（單例）
│   │   ├── router.py           # TaskRouter: 任務路由與分發
│   │   ├── executor.py         # ModelExecutor: 模型執行協調器
│   │   ├── cache.py            # ModelCache: LRU 模型緩存
│   │   ├── registry.py         # ModelRegistry: 模型處理器註冊表
│   │   ├── engines/          # 推理引擎層
│   │   │   ├── base.py           # BaseEngine: 引擎抽象基類
│   │   │   ├── ollama.py         # OllamaEngine: Ollama 引擎實現
│   │   │   └── transformers.py   # TransformersEngine: HuggingFace 引擎
│   │   └── models/           # 模型處理器層（預處理/後處理）
│   │       ├── base.py               # BaseModelHandler: 處理器基類
│   │       ├── text_generation.py    # 文本生成處理器
│   │       ├── vlm.py                # 視覺語言模型處理器
│   │       ├── asr.py                # 語音識別處理器
│   │       ├── ocr.py                # 文字識別處理器
│   │       ├── audio_classification.py
│   │       ├── video_analysis.py
│   │       └── document_analysis.py
│   ├── main.py               # FastAPI 應用入口
│   └── __init__.py
├── tmp/               # 模型和數據集暫存區
│   ├── models/           # 下載的模型暫存
│   └── datasets/         # 下載的數據集暫存
├── data/              # 模型和數據集儲存區(已註冊，從lakeFS下載下來的，推理用)
│   ├── models/           
│   └── datasets/         
├── .env                  # 環境變數 
├── docker-compose.yaml   # 服務編排文件
├── Dockerfile            # API 服務的容器定義
└── requirements.txt      # Python 依賴列表
```

## 工作流設計
```mermaid
---
config:
  layout: elk
---
flowchart TD
 subgraph s1["模型準備與版本控制"]
        B("本地暫存區 /tmp/models")
        A["外部來源 HuggingFace"]
        C["本地 Ollama 模型"]
        D{"LakeFS 模型儲存庫"}
        E["生成唯一的 Commit ID"]
  end
 subgraph s2["模型註冊與生命週期管理"]
        F("MLflow Model Registry")
        G["模型版本, 包含指向LakeFS的標籤"]
        H@{ label: "設置標籤: stage='production'" }
  end
 subgraph s3["線上推論服務"]
        J("統一推論端點 /infer")
        I["用戶端應用"]
        K["InferenceManager"]
        L{"Ollama / Transformers"}
        M["模型處理器"]
  end
    A -- 下載 --> B
    C -- (本地已存在) --> B
    B -- API觸發上傳 --> D
    D -- Commit --> E
    E -- API觸發註冊 --> F
    F -- 註冊模型版本 --> G
    G -- API觸發階段轉換 --> H
    I -- API請求 --> J
    J -- "1.任務路由" --> K
    K -- "2.查詢 MLflow" --> H
    K -- "3.選擇引擎" --> L
    L -- "4.執行推論" --> M
    M -- "5.預處理/後處理" --> L
    L -- 返回結果 --> K
    K -- 統一響應 --> J
    J -- 回應 --> I

    H@{ shape: diamond}
```

---

## 統一推理系統

### 概述
本平台提供簡化、統一的 AI 模型推理接口，採用清晰的 3 層架構設計，兼具高可維護性和優異性能。

### 架構設計

```mermaid
---
config:
  layout: elk
---
flowchart TD
    subgraph A[API Layer]
        A1[統一推理端點<br>/inference/infer<br>處理所有任務類型]
    end

    subgraph B[Manager Layer]
        B1[InferenceManager<br>Singleton]
        B1 --> B2[任務路由與執行協調]
        B1 --> B3[資源管理與緩存控制]
        B1 --> B4[統計與監控]
    end

    subgraph C[执行层]
        C1[TaskRouter<br>• 任務類型識別<br>• 引擎選擇<br>• 處理器映射]
        C2[ModelExecutor<br>• 模型加載與緩存<br>• 預處理/後處理協調<br>• 推理執行]
    end

    subgraph D[Engine Layer]
        D1[OllamaEngine<br>• Ollama API 調用<br>• 模型管理<br>• 自動拉取]
        D2[TransformersEngine<br>• HuggingFace Pipeline<br>• 多模態支持<br>• GPU/CPU 自動選擇]
    end

    subgraph E[Model Handlers Layer]
        E1[TextGeneration]
        E2[VLM]
        E3[ASR]
        E4[OCR]
        E5[Audio]
        E6[Video]
        E7[Doc]
    end

    A1 --> B1
    B2 --> C1
    B3 --> C2
    C1 --> D1
    C1 --> D2
    C2 --> D1
    C2 --> D2
    D1 --> E1
    D2 --> E2
    D2 --> E3
    D2 --> E4
    D2 --> E5
    D2 --> E6
    D2 --> E7
```

### 核心組件

#### 1. InferenceManager - 推理管理器
統一的推理入口（單例模式），負責：
- 請求路由與分發
- 資源管理與優化
- 緩存控制與清理
- 統計數據收集

#### 2. TaskRouter - 任務路由器
智能任務分發，負責：
- 任務類型驗證
- 引擎選擇與實例化
- 處理器映射
- 執行器創建

#### 3. ModelExecutor - 模型執行器
模型執行協調，負責：
- 模型加載與管理
- 預處理/後處理協調
- 推理執行
- 錯誤處理

#### 4. ModelCache - 模型緩存
LRU 緩存系統，負責：
- 模型實例緩存
- 內存自動管理
- 自動清理機制
- 緩存統計

### 支援的任務類型

| 任務類型 | 引擎支持 | 說明 | 輸入格式 |
|---------|---------|------|---------|
| `text-generation` | ollama, transformers | 文本生成 | `{"inputs": str}` |
| `vlm` | transformers | 視覺語言模型 | `{"image": str/PIL.Image, "prompt": str}` |
| `asr` | transformers | 自動語音識別 | `{"audio": str}` |
| `ocr` | transformers | 光學字符識別 | `{"image": str}` |
| `audio-classification` | transformers | 音頻分類 | `{"audio": str}` |
| `video-analysis` | transformers | 視頻分析 | `{"video": str, "prompt": str}` |
| `document-analysis` | transformers | 文檔分析 | `{"document": str, "query": str}` |

### 使用示例

#### 1. 文本生成 (Ollama)
```python
from src.inference import inference_manager

result = inference_manager.infer(
    task="text-generation",
    engine="ollama",
    model_name="llama2-7b",
    data={"inputs": "請解釋什麼是深度學習"},
    options={
        "max_length": 200,
        "temperature": 0.7
    }
)
```

#### 2. 視覺語言模型 (VLM)
```python
result = inference_manager.infer(
    task="vlm",
    engine="transformers",
    model_name="llava-1.5-7b",
    data={
        "image": "/path/to/image.jpg",
        "prompt": "描述這張圖片"
    },
    options={"max_length": 256}
)
```

#### 3. 語音識別 (ASR)
```python
result = inference_manager.infer(
    task="asr",
    engine="transformers",
    model_name="whisper-large-v3",
    data={"audio": "/path/to/audio.wav"},
    options={"language": "zh"}
)
```

### 核心特性

#### 智能緩存機制
- **LRU 策略**: 自動管理模型實例，優先保留常用模型
- **內存優化**: 自動卸載長時間未使用的模型
- **統計追蹤**: 記錄緩存命中率和使用情況

#### MLflow 整合
- **自動模型獲取**: 從 MLflow Model Registry 自動獲取模型路徑
- **標籤管理**: 支持通過標籤篩選和識別模型
- **Ollama 映射**: 自動處理 Ollama 模型名稱映射

#### 可擴展架構
- **插件式處理器**: 輕鬆添加新的任務處理器
- **統一接口**: 所有處理器遵循相同的接口規範
- **配置驅動**: 通過配置文件管理引擎和處理器設置

---

## 部署與使用指南
### 前置準備
1.  **容器化工具**: 安裝 Docker。
2.  **Python**: 本地開發建議 3.8+ 版本。
3.  **核心服務**:
    *   **LakeFS**: 確保您有一個正在運行的 LakeFS 實例，並取得其 Endpoint、Access Key 和 Secret Key。
    *   **Ollama**: 確保您在本機或可訪問的伺服器上安裝並運行了 Ollama 服務。

### 步驟一：設定環境
1.  **複製專案**: `git clone <your-repo-url>`
2.  **創建 `.env` 文件**: 在專案根目錄下，根據 `.env.example` (或以下範本) 創建 `.env` 文件，並填入您的 LakeFS 和 MLflow 設定。

    ```ini
    # .env 文件範例

    # --- Postgres for MLflow Backend Store ---
    POSTGRES_USER=mlflow
    POSTGRES_PASSWORD=mlflow
    POSTGRES_DB=mlflow

    # --- LakeFS S3-Compatible Gateway for MLflow Artifact Store ---
    # 這應指向您的 LakeFS S3 Gateway
    MLFLOW_S3_ENDPOINT_URL=http://<YOUR_LAKEFS_IP>:8001
    AWS_ACCESS_KEY_ID=<YOUR_LAKEFS_ACCESS_KEY>
    AWS_SECRET_ACCESS_KEY=<YOUR_LAKEFS_SECRET_KEY>

    # --- MLflow Server Settings ---
    # 在您的 LakeFS 中用於存放 MLflow 成品的 bucket 和路徑
    MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/
    
    # MLflow Server URIs (由 Compose 自動構建)
    MLFLOW_BACKEND_STORE_URI=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
    ```
3.  **配置 `base.yaml`**: 檢查 `src/core/configs/base.yaml` 文件，確保 `mlflow`, `lakefs`, 和 `ollama` 的 `endpoint`/`uri` 指向正確的位址。從容器內訪問宿主機的服務，通常使用 `http://host.docker.internal`。

### 步驟二：啟動所有服務
在專案根目錄下，執行單一指令即可啟動 FastAPI 應用、MLflow Server 和 PostgreSQL 資料庫。
```bash
docker compose up --build -d
```
`--build` 參數會確保您的 API 服務是使用最新的程式碼構建的。  
`-d` 參數會讓服務在背景執行。

### 步驟三：訪問服務介面
-   **FastAPI API 文檔 (Swagger UI)**: [http://localhost:8009/docs](http://localhost:8009/docs)
-   **MLflow UI**: [http://localhost:5000](http://localhost:5000)

### 範例工作流
以下提供常見的工作流程，您可以直接在 Swagger UI 上操作。

#### 工作流 A：Ollama 模型快速註冊與推論
此流程適用於您本地 Ollama 已有並想納管的模型，是最快速的部署方式。

##### 步驟 1：確認本地 Ollama 模型
您可以通過以下兩種方式確認本地 Ollama 模型：

**方式 1：使用終端命令**
```bash
ollama list
```
範例輸出：
```
NAME              ID              SIZE    MODIFIED
qwen3:1.7b        abc123def456    1.0 GB  2 days ago
llama3:8b         def789ghi012    4.7 GB  1 week ago
```

**方式 2：使用 API 查詢**

**API 端點**: `GET /models/local?model_source=ollama`

使用 curl 命令：
```bash
curl "http://localhost:8009/models/local?model_source=ollama"
```

響應範例：
```json
[
  {
    "name": "qwen3:1.7b",
    "size": "1.0 GB",
    "modified": "2 days ago"
  },
  {
    "name": "llama3:8b",
    "size": "4.7 GB",
    "modified": "1 week ago"
  }
]
```

**參數說明**:
- 不帶參數 `GET /models/local`: 列出 HuggingFace 下載的本地模型
- 帶參數 `GET /models/local?model_source=ollama`: 列出 Ollama 本地模型

##### 步驟 2：註冊 Ollama 模型到 MLflow
使用 API 將本地 Ollama 模型註冊到 MLflow Model Registry：

**API 端點**: `POST /models/register_ollama`

**請求參數說明**:
- `local_model_name` (必填): 本地 Ollama 模型的完整名稱（必須包含標籤，如 `:latest`、`:1.7b`）
- `task` (必填): 任務類型，Ollama 模型使用 `"text-generation-ollama"`
- `model_params` (可選): 模型參數量（單位：B），不填則自動估算
- `registered_name` (可選): 在 MLflow 中的註冊名稱，不填則使用 `local_model_name`
- `version_description` (可選): 版本描述
- `stage` (可選): 註冊後直接設定階段，可選 `"production"`, `"staging"`, `"archived"`, `"none"`
- `set_priority_to_one` (可選): 是否設為最高推理優先級

**請求範例**:
```json
{
  "local_model_name": "qwen3:1.7b",
  "task": "text-generation-ollama",
  "model_params": 1.7,
  "registered_name": "qwen3:1.7b-ollama-1",
  "version_description": "Qwen3 1.7B model for production",
  "stage": "staging"
}
```

**響應範例**:
```json
{
  "message": "Ollama 模型 'qwen3:1.7b' 註冊成功",
  "result": {
    "name": "qwen3:1.7b-ollama-1",
    "version": "1",
    "stage": "staging",
    "tags": {
      "inference_task": "text-generation",
      "engine": "ollama",
      "ollama_model_name": "qwen3:1.7b",
      "model_params": "1.7"
    }
  }
}
```

##### 步驟 3：（可選）轉換模型階段
如果在步驟 2 中未指定 `stage`，或需要更改模型階段，使用階段轉換 API：

**API 端點**: `POST /models/transition_stage`

**請求參數說明**:
- `model_name` (必填): MLflow 中註冊的模型名稱
- `version` (必填): 要轉換的模型版本號（字符串格式）
- `stage` (必填): 目標階段，可選 `"production"`, `"staging"`, `"archived"`, `"none"`
- `archive_existing_versions` (可選): 是否將該階段中的其他版本封存，默認 `false`

**請求範例**:
```json
{
  "model_name": "qwen3:1.7b-ollama-1",
  "version": "1",
  "stage": "production",
  "archive_existing_versions": true
}
```

**響應範例**:
```json
{
  "message": "模型 'qwen3:1.7b-ollama-1' 版本 '1' 已成功轉換到階段 'production'"
}
```

##### 步驟 4：執行推論
模型註冊完成後，即可通過統一推理 API 進行調用：

**API 端點**: `POST /inference/infer`

**請求參數說明**:
- `task` (必填): 任務類型，使用 `"text-generation-ollama"` 或 `"text-generation"`
- `engine` (必填): 引擎類型，使用 `"ollama"`
- `model_name` (必填): MLflow 中註冊的模型名稱
- `data` (必填): 輸入數據，包含 `inputs` 字段
- `options` (可選): 推理選項，如 `temperature`, `max_length` 等

**請求範例**:
```json
{
  "task": "text-generation-ollama",
  "engine": "ollama",
  "model_name": "qwen3:1.7b-ollama-1",
  "data": {
    "inputs": "請用繁體中文介紹一下 AI 模型生命週期管理的重要性"
  },
  "options": {
    "temperature": 0.7,
    "max_length": 500
  }
}
```

**響應範例**:
```json
{
  "success": true,
  "result": {
    "generated_text": "AI 模型生命週期管理是現代 MLOps 實踐中至關重要的一環...",
    "metadata": {
      "model": "qwen3:1.7b",
      "tokens": 245,
      "finish_reason": "stop"
    }
  },
  "task": "text-generation-ollama",
  "engine": "ollama",
  "model_name": "qwen3:1.7b-ollama-1",
  "processing_time": 3.85,
  "timestamp": 1697234567.89
}
```

**重要說明**:
- 系統會自動將 MLflow 中註冊的模型名稱（如 `qwen3:1.7b-ollama-1`）轉換為本地 Ollama 模型名稱（如 `qwen3:1.7b`）
- 這個轉換通過 MLflow 模型標籤中的 `ollama_model_name` 實現，確保推理時使用正確的模型
- 使用 `"text-generation"` 任務類型，系統會根據 engine 自動路由到正確的處理器

#### 工作流 B：HuggingFace -> LakeFS -> MLflow
此流程展示了從外部下載模型並進行完整版本控制的過程。

1.  **下載 HuggingFace 模型**:
    *   `POST /models/download`
    *   請求內文:
        ```json
        {
          "model_source": "huggingface",
          "model_name": "google/gemma-2b"
        }
        ```
2.  **上傳模型至 LakeFS**:
    *   `POST /models/upload_to_lakefs`
    *   請求內文 (注意 `local_model_name` 中的 `/` 已被替換為 `_`):
        ```json
        {
          "repo_name": "gemma-models",
          "local_model_name": "google_gemma-2b"
        }
        ```
    *   從回應中記下 `commit_id`。
3.  **從 LakeFS 註冊模型到 MLflow**:
    *   `POST /models/register_from_lakefs`
    *   請求內文:
        ```json
        {
          "registered_name": "gemma-2b-translator",
          "task": "text-generation",
          "lakefs_repo": "gemma-models",
          "commit_id": "<YOUR_COMMIT_ID_FROM_STEP_2>"
        }
        ```
4.  **設為生產環境並推論**: 重複工作流 A 的步驟 3 和 4。

#### 工作流 C：多模態任務推論
此流程展示了統一推理接口如何支持各種多模態任務。

1.  **視覺語言模型（VLM）推論**:
    *   `POST /inference/infer`
    *   請求內文:
        ```json
        {
          "task": "vlm",
          "engine": "transformers",
          "model_name": "llava-1.5-7b",
          "data": {
            "image": "/path/to/image.jpg",
            "prompt": "請詳細描述這張圖片的內容"
          },
          "options": {"max_length": 256}
        }
        ```

2.  **自動語音識別（ASR）**:
    *   `POST /inference/infer`
    *   請求內文:
        ```json
        {
          "task": "asr",
          "engine": "transformers",
          "model_name": "whisper-large-v3",
          "data": {"audio": "/path/to/audio.wav"},
          "options": {"language": "zh", "task": "transcribe"}
        }
        ```

3.  **光學字符識別（OCR）**:
    *   `POST /inference/infer`
    *   請求內文:
        ```json
        {
          "task": "ocr",
          "engine": "transformers",
          "model_name": "trocr-base",
          "data": {"image": "/path/to/document.jpg"},
          "options": {}
        }
        ```

4.  **視頻分析**:
    *   `POST /inference/infer`
    *   請求內文:
        ```json
        {
          "task": "video-analysis",
          "engine": "transformers",
          "model_name": "video-llava",
          "data": {
            "video": "/path/to/video.mp4",
            "prompt": "分析這段視頻的主要內容"
          },
          "options": {"max_length": 300}
        }
        ```

## API 端點說明
所有端點都可以在 [http://localhost:8009/docs](http://localhost:8009/docs) 找到詳細的說明和線上測試界面。

### 模型管理 (`/models`)
| 端點 | 方法 | 主要請求內容/參數 | 描述 |
| --- | --- | --- | --- |
| `/download` | POST | `{ "model_source": string, "model_name": string }` | 從 HuggingFace 或 Ollama 下載/拉取模型。 |
| `/local` | GET | `?model_source=ollama` (可選) | 列出本地暫存的模型，可篩選 Ollama。 |
| `/upload_to_lakefs` | POST | `{ "repo_name": string, "local_model_name": string }` | 將本地模型上傳到 LakeFS 並生成一個 commit。 |
| `/lakefs_repos` | GET | - | 列出 LakeFS 中所有的模型儲存庫。 |
| `/register_from_lakefs` | POST | `{ "registered_name": string, "task": string, "engine": string, ... }` | 將 LakeFS 的一個 commit 註冊為 MLflow 中的模型版本（需指定 task 和 engine）。 |
| `/register_ollama` | POST | `{ "local_model_name": string, "task": string, ... }` | **將本地 Ollama 模型註冊到 MLflow**（自動記錄 ollama_model_name 標籤）。 |
| `/registered_in_mlflow` | GET | `?show_all=bool` (可選) | 列出 MLflow 中所有已註冊的模型。 |
| `/registered_in_mlflow/{model_name}` | GET | `?all_version=bool` (可選) | 獲取特定註冊模型的版本詳細資訊。 |
| `/transition_stage` | POST | `{ "model_name": string, "version": string, "stage": string }` | 轉換 MLflow 中模型版本的階段標籤。 |
| `/batch_download` | POST | `{ "downloads": array }` | 批量下載多個模型。 |
| `/stats` | GET | `?show_all=bool` (可選) | 獲取模型統計信息（階段分布、任務分布等）。 |
| `/health` | GET | - | 檢查模型管理服務健康狀態（MLflow、lakeFS、Ollama 連接狀態）。 |

### 推理服務 (`/inference`)
| 端點 | 方法 | 主要請求內容 | 描述 |
| --- | --- | --- | --- |
| `/infer` | POST | `{ "task": string, "engine": string, "model_name": string, "data": object, "options": object }` | **統一推理接口**：支持所有任務類型的推論（文本生成、VLM、ASR、OCR 等）。 |
| `/health` | GET | - | 推理系統健康檢查（管理器、路由器、緩存狀態）。 |
| `/stats` | GET | - | 獲取推理系統統計信息（推理次數、成功率、緩存命中率、已加載模型數等）。 |
| `/cache/clear` | POST | - | 清理推理系統的模型緩存，釋放資源。 |

---

## 常見問題

### Q1: 如何查看本地 Ollama 模型？
**方法 1 - 使用終端命令**:
```bash
ollama list
```

**方法 2 - 使用 API**:
```bash
# 查看 Ollama 模型
curl "http://localhost:8009/models/local?model_source=ollama"

# 查看 HuggingFace 下載的模型
curl "http://localhost:8009/models/local"
```

### Q2: Ollama 模型名稱映射問題
**問題**: 推論時出現 "model 'xxx-ollama-1' not found" 錯誤

**解答**: 系統會自動處理 MLflow 註冊名稱與本地 Ollama 模型名稱的映射。註冊時會在 MLflow 標籤中記錄 `ollama_model_name`，推論時自動轉換。確保：
- 註冊時使用 `task: "text-generation-ollama"`
- 推論時 `engine: "ollama"`
- 本地 Ollama 中確實存在該模型（使用上述方法確認）

### Q3: 支持哪些任務類型？
**統一推理系統支持的任務類型**:

| 任務類型 | 引擎支持 | 說明 |
|---------|---------|------|
| `text-generation` | ollama, transformers | 文本生成（對話、寫作、翻譯等） |
| `vlm` | transformers | 視覺語言模型（圖像描述、視覺問答） |
| `asr` | transformers | 自動語音識別 |
| `ocr` | transformers | 光學字符識別 |
| `audio-classification` | transformers | 音頻分類 |
| `video-analysis` | transformers | 視頻內容分析 |
| `document-analysis` | transformers | 文檔理解與分析 |

查看完整任務列表：`GET /inference/stats` 會顯示所有已註冊的任務處理器

### Q4: 如何查看本地和已註冊的模型？
```bash
# 查看本地 Ollama 模型
curl http://localhost:8009/models/local?model_source=ollama

# 查看本地 HuggingFace 模型
curl http://localhost:8009/models/local

# 列出 MLflow 中所有註冊模型
curl http://localhost:8009/models/registered_in_mlflow

# 查看特定模型詳情
curl http://localhost:8009/models/registered_in_mlflow/my-qwen-model

# 查看模型統計（階段分布、任務分布等）
curl http://localhost:8009/models/stats
```

### Q5: 推論參數配置
在 `options` 中可以設置的常用參數：

**通用參數**:
- `temperature`: 生成溫度（0.0-2.0），越高越隨機
- `max_length`: 最大生成長度
- `top_p`: nucleus sampling 參數
- `top_k`: top-k sampling 參數
- `do_sample`: 是否使用採樣（boolean）

**任務特定參數**:
- ASR: `language`, `task` (transcribe/translate)
- VLM: `max_new_tokens`, `num_beams`

範例：
```json
{
  "task": "text-generation-ollama",
  "engine": "ollama",
  "model_name": "my-model",
  "data": {"inputs": "你好"},
  "options": {
    "temperature": 0.7,
    "max_length": 500,
    "do_sample": true
  }
}
```

### Q6: 服務健康檢查
```bash
# 檢查模型管理服務
curl http://localhost:8009/models/health

# 檢查推理服務
curl http://localhost:8009/inference/health

# 檢查 MLflow UI
open http://localhost:5000
```

### Q7: 如何更新模型版本？
1. 註冊新版本（版本號自動遞增）
2. 使用 `/transition_stage` 轉換階段
3. 可選：設置 `archive_existing_versions: true` 自動封存舊版本

### Q8: 推理系統的主要特性是什麼？
**架構設計**:
- 清晰的 3 層架構設計，易於理解和維護
- 統一 API 端點，所有任務通過 `/infer` 處理

**性能優化**:
- LRU 模型緩存，自動管理內存
- 智能資源分配，優化 GPU/CPU 使用

**開發體驗**:
- 完整的函數註解
- 深度整合 MLflow
- 插件式架構，易於擴展

**使用方式**:
```python
# 簡潔的統一接口
result = inference_manager.infer(task, engine, model_name, data)
```

### Q9: 如何查看推理系統狀態？
```bash
# 健康檢查
curl http://localhost:8009/inference/health

# 統計信息（推理次數、緩存命中率等）
curl http://localhost:8009/inference/stats

# 清理緩存
curl -X POST http://localhost:8009/inference/cache/clear
```

