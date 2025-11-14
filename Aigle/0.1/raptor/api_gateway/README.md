# RAPTOR API Gateway - Sample Code

## 專案簡介

本專案提供 RAPTOR API Gateway 的完整範例程式碼與使用說明，協助開發者快速整合 API 功能。專案包含 Python 客戶端、cURL 腳本範例、OpenAPI 規格文件，以及一個基於 Flask 的 Web 前端示範應用。

RAPTOR API Gateway 提供多媒體資產管理、檔案上傳處理、智能搜尋、聊天互動等功能，適用於需要處理影片、音訊、文件和圖片的應用場景。

## API URL

- **API Base URL**: http://raptor_open_0_1_api.dhtsolution.com:8012/
- **API 文件 (Swagger UI)**: http://raptor_open_0_1_api.dhtsolution.com:8012/docs

## 快速開始

1. **使用 Python 客戶端**：查看 `sample_code_python.py` 或參考 `SAMPLE_CODE_GUIDE.md`
2. **使用 cURL 腳本**：執行 `sample_code_curl.sh` 中的範例命令
3. **使用 Web 前端**：進入 `web_frontend/` 目錄，參考其 README 啟動 Flask 應用

## 檔案索引

### 核心文件

#### `README.md`
本文件，提供專案概述與檔案索引。

#### `SAMPLE_CODE_GUIDE.md`
詳細的使用指南，包含：
- Python 和 cURL 兩種使用方式的完整說明
- 所有 API 端點的使用範例
- 認證、檔案上傳、搜尋、資產管理、處理、聊天等功能的詳細教學
- 常見問題與注意事項

#### `openapi.json`
OpenAPI 3.1.0 規格文件，包含完整的 API 定義、端點、參數、回應格式等資訊。可匯入 Postman 或其他 API 測試工具使用。

### Python 範例

#### `sample_code_python.py`
完整的 Python 客戶端實作，包含：
- `RaptorAPIClient` 類別，封裝所有 API 端點
- 認證功能（註冊、登入、Token 管理）
- 檔案上傳（單檔、批次、上傳即分析）
- 多媒體搜尋（影片、音訊、文件、圖片、跨集合搜尋）
- 資產管理（列出版本、下載、封存、刪除）
- 資料處理（觸發處理、快取查詢、系統訊息查詢）
- 聊天功能（發送訊息、查詢聊天記錄）
- 健康檢查

**使用方式**：
```python
from sample_code_python import RaptorAPIClient
client = RaptorAPIClient()
client.register_user("username", "email@example.com", "password")
client.login("username", "password")
```

### Shell 腳本範例

#### `sample_code_curl.sh`
使用 cURL 呼叫所有 API 端點的 Shell 腳本範例，包含：
- 認證（註冊、登入）
- 檔案上傳的各種方式
- 所有搜尋端點的呼叫範例
- 資產管理操作
- 資料處理與聊天功能

**使用方式**：
```bash
chmod +x sample_code_curl.sh
# 可以直接執行或複製其中的 curl 命令使用
./sample_code_curl.sh
```

### Web 前端

#### `web_frontend/`
基於 Flask 的示範前端應用，提供圖形化介面來操作 RAPTOR API Gateway。

**主要檔案**：
- `app.py` - Flask 應用主程式
- `requirements.txt` - Python 依賴套件列表
- `templates/index.html` - 前端 HTML 模板
- `static/styles.css` - 樣式表
- `README.md` - Web 前端詳細說明

**功能特色**：
- 圖形化的 API 操作界面
- 分頁導覽（搜尋、上傳、資產、處理、聊天等）
- JWT Token 自動管理
- 資產路徑與版本 ID 自動記錄
- 支援所有 API 功能

**啟動方式**：
```bash
cd web_frontend

# 創建環境
conda create -n raptor_env python=3.10
conda activate raptor_env
pip install uv
uv pip install -r requirements.txt
python app.py
```

## 主要功能

### 認證系統
- 使用者註冊與登入
- JWT Token 驗證機制

### 檔案上傳
- 單檔上傳
- 批次上傳
- 上傳即分析（Upload-and-Analyze）

### 智能搜尋
- 影片內容搜尋（Video Search）
- 音訊內容搜尋（Audio Search）
- 文件內容搜尋（Document Search）
- 圖片內容搜尋（Image Search）
- 跨集合搜尋（Cross-Collection Search）

### 資產管理
- 列出資產版本
- 下載資產檔案
- 封存資產
- 刪除資產

### 資料處理
- 觸發檔案處理
- 快取資料查詢
- 系統訊息查詢

### 聊天功能
- 發送聊天訊息
- 查詢聊天記錄

### 系統健康
- API 健康檢查端點

## 技術支援

如有問題或需要協助，請參考：
- API 文件：http://raptor_open_0_1_api.dhtsolution.com:8012/docs
- 範例程式碼使用指南：`SAMPLE_CODE_GUIDE.md`

