# RAPTOR API Gateway - Sample Code Usage Guide

## Overview

This directory contains complete sample code for the RAPTOR API Gateway, including both Python and cURL methods.

- **API Base URL**: http://raptor_open_0_1_api.dhtsolution.com:8012
- **API Documentation**: http://raptor_open_0_1_api.dhtsolution.com:8012/docs

## File Descriptions

### 1. `sample_code_python.py`
Complete Python client implementation containing wrappers for all API endpoints.

### 2. `sample_code_curl.sh`
Shell script examples using cURL to call all API endpoints.

### 3. `openapi.json`
OpenAPI 3.1.0 specification file containing complete API definitions.

---

## Python Usage

### Install Dependencies

```bash
pip install requests
```

### Basic Usage

#### 1. Import and Initialize Client

```python
from sample_code_python import RaptorAPIClient

# Initialize client
client = RaptorAPIClient()
```

#### 2. Register and Login

```python
# Register new user
client.register_user(
    username="your_username",
    email="your_email@example.com",
    password="your_password"
)

# Login to get token
token = client.login(
    username="your_username",
    password="your_password"
)
print(f"Token: {token}")
```

#### 3. Search Functions

##### Video Search
```python
result = client.video_search(
    query_text="machine learning",
    embedding_type="text",
    filename=["MV.mp4"],
    speaker=["SPEAKER_00"],
    limit=5
)
print(result)
```

##### Audio Search
```python
result = client.audio_search(
    query_text="machine learning",
    embedding_type="text",
    filename=["audio.mp3"],
    limit=5
)
```

##### Document Search
```python
result = client.document_search(
    query_text="financial report",
    embedding_type="text",
    filename=["EF25Y01.csv"],
    source="csv",
    limit=5
)
```

##### Image Search
```python
result = client.image_search(
    query_text="landscape photo",
    embedding_type="summary",
    filename=["photo.jpg"],
    source="jpg",
    limit=5
)
```

##### Unified Search (Across Multiple Collections)
```python
result = client.unified_search(
    query_text="machine learning and artificial intelligence",
    embedding_type="text",
    filters={
        "video": {"filename": ["MV.mp4"]},
        "document": {"source": "csv"}
    },
    limit_per_collection=5,
    global_limit=20,
    score_threshold=0.5
)
```

#### 4. File Upload

##### Upload Single File
```python
result = client.upload_file(
    file_path="/path/to/your/file.pdf",
    archive_ttl=30,
    destroy_ttl=30
)
```

##### Batch Upload
```python
result = client.upload_files_batch(
    file_paths=[
        "/path/to/file1.pdf",
        "/path/to/file2.pdf",
        "/path/to/file3.pdf"
    ],
    archive_ttl=30,
    destroy_ttl=30,
    concurrency=4
)
```

##### Upload with Automatic Analysis
```python
result = client.upload_file_with_analysis(
    file_path="/path/to/document.pdf",
    processing_mode="default",
    archive_ttl=30,
    destroy_ttl=30
)
```

##### Batch Upload with Analysis
```python
result = client.upload_files_batch_with_analysis(
    file_paths=["/path/to/file1.pdf", "/path/to/file2.pdf"],
    processing_mode="default",
    archive_ttl=30,
    destroy_ttl=30,
    concurrency=4
)
```

#### 5. Asset Management

##### List File Versions
```python
versions = client.list_file_versions(
    asset_path="my_assets",
    filename="document.pdf"
)
```

##### Download File
```python
file_content = client.download_asset(
    asset_path="my_assets",
    version_id="v1234",
    return_file_content=True
)

# Save to local
with open("downloaded_file.pdf", "wb") as f:
    f.write(file_content)
```

##### Archive File
```python
result = client.archive_asset(
    asset_path="my_assets",
    version_id="v1234"
)
```

##### Delete File
```python
result = client.delete_asset(
    asset_path="my_assets",
    version_id="v1234"
)
```

#### 6. Processing Functions

##### Process Uploaded File
```python
result = client.process_file(
    upload_result={
        "asset_path": "uploads/document.pdf",
        "version_id": "v1234",
        "filename": "document.pdf"
    }
)
```

##### Get Cache
```python
# Get specific cache
cached_value = client.get_cached_value(
    m_type="document",
    key="my_cache_key"
)

# Get all cache
all_cache = client.get_all_cache()
```

#### 7. Chat Functions

##### Send Message
```python
response = client.send_chat(
    user_id="your_user_id",
    message="Please summarize the key points of machine learning for me"
)
print(response)
```

##### Chat with Search Results
```python
# First perform search
search_results = client.document_search(
    query_text="machine learning",
    limit=5
)

# Pass search results to chat
response = client.send_chat(
    user_id="your_user_id",
    message="Please summarize the key points based on these search results",
    search_results=search_results.get("items", [])
)
```

##### Manage Chat Memory
```python
# Get chat memory
memory = client.get_chat_memory(user_id="your_user_id")

# Clear chat memory
client.clear_chat_memory(user_id="your_user_id")
```

#### 8. Health Check

```python
health = client.health_check()
print(health)
```

### Complete Workflow Example

```python
from sample_code_python import RaptorAPIClient

# Initialize client
client = RaptorAPIClient()

# 1. Login
token = client.login(username="your_username", password="your_password")

# 2. Upload file with automatic analysis
upload_result = client.upload_file_with_analysis(
    file_path="/path/to/document.pdf",
    processing_mode="default"
)

# 3. Search for relevant content
search_results = client.document_search(
    query_text="important information",
    embedding_type="text",
    limit=5
)

# 4. Chat with AI using search results
chat_response = client.send_chat(
    user_id="your_user_id",
    message="Please summarize the important information based on search results",
    search_results=search_results.get("items", [])
)

print("AI Response:", chat_response.get("response"))
```

### Run Sample Program

```bash
# Run the built-in sample program
python sample_code_python.py
```

---

## cURL Usage

### Preparation

1. Ensure `curl` and `jq` (for JSON processing) are installed

```bash
# Ubuntu/Debian
sudo apt-get install curl jq

# macOS
brew install curl jq
```

2. Set execution permissions

```bash
chmod +x sample_code_curl.sh
```

### Basic Usage

#### 1. Manually Execute Individual Commands

##### Register User
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "password": "secure_password"
  }'
```

##### Login
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password"
```

Extract token:
```bash
TOKEN=$(curl -s -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password" | jq -r '.access_token')
```

#### 2. Search Functions

##### Video Search
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/search/video_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning",
    "embedding_type": "text",
    "filename": ["MV.mp4"],
    "limit": 5
  }'
```

##### Document Search
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/search/document_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "financial report",
    "embedding_type": "text",
    "source": "csv",
    "limit": 5
  }'
```

##### Unified Search
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/search/unified_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning",
    "embedding_type": "text",
    "limit_per_collection": 5,
    "global_limit": 20
  }'
```

#### 3. File Upload

##### Upload Single File
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileupload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/your/file.pdf" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30"
```

##### Batch Upload
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileupload_batch" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_files=@/path/to/file1.pdf" \
  -F "primary_files=@/path/to/file2.pdf" \
  -F "concurrency=4"
```

##### Upload with Analysis
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileupload_analysis" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/document.pdf" \
  -F "processing_mode=default"
```

#### 4. Asset Management

##### List Versions
```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileversions?asset_path=my_assets&filename=document.pdf" \
  -H "Authorization: Bearer ${TOKEN}"
```

##### Download File
```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/filedownload?asset_path=my_assets&version_id=v1234&return_file_content=true" \
  -H "Authorization: Bearer ${TOKEN}" \
  -o downloaded_file.pdf
```

##### Archive File
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/filearchive?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"
```

##### Delete File
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/delfile?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"
```

#### 5. Chat Functions

##### Send Message
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "Please summarize the key points of machine learning for me"
  }'
```

##### Get Chat Memory
```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"
```

##### Clear Chat Memory
```bash
curl -X DELETE "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"
```

#### 6. Health Check

```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/health"
```

### Execute Complete Script

```bash
# Edit script, set your token
nano sample_code_curl.sh

# Execute all examples
./sample_code_curl.sh

# Or execute specific parts
bash -x sample_code_curl.sh 2>&1 | grep "=== 2.1"
```

---

## API Endpoint Overview

### Authentication
- `POST /api/v1/auth/register` - Register user
- `POST /api/v1/auth/login` - Login to get JWT token

### Search
- `POST /api/v1/search/video_search` - Video search
- `POST /api/v1/search/audio_search` - Audio search
- `POST /api/v1/search/document_search` - Document search
- `POST /api/v1/search/image_search` - Image search
- `POST /api/v1/search/unified_search` - Unified cross-collection search

### File Upload
- `POST /api/v1/asset/fileupload` - Upload single file
- `POST /api/v1/asset/fileupload_batch` - Batch upload
- `POST /api/v1/asset/fileupload_analysis` - Upload with analysis
- `POST /api/v1/asset/fileupload_analysis_batch` - Batch upload with analysis

### Asset Management
- `GET /api/v1/asset/fileversions` - List file versions
- `GET /api/v1/asset/filedownload` - Download asset
- `POST /api/v1/asset/filearchive` - Archive asset
- `POST /api/v1/asset/delfile` - Delete asset

### Processing
- `POST /api/v1/processing/process-file` - Process file
- `GET /api/v1/processing/processing/cache/{m_type}/{key}` - Get cache value
- `GET /api/v1/processing/cache/all` - Get all cache

### Chat
- `POST /api/v1/chat/chat` - Send chat message
- `GET /api/v1/chat/memory/{user_id}` - Get chat memory
- `DELETE /api/v1/chat/memory/{user_id}` - Clear chat memory

### Health
- `GET /` - Root path health check
- `GET /health` - Health check endpoint

---

## Notes

1. **Authentication**: Most APIs require a JWT token, please login first to obtain a token
2. **File Paths**: Use correct file paths when uploading files
3. **Concurrent Upload**: For batch uploads, concurrency parameter ranges from 1-16
4. **Search Limits**:
   - Maximum 50 results per collection
   - Unified search maximum 100 total results
5. **TTL Settings**: archive_ttl and destroy_ttl are in days

---

## Troubleshooting

### Common Errors

#### 401 Unauthorized
- Check if token is correct
- Verify login status
- Token may have expired, please login again

#### 422 Validation Error
- Check if request parameters are correct
- Verify all required fields are provided
- Check data types are correct

#### 404 Not Found
- Verify API endpoint path is correct
- Check if resource exists

#### Connection Error
- Verify API server is running
- Check network connection
- Verify BASE_URL is correct

---

## More Information

- **Complete API Documentation**: http://raptor_open_0_1_api.dhtsolution.com:8012/docs
- **OpenAPI Specification**: See `openapi.json`

---

## Contributing

If you have questions or suggestions, feel free to submit an issue or pull request.

---

# RAPTOR API Gateway - Sample Code 使用指南

## 概述

本目錄包含 RAPTOR API Gateway 的完整範例程式碼，包括 Python 和 cURL 兩種方式。

- **API Base URL**: http://raptor_open_0_1_api.dhtsolution.com:8012
- **API 文件**: http://raptor_open_0_1_api.dhtsolution.com:8012/docs

## 檔案說明

### 1. `sample_code_python.py`
完整的 Python 客戶端實作，包含所有 API 端點的封裝。

### 2. `sample_code_curl.sh`
使用 cURL 呼叫所有 API 端點的 Shell 腳本範例。

### 3. `openapi.json`
OpenAPI 3.1.0 規格檔案，包含完整的 API 定義。

---

## Python 使用方式

### 安裝依賴

```bash
pip install requests
```

### 基本使用

#### 1. 匯入並初始化客戶端

```python
from sample_code_python import RaptorAPIClient

# 初始化客戶端
client = RaptorAPIClient()
```

#### 2. 註冊和登入

```python
# 註冊新使用者
client.register_user(
    username="your_username",
    email="your_email@example.com",
    password="your_password"
)

# 登入取得 token
token = client.login(
    username="your_username",
    password="your_password"
)
print(f"Token: {token}")
```

#### 3. 搜尋功能

##### 影片搜尋
```python
result = client.video_search(
    query_text="機器學習",
    embedding_type="text",
    filename=["MV.mp4"],
    speaker=["SPEAKER_00"],
    limit=5
)
print(result)
```

##### 音訊搜尋
```python
result = client.audio_search(
    query_text="機器學習",
    embedding_type="text",
    filename=["audio.mp3"],
    limit=5
)
```

##### 文件搜尋
```python
result = client.document_search(
    query_text="財務報告",
    embedding_type="text",
    filename=["EF25Y01.csv"],
    source="csv",
    limit=5
)
```

##### 圖片搜尋
```python
result = client.image_search(
    query_text="風景照片",
    embedding_type="summary",
    filename=["photo.jpg"],
    source="jpg",
    limit=5
)
```

##### 統一搜尋 (跨多個集合)
```python
result = client.unified_search(
    query_text="機器學習和人工智慧",
    embedding_type="text",
    filters={
        "video": {"filename": ["MV.mp4"]},
        "document": {"source": "csv"}
    },
    limit_per_collection=5,
    global_limit=20,
    score_threshold=0.5
)
```

#### 4. 檔案上傳

##### 上傳單一檔案
```python
result = client.upload_file(
    file_path="/path/to/your/file.pdf",
    archive_ttl=30,
    destroy_ttl=30
)
```

##### 批次上傳
```python
result = client.upload_files_batch(
    file_paths=[
        "/path/to/file1.pdf",
        "/path/to/file2.pdf",
        "/path/to/file3.pdf"
    ],
    archive_ttl=30,
    destroy_ttl=30,
    concurrency=4
)
```

##### 上傳並自動分析
```python
result = client.upload_file_with_analysis(
    file_path="/path/to/document.pdf",
    processing_mode="default",
    archive_ttl=30,
    destroy_ttl=30
)
```

##### 批次上傳並分析
```python
result = client.upload_files_batch_with_analysis(
    file_paths=["/path/to/file1.pdf", "/path/to/file2.pdf"],
    processing_mode="default",
    archive_ttl=30,
    destroy_ttl=30,
    concurrency=4
)
```

#### 5. 資產管理

##### 列出檔案版本
```python
versions = client.list_file_versions(
    asset_path="my_assets",
    filename="document.pdf"
)
```

##### 下載檔案
```python
file_content = client.download_asset(
    asset_path="my_assets",
    version_id="v1234",
    return_file_content=True
)

# 儲存到本地
with open("downloaded_file.pdf", "wb") as f:
    f.write(file_content)
```

##### 封存檔案
```python
result = client.archive_asset(
    asset_path="my_assets",
    version_id="v1234"
)
```

##### 刪除檔案
```python
result = client.delete_asset(
    asset_path="my_assets",
    version_id="v1234"
)
```

#### 6. 處理功能

##### 處理已上傳的檔案
```python
result = client.process_file(
    upload_result={
        "asset_path": "uploads/document.pdf",
        "version_id": "v1234",
        "filename": "document.pdf"
    }
)
```

##### 取得快取
```python
# 取得特定快取
cached_value = client.get_cached_value(
    m_type="document",
    key="my_cache_key"
)

# 取得所有快取
all_cache = client.get_all_cache()
```

#### 7. 聊天功能

##### 發送訊息
```python
response = client.send_chat(
    user_id="your_user_id",
    message="請幫我總結一下機器學習的重點"
)
print(response)
```

##### 帶搜尋結果的聊天
```python
# 先進行搜尋
search_results = client.document_search(
    query_text="機器學習",
    limit=5
)

# 將搜尋結果傳給聊天
response = client.send_chat(
    user_id="your_user_id",
    message="請根據這些搜尋結果總結重點",
    search_results=search_results.get("items", [])
)
```

##### 管理聊天記憶
```python
# 取得聊天記憶
memory = client.get_chat_memory(user_id="your_user_id")

# 清除聊天記憶
client.clear_chat_memory(user_id="your_user_id")
```

#### 8. 健康檢查

```python
health = client.health_check()
print(health)
```

### 完整工作流程範例

```python
from sample_code_python import RaptorAPIClient

# 初始化客戶端
client = RaptorAPIClient()

# 1. 登入
token = client.login(username="your_username", password="your_password")

# 2. 上傳檔案並自動分析
upload_result = client.upload_file_with_analysis(
    file_path="/path/to/document.pdf",
    processing_mode="default"
)

# 3. 搜尋相關內容
search_results = client.document_search(
    query_text="重要資訊",
    embedding_type="text",
    limit=5
)

# 4. 使用搜尋結果與 AI 對話
chat_response = client.send_chat(
    user_id="your_user_id",
    message="請根據搜尋結果總結重要資訊",
    search_results=search_results.get("items", [])
)

print("AI 回應:", chat_response.get("response"))
```

### 執行範例程式

```bash
# 執行內建的範例程式
python sample_code_python.py
```

---

## cURL 使用方式

### 準備工作

1. 確保已安裝 `curl` 和 `jq` (用於 JSON 處理)

```bash
# Ubuntu/Debian
sudo apt-get install curl jq

# macOS
brew install curl jq
```

2. 設定執行權限

```bash
chmod +x sample_code_curl.sh
```

### 基本使用

#### 1. 手動執行個別命令

##### 註冊使用者
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "password": "secure_password"
  }'
```

##### 登入
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password"
```

提取 token:
```bash
TOKEN=$(curl -s -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password" | jq -r '.access_token')
```

#### 2. 搜尋功能

##### 影片搜尋
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/search/video_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "機器學習",
    "embedding_type": "text",
    "filename": ["MV.mp4"],
    "limit": 5
  }'
```

##### 文件搜尋
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/search/document_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "財務報告",
    "embedding_type": "text",
    "source": "csv",
    "limit": 5
  }'
```

##### 統一搜尋
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/search/unified_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "機器學習",
    "embedding_type": "text",
    "limit_per_collection": 5,
    "global_limit": 20
  }'
```

#### 3. 檔案上傳

##### 上傳單一檔案
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileupload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/your/file.pdf" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30"
```

##### 批次上傳
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileupload_batch" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_files=@/path/to/file1.pdf" \
  -F "primary_files=@/path/to/file2.pdf" \
  -F "concurrency=4"
```

##### 上傳並分析
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileupload_analysis" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/document.pdf" \
  -F "processing_mode=default"
```

#### 4. 資產管理

##### 列出版本
```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/fileversions?asset_path=my_assets&filename=document.pdf" \
  -H "Authorization: Bearer ${TOKEN}"
```

##### 下載檔案
```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/filedownload?asset_path=my_assets&version_id=v1234&return_file_content=true" \
  -H "Authorization: Bearer ${TOKEN}" \
  -o downloaded_file.pdf
```

##### 封存檔案
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/filearchive?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"
```

##### 刪除檔案
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/asset/delfile?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"
```

#### 5. 聊天功能

##### 發送訊息
```bash
curl -X POST "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "請幫我總結一下機器學習的重點"
  }'
```

##### 取得聊天記憶
```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"
```

##### 清除聊天記憶
```bash
curl -X DELETE "http://raptor_open_0_1_api.dhtsolution.com:8012/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"
```

#### 6. 健康檢查

```bash
curl -X GET "http://raptor_open_0_1_api.dhtsolution.com:8012/health"
```

### 執行完整腳本

```bash
# 編輯腳本，設定您的 token
nano sample_code_curl.sh

# 執行所有範例
./sample_code_curl.sh

# 或執行特定部分
bash -x sample_code_curl.sh 2>&1 | grep "=== 2.1"
```

---

## API 端點總覽

### Authentication (認證)
- `POST /api/v1/auth/register` - 註冊使用者
- `POST /api/v1/auth/login` - 登入取得 JWT token

### Search (搜尋)
- `POST /api/v1/search/video_search` - 影片搜尋
- `POST /api/v1/search/audio_search` - 音訊搜尋
- `POST /api/v1/search/document_search` - 文件搜尋
- `POST /api/v1/search/image_search` - 圖片搜尋
- `POST /api/v1/search/unified_search` - 統一跨集合搜尋

### File Upload (檔案上傳)
- `POST /api/v1/asset/fileupload` - 上傳單一檔案
- `POST /api/v1/asset/fileupload_batch` - 批次上傳
- `POST /api/v1/asset/fileupload_analysis` - 上傳並分析
- `POST /api/v1/asset/fileupload_analysis_batch` - 批次上傳並分析

### Asset Management (資產管理)
- `GET /api/v1/asset/fileversions` - 列出檔案版本
- `GET /api/v1/asset/filedownload` - 下載資產
- `POST /api/v1/asset/filearchive` - 封存資產
- `POST /api/v1/asset/delfile` - 刪除資產

### Processing (處理)
- `POST /api/v1/processing/process-file` - 處理檔案
- `GET /api/v1/processing/processing/cache/{m_type}/{key}` - 取得快取值
- `GET /api/v1/processing/cache/all` - 取得所有快取

### Chat (聊天)
- `POST /api/v1/chat/chat` - 發送聊天訊息
- `GET /api/v1/chat/memory/{user_id}` - 取得聊天記憶
- `DELETE /api/v1/chat/memory/{user_id}` - 清除聊天記憶

### Health (健康檢查)
- `GET /` - 根路徑健康檢查
- `GET /health` - 健康檢查端點

---

## 注意事項

1. **認證**: 大部分 API 都需要 JWT token，請先登入取得 token
2. **檔案路徑**: 上傳檔案時請使用正確的檔案路徑
3. **並行上傳**: 批次上傳時，concurrency 參數範圍為 1-16
4. **搜尋限制**: 
   - 每個集合最多返回 50 個結果
   - 統一搜尋最多返回 100 個總結果
5. **TTL 設定**: archive_ttl 和 destroy_ttl 的單位為天

---

## 故障排除

### 常見錯誤

#### 401 Unauthorized
- 檢查 token 是否正確
- 確認是否已登入
- Token 可能已過期，請重新登入

#### 422 Validation Error
- 檢查請求參數是否正確
- 確認必填欄位是否都有提供
- 檢查資料型別是否正確

#### 404 Not Found
- 確認 API 端點路徑是否正確
- 檢查資源是否存在

#### Connection Error
- 確認 API 伺服器是否運行
- 檢查網路連線
- 確認 BASE_URL 是否正確

---

## 更多資訊

- **完整 API 文件**: http://raptor_open_0_1_api.dhtsolution.com:8012/docs
- **OpenAPI 規格**: 參考 `openapi.json`

---

## 貢獻

如有問題或建議，歡迎提交 issue 或 pull request。
