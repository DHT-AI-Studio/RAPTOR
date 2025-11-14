#!/bin/bash

################################################################################
# RAPTOR API Gateway - cURL Sample Code
# API Base URL: http://raptor_open_0_1_api.dhtsolution.com:8012
# API Docs: http://raptor_open_0_1_api.dhtsolution.com:8012/docs
################################################################################

BASE_URL="http://raptor_open_0_1_api.dhtsolution.com:8012"
TOKEN=""

################################################################################
# 1. Authentication (認證)
################################################################################

echo "=== 1.1 註冊使用者 ==="
curl -X POST "${BASE_URL}/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "password": "secure_password"
  }'

echo -e "\n\n=== 1.2 登入取得 Token ==="
# 注意: OAuth2 password flow 使用 form data
RESPONSE=$(curl -X POST "${BASE_URL}/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password")

echo $RESPONSE

# 提取 token (需要 jq 工具)
TOKEN=$(echo $RESPONSE | jq -r '.access_token')
# 或手動設定:
# TOKEN="YOUR_JWT_TOKEN_HERE"

################################################################################
# 2. Search (搜尋)
################################################################################

echo -e "\n\n=== 2.1 影片搜尋 ==="
curl -X POST "${BASE_URL}/api/v1/search/video_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "機器學習",
    "embedding_type": "text",
    "filename": ["MV.mp4"],
    "speaker": ["SPEAKER_00"],
    "limit": 5
  }'

echo -e "\n\n=== 2.2 音訊搜尋 ==="
curl -X POST "${BASE_URL}/api/v1/search/audio_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "機器學習",
    "embedding_type": "text",
    "filename": ["audio.mp3"],
    "speaker": ["SPEAKER_00"],
    "limit": 5
  }'

echo -e "\n\n=== 2.3 文件搜尋 ==="
curl -X POST "${BASE_URL}/api/v1/search/document_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "財務報告",
    "embedding_type": "text",
    "filename": ["EF25Y01.csv"],
    "source": "csv",
    "limit": 5
  }'

echo -e "\n\n=== 2.4 圖片搜尋 ==="
curl -X POST "${BASE_URL}/api/v1/search/image_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "風景照片",
    "embedding_type": "summary",
    "filename": ["發票.jpg"],
    "source": "jpg",
    "limit": 5
  }'

echo -e "\n\n=== 2.5 統一搜尋 (跨多個集合) ==="
curl -X POST "${BASE_URL}/api/v1/search/unified_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "機器學習和人工智慧",
    "embedding_type": "text",
    "filters": {
      "video": {
        "filename": ["MV.mp4"]
      },
      "document": {
        "source": "csv"
      }
    },
    "limit_per_collection": 5,
    "global_limit": 20,
    "score_threshold": 0.5
  }'

################################################################################
# 3. File Upload (檔案上傳)
################################################################################

echo -e "\n\n=== 3.1 上傳單一檔案 ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/your/file.pdf" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30"

echo -e "\n\n=== 3.2 批次上傳多個檔案 ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload_batch" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_files=@/path/to/file1.pdf" \
  -F "primary_files=@/path/to/file2.pdf" \
  -F "primary_files=@/path/to/file3.pdf" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30" \
  -F "concurrency=4"

echo -e "\n\n=== 3.3 上傳檔案並自動分析 ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload_analysis" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/your/document.pdf" \
  -F "processing_mode=default" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30"

echo -e "\n\n=== 3.4 批次上傳並分析 ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload_analysis_batch" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_files=@/path/to/file1.pdf" \
  -F "primary_files=@/path/to/file2.pdf" \
  -F "processing_mode=default" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30" \
  -F "concurrency=4"

################################################################################
# 4. Asset Management (資產管理)
################################################################################

echo -e "\n\n=== 4.1 列出檔案版本 ==="
curl -X GET "${BASE_URL}/api/v1/asset/fileversions?asset_path=my_assets&filename=document.pdf" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n=== 4.2 下載資產 ==="
curl -X GET "${BASE_URL}/api/v1/asset/filedownload?asset_path=my_assets&version_id=v1234&return_file_content=true" \
  -H "Authorization: Bearer ${TOKEN}" \
  -o downloaded_file.pdf

echo -e "\n\n=== 4.3 封存資產 ==="
curl -X POST "${BASE_URL}/api/v1/asset/filearchive?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n=== 4.4 刪除資產 ==="
curl -X POST "${BASE_URL}/api/v1/asset/delfile?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"

################################################################################
# 5. Processing (處理)
################################################################################

echo -e "\n\n=== 5.1 處理已上傳的檔案 ==="
curl -X POST "${BASE_URL}/api/v1/processing/process-file" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "upload_result": {
      "asset_path": "uploads/document.pdf",
      "version_id": "v1234",
      "filename": "document.pdf"
    }
  }'

echo -e "\n\n=== 5.2 取得快取值 ==="
curl -X GET "${BASE_URL}/api/v1/processing/processing/cache/document/my_cache_key"

echo -e "\n\n=== 5.3 取得所有快取 ==="
curl -X GET "${BASE_URL}/api/v1/processing/cache/all"

################################################################################
# 6. Chat (聊天)
################################################################################

echo -e "\n\n=== 6.1 發送聊天訊息 ==="
curl -X POST "${BASE_URL}/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "請幫我總結一下機器學習的重點"
  }'

echo -e "\n\n=== 6.2 發送聊天訊息 (帶搜尋結果) ==="
curl -X POST "${BASE_URL}/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "根據這些搜尋結果，請說明機器學習的應用",
    "search_results": [
      {
        "collection": "document",
        "score": 0.95,
        "payload": {
          "text": "機器學習是人工智慧的一個分支..."
        }
      }
    ]
  }'

echo -e "\n\n=== 6.3 取得使用者聊天記憶 ==="
curl -X GET "${BASE_URL}/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n=== 6.4 清除使用者聊天記憶 ==="
curl -X DELETE "${BASE_URL}/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"

################################################################################
# 7. Health Check (健康檢查)
################################################################################

echo -e "\n\n=== 7.1 根路徑健康檢查 ==="
curl -X GET "${BASE_URL}/"

echo -e "\n\n=== 7.2 健康檢查端點 ==="
curl -X GET "${BASE_URL}/health"

################################################################################
# 完整範例工作流程
################################################################################

echo -e "\n\n=== 完整工作流程範例 ==="

# 步驟 1: 登入
echo "步驟 1: 登入"
LOGIN_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password")
# TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

# 步驟 2: 上傳檔案並分析
echo "步驟 2: 上傳檔案"
# UPLOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/asset/fileupload_analysis" \
#   -H "Authorization: Bearer ${TOKEN}" \
#   -F "primary_file=@/path/to/document.pdf" \
#   -F "processing_mode=default")

# 步驟 3: 搜尋相關內容
echo "步驟 3: 搜尋文件"
SEARCH_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/search/document_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "重要資訊",
    "embedding_type": "text",
    "limit": 5
  }')

# 步驟 4: 使用搜尋結果進行聊天
echo "步驟 4: 與 AI 聊天"
CHAT_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_id\": \"test_user\",
    \"message\": \"請根據搜尋結果總結重要資訊\",
    \"search_results\": ${SEARCH_RESPONSE}
  }")

echo "完成!"
