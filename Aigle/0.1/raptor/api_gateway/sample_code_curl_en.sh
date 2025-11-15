#!/bin/bash

################################################################################
# RAPTOR API Gateway - cURL Sample Code
# API Base URL: http://raptor_open_0_1_api.dhtsolution.com:8012
# API Docs: http://raptor_open_0_1_api.dhtsolution.com:8012/docs
################################################################################

BASE_URL="http://raptor_open_0_1_api.dhtsolution.com:8012"
TOKEN=""

################################################################################
# 1. Authentication
################################################################################

echo "=== 1.1 Register User ==="
curl -X POST "${BASE_URL}/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "password": "secure_password"
  }'

echo -e "\n\n=== 1.2 Login to Get Token ==="
# Note: OAuth2 password flow uses form data
RESPONSE=$(curl -X POST "${BASE_URL}/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password")

echo $RESPONSE

# Extract token (requires jq tool)
TOKEN=$(echo $RESPONSE | jq -r '.access_token')
# Or set manually:
# TOKEN="YOUR_JWT_TOKEN_HERE"

################################################################################
# 2. Search
################################################################################

echo -e "\n\n=== 2.1 Video Search ==="
curl -X POST "${BASE_URL}/api/v1/search/video_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning",
    "embedding_type": "text",
    "filename": ["MV.mp4"],
    "speaker": ["SPEAKER_00"],
    "limit": 5
  }'

echo -e "\n\n=== 2.2 Audio Search ==="
curl -X POST "${BASE_URL}/api/v1/search/audio_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning",
    "embedding_type": "text",
    "filename": ["audio.mp3"],
    "speaker": ["SPEAKER_00"],
    "limit": 5
  }'

echo -e "\n\n=== 2.3 Document Search ==="
curl -X POST "${BASE_URL}/api/v1/search/document_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "financial report",
    "embedding_type": "text",
    "filename": ["EF25Y01.csv"],
    "source": "csv",
    "limit": 5
  }'

echo -e "\n\n=== 2.4 Image Search ==="
curl -X POST "${BASE_URL}/api/v1/search/image_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "landscape photo",
    "embedding_type": "summary",
    "filename": ["invoice.jpg"],
    "source": "jpg",
    "limit": 5
  }'

echo -e "\n\n=== 2.5 Unified Search (Across Multiple Collections) ==="
curl -X POST "${BASE_URL}/api/v1/search/unified_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning and artificial intelligence",
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
# 3. File Upload
################################################################################

echo -e "\n\n=== 3.1 Upload Single File ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/your/file.pdf" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30"

echo -e "\n\n=== 3.2 Batch Upload Multiple Files ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload_batch" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_files=@/path/to/file1.pdf" \
  -F "primary_files=@/path/to/file2.pdf" \
  -F "primary_files=@/path/to/file3.pdf" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30" \
  -F "concurrency=4"

echo -e "\n\n=== 3.3 Upload File with Automatic Analysis ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload_analysis" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_file=@/path/to/your/document.pdf" \
  -F "processing_mode=default" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30"

echo -e "\n\n=== 3.4 Batch Upload with Analysis ==="
curl -X POST "${BASE_URL}/api/v1/asset/fileupload_analysis_batch" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "primary_files=@/path/to/file1.pdf" \
  -F "primary_files=@/path/to/file2.pdf" \
  -F "processing_mode=default" \
  -F "archive_ttl=30" \
  -F "destroy_ttl=30" \
  -F "concurrency=4"

################################################################################
# 4. Asset Management
################################################################################

echo -e "\n\n=== 4.1 List File Versions ==="
curl -X GET "${BASE_URL}/api/v1/asset/fileversions?asset_path=my_assets&filename=document.pdf" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n=== 4.2 Download Asset ==="
curl -X GET "${BASE_URL}/api/v1/asset/filedownload?asset_path=my_assets&version_id=v1234&return_file_content=true" \
  -H "Authorization: Bearer ${TOKEN}" \
  -o downloaded_file.pdf

echo -e "\n\n=== 4.3 Archive Asset ==="
curl -X POST "${BASE_URL}/api/v1/asset/filearchive?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n=== 4.4 Delete Asset ==="
curl -X POST "${BASE_URL}/api/v1/asset/delfile?asset_path=my_assets&version_id=v1234" \
  -H "Authorization: Bearer ${TOKEN}"

################################################################################
# 5. Processing
################################################################################

echo -e "\n\n=== 5.1 Process Uploaded File ==="
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

echo -e "\n\n=== 5.2 Get Cache Value ==="
curl -X GET "${BASE_URL}/api/v1/processing/processing/cache/document/my_cache_key"

echo -e "\n\n=== 5.3 Get All Cache ==="
curl -X GET "${BASE_URL}/api/v1/processing/cache/all"

################################################################################
# 6. Chat
################################################################################

echo -e "\n\n=== 6.1 Send Chat Message ==="
curl -X POST "${BASE_URL}/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "Please summarize the key points of machine learning for me"
  }'

echo -e "\n\n=== 6.2 Send Chat Message (with Search Results) ==="
curl -X POST "${BASE_URL}/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "Based on these search results, please explain the applications of machine learning",
    "search_results": [
      {
        "collection": "document",
        "score": 0.95,
        "payload": {
          "text": "Machine learning is a branch of artificial intelligence..."
        }
      }
    ]
  }'

echo -e "\n\n=== 6.3 Get User Chat Memory ==="
curl -X GET "${BASE_URL}/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n=== 6.4 Clear User Chat Memory ==="
curl -X DELETE "${BASE_URL}/api/v1/chat/memory/test_user" \
  -H "Authorization: Bearer ${TOKEN}"

################################################################################
# 7. Health Check
################################################################################

echo -e "\n\n=== 7.1 Root Path Health Check ==="
curl -X GET "${BASE_URL}/"

echo -e "\n\n=== 7.2 Health Check Endpoint ==="
curl -X GET "${BASE_URL}/health"

################################################################################
# Complete Example Workflow
################################################################################

echo -e "\n\n=== Complete Workflow Example ==="

# Step 1: Login
echo "Step 1: Login"
LOGIN_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=secure_password")
# TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

# Step 2: Upload file with analysis
echo "Step 2: Upload file"
# UPLOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/asset/fileupload_analysis" \
#   -H "Authorization: Bearer ${TOKEN}" \
#   -F "primary_file=@/path/to/document.pdf" \
#   -F "processing_mode=default")

# Step 3: Search for relevant content
echo "Step 3: Search documents"
SEARCH_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/search/document_search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "important information",
    "embedding_type": "text",
    "limit": 5
  }')

# Step 4: Chat using search results
echo "Step 4: Chat with AI"
CHAT_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/chat/chat" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_id\": \"test_user\",
    \"message\": \"Please summarize the important information based on search results\",
    \"search_results\": ${SEARCH_RESPONSE}
  }")

echo "Complete!"
