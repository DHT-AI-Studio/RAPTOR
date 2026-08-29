#!/usr/bin/env bash
# scripts/common.sh — 各模型使用腳本的共用函式庫
#
# 用法：在各腳本開頭 `source "$(dirname "$0")/common.sh"`
#
# 可用環境變數：
#   API_BASE   AI Lifecycle API 位址（預設 http://localhost:8010；dev stack 用 http://localhost:9997）
#   STAGE      註冊後切換的階段（預設 staging）
#   SKIP_DOWNLOAD=1   跳過 下載→上傳 lakeFS（模型已在 lakeFS 時）
#   SKIP_REGISTER=1   跳過 註冊（模型已在 MLflow 時，直接推理）

set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8010}"
STAGE="${STAGE:-staging}"

# ---------- 輸出 ----------

step()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
info()  { printf '\033[0;36m    %s\033[0m\n' "$*"; }
die()   { printf '\033[0;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

pretty() {  # stdin JSON → 縮排輸出（沒有 python3 時原樣印）
  python3 -m json.tool --no-ensure-ascii 2>/dev/null || cat
}

# ---------- HTTP ----------

api_get() {  # api_get <path>
  curl -sf "$API_BASE$1" || die "GET $1 failed"
}

api_post() {  # api_post <path> <json-body>
  local out
  out=$(curl -s -X POST "$API_BASE$1" -H 'Content-Type: application/json' -d "$2")
  if printf '%s' "$out" | grep -q '"detail"'; then
    printf '%s\n' "$out" | pretty >&2
    die "POST $1 failed"
  fi
  printf '%s' "$out"
}

check_health() {
  step "檢查服務健康（$API_BASE）"
  api_get /health >/dev/null || die "API 不可達，請確認服務已啟動且 API_BASE 正確"
  info "OK"
}

# ---------- HF → lakeFS → MLflow 流程 ----------

hf_download() {  # hf_download <hf-repo-id>
  step "下載 $1（HuggingFace → 本地暫存）"
  api_post /models/download "{\"model_source\": \"huggingface\", \"model_name\": \"$1\"}" | pretty
}

lakefs_upload() {  # lakefs_upload <lakefs-repo> <hf-repo-id>
  local local_name="${2//\//_}"   # HF repo-id 的 '/' 在暫存區被換成 '_'
  step "上傳到 lakeFS（repo=$1, local=$local_name）"
  api_post /models/upload_to_lakefs "{\"repo_name\": \"$1\", \"local_model_name\": \"$local_name\"}" | pretty
}

register_from_lakefs() {  # register_from_lakefs <json-body>
  step "註冊到 MLflow"
  api_post /models/register_from_lakefs "$1" | pretty
}

infer() {  # infer <json-body>
  api_post /inference/infer "$1"
}

show_model() {  # show_model <registered-name>
  step "MLflow 中的註冊資訊：$1"
  api_get "/models/registered_in_mlflow/$1" | pretty
}
