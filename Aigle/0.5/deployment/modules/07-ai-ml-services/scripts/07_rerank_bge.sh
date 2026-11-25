#!/usr/bin/env bash
# Rerank（bge-reranker-v2-m3，17-hybrid-search / 21-agent 使用的重排模型）：
# 下載 → 上傳 lakeFS → 註冊 → 重排
#
# 用法：
#   ./07_rerank_bge.sh
#   HF_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 MODEL_PARAMS=0.02 ./07_rerank_bge.sh   # 小模型驗證流程

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-BAAI/bge-reranker-v2-m3}"
MODEL_PARAMS="${MODEL_PARAMS:-0.6}"
BASENAME=$(basename "$HF_MODEL" | tr '[:upper:]._' '[:lower:]--')
LAKEFS_REPO="${LAKEFS_REPO:-$BASENAME}"
REGISTERED_NAME="${REGISTERED_NAME:-$BASENAME}"

check_health

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  hf_download "$HF_MODEL"
  lakefs_upload "$LAKEFS_REPO" "$HF_MODEL"
fi

if [ "${SKIP_REGISTER:-0}" != "1" ]; then
  register_from_lakefs "{
    \"registered_name\": \"$REGISTERED_NAME\",
    \"task\": \"rerank\",
    \"engine\": \"transformers\",
    \"model_params\": $MODEL_PARAMS,
    \"lakefs_repo\": \"$LAKEFS_REPO\",
    \"stage\": \"$STAGE\",
    \"model_class\": \"AutoModelForSequenceClassification\",
    \"processor_class\": \"AutoTokenizer\"
  }"
fi

step "推理（jina/xinference 相容 /v1/rerank）"
api_post /v1/rerank "{
  \"model\": \"$REGISTERED_NAME\",
  \"query\": \"什麼是機器學習\",
  \"documents\": [
    \"機器學習是人工智慧的分支，讓系統從資料中學習。\",
    \"台北今天下雨。\",
    \"深度學習使用神經網路學習資料表徵。\"
  ],
  \"top_n\": 2
}" | pretty

step "推理（原生 /inference/infer — 可帶 normalize 讓分數變 0~1）"
infer "{
  \"model_name\": \"$REGISTERED_NAME\",
  \"data\": {\"query\": \"machine learning\", \"documents\": [\"ML is great\", \"the sky is blue\"]},
  \"options\": {\"normalize\": true, \"top_n\": 1}
}" | pretty
