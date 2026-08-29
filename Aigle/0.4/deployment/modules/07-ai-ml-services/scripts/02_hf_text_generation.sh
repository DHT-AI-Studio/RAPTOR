#!/usr/bin/env bash
# HF 文字生成 LLM：下載 → 上傳 lakeFS → 註冊（pipeline 捷徑）→ 推理
#
# 用法：
#   ./02_hf_text_generation.sh                                    # 預設 gemma-3-270m-it（小，適合驗證流程）
#   HF_MODEL=google/gemma-2-2b-it MODEL_PARAMS=2 ./02_hf_text_generation.sh

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-google/gemma-3-270m-it}"
MODEL_PARAMS="${MODEL_PARAMS:-0.27}"           # 參數量（B），用於 VRAM 估算
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
    \"task\": \"text-generation\",
    \"engine\": \"transformers\",
    \"model_params\": $MODEL_PARAMS,
    \"lakefs_repo\": \"$LAKEFS_REPO\",
    \"stage\": \"$STAGE\",
    \"pipeline_task\": \"text-generation\"
  }"
fi

step "推理（原生 /inference/infer）"
infer "{
  \"model_name\": \"$REGISTERED_NAME\",
  \"data\": {\"inputs\": \"Explain MLOps in one sentence.\"},
  \"options\": {\"max_new_tokens\": 100, \"temperature\": 0.7}
}" | pretty

step "推理（OpenAI 相容 /v1/completions）"
api_post /v1/completions "{
  \"model\": \"$REGISTERED_NAME\",
  \"prompt\": \"The three pillars of MLOps are\",
  \"max_tokens\": 80
}" | pretty
