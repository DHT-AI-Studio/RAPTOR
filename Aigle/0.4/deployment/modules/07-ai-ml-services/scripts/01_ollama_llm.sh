#!/usr/bin/env bash
# Ollama LLM：註冊本地 Ollama 模型 → 推理（原生 + OpenAI 相容兩種調用）
#
# 前置：Ollama daemon 上已有該模型（`ollama pull qwen2.5:7b`），
#       且 07 的 OLLAMA_API_BASE 指向該 daemon。Ollama 模型不需下載/上傳 lakeFS。
#
# 用法：
#   ./01_ollama_llm.sh                          # 預設 qwen2.5:7b
#   OLLAMA_MODEL=qwen3.5:9b ./01_ollama_llm.sh

source "$(dirname "$0")/common.sh"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
REGISTERED_NAME="${REGISTERED_NAME:-${OLLAMA_MODEL//[:.]/-}-ollama}"

check_health

step "確認 daemon 上有 $OLLAMA_MODEL"
api_get "/models/local?model_source=ollama" | pretty

if [ "${SKIP_REGISTER:-0}" != "1" ]; then
  step "註冊 Ollama 模型到 MLflow：$REGISTERED_NAME"
  api_post /models/register_ollama "{
    \"local_model_name\": \"$OLLAMA_MODEL\",
    \"task\": \"text-generation\",
    \"registered_name\": \"$REGISTERED_NAME\",
    \"stage\": \"$STAGE\"
  }" | pretty
fi

step "推理（原生 /inference/infer）"
infer "{
  \"model_name\": \"$REGISTERED_NAME\",
  \"data\": {\"inputs\": \"請用繁體中文一句話介紹 MLOps。\"},
  \"options\": {\"temperature\": 0.7, \"max_length\": 200}
}" | pretty

step "推理（OpenAI 相容 /v1/chat/completions）"
api_post /v1/chat/completions "{
  \"model\": \"$REGISTERED_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"用一句話說明什麼是模型註冊表\"}],
  \"max_tokens\": 128
}" | pretty
