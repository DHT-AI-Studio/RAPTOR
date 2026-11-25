#!/usr/bin/env bash
# Whisper 語音辨識（ASR）：下載 → 上傳 lakeFS → 註冊（pipeline 捷徑）→ 音訊推理
#
# 用法：
#   AUDIO=/path/to/clip.wav ./05_asr_whisper.sh
#   HF_MODEL=openai/whisper-small MODEL_PARAMS=0.24 AUDIO=... ./05_asr_whisper.sh
#
# AUDIO 需為 07 容器可見的路徑（如 /app/data/...）；
# OpenAI 相容調用（multipart 上傳）則接受本機檔案。

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-openai/whisper-large-v3}"
MODEL_PARAMS="${MODEL_PARAMS:-1.5}"
BASENAME=$(basename "$HF_MODEL" | tr '[:upper:]._' '[:lower:]--')
LAKEFS_REPO="${LAKEFS_REPO:-$BASENAME}"
REGISTERED_NAME="${REGISTERED_NAME:-$BASENAME}"
AUDIO="${AUDIO:?請設定 AUDIO=<音訊路徑>}"

check_health

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  hf_download "$HF_MODEL"
  lakefs_upload "$LAKEFS_REPO" "$HF_MODEL"
fi

if [ "${SKIP_REGISTER:-0}" != "1" ]; then
  register_from_lakefs "{
    \"registered_name\": \"$REGISTERED_NAME\",
    \"task\": \"asr\",
    \"engine\": \"transformers\",
    \"model_params\": $MODEL_PARAMS,
    \"lakefs_repo\": \"$LAKEFS_REPO\",
    \"stage\": \"$STAGE\",
    \"pipeline_task\": \"automatic-speech-recognition\"
  }"
fi

step "推理（原生 /inference/infer，容器內路徑）"
infer "{
  \"model_name\": \"$REGISTERED_NAME\",
  \"data\": {\"audio\": \"$AUDIO\"},
  \"options\": {\"language\": \"zh\"}
}" | pretty

if [ -f "$AUDIO" ]; then
  step "推理（OpenAI 相容 /v1/audio/transcriptions，multipart 上傳本機檔案）"
  curl -sf -X POST "$API_BASE/v1/audio/transcriptions" \
    -F "file=@$AUDIO" -F "model=$REGISTERED_NAME" | pretty
fi
