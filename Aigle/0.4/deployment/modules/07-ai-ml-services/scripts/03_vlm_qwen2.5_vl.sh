#!/usr/bin/env bash
# Qwen2.5-VL 視覺語言模型：下載 → 上傳 lakeFS → 註冊（model_class 路線）→ 圖片推理
#
# 用法：
#   IMAGE=/path/to/photo.jpg ./03_vlm_qwen2.5_vl.sh
#   HF_MODEL=Qwen/Qwen2.5-VL-3B-Instruct MODEL_PARAMS=3 IMAGE=... ./03_vlm_qwen2.5_vl.sh
#
# IMAGE 可為容器可見的路徑（如 /app/data/...）或本機檔案（自動轉 base64 上傳）。

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_PARAMS="${MODEL_PARAMS:-7}"
BASENAME=$(basename "$HF_MODEL" | tr '[:upper:]._' '[:lower:]--')
LAKEFS_REPO="${LAKEFS_REPO:-$BASENAME}"
REGISTERED_NAME="${REGISTERED_NAME:-$BASENAME}"
IMAGE="${IMAGE:?請設定 IMAGE=<圖片路徑>}"

check_health

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  hf_download "$HF_MODEL"
  lakefs_upload "$LAKEFS_REPO" "$HF_MODEL"
fi

if [ "${SKIP_REGISTER:-0}" != "1" ]; then
  register_from_lakefs "{
    \"registered_name\": \"$REGISTERED_NAME\",
    \"task\": \"vlm\",
    \"engine\": \"transformers\",
    \"model_params\": $MODEL_PARAMS,
    \"lakefs_repo\": \"$LAKEFS_REPO\",
    \"stage\": \"$STAGE\",
    \"model_class\": \"Qwen2_5_VLForConditionalGeneration\",
    \"processor_class\": \"AutoProcessor\",
    \"quantization\": \"4bit\"
  }"
fi

# 本機檔案 → base64；容器內路徑 → 原樣傳
if [ -f "$IMAGE" ]; then
  IMAGE_PAYLOAD="data:image/jpeg;base64,$(base64 -w0 "$IMAGE")"
else
  IMAGE_PAYLOAD="$IMAGE"
fi

step "推理（task=vlm）"
printf '{"model_name": "%s", "data": {"image": "%s", "prompt": "詳細描述這張圖片"}, "options": {"max_new_tokens": 256}}' \
  "$REGISTERED_NAME" "$IMAGE_PAYLOAD" \
  | curl -sf -X POST "$API_BASE/inference/infer" -H 'Content-Type: application/json' -d @- | pretty
