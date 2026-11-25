#!/usr/bin/env bash
# InternVL 視覺語言模型（10/11/12 模組共用）：下載 → 上傳 lakeFS → 註冊 → 圖片推理
#
# InternVL 走 .chat() 介面 + 動態切圖，07 已內建 InternVLHandler：
#   - 由 physical_path / model_class 中的 "InternVL" 字樣自動偵測
#   - 註冊時也可顯式填 custom_handler="internvl"（本腳本採顯式，lakeFS repo 名不含 internvl 字樣時較保險）
#   - torch_dtype 建議 bf16（fp16 會溢位）
#
# 用法：
#   IMAGE=/path/to/photo.jpg ./04_vlm_internvl.sh
#   HF_MODEL=OpenGVLab/InternVL3_5-4B MODEL_PARAMS=4 IMAGE=... ./04_vlm_internvl.sh
# 已經下載到 lakeFS 且已註冊的模型可跳過下載與註冊：
  # API_BASE=http://localhost:9997 IMAGE=/opt/home/george/george-test/vibevoice-test040926/teddy.jpg SKIP_DOWNLOAD=1 SKIP_REGISTER=1 REGISTERED_NAME=intervl3_5-4b ./04_vlm_internvl.sh
source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-OpenGVLab/InternVL3_5-1B}"
MODEL_PARAMS="${MODEL_PARAMS:-1}"
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
    \"model_class\": \"AutoModel\",
    \"processor_class\": \"AutoTokenizer\",
    \"torch_dtype\": \"bf16\",
    \"custom_handler\": \"internvl\"
  }"
fi

if [ -f "$IMAGE" ]; then
  IMAGE_PAYLOAD="data:image/jpeg;base64,$(base64 -w0 "$IMAGE")"
else
  IMAGE_PAYLOAD="$IMAGE"
fi

step "推理（task=vlm，InternVL handler）"
printf '{"model_name": "%s", "data": {"image": "%s", "prompt": "詳細描述這張圖片"}, "options": {"max_new_tokens": 256, "max_num": 12}}' \
  "$REGISTERED_NAME" "$IMAGE_PAYLOAD" \
  | curl -sf -X POST "$API_BASE/inference/infer" -H 'Content-Type: application/json' -d @- | pretty
