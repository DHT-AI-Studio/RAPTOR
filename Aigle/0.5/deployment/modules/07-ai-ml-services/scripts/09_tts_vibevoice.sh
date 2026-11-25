#!/usr/bin/env bash
# VibeVoice TTS（實驗性）：註冊 → 調用（09 audio_tts_service 的目標模型 vibevoice-tts）
#
# ⚠️ 前置（跟其他腳本不同，需要手動準備）：
#   1. 07 映像檔需安裝 vibevoice 套件（官方 repo 已下架，用社群 fork）：
#        pip install git+https://github.com/vibevoice-community/VibeVoice.git
#   2. 模型權重（官方 HF repo 也已下架，找鏡像如 aoi-ot/VibeVoice-Large 或內部備份）
#      放到 07 的暫存區後上傳 lakeFS，或直接跑本腳本的下載步驟（若鏡像可用）
#   3. 一段參考語音 wav（voice cloning 用），路徑要容器可見：
#        VOICE=/app/data/voices/reference.wav
#      或在 07 的環境變數設 VIBEVOICE_DEFAULT_VOICE
#
# 用法：
#   HF_MODEL=aoi-ot/VibeVoice-Large VOICE=/app/data/voices/ref.wav ./09_tts_vibevoice.sh
#   SKIP_DOWNLOAD=1 VOICE=... ./09_tts_vibevoice.sh    # 權重已在 lakeFS

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-aoi-ot/VibeVoice-Large}"
MODEL_PARAMS="${MODEL_PARAMS:-1.5}"
LAKEFS_REPO="${LAKEFS_REPO:-vibevoice-tts}"
REGISTERED_NAME="${REGISTERED_NAME:-vibevoice-tts}"   # 09 audio_tts_service 期望的名字
TEXT="${TEXT:-歡迎使用 Raptor 統一模型服務。}"
VOICE="${VOICE:-}"
OUT="${OUT:-./vibevoice_output.wav}"

check_health

step "確認 vibevoice 套件已安裝在 07 容器"
if ! api_get /health >/dev/null; then die "API down"; fi
info "（無法遠端確認套件；若推理回 'cannot import class' 請先在映像檔安裝 vibevoice）"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  hf_download "$HF_MODEL"
  lakefs_upload "$LAKEFS_REPO" "$HF_MODEL"
fi

if [ "${SKIP_REGISTER:-0}" != "1" ]; then
  register_from_lakefs "{
    \"registered_name\": \"$REGISTERED_NAME\",
    \"task\": \"tts\",
    \"engine\": \"transformers\",
    \"model_params\": $MODEL_PARAMS,
    \"lakefs_repo\": \"$LAKEFS_REPO\",
    \"stage\": \"$STAGE\",
    \"model_class\": \"vibevoice.modular.modeling_vibevoice_inference.VibeVoiceForConditionalGenerationInference\",
    \"processor_class\": \"vibevoice.processor.vibevoice_processor.VibeVoiceProcessor\",
    \"torch_dtype\": \"bf16\",
    \"custom_handler\": \"vibevoice\"
  }"
fi

step "推理（POST /inference/tts — 09 audio_tts_service 走的端點）"
VOICE_OPT=""
[ -n "$VOICE" ] && VOICE_OPT=", \"options\": {\"voice\": \"$VOICE\"}"
api_post /inference/tts "{
  \"text\": \"$TEXT\",
  \"model_name\": \"$REGISTERED_NAME\",
  \"output_format\": \"wav\"$VOICE_OPT
}" | python3 -c "
import json, sys, base64
r = json.load(sys.stdin)
res = r['result']
open('$OUT', 'wb').write(base64.b64decode(res['audio_base64']))
print(f\"  sample_rate={res['sample_rate']} duration={res['metadata']['duration_seconds']}s → $OUT\")"

info "註冊完成後，09 的 audio_tts_service 不帶 model_name 的請求也會命中本模型"
info "（07 的 DEFAULT_TTS_MODEL 預設就是 vibevoice-tts）"
