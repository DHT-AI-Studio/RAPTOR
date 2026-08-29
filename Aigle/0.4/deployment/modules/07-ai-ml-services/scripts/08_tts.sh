#!/usr/bin/env bash
# TTS 文字轉語音：下載 → 上傳 lakeFS → 註冊（pipeline 捷徑）→ 合成語音存檔
#
# 用法：
#   ./08_tts.sh                                  # 預設 facebook/mms-tts-eng（英文 VITS，145MB）
#   TEXT="hello world" OUT=/tmp/out.wav ./08_tts.sh
#
# 標準 HF TTS（VITS / SpeechT5 / Bark）填 pipeline_task=text-to-speech 即可；
# VibeVoice 等非標準介面需另寫 custom_handler（見 README 工作流 D）。
# 09-audio-processing 的 audio_tts_service 用的就是本腳本示範的 /inference/tts 端點
#（其 payload 不帶 model_name，由 07 的環境變數 DEFAULT_TTS_MODEL 決定模型）。

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-facebook/mms-tts-eng}"
MODEL_PARAMS="${MODEL_PARAMS:-0.1}"
BASENAME=$(basename "$HF_MODEL" | tr '[:upper:]._' '[:lower:]--')
LAKEFS_REPO="${LAKEFS_REPO:-$BASENAME}"
REGISTERED_NAME="${REGISTERED_NAME:-$BASENAME}"
TEXT="${TEXT:-Hello from the unified Raptor model service.}"
OUT="${OUT:-./tts_output.wav}"

check_health

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
    \"pipeline_task\": \"text-to-speech\"
  }"
fi

step "推理（POST /inference/tts → base64 JSON）"
api_post /inference/tts "{
  \"text\": \"$TEXT\",
  \"model_name\": \"$REGISTERED_NAME\",
  \"output_format\": \"wav\"
}" | python3 -c "
import json, sys, base64
r = json.load(sys.stdin)
res = r['result']
open('$OUT', 'wb').write(base64.b64decode(res['audio_base64']))
print(f\"  sample_rate={res['sample_rate']} duration={res['metadata']['duration_seconds']}s → $OUT\")"

step "推理（OpenAI 相容 /v1/audio/speech → 直接回二進位 WAV）"
curl -sf -X POST "$API_BASE/v1/audio/speech" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$REGISTERED_NAME\", \"input\": \"$TEXT\"}" \
  -o "${OUT%.wav}_openai.wav" -w "  HTTP %{http_code}, %{size_download} bytes → ${OUT%.wav}_openai.wav\n"
