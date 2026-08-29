#!/usr/bin/env bash
# Embedding（bge-m3，17-hybrid-search 使用的向量模型）：下載 → 上傳 lakeFS → 註冊 → 向量化
#
# 用法：
#   ./06_embedding_bge_m3.sh
#   HF_MODEL=sentence-transformers/all-MiniLM-L6-v2 MODEL_PARAMS=0.02 ./06_embedding_bge_m3.sh   # 小模型驗證流程
#
# 註：BGE 系列用 CLS pooling（07 的預設）；MiniLM 等 sentence-transformers 模型請在
#     options 加 {"pooling": "mean"}。

source "$(dirname "$0")/common.sh"

HF_MODEL="${HF_MODEL:-BAAI/bge-m3}"
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
    \"task\": \"embedding\",
    \"engine\": \"transformers\",
    \"model_params\": $MODEL_PARAMS,
    \"lakefs_repo\": \"$LAKEFS_REPO\",
    \"stage\": \"$STAGE\",
    \"model_class\": \"AutoModel\",
    \"processor_class\": \"AutoTokenizer\"
  }"
fi

step "推理（OpenAI 相容 /v1/embeddings）"
api_post /v1/embeddings "{
  \"model\": \"$REGISTERED_NAME\",
  \"input\": [\"什麼是 MLOps？\", \"向量檢索與語義搜尋\"]
}" | python3 -c "
import json, sys
r = json.load(sys.stdin)
for d in r['data']:
    v = d['embedding']
    print(f\"  index={d['index']} dim={len(v)} head={[round(x,4) for x in v[:4]]}\")
print('  usage:', r['usage'])"

step "推理（原生 /inference/infer — 可帶 pooling / normalize 選項）"
infer "{
  \"model_name\": \"$REGISTERED_NAME\",
  \"data\": {\"inputs\": \"hello world\"},
  \"options\": {\"pooling\": \"cls\", \"normalize\": true}
}" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print('  dim:', r['result']['metadata']['dim'], ' count:', r['result']['metadata']['count'])"
