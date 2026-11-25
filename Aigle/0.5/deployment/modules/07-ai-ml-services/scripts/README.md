# 模型使用腳本

每個腳本示範一類模型的完整生命週期：**下載 → 上傳 lakeFS → 註冊 MLflow → 調用推理**。
全部透過 07 的 HTTP API 操作（curl），不直接碰 MLflow / lakeFS SDK。

## 前置

- 07 服務已啟動（`docker compose up -d`），MLflow / lakeFS 可用
- Ollama 腳本需要 daemon 上已有模型；HF 腳本需要 `HF_TOKEN`（gated 模型如 gemma）
- 本機需要 `curl` 與 `python3`（美化輸出用）

## 共用環境變數（所有腳本）

| 變數 | 預設 | 說明 |
|------|------|------|
| `API_BASE` | `http://localhost:8010` | 07 API 位址；dev stack 用 `http://localhost:9997` |
| `STAGE` | `staging` | 註冊後切換的階段 |
| `SKIP_DOWNLOAD=1` | – | 模型已在 lakeFS 時跳過下載/上傳 |
| `SKIP_REGISTER=1` | – | 模型已在 MLflow 時跳過註冊，直接推理 |
| `HF_MODEL` / `MODEL_PARAMS` / `LAKEFS_REPO` / `REGISTERED_NAME` | 各腳本有預設 | 換模型時覆寫 |

## 腳本一覽

| 腳本 | 模型（預設） | task | 註冊方式 | 調用端點 |
|------|--------------|------|----------|----------|
| `01_ollama_llm.sh` | gemma4:12b | text-generation | `register_ollama`（不經 lakeFS） | `/inference/infer`、`/v1/chat/completions` |
| `02_hf_text_generation.sh` | google/gemma-3-270m-it | text-generation | `pipeline_task=text-generation` | `/inference/infer`、`/v1/completions` |
| `03_vlm_qwen2.5_vl.sh` | Qwen/Qwen2.5-VL-7B-Instruct | vlm | `model_class=Qwen2_5_VLForConditionalGeneration` + 4bit | `/inference/infer` |
| `04_vlm_internvl.sh` | OpenGVLab/InternVL3_5-1B | vlm | `custom_handler=internvl` + `torch_dtype=bf16` | `/inference/infer` |
| `05_asr_whisper.sh` | openai/whisper-large-v3 | asr | `pipeline_task=automatic-speech-recognition` | `/inference/infer`、`/v1/audio/transcriptions` |
| `06_embedding_bge_m3.sh` | BAAI/bge-m3 | embedding | `model_class=AutoModel` | `/v1/embeddings`、`/inference/infer` |
| `07_rerank_bge.sh` | BAAI/bge-reranker-v2-m3 | rerank | `model_class=AutoModelForSequenceClassification` | `/v1/rerank`、`/inference/infer` |
| `08_tts.sh` | facebook/mms-tts-eng | tts | `pipeline_task=text-to-speech` | `/inference/tts`、`/v1/audio/speech` |
| `09_tts_vibevoice.sh` ⚠️ | VibeVoice（需鏡像權重 + 套件） | tts | `custom_handler=vibevoice` + dotted class | `/inference/tts` |

> ⚠️ `09_tts_vibevoice.sh` 為實驗性：官方 repo/權重已下架，需先手動安裝社群 fork 套件與
> 取得權重鏡像（腳本開頭有前置說明）。跑通後 09-audio-processing 的 TTS 即完全接通。

## 使用範例

```bash
# dev stack 上先用小模型驗證整條流程（下載僅 ~90MB）
API_BASE=http://localhost:9997 \
HF_MODEL=sentence-transformers/all-MiniLM-L6-v2 MODEL_PARAMS=0.02 \
./06_embedding_bge_m3.sh

# 正式跑 bge-m3
API_BASE=http://localhost:9997 ./06_embedding_bge_m3.sh

# 模型已註冊過，只想重新推理
SKIP_DOWNLOAD=1 SKIP_REGISTER=1 ./06_embedding_bge_m3.sh

# InternVL 圖片描述（IMAGE 給本機檔案會自動轉 base64）
IMAGE=./sample.jpg ./04_vlm_internvl.sh

# TTS 合成並存檔
TEXT="歡迎使用 Raptor" OUT=./hello.wav ./08_tts.sh
```

## 對應其他模組的接入

- **09 audio_tts_service** → `08_tts.sh` 的 `/inference/tts`（09 的 payload 不帶 `model_name`，
  由 07 環境變數 `DEFAULT_TTS_MODEL` 決定；預設 `vibevoice-tts`）
- **10/11/12 的 InternVL** → `04_vlm_internvl.sh`（註冊一次三個模組共用）
- **17 hybrid-search** → `06_embedding_bge_m3.sh` + `07_rerank_bge.sh`
- **21 reranker fallback** → `07_rerank_bge.sh`（`HF_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`）
- **各模組 Ollama LLM** → `01_ollama_llm.sh`（註冊後 base_url 改指 07 `/v1` 即可）

## 疑難排解

- 註冊/推理錯誤會直接印出 API 的 `detail` JSON（`error_type` / `message`）
- GPU OOM：`curl -X POST $API_BASE/inference/cache/clear`，或註冊時加 `"quantization": "4bit"`
- 查看已載入模型與閒置狀態：`curl $API_BASE/inference/loaded-models`
- 一次性請求想用完即卸：options 加 `"keep_alive": 0`
