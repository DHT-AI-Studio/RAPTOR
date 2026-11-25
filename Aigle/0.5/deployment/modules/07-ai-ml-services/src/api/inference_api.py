# src/api/inference_api.py
"""
推理 API — FastAPI router

對外 API 形狀（task / engine / model_name / data / options）保持不變，
但內部已切到 spec-driven 的 InferenceService。
engine 欄位現在是可選欄位：模型已註冊時 runtime 由 MLflow tag 決定；
模型未註冊且 engine="ollama" 時，把 model_name 當 Ollama daemon tag 直接呼叫。

新增（皆為可選／附加欄位，舊 caller 不受影響）：
    - request：model_version / model_alias（互斥）選定版本
    - response：usage（text-generation 真實 token 計數）
    - error：detail.code（穩定錯誤碼）及必要欄位（如 model_limit）
    - GET /inference/model-info：只讀模型能力與限制查詢
"""

import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from ..inference.exceptions import (
    InferenceError,
    PolicyViolationError,
    ResourceNotFoundError,
    error_detail,
)
from ..inference.service import inference_service

# Guardrail hook (GB-6, module 23) — no-op unless GUARDRAIL_URL is set
from . import guardrail_hook

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/inference", tags=["推理 (Inference)"])


# ===== 請求模型 =====


class InferenceRequest(BaseModel):
    """統一推理請求模型。

    runtime（ollama / hf-transformers）由 MLflow tag 決定，呼叫端不需指定；
    engine 只在模型未註冊時作為 fallback 依據（engine="ollama" → 當 daemon tag 呼叫）。
    """

    task: str | None = Field(
        default=None,
        description="任務類型（可省略 — 省略時用模型註冊的 task_family）；"
                    "舊名（如 text-generation-ollama / asr-hf）會自動 canonicalize",
        example="text-generation",
    )
    model_name: str = Field(..., description="MLflow 中註冊的模型名稱", example="gemma4-12b-ollama")
    data: dict[str, Any] = Field(
        ...,
        description="輸入資料；欄位依 task 而異（見端點說明）",
        example={"inputs": "請用繁體中文寫一段對 MLOps 的介紹"},
    )
    engine: str | None = Field(
        default=None,
        description="模型已註冊時忽略（runtime 由 MLflow tag 決定）；模型未註冊且填 'ollama' 時，"
                    "把 model_name 當 Ollama daemon 上的 tag 直接呼叫。",
    )
    model_version: str | None = Field(
        default=None,
        description="指定 MLflow 版本號（與 model_alias 互斥）；不存在回 404 model_version_not_found，不回退。"
                    "未提供 selector 時使用最大版本號。",
        example="3",
    )
    model_alias: str | None = Field(
        default=None,
        description="指定 MLflow alias（如 production；與 model_version 互斥），於本次請求開始時解析為固定版本；"
                    "未設定回 404 model_alias_not_found，不回退。",
        example="production",
    )
    options: dict[str, Any] | None = Field(
        default_factory=dict,
        description="推理選項（max_tokens / temperature / stop / top_p / ...）",
        example={"temperature": 0.7, "max_tokens": 500},
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "task": "text-generation",
            "model_name": "gemma4-12b-ollama",
            "data": {"inputs": "請用繁體中文寫一段對 MLOps 的介紹"},
            "options": {"temperature": 0.7, "max_tokens": 500},
        }
    })


# ===== 主要端點 =====


@router.post("/infer", summary="統一推理接口")
def unified_inference(request: InferenceRequest):
    """
    執行模型推理（所有 task 共用此單一端點）

    **輸入參數說明：**
    - **task** (可選): 任務類型；省略時直接採用模型註冊的 task_family（xinference 式：只給模型名即可）。
      有填時會與註冊值核對，不一致回 400。舊名會被自動轉成 canonical 名
      - canonical 名：`text-generation`, `vlm`, `asr`, `ocr`, `audio-classification`,
        `video-analysis`, `document-analysis`, `image-captioning`
      - 舊名（仍接受）：`text-generation-ollama`, `text-generation-hf`, `asr-hf`,
        `vad-hf`, `ocr-hf`, `audio-transcription`, `scene-detection`, `video-summary`
    - **model_name** (必填): MLflow Registry 中的註冊名稱；runtime 由其 tag 決定
    - **model_version** / **model_alias** (可選, 互斥): 選定版本；同時提供回 400 `conflicting_parameters`；
      指定的版本／alias 不存在回 404（`model_version_not_found` / `model_alias_not_found`），不回退其他版本。
      未提供時沿用「最大版本號」
    - **data** (必填): 輸入資料，欄位依 task 而異：
      - text-generation: `{"inputs": str}` 或 `{"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}`
      - vlm / image-captioning: `{"image": <path|PIL|base64>, "prompt": str}`
      - asr / audio-classification: `{"audio": <path|numpy>}`
      - ocr: `{"image": <path|PIL>}`
      - video-analysis: `{"video": str}` (`prompt` 可選)
      - document-analysis: `{"document": str}` (`query` 可選)
    - **options** (可選): 推理選項
      - text-generation 共用欄位（Ollama / HF raw / HF pipeline / 串流皆實際套用，或明確回 400）：
        - `max_tokens`（= `max_new_tokens`；Ollama → `num_predict`、HF → `max_new_tokens`）；
          超過模型已配置的 `max_output_tokens` 回 422 `max_tokens_exceeded`（含 `model_limit`）
        - `stop`（= `stop_sequences`）：str 或 list[str]；回應不含停止字串及其後文字
        - `temperature`：0～2；0 = greedy（與 `do_sample=true` 同時提供回 400）
        - `max_length`：保留既有語意（HF = 含 prompt 的總長度、Ollama = 生成上限），
          不可與 `max_tokens` / `max_new_tokens` 混用（400 `conflicting_parameters`）
        - 同義欄位同時提供且值不同 → 400 `conflicting_parameters`
      - 其他通用：`top_p`, `top_k`, `do_sample`, `think`（預設 false；HF 以 chat template 的 enable_thinking 套用）
      - **`format`**（結構化輸出）：OpenAI `response_format` 形式（`{"type": "json_object"}` /
        `{"type": "json_schema", "json_schema": {"name", "schema"}}`）、`"json"`，或直接一份 JSON Schema。
        只有 Ollama text-generation 能照辦（其他回 400 `unsupported_capability`）；schema 只接受
        型別 / properties / required / items / enum / 數值與長度限制等關鍵字（其他回 400 `invalid_parameter`）。
        輸出一定經驗證：不合格回 502 `structured_output_error`（含 `violations`）；合格時 `result.response`
        是正規化後的 JSON 文字，`result.parsed` 是解析後的值
      - **`keep_alive`**（ollama 式）：本次推理後模型在快取/GPU 的存活時間。
        秒數或 `"30s"`/`"5m"`/`"1h"`；`0` = 推理完立即卸載；負值 = 常駐。
        未填時用 `inference.yaml` 的 `idle_timeout`（HF 預設 300 秒）
      - HF 專屬：`num_beams`, `repetition_penalty`, `length_penalty`, `early_stopping`
      - Ollama 專屬：`repeat_penalty`, `stop`, `mirostat`, `num_ctx`
      - ASR：`language`, `task`（`transcribe`/`translate`）
    - **engine** (可選): 模型已註冊時忽略；模型未註冊（且未指定版本 selector）時填 `ollama`，
      會把 model_name 當 Ollama daemon tag 直接呼叫（registry 無法連線時不會觸發此 fallback）

    **使用範例（Ollama 文本生成）：**
    ```json
    {
      "task": "text-generation",
      "model_name": "gemma4-12b-ollama",
      "data": {"inputs": "請寫一首關於春天的詩"},
      "options": {"max_tokens": 200, "temperature": 0.8}
    }
    ```

    **使用範例（指定版本 + messages + 停止字串）：**
    ```json
    {
      "model_name": "gemma4-12b-ollama",
      "model_alias": "production",
      "data": {"messages": [
        {"role": "system", "content": "你是翻譯助理，只輸出譯文。"},
        {"role": "user", "content": "Translate to English: 早安"}
      ]},
      "options": {"max_tokens": 128, "temperature": 0, "stop_sequences": ["\\n\\n"]}
    }
    ```

    **使用範例（VLM 圖像理解）：**
    ```json
    {
      "task": "vlm",
      "model_name": "qwen2.5-vl-7b",
      "data": {
        "image": "/data/images/scene.jpg",
        "prompt": "詳細描述這張圖片"
      },
      "options": {"max_new_tokens": 256}
    }
    ```

    **使用範例（語音識別）：**
    ```json
    {
      "task": "asr",
      "model_name": "whisper-large-v3",
      "data": {"audio": "/data/audio/clip.wav"},
      "options": {"language": "zh", "task": "transcribe"}
    }
    ```

    **返回值：**
    - **success** (bool): 是否成功
    - **result** (object): 推理結果，主要欄位依 task 而異（`response` / `text` / `classifications`）
    - **task** (str): 實際執行的 canonical task family
    - **engine** (str): 實際使用的 runtime（`ollama` / `hf-transformers`）
    - **model_name** (str): 模型名稱
    - **model_version** (str): 實際執行的 MLflow 模型版本（alias 已解析；未註冊的 ollama fallback 為 `unregistered`）
    - **usage** (object|null): `{"prompt_tokens", "completion_tokens", "total_tokens"}`，text-generation 的
      真實 token 計數（HF 以實際送入／新生成的 token IDs、Ollama 採 daemon 回報值；不同 runtime 不保證一致）；
      非文字 task、custom handler 或 runtime 未回報計數時為 null
    - **processing_time** (float): service 內推理耗時（秒）
    - **api_processing_time** (float): API 層總耗時（秒）
    - **timestamp** (float): 完成時間

    **錯誤碼：** `detail` 固定含 `error_type` / `message` / `code`
    - **400**: `invalid_request`（data 缺欄位等）、`unsupported_task`、`invalid_parameter`、
      `conflicting_parameters`、`unsupported_parameter`、`policy_violation`
    - **404**: `model_not_found`、`model_version_not_found`、`model_alias_not_found`（含 `model_name`）
    - **400**: `unsupported_capability`（要求結構化輸出，但模型／runtime 做不到）
    - **422**: `max_tokens_exceeded`（含 `model_limit`、`requested`）
    - **502**: `structured_output_error`（輸出不是 JSON 或不符 schema；含 `violations`）
    - **500**: `model_load_failed`、`inference_failed`、`engine_error`、`inference_error`
    - **503**: `registry_unavailable`、`runtime_unavailable`、`resource_exhausted`（含 `retry_after`=60）
    - **504**: `inference_timeout`（上游 runtime 逾時）
    """
    t0 = time.time()
    try:
        request_id = str(uuid.uuid4())

        # GB-6 前置護欄：送入 LLM 前檢查輸入。text-generation 有 data["inputs"]
        # (prompt string) 跟 data["messages"] (chat) 兩種真實形狀 -- extract_input_text
        # 兩種都認得(見其 docstring),不只 inputs 字串那種。
        # module 23 的 POST /guardrail/policy/decide 回 allow/warn/redact/block：
        # block 擋下；warn 放行（hook 會記 log）；redact 把 redacted_content 寫回
        # request.data 原本取文字的位置，再送進 LLM。
        gin = guardrail_hook.check(guardrail_hook.extract_input_text(request.data), "input", request.task, request_id)
        if guardrail_hook.is_blocked(gin):
            raise PolicyViolationError(guardrail_hook.blocked_category(gin), "input")
        guardrail_hook.apply_input_redaction(request.data, gin)

        result = inference_service.infer(
            task=request.task,
            model_name=request.model_name,
            data=request.data,
            engine=request.engine,
            options=request.options,
            model_version=request.model_version,
            model_alias=request.model_alias,
        )

        # GB-6 後置護欄：回傳前檢查產生的輸出。result["result"] 本身是個 dict
        # ({"response": ...} 等) 不是字串 -- 直接丟給 check() 一定 fail isinstance(str)
        # 靜默 no-op，extract_output_text 先把實際文字挖出來。
        # redact 時把 redacted_content 寫回 result["result"] 原本取文字的位置。
        gout = guardrail_hook.check(guardrail_hook.extract_output_text(result.get("result")), "output", request.task, request_id)
        if guardrail_hook.is_blocked(gout):
            raise PolicyViolationError(guardrail_hook.blocked_category(gout), "output")
        guardrail_hook.apply_output_redaction(result.get("result"), gout)

        result["api_processing_time"] = time.time() - t0
        result["success"] = True
        return result

    except PolicyViolationError as e:
        logger.warning(f"guardrail blocked ({e.direction}): category={e.category}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_type": "PolicyViolationError",
                "message": str(e),
                "code": e.code,
                "category": e.category,
                "direction": e.direction,
            },
        )
    except InferenceError as e:
        raise _http_error(e, model_name=request.model_name)
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_type": "UnexpectedError", "message": str(e), "code": "internal_error"},
        )


def _http_error(e: InferenceError, model_name: str | None = None) -> HTTPException:
    """推理異常 → HTTPException（沿用 detail.error_type / message，新增 code 與必要欄位）。"""
    status_code, detail = error_detail(e)
    if isinstance(e, ResourceNotFoundError) and model_name:
        detail.setdefault("model_name", model_name)
    if status_code >= 500:
        logger.error(f"inference error [{detail['code']}]: {e}", exc_info=status_code == 500)
    else:
        logger.warning(f"inference request rejected [{detail['code']}]: {e}")
    return HTTPException(status_code=status_code, detail=detail)


# ===== 模型能力查詢 =====


@router.get("/model-info", summary="查詢模型能力與限制（只讀，不載入模型）")
def model_info(
    model_name: str = Query(..., description="MLflow 註冊名稱（含 / 的名稱亦可，放在 query）"),
    model_version: str | None = Query(None, description="指定版本（與 model_alias 互斥）"),
    model_alias: str | None = Query(None, description="指定 alias，如 production（與 model_version 互斥）"),
    engine: str | None = Query(None, description="模型未註冊時填 ollama，查詢 Ollama-only 模型的設定"),
):
    """
    查詢模型版本的推論能力與限制；與 `/inference/infer` 共用同一個版本解析器，
    不會觸發模型載入、pull 或權重下載

    **使用範例：**
    - GET /inference/model-info?model_name=gemma4-12b-ollama&model_alias=production

    **返回值：**
    - **model_name** / **model_version** (str): 解析後的模型與版本（推論帶相同 selector 時會用同一版）
    - **runtime** (str): `ollama` / `hf-transformers`
    - **task_family** (str)
    - **runtime_model** (str|null): Ollama daemon 上的實體 tag（HF 為 null）
    - **supported_options** (list): 實際可套用的共用 options（非文字 task 為空）
    - **limits.max_output_tokens** / **limits.context_window_tokens** (int|null): 未配置為 null
    - **usage_supported** (bool): 此執行路徑是否回傳統一 usage
    - **structured_output** (bool): 是否接受 `format` / `response_format`（Ollama text-generation）

    **錯誤碼：** 同 `/inference/infer` 的 400 / 404 / 503
    """
    try:
        return inference_service.model_info(
            model_name, model_version=model_version, model_alias=model_alias, engine=engine
        )
    except InferenceError as e:
        raise _http_error(e, model_name=model_name)


# ===== TTS 端點 =====


class TTSRequest(BaseModel):
    """文字轉語音請求（09-audio-processing 的 audio_tts_service 依此合約呼叫）。"""

    text: str = Field(..., description="要合成的文字", example="你好，歡迎使用 Raptor。")
    model_name: str | None = Field(
        default=None,
        description="MLflow 中註冊的 tts 模型名；省略時用環境變數 DEFAULT_TTS_MODEL",
    )
    voice: str | None = Field(default=None, description="語音（模型支援才有作用）")
    speed: float | None = Field(default=None, description="語速（模型支援才有作用）")
    output_format: str = Field(default="wav", description="輸出格式（目前僅 wav）")
    options: dict[str, Any] | None = Field(default_factory=dict, description="其他推理選項（keep_alive 等）")


@router.post("/tts", summary="文字轉語音（TTS）")
def text_to_speech(request: TTSRequest):
    """
    文字轉語音端點；等價於 `POST /inference/infer`（task=tts）的便捷包裝

    **輸入參數說明：**
    - **text** (必填): 要合成的文字
    - **model_name** (可選): MLflow 已註冊的 tts 模型；省略時取環境變數
      `DEFAULT_TTS_MODEL`（預設 `vibevoice-tts`）
    - **voice** / **speed** (可選): 傳遞給模型的語音與語速（模型支援才有作用）
    - **output_format** (可選): 目前僅支援 `wav`

    **返回值：**
    - **success** (bool)
    - **result.audio_base64** (str): base64 編碼的 WAV 音訊
    - **result.format** (str) / **result.sample_rate** (int)
    - 其餘欄位同 `/inference/infer`
    """
    model_name = request.model_name or os.getenv("DEFAULT_TTS_MODEL", "vibevoice-tts")
    options = dict(request.options or {})
    if request.voice is not None:
        options["voice"] = request.voice
    if request.speed is not None:
        options["speed"] = request.speed
    if request.output_format and request.output_format != "wav":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_type": "ValidationError",
                    "message": f"output_format '{request.output_format}' not supported yet (only 'wav')"},
        )
    return unified_inference(InferenceRequest(
        task="tts",
        model_name=model_name,
        data={"text": request.text},
        options=options,
    ))


# ===== 內省與監控端點 =====


@router.get("/health", summary="檢查推理服務健康狀態")
def health_check():
    """
    檢查推理服務（service 與已初始化 adapter）的整體健康狀態

    **無需輸入參數**

    **使用範例：**
    - GET /inference/health

    **返回值：**
    - **status** (str): 整體狀態（`healthy` / `unhealthy` / `error`）
    - **adapters_initialized** (list): 目前已初始化的 adapter runtime 名稱
      - 註：adapter 採 lazy-init，第一次有對應請求時才會建立
    - **stats** (object): 推理計數快照（內容同 /stats）
    - **timestamp** (float): 檢查時間
    """
    return inference_service.health_check()


@router.get("/supported-tasks", summary="列出支持的任務類型")
def get_supported_tasks():
    """
    列出所有支持的 task family 與其 data 必填欄位

    **無需輸入參數**

    **使用範例：**
    - GET /inference/supported-tasks

    **返回值：**
    - **tasks** (object): 各 task family 的設定，鍵為 task 名，值包含：
      - **required_fields** (list): 該任務 data 中必填的欄位名
    - **total_tasks** (int): 支持的 task family 總數

    **範例返回：**
    ```json
    {
      "tasks": {
        "text-generation": {"required_fields": ["inputs"]},
        "vlm": {"required_fields": ["image", "prompt"]},
        "asr": {"required_fields": ["audio"]},
        "ocr": {"required_fields": ["image"]}
      },
      "total_tasks": 8
    }
    ```
    """
    tasks = inference_service.get_supported_tasks()
    return {"tasks": tasks, "total_tasks": len(tasks)}


@router.get("/stats", summary="取得推理統計資訊")
def get_stats():
    """
    取得推理服務啟動以來的累計統計

    **無需輸入參數**

    **使用範例：**
    - GET /inference/stats

    **返回值：**
    - **stats** (object): 統計資料
      - **total** (int): 累計推理請求數
      - **success** (int): 成功數
      - **failed** (int): 失敗數
      - **success_rate** (float): 成功率（0–1）
      - **loaded** (object): 已載入模型的快照（內容同 /loaded-models）
    - **timestamp** (float): 取得統計的時間
    """
    return {"stats": inference_service.get_stats(), "timestamp": time.time()}


@router.get("/loaded-models", summary="列出所有已載入模型")
def get_loaded_models():
    """
    列出每個 adapter 內 LRU 快取中的模型清單

    **無需輸入參數**

    **使用範例：**
    - GET /inference/loaded-models

    **返回值：**
    - **total** (int): 已載入模型總數（跨 adapter 加總）
    - **by_runtime** (object): 依 runtime 分組的模型名稱
      - **ollama** (list): Ollama adapter LRU 中的模型
      - **hf-transformers** (list): HF adapter LRU 中的模型
    - **details** (object): 依 runtime 分組的完整快取狀態，每個模型含
      `loaded_at` / `last_used` / `idle_seconds` / `expires_in_seconds`（None = 常駐）/
      `use_count` / `in_flight`
    - **timestamp** (float): 取得時間

    **範例返回：**
    ```json
    {
      "total": 2,
      "by_runtime": {
        "ollama": ["gemma4-12b-ollama"],
        "hf-transformers": ["qwen2.5-vl-7b"]
      },
      "timestamp": 1714195200.0
    }
    ```
    """
    return {**inference_service.loaded_models(), "timestamp": time.time()}


# ===== 模型生命週期端點 =====


@router.post("/cache/clear", summary="清空所有 adapter 的模型快取")
def clear_cache():
    """
    卸載所有 adapter 中 LRU 快取的模型實例，並清理 GPU cache

    **無需輸入參數**

    **使用範例：**
    - POST /inference/cache/clear

    **返回值：**
    - **success** (bool): 是否成功
    - **timestamp** (float): 清理完成時間

    **建議使用時機：**
    - GPU OOM 後快速釋放資源
    - 切換到不同的模型集前
    - 長時間 idle 後想釋放記憶體

    **注意：** 後續推理請求會重新載入模型，首次推理會比較慢
    """
    inference_service.clear_cache()
    return {"success": True, "timestamp": time.time()}


@router.post("/clear-cache", summary="清空快取（舊路徑，建議改用 /cache/clear）")
def clear_cache_legacy():
    """
    向下相容路徑；行為等同 `POST /inference/cache/clear`

    **建議使用 `/cache/clear` 取代此端點。**
    """
    return clear_cache()


@router.post("/unload-model", summary="卸載指定模型")
def unload_model(model_name: str, task: str | None = None, engine: str | None = None):
    """
    卸載指定模型在所有 adapter 中的實例，並清理 GPU cache

    **查詢參數說明：**
    - **model_name** (必填): 要卸載的模型名稱（在 MLflow 中的註冊名）
    - **task** (可選, 已棄用): 為向下相容保留；新版 service 不需要此參數
    - **engine** (可選, 已棄用): 為向下相容保留；新版 service 不需要此參數

    **使用範例：**
    - POST /inference/unload-model?model_name=qwen2.5-vl-7b
    - POST /inference/unload-model?model_name=gemma4-12b-ollama

    **返回值：**
    - **success** (bool): 是否成功
    - **model_name** (str): 被卸載的模型名稱
    - **unloaded_runtimes** (list): 該模型曾被載入的 runtime 名稱（如 `["hf-transformers"]`）
      - 若空 list 表示該模型本來就不在任何 adapter 的 cache 中
    - **gpu_memory_freed** (object): GPU 記憶體釋放資訊
      - cuda_available (bool)
      - memory_before_gb / memory_after_gb / memory_freed_gb (float)
    - **timestamp** (float): 完成時間
    """
    result = inference_service.unload_model(model_name)
    return {**result, "timestamp": time.time()}


@router.post("/unload-all-models", summary="卸載所有已載入模型")
def unload_all_models():
    """
    卸載所有 adapter 內目前已載入的模型實例，並清理 GPU cache

    **無需輸入參數**

    **使用範例：**
    - POST /inference/unload-all-models

    **返回值：**
    - **success** (bool): 是否成功
    - **total_unloaded** (int): 卸載的模型實例總數
    - **gpu_memory_freed** (object): GPU 記憶體釋放資訊（同 /unload-model）
    - **timestamp** (float): 完成時間

    **注意：** 後續推理請求會重新載入模型，首次推理會比較慢；
    與 `/cache/clear` 的差別只在回傳欄位（此端點多回 `total_unloaded`）。
    """
    result = inference_service.unload_all()
    return {**result, "timestamp": time.time()}


# ===== 向下相容端點 =====


@router.post("/infer_fixed", summary="固定模型推理（向下相容，建議改用 /infer）")
def infer_fixed_compatibility(request: InferenceRequest):
    """
    向下相容路徑；行為等同 `POST /inference/infer`

    新版本沒有「fixed」與「dynamic」推理之分，所有推理走統一 spec-driven 流程。
    **建議使用 `/infer` 取代此端點。**
    """
    return unified_inference(request)


@router.post("/infer_multimodal", summary="多模態推理（向下相容，建議改用 /infer）")
def infer_multimodal_compatibility(request: InferenceRequest):
    """
    向下相容路徑；行為等同 `POST /inference/infer`

    新版本所有任務（含多模態）皆走統一 `/infer` 端點。
    **建議使用 `/infer` 取代此端點。**
    """
    return unified_inference(request)
