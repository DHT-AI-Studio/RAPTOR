# src/api/openai_api.py
"""
OpenAI 相容 API（xinference 式）— FastAPI router

讓任何 OpenAI SDK / client 直接對接本平台已註冊在 MLflow 的模型：

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8010/v1", api_key="not-needed")
    client.chat.completions.create(model="qwen3-1.7b-ollama", messages=[...])

端點：
    GET  /v1/models                 — 列出 MLflow 已註冊模型
    POST /v1/chat/completions       — chat（text-generation 與 vlm 模型皆可；vlm 用 image_url parts）
    POST /v1/completions            — 傳統 prompt completion
    POST /v1/audio/transcriptions   — Whisper 式語音轉文字（asr 模型）
    POST /v1/audio/speech           — 文字轉語音（tts 模型；回傳二進位音訊）
    POST /v1/embeddings             — 文本向量化（embedding 模型）
    POST /v1/rerank                 — cross-encoder 重排（rerank 模型；xinference/jina 式）

呼叫端不需要知道 task / runtime — 全部由 MLflow tag 決定（spec-driven）。
額外欄位 `keep_alive`（ollama 式）可控制推理後模型在 GPU 上的存活時間。

串流：`stream=true` 時 text-generation 模型（ollama / hf-transformers）走
真·逐 token SSE；其他 task（如 VLM）fallback 為單 chunk 偽串流以維持相容。
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..inference.exceptions import (
    EngineError,
    InferenceError,
    InferenceExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    ResourceExhaustedError,
    UnsupportedTaskError,
    ValidationError,
)
from ..inference.service import inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI 相容 API (v1)"])


# ===== 請求模型 =====


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # 容忍 OpenAI client 送來的其他欄位

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    keep_alive: Optional[Union[int, float, str]] = Field(
        default=None,
        description="ollama 式擴充欄位：推理後模型存活秒數（0=立即卸載、負值=常駐、'5m' 等字串亦可）",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="OpenAI 格式的 function-calling 定義（[{\"type\":\"function\",\"function\":{...}}]）。"
                    "只有 Ollama runtime、且模型本身支援 tools（見 `ollama show <model>` 的 capabilities）"
                    "才有效；其他 runtime 目前忽略此欄位。",
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="目前未轉發給 Ollama（daemon 沒有對應參數，模型自行決定要不要呼叫）——"
                    "接受這個欄位只是為了不讓帶了它的 OpenAI client 出錯，不是真的支援強制/停用。",
    )
    engine: Optional[str] = Field(
        default=None,
        description="模型未在 MLflow 註冊時的 fallback 依據，見 /inference/infer 的同名欄位。"
                    "例如 engine='ollama' 讓呼叫端可以直接用 Ollama daemon 上的原生 tag，不用先註冊。",
    )
    num_ctx: Optional[int] = Field(
        default=None,
        description="ollama 式擴充欄位：context window 大小。Ollama 自己的 /v1 OpenAI 相容層會"
                    "靜默忽略這個值（已實測確認）；這裡透過 options → Ollama 原生 /api/chat 的"
                    "options.num_ctx 傳遞，才是唯一真的能從 OpenAI 相容介面設定它的路徑。",
    )
    think: Optional[bool] = Field(
        default=None,
        description="ollama 式擴充欄位：qwen3 系列模型預設會開啟思考模式，明確傳 false 可跳過，"
                    "省下大量延遲（其餘 runtime 忽略此欄位）。不傳則沿用 OllamaAdapter._chat() 的"
                    "既有預設（think=false）。",
    )


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    keep_alive: Optional[Union[int, float, str]] = None


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: Union[str, List[str]]
    encoding_format: str = "float"
    keep_alive: Optional[Union[int, float, str]] = None


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: bool = True
    keep_alive: Optional[Union[int, float, str]] = None


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: Optional[float] = None
    keep_alive: Optional[Union[int, float, str]] = None


# ===== 端點 =====


@router.get("/models", summary="列出可用模型（OpenAI 格式）")
def list_models():
    """回傳 MLflow Registry 中所有已註冊模型（最新版本），OpenAI `GET /v1/models` 格式。

    每個模型的 `metadata` 額外帶出 task_family / runtime / stage / version，
    方便呼叫端判斷該模型可以用哪個端點調用。
    """
    from ..core.model_manager import model_manager

    data = []
    for m in model_manager.list_mlflow_models(show_all=False):
        for v in m.get("versions", []):
            tags = v.get("tags", {}) or {}
            data.append({
                "id": m["name"],
                "object": "model",
                "created": _to_epoch(v.get("creation_timestamp")),
                "owned_by": "mlflow-registry",
                "metadata": {
                    "task_family": tags.get("task_family") or tags.get("inference_task"),
                    "runtime": tags.get("runtime") or tags.get("inference_engine"),
                    "stage": v.get("stage"),
                    "version": v.get("version"),
                },
            })
    return {"object": "list", "data": data}


@router.post("/chat/completions", summary="Chat Completions（LLM 與 VLM 模型通用）")
def chat_completions(request: ChatCompletionRequest):
    """OpenAI 相容 chat 端點。

    - **text-generation 模型**：messages 直接轉交 runtime（Ollama 走 /api/chat，
      HF 走 tokenizer chat template）。
    - **vlm 模型**：最後一則 user 訊息中的 `image_url` part 作為影像輸入、
      text part 作為 prompt（image_url 接受 http(s) URL / data: URI / base64 / 本機路徑）。
    - `keep_alive` 為 ollama 式擴充欄位，控制推理後模型的 GPU 存活時間。
    """
    spec = _describe(request.model, engine=request.engine)
    options = _build_options(request)

    if spec.task_family == "text-generation":
        data = {"messages": [_plain_message(m) for m in request.messages]}
        if request.tools:
            data["tools"] = request.tools
    elif spec.task_family in {"vlm", "image-captioning"}:
        image, prompt = _extract_image_and_prompt(request.messages)
        if image is None:
            raise _openai_error(400, f"model '{request.model}' is a {spec.task_family} model; "
                                     "include an image_url content part in the last user message")
        data = {"image": image, "prompt": prompt or "Describe this image."}
    else:
        raise _openai_error(
            400,
            f"model '{request.model}' is registered as task '{spec.task_family}', "
            f"which /v1/chat/completions does not serve. Use /inference/infer for this model.",
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # tools + real token-streaming together isn't implemented (would need
    # incremental tool_call.arguments deltas, which Ollama's own streaming
    # protocol doesn't hand back in an easily-forwardable shape) -- skip the
    # real-stream attempt when tools are present so this always goes through
    # _infer() below, which correctly returns tool_calls either way. A
    # request with stream=True and tools still gets a response (pseudo-stream
    # single chunk further down), just not real token-by-token streaming.
    if request.stream and not request.tools:
        stream_gen = _try_stream(request.model, data, options, engine=request.engine)
        if stream_gen is not None:  # 真·逐 token 串流（text-generation on ollama / hf）
            return _sse_chat_real_stream(completion_id, created, request.model, stream_gen)

    result = _infer(request.model, data, options, engine=request.engine)
    content = _content_of(result)
    usage = _usage_of(result)
    tool_calls = _tool_calls_of(result)

    if request.stream:  # 該 task/runtime 不支援串流，或帶了 tools → 偽串流（單 chunk）維持相容
        return _sse_chat_stream(completion_id, created, request.model, content, usage, tool_calls=tool_calls)

    message: Dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


@router.post("/completions", summary="Completions（傳統 prompt 形式）")
def completions(request: CompletionRequest):
    """OpenAI 相容 completion 端點；只支援 text-generation 模型。"""
    spec = _describe(request.model)
    if spec.task_family != "text-generation":
        raise _openai_error(
            400,
            f"model '{request.model}' is registered as task '{spec.task_family}'; "
            f"/v1/completions only serves text-generation models.",
        )
    prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
    options = _build_options(request)
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if request.stream:
        stream_gen = _try_stream(request.model, {"inputs": prompt}, options)
        if stream_gen is not None:
            return _sse_text_real_stream(completion_id, created, request.model, stream_gen)

    result = _infer(request.model, {"inputs": prompt}, options)
    content = _content_of(result)
    usage = _usage_of(result)

    if request.stream:
        return _sse_text_stream(completion_id, created, request.model, content, usage)

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "text": content, "finish_reason": "stop", "logprobs": None}],
        "usage": usage,
    }


@router.post("/audio/transcriptions", summary="語音轉文字（Whisper 式）")
def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
):
    """OpenAI 相容 `/v1/audio/transcriptions`；`model` 需為已註冊的 asr 模型。"""
    spec = _describe(model)
    if spec.task_family != "asr":
        raise _openai_error(
            400,
            f"model '{model}' is registered as task '{spec.task_family}'; "
            f"/v1/audio/transcriptions only serves asr models.",
        )

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        audio_path = tmp.name

    options: Dict[str, Any] = {}
    if language:
        options["language"] = language
    try:
        result = _infer(model, {"audio": audio_path}, options)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    text = result["result"].get("text") or result["result"].get("response", "")
    if response_format == "text":
        return text
    return {"text": text}


@router.post("/audio/speech", summary="文字轉語音（OpenAI TTS 式）")
def audio_speech(request: SpeechRequest):
    """OpenAI 相容 `/v1/audio/speech`；`model` 需為已註冊的 tts 模型。回傳二進位音訊。"""
    import base64

    from fastapi.responses import Response

    spec = _describe(request.model)
    if spec.task_family != "tts":
        raise _openai_error(
            400,
            f"model '{request.model}' is registered as task '{spec.task_family}'; "
            f"/v1/audio/speech only serves tts models.",
        )
    if request.response_format not in {"wav"}:
        raise _openai_error(400, f"response_format '{request.response_format}' not supported yet (only 'wav')")

    options: Dict[str, Any] = {}
    if request.voice is not None:
        options["voice"] = request.voice
    if request.speed is not None:
        options["speed"] = request.speed
    if request.keep_alive is not None:
        options["keep_alive"] = request.keep_alive

    result = _infer(request.model, {"text": request.input}, options)
    r = result.get("result", {}) or {}
    audio_b64 = r.get("audio_base64")
    if not audio_b64:
        raise _openai_error(500, "tts model returned no audio", err_type="server_error")
    return Response(content=base64.b64decode(audio_b64), media_type="audio/wav")


@router.post("/embeddings", summary="文本向量化（OpenAI Embeddings 式）")
def embeddings(request: EmbeddingsRequest):
    """OpenAI 相容 `/v1/embeddings`；`model` 需為已註冊的 embedding 模型（如 bge-m3）。"""
    spec = _describe(request.model)
    if spec.task_family != "embedding":
        raise _openai_error(
            400,
            f"model '{request.model}' is registered as task '{spec.task_family}'; "
            f"/v1/embeddings only serves embedding models.",
        )
    if request.encoding_format != "float":
        raise _openai_error(400, f"encoding_format '{request.encoding_format}' not supported (only 'float')")

    options: Dict[str, Any] = {}
    if request.keep_alive is not None:
        options["keep_alive"] = request.keep_alive

    result = _infer(request.model, {"inputs": request.input}, options)
    r = result.get("result", {}) or {}
    vectors = r.get("embeddings", [])
    meta = r.get("metadata", {}) or {}
    prompt_tokens = int(meta.get("prompt_eval_count", 0))
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vec, "index": i}
            for i, vec in enumerate(vectors)
        ],
        "model": request.model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


@router.post("/rerank", summary="文檔重排（xinference/jina 式）")
def rerank(request: RerankRequest):
    """xinference/jina 相容 `/v1/rerank`；`model` 需為已註冊的 rerank 模型（如 bge-reranker-v2-m3）。"""
    spec = _describe(request.model)
    if spec.task_family != "rerank":
        raise _openai_error(
            400,
            f"model '{request.model}' is registered as task '{spec.task_family}'; "
            f"/v1/rerank only serves rerank models.",
        )

    options: Dict[str, Any] = {"return_documents": request.return_documents}
    if request.top_n is not None:
        options["top_n"] = request.top_n
    if request.keep_alive is not None:
        options["keep_alive"] = request.keep_alive

    result = _infer(request.model, {"query": request.query, "documents": request.documents}, options)
    r = result.get("result", {}) or {}
    results = r.get("results", [])
    if request.return_documents:
        for item in results:
            if "document" in item:
                item["document"] = {"text": item["document"]}  # jina 式巢狀格式
    return {
        "id": f"rerank-{uuid.uuid4().hex[:24]}",
        "model": request.model,
        "results": results,
        "usage": {"total_tokens": 0},
    }


# ===== 內部 helpers =====


def _describe(model_name: str, engine: Optional[str] = None):
    try:
        return inference_service.describe_model(model_name, engine=engine)
    except ModelNotFoundError as e:
        raise _openai_error(404, str(e), err_type="invalid_request_error", code="model_not_found")
    except (ValidationError, UnsupportedTaskError) as e:
        raise _openai_error(400, str(e))


def _infer(model_name: str, data: Dict[str, Any], options: Dict[str, Any],
          engine: Optional[str] = None) -> Dict[str, Any]:
    try:
        return inference_service.infer(task=None, model_name=model_name, data=data,
                                       options=options, engine=engine)
    except (ValidationError, UnsupportedTaskError) as e:
        raise _openai_error(400, str(e))
    except ModelNotFoundError as e:
        raise _openai_error(404, str(e), err_type="invalid_request_error", code="model_not_found")
    except ResourceExhaustedError as e:
        raise _openai_error(503, str(e), err_type="server_error", code="overloaded")
    except (ModelLoadError, InferenceExecutionError, EngineError, InferenceError) as e:
        logger.error(f"/v1 inference error: {e}", exc_info=True)
        raise _openai_error(500, str(e), err_type="server_error")


def _try_stream(model_name: str, data: Dict[str, Any], options: Dict[str, Any],
                engine: Optional[str] = None):
    """嘗試建立真串流 generator；該 task/runtime 不支援時回 None（呼叫端走偽串流）。

    驗證/載入/連線錯誤在這裡就會發生（headers 尚未送出，能回正常錯誤碼）。
    """
    try:
        return inference_service.infer_stream(task=None, model_name=model_name, data=data,
                                              options=options, engine=engine)
    except UnsupportedTaskError:
        return None
    except ValidationError as e:
        raise _openai_error(400, str(e))
    except ModelNotFoundError as e:
        raise _openai_error(404, str(e), err_type="invalid_request_error", code="model_not_found")
    except ResourceExhaustedError as e:
        raise _openai_error(503, str(e), err_type="server_error", code="overloaded")
    except (ModelLoadError, InferenceExecutionError, EngineError, InferenceError) as e:
        logger.error(f"/v1 stream setup error: {e}", exc_info=True)
        raise _openai_error(500, str(e), err_type="server_error")


def _openai_error(status_code: int, message: str, err_type: str = "invalid_request_error",
                  code: Optional[str] = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": err_type, "code": code}},
    )


def _build_options(request) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["max_new_tokens"] = request.max_tokens  # HF 名稱；Ollama 端會映射成 num_predict
    if request.stop is not None:
        options["stop"] = request.stop
    if request.keep_alive is not None:
        options["keep_alive"] = request.keep_alive
    # num_ctx/think are ChatCompletionRequest-only fields -- CompletionRequest
    # (the other caller of this helper) doesn't declare them, so use getattr
    # rather than a direct attribute access.
    num_ctx = getattr(request, "num_ctx", None)
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    think = getattr(request, "think", None)
    if think is not None:
        options["think"] = think
    return options


def _plain_message(m: ChatMessage) -> Dict[str, Any]:
    """OpenAI message → Ollama 形式的 message dict（多模態 parts 只保留文字）。

    tool_calls / tool_call_id / name 是 ChatMessage 的 extra="allow" 欄位
    (不在 schema 裡明確定義)，只有在多輪 tool-calling 對話中，client 把
    "assistant 呼叫了工具" 跟 "工具回傳結果" 這兩種訊息重新送回來當歷史
    時才會出現 -- 沒有這幾行，這些欄位會被靜默丟掉，模型收到的訊息序列會
    缺少呼叫過的工具跟其結果，等於忘記自己剛才做了什麼。
    """
    content = m.content
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    out: Dict[str, Any] = {"role": m.role, "content": content or ""}
    extra = m.model_extra or {}
    if extra.get("tool_calls"):
        out["tool_calls"] = extra["tool_calls"]
    if extra.get("tool_call_id"):
        out["tool_call_id"] = extra["tool_call_id"]
    if extra.get("name"):
        out["name"] = extra["name"]
    return out


def _extract_image_and_prompt(messages: List[ChatMessage]):
    """從最後一則 user 訊息取出 (image, prompt)；image 可為 URL / data URI / base64 / 路徑。"""
    image = None
    prompt_parts: List[str] = []
    for m in reversed(messages):
        if m.role != "user":
            continue
        if isinstance(m.content, str):
            prompt_parts.append(m.content)
        elif isinstance(m.content, list):
            for p in m.content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    prompt_parts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    if url:
                        image = url
        break
    return image, " ".join(prompt_parts).strip()


def _content_of(result: Dict[str, Any]) -> str:
    r = result.get("result", {}) or {}
    return r.get("response") or r.get("text") or ""


def _tool_calls_of(result: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    r = result.get("result", {}) or {}
    return r.get("tool_calls") or None


def _usage_of(result: Dict[str, Any]) -> Dict[str, int]:
    meta = (result.get("result", {}) or {}).get("metadata", {}) or {}
    prompt_tokens = int(meta.get("prompt_eval_count") or meta.get("input_length") or 0)
    completion_tokens = int(meta.get("eval_count") or meta.get("output_length") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _to_epoch(ts: Optional[str]) -> int:
    if not ts:
        return 0
    try:
        return int(time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")))
    except (ValueError, OverflowError):
        return 0


# ===== SSE（真串流：逐 token chunk）=====


def _sse_chat_real_stream(completion_id: str, created: int, model: str, stream_gen):
    def gen():
        base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model}
        yield "data: " + json.dumps({
            **base,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }, ensure_ascii=False) + "\n\n"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            for piece in stream_gen:
                if isinstance(piece, dict):  # 最後的 metadata sentinel
                    usage = _usage_of({"result": piece})
                    continue
                yield "data: " + json.dumps({
                    **base,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }, ensure_ascii=False) + "\n\n"
        except Exception as e:  # headers 已送出，只能以 SSE error 事件收尾
            logger.error(f"/v1 mid-stream error: {e}", exc_info=True)
            yield "data: " + json.dumps(
                {"error": {"message": str(e), "type": "server_error", "code": None}},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        yield "data: " + json.dumps({
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        }, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse_text_real_stream(completion_id: str, created: int, model: str, stream_gen):
    def gen():
        base = {"id": completion_id, "object": "text_completion", "created": created, "model": model}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            for piece in stream_gen:
                if isinstance(piece, dict):
                    usage = _usage_of({"result": piece})
                    continue
                yield "data: " + json.dumps({
                    **base,
                    "choices": [{"index": 0, "text": piece, "finish_reason": None, "logprobs": None}],
                }, ensure_ascii=False) + "\n\n"
        except Exception as e:
            logger.error(f"/v1 mid-stream error: {e}", exc_info=True)
            yield "data: " + json.dumps(
                {"error": {"message": str(e), "type": "server_error", "code": None}},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        yield "data: " + json.dumps({
            **base,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop", "logprobs": None}],
            "usage": usage,
        }, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ===== SSE（偽串流：單 chunk + [DONE]；串流不支援的 task 用）=====


def _sse_chat_stream(completion_id: str, created: int, model: str, content: str, usage: Dict[str, int],
                     tool_calls: Optional[List[Dict[str, Any]]] = None):
    """單 chunk 偽串流。tool_calls 存在時整包放進同一個 delta（不是逐個 token/
    參數片段真的串流出去 -- Ollama 的串流協定不會逐步吐出 tool_call.arguments
    的片段，這裡老實用一個 chunk 表達完整結果，而不是假裝有更細的粒度）。"""
    def gen():
        head = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        delta: Dict[str, Any] = {"content": content}
        if tool_calls:
            delta = {"tool_calls": tool_calls}
        body = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        tail = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": usage,
        }
        for chunk in (head, body, tail):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse_text_stream(completion_id: str, created: int, model: str, content: str, usage: Dict[str, int]):
    def gen():
        body = {
            "id": completion_id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": content, "finish_reason": None, "logprobs": None}],
        }
        tail = {
            "id": completion_id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop", "logprobs": None}],
            "usage": usage,
        }
        yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
