# src/inference/adapters/ollama.py
"""
Ollama Runtime Adapter

模型由 Ollama daemon 管理，本 adapter 只負責：
    1. 把 spec.physical_path / ollama_model_name 對應到 daemon 上的真名
    2. 必要時自動 pull
    3. POST /api/generate（messages 形式時 /api/chat、embedding task 時 /api/embed）
       並把結果整理為統一格式
    4. 把 keep_alive 透傳給 daemon，讓 Ollama 自己的 GPU 釋放節奏與本服務一致

共用生成參數（已由 service 正規化）：max_tokens → num_predict、stop → stop（list）、
temperature 原樣（0 即 greedy）。usage 採 daemon 回傳的 prompt_eval_count / eval_count。
結構化輸出：service 驗證過的 options["format"]（"json" 或 JSON Schema）放進 payload 最外層的
`format`，由 daemon 約束生成；輸出是否合格由 service 檢查（structured.py）。

錯誤分類：逾時 → InferenceTimeoutError(504)、無法連線 → RuntimeUnavailableError(503)、
daemon 回報記憶體不足 → ResourceExhaustedError(503)、其餘非 200 → EngineError(500)。

多個 registry 版本可能指向同一個 daemon tag：明確卸載某版本時，若其他快取 entry
仍有請求在用同一個 daemon 模型，不請 daemon 釋放。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from ..exceptions import (
    EngineError,
    InferenceTimeoutError,
    ModelLoadError,
    ModelNotFoundError,
    ResourceExhaustedError,
    RuntimeUnavailableError,
)
from ..options import StopStringFilter, truncate_at_stop
from ..spec import RUNTIME_OLLAMA, ModelSpec
from .base import BaseAdapter, CacheEntry, parse_keep_alive

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseAdapter):
    runtime = RUNTIME_OLLAMA

    def __init__(self, config: dict[str, Any] | None = None):
        # 本地 cache 只存「daemon 上的模型名」字串，幾乎無成本 → 預設不做 idle 回收；
        # GPU 釋放交給 daemon 的 keep_alive 機制。
        super().__init__(config=config, max_cached_models=64, default_idle_timeout=-1)
        self.base_url = (
            self.config.get("base_url")
            or self.config.get("api_base")
            or "http://localhost:11434"
        )
        self.timeout = int(self.config.get("timeout", 300))
        self.auto_pull = bool(self.config.get("auto_pull", True))
        # daemon 端模型存活時間預設值（未逐請求指定 keep_alive 時使用）
        self.default_keep_alive: float | None = parse_keep_alive(self.config.get("keep_alive"))
        logger.info(
            f"OllamaAdapter ready — base_url={self.base_url} timeout={self.timeout} "
            f"keep_alive={self.default_keep_alive}"
        )

    # ----- 由 BaseAdapter.run() 呼叫 -----

    def load_model(self, spec: ModelSpec) -> str:
        """對 Ollama 而言，"load" = 確認 daemon 上有這個模型，回傳 daemon 上的名字。"""
        target = spec.ollama_model_name or spec.physical_path
        if not target:
            raise ModelLoadError(
                f"Ollama spec for '{spec.model_name}' missing both ollama_model_name and physical_path"
            )

        if not self._is_available(target):
            if self.auto_pull:
                self._pull(target)
            else:
                raise ModelNotFoundError(
                    f"Ollama model '{target}' not on daemon, and auto_pull is disabled"
                )
        return target

    def infer(self, model: str, spec: ModelSpec, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        if spec.task_family == "embedding":
            return self._embed(model, data, options)
        if "messages" in data:
            return self._chat(model, data["messages"], options, tools=data.get("tools"),
                              limit=spec.max_output_tokens)
        if "inputs" not in data:
            raise EngineError("Ollama generate requires data['inputs'] (prompt string) or data['messages']")

        payload = {
            "model": model,
            "prompt": data["inputs"],
            "stream": False,
            # think=False by default: gemma4-family models emit a hidden
            # reasoning trace and leave "response" empty unless this is set —
            # it's a top-level Ollama API field, not an "options" sub-field.
            "think": options.get("think", False),
            "options": _to_ollama_options(options, spec.max_output_tokens),
        }
        _apply_format(payload, options)
        self._apply_keep_alive(payload, options)
        body = self._post("/api/generate", payload)
        text, _ = truncate_at_stop(body.get("response", ""), options.get("stop") or [])
        return {
            "response": text,
            "model": model,
            "metadata": _metadata(body),
        }

    def _chat(self, model: str, messages: list, options: dict[str, Any],
             tools: list | None = None, limit: int | None = None) -> dict[str, Any]:
        """OpenAI 形式的 messages → Ollama /api/chat（daemon 會套模型自己的 chat template）。

        tools: OpenAI 格式的 function-calling 定義列表（[{"type":"function",
        "function":{"name":...,"description":...,"parameters":{...}}}, ...]）。
        Ollama 原生 /api/chat 直接接受這個格式，最外層的 "tools" 欄位，不是
        options 的子欄位 -- 不透過 _to_ollama_options()（那是白名單過濾器，
        且是給推理參數用的，tools 是結構完全不同的東西）。只有真的有工具
        時才加這個欄位，避免對不支援 tools 的模型送出多餘欄位。
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": options.get("think", False),
            "options": _to_ollama_options(options, limit),
        }
        if tools:
            payload["tools"] = tools
        _apply_format(payload, options)
        self._apply_keep_alive(payload, options)
        body = self._post("/api/chat", payload)
        message = body.get("message") or {}
        text, _ = truncate_at_stop(message.get("content", ""), options.get("stop") or [])
        result = {
            "response": text,
            "model": model,
            "metadata": _metadata(body),
        }
        # Ollama returns tool_calls on message.tool_calls when the model
        # decides to call one, in an OpenAI-ish shape (function.name /
        # function.arguments) but with one real deviation confirmed live:
        # function.arguments comes back as an already-parsed JSON object,
        # not the JSON-encoded *string* the real OpenAI API contract (and
        # every OpenAI-compatible client, including LangChain's tool-call
        # parser) expects -- re-serialize it here so callers one layer up
        # don't each have to know about this quirk.
        if message.get("tool_calls"):
            tool_calls = []
            for i, tc in enumerate(message["tool_calls"]):
                fn = tc.get("function", {}) or {}
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append({
                    "id": tc.get("id") or f"call_{model}_{i}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
            result["tool_calls"] = tool_calls
        return result

    def _embed(self, model: str, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        """向量化：daemon 的 /api/embed（接受 str 或 list[str]，一次批次完成）。

        回傳格式對齊 HF EmbeddingHandler（embeddings + metadata.dim/count），
        讓呼叫端不需分辨 runtime。daemon 已對向量做 L2 normalize。
        """
        texts = data.get("inputs", data.get("input"))
        if texts is None:
            raise EngineError("Ollama embedding requires data['inputs'] (str or list[str])")
        payload = {"model": model, "input": texts}
        self._apply_keep_alive(payload, options)
        body = self._post("/api/embed", payload)
        vectors = body.get("embeddings", [])
        return {
            "embeddings": vectors,
            "model": model,
            "metadata": {
                "dim": len(vectors[0]) if vectors else 0,
                "count": len(vectors),
                "normalized": True,
                "prompt_eval_count": body.get("prompt_eval_count", 0),
                "total_duration": body.get("total_duration", 0),
                "load_duration": body.get("load_duration", 0),
            },
        }

    # ----- 串流 -----

    def infer_stream(self, model: str, spec: ModelSpec, data: dict[str, Any], options: dict[str, Any]):
        """逐 token 串流：daemon 的 /api/chat 或 /api/generate（stream=true，JSONL）。

        依 base 合約：本方法在呼叫時就建立連線（錯誤在這裡拋，HTTP 層還能回錯誤碼），
        回傳的 generator 逐段 yield 文字增量，最後 yield {"metadata": ...}。
        """
        if "messages" in data:
            path = "/api/chat"
            payload: dict[str, Any] = {"model": model, "messages": data["messages"]}
        elif "inputs" in data:
            path = "/api/generate"
            payload = {"model": model, "prompt": data["inputs"]}
        else:
            raise EngineError("Ollama stream requires data['inputs'] or data['messages']")
        payload["stream"] = True
        payload["think"] = options.get("think", False)
        payload["options"] = _to_ollama_options(options, spec.max_output_tokens)
        self._apply_keep_alive(payload, options)

        try:
            r = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout, stream=True)
        except requests.RequestException as e:
            raise self._request_error(e) from e
        if r.status_code != 200:
            body = r.text
            r.close()
            raise _status_error(path, r.status_code, body)

        # daemon 本身會套用 stop；此過濾器是保險 — 保證停止字串（含跨 chunk 的前綴）不外洩
        stop_filter = StopStringFilter(options.get("stop"))

        def gen():
            try:
                for line in r.iter_lines():
                    if not line:
                        continue
                    body = json.loads(line)
                    if body.get("error"):
                        raise _message_error(f"Ollama stream error: {body['error']}", str(body["error"]))
                    delta = (
                        (body.get("message") or {}).get("content", "")
                        if path == "/api/chat" else body.get("response", "")
                    )
                    out = stop_filter.feed(delta) if delta else ""
                    if out:
                        yield out
                    if body.get("done"):
                        tail = stop_filter.flush()
                        if tail:
                            yield tail
                        yield {"metadata": _metadata(body)}
                        return
                raise EngineError("Ollama stream ended without a final 'done' message")
            except requests.RequestException as e:
                raise self._request_error(e) from e
            finally:
                r.close()

        return gen()

    # ----- 卸載：請 daemon 立刻釋放該模型的 GPU 記憶體 -----

    def _on_explicit_unload(self, entry: CacheEntry) -> None:
        """明確卸載後請 daemon 釋放；其他版本的 entry 仍在用同一個 daemon 模型時跳過。"""
        daemon_name = entry.model
        busy = [e for e in self._entries_using(daemon_name) if e is not entry and e.in_flight > 0]
        if busy:
            logger.info(
                f"not releasing '{daemon_name}' on daemon: still in use by "
                f"{[f'{e.model_name}@{e.version}' for e in busy]}"
            )
            return
        self._release_on_daemon(daemon_name)

    def _release_on_daemon(self, daemon_name: str) -> None:
        """keep_alive=0 的空 generate 會讓 daemon 立即卸載該模型（官方建議做法）。"""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": daemon_name, "keep_alive": 0},
                timeout=30,
            )
            logger.info(f"asked Ollama daemon to release '{daemon_name}'")
        except requests.RequestException as e:
            logger.warning(f"failed to ask daemon to release '{daemon_name}': {e}")

    # ----- helpers -----

    def _apply_keep_alive(self, payload: dict[str, Any], options: dict[str, Any]) -> None:
        ka = parse_keep_alive(options.get("keep_alive"))
        if ka is None:
            ka = self.default_keep_alive
        if ka is not None:
            # Ollama 接受數字（秒）；負值 = 常駐
            payload["keep_alive"] = int(ka) if float(ka).is_integer() else ka

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise self._request_error(e) from e
        if r.status_code != 200:
            raise _status_error(path, r.status_code, r.text)
        return r.json()

    def _request_error(self, e: requests.RequestException) -> Exception:
        """requests 例外 → 可辨識的推論錯誤（等待回應逾時 504 / 無法連線（含連線逾時）503）。"""
        if isinstance(e, requests.ConnectTimeout):
            return RuntimeUnavailableError(f"Ollama daemon unreachable ({self.base_url}): {e}")
        if isinstance(e, requests.Timeout) or "read timed out" in str(e).lower():
            return InferenceTimeoutError(
                f"Ollama daemon ({self.base_url}) did not respond within {self.timeout}s: {e}"
            )
        return RuntimeUnavailableError(f"Ollama daemon unreachable ({self.base_url}): {e}")

    def _is_available(self, name: str) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        except requests.RequestException as e:
            raise self._request_error(e) from e
        if r.status_code != 200:
            logger.warning(f"Ollama /api/tags returned {r.status_code}: {r.text}")
            return False
        names = [m["name"] for m in r.json().get("models", [])]
        return name in names or any(name in n for n in names)

    def get_context_length(self, name: str) -> int | None:
        """The model's native context window, straight from the daemon's own metadata
        (``POST /api/show``'s ``model_info['<arch>.context_length']`` -- the key's arch
        prefix varies per model family, e.g. ``qwen3.context_length``, so match by
        suffix). Best-effort: any failure (daemon unreachable, model not pulled yet,
        unexpected shape) returns None rather than raising -- this is only ever used to
        auto-fill a limit that was otherwise going to be None anyway.
        """
        try:
            r = requests.post(f"{self.base_url}/api/show", json={"name": name}, timeout=10)
            if r.status_code != 200:
                return None
            model_info = r.json().get("model_info") or {}
            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                    return value
        except (requests.RequestException, ValueError):
            pass
        return None

    def _pull(self, name: str) -> None:
        logger.info(f"pulling Ollama model: {name}")
        try:
            r = requests.post(f"{self.base_url}/api/pull", json={"name": name}, timeout=1200)
        except requests.ConnectionError as e:
            raise RuntimeUnavailableError(f"Ollama daemon unreachable ({self.base_url}) while pulling '{name}': {e}") from e
        except requests.RequestException as e:
            raise ModelLoadError(f"failed to pull '{name}' from Ollama: {e}") from e
        if r.status_code != 200:
            raise ModelLoadError(f"failed to pull '{name}': {r.status_code} {r.text}")


def _apply_format(payload: dict[str, Any], options: dict[str, Any]) -> None:
    """結構化輸出的 format 是 Ollama API 最外層欄位（不是 options 子欄位）；沒要求時不送。"""
    if options.get("format") is not None:
        payload["format"] = options["format"]


_OOM_MARKERS = ("out of memory", "requires more system memory", "insufficient memory", "cudaMalloc failed")


def _message_error(message: str, raw: str) -> Exception:
    if any(m.lower() in raw.lower() for m in _OOM_MARKERS):
        return ResourceExhaustedError(message)
    return EngineError(message)


def _status_error(path: str, status_code: int, body: str) -> Exception:
    message = f"Ollama API {path} {status_code}: {body}"
    if status_code == 404:
        return ModelLoadError(message)
    if status_code in (408, 504):
        return InferenceTimeoutError(message)
    if status_code == 503 and not any(m.lower() in body.lower() for m in _OOM_MARKERS):
        return RuntimeUnavailableError(message)
    return _message_error(message, body)


def _metadata(body: dict[str, Any]) -> dict[str, Any]:
    """保留既有欄位（缺值填 0 以維持相容）；prompt_tokens / completion_tokens
    只在 daemon 真的回報計數時才填，供統一 usage 使用（缺值不偽造）。"""
    meta = {
        "total_duration": body.get("total_duration", 0),
        "load_duration": body.get("load_duration", 0),
        "prompt_eval_count": body.get("prompt_eval_count", 0),
        "eval_count": body.get("eval_count", 0),
    }
    if body.get("done_reason"):
        meta["done_reason"] = body["done_reason"]
    if isinstance(body.get("prompt_eval_count"), int) and isinstance(body.get("eval_count"), int):
        meta["prompt_tokens"] = body["prompt_eval_count"]
        meta["completion_tokens"] = body["eval_count"]
    return meta


# ----- options 映射 -----

# 正規化後 text-generation 只會出現 max_tokens / stop；max_new_tokens / stop_sequences
# 保留給未經正規化的其他路徑（例如 embedding 以外的舊呼叫端）。
_OLLAMA_OPT_MAP = {
    "max_length": "num_predict",
    "max_tokens": "num_predict",
    "max_new_tokens": "num_predict",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "repeat_penalty": "repeat_penalty",
    "stop": "stop",
    "stop_sequences": "stop",
}

_OLLAMA_PASSTHROUGH = {
    "mirostat", "mirostat_eta", "mirostat_tau",
    "num_ctx", "num_batch", "num_gpu", "main_gpu", "low_vram",
    "f16_kv", "use_mmap", "use_mlock", "num_thread",
}


def _to_ollama_options(options: dict[str, Any], max_output_tokens: int | None = None) -> dict[str, Any]:
    """options → Ollama options。max_output_tokens：模型已知上限，呼叫端沒指定長度時作為 num_predict。"""
    out: dict[str, Any] = {}
    for k, v in options.items():
        if v is None:
            continue
        if k in _OLLAMA_OPT_MAP:
            if k in ("stop", "stop_sequences") and isinstance(v, str):
                v = [v]
            out[_OLLAMA_OPT_MAP[k]] = v
        elif k in _OLLAMA_PASSTHROUGH:
            out[k] = v
    if "num_predict" not in out and max_output_tokens is not None:
        out["num_predict"] = max_output_tokens
    return out
