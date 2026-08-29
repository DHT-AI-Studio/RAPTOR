# src/inference/adapters/ollama.py
"""
Ollama Runtime Adapter

模型由 Ollama daemon 管理，本 adapter 只負責：
    1. 把 spec.physical_path / ollama_model_name 對應到 daemon 上的真名
    2. 必要時自動 pull
    3. POST /api/generate（messages 形式時 /api/chat、embedding task 時 /api/embed）
       並把結果整理為統一格式
    4. 把 keep_alive 透傳給 daemon，讓 Ollama 自己的 GPU 釋放節奏與本服務一致
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import requests

from ..exceptions import EngineError, ModelLoadError, ModelNotFoundError
from ..spec import ModelSpec, RUNTIME_OLLAMA
from .base import BaseAdapter, parse_keep_alive

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseAdapter):
    runtime = RUNTIME_OLLAMA

    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        self.default_keep_alive: Optional[float] = parse_keep_alive(self.config.get("keep_alive"))
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

    def infer(self, model: str, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        if spec.task_family == "embedding":
            return self._embed(model, data, options)
        if "messages" in data:
            return self._chat(model, data["messages"], options, tools=data.get("tools"))
        if "inputs" not in data:
            raise EngineError("Ollama generate requires data['inputs'] (prompt string) or data['messages']")

        payload = {
            "model": model,
            "prompt": data["inputs"],
            "stream": False,
            # think=False by default: qwen3.x-family models emit a hidden
            # reasoning trace and leave "response" empty unless this is set —
            # it's a top-level Ollama API field, not an "options" sub-field.
            "think": options.get("think", False),
            "options": _to_ollama_options(options),
        }
        self._apply_keep_alive(payload, options)
        body = self._post("/api/generate", payload)
        return {
            "response": body.get("response", ""),
            "model": model,
            "metadata": _metadata(body),
        }

    def _chat(self, model: str, messages: list, options: Dict[str, Any],
             tools: Optional[list] = None) -> Dict[str, Any]:
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
            "options": _to_ollama_options(options),
        }
        if tools:
            payload["tools"] = tools
        self._apply_keep_alive(payload, options)
        body = self._post("/api/chat", payload)
        message = body.get("message") or {}
        result = {
            "response": message.get("content", ""),
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

    def _embed(self, model: str, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
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

    def infer_stream(self, model: str, spec: ModelSpec, data: Dict[str, Any], options: Dict[str, Any]):
        """逐 token 串流：daemon 的 /api/chat 或 /api/generate（stream=true，JSONL）。

        依 base 合約：本方法在呼叫時就建立連線（錯誤在這裡拋，HTTP 層還能回錯誤碼），
        回傳的 generator 逐段 yield 文字增量，最後 yield {"metadata": ...}。
        """
        if "messages" in data:
            path = "/api/chat"
            payload: Dict[str, Any] = {"model": model, "messages": data["messages"]}
        elif "inputs" in data:
            path = "/api/generate"
            payload = {"model": model, "prompt": data["inputs"]}
        else:
            raise EngineError("Ollama stream requires data['inputs'] or data['messages']")
        payload["stream"] = True
        payload["think"] = options.get("think", False)
        payload["options"] = _to_ollama_options(options)
        self._apply_keep_alive(payload, options)

        try:
            r = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout, stream=True)
        except requests.RequestException as e:
            raise EngineError(f"Ollama daemon unreachable ({self.base_url}): {e}") from e
        if r.status_code != 200:
            body = r.text
            r.close()
            raise EngineError(f"Ollama API {path} {r.status_code}: {body}")

        def gen():
            try:
                for line in r.iter_lines():
                    if not line:
                        continue
                    body = json.loads(line)
                    if body.get("error"):
                        raise EngineError(f"Ollama stream error: {body['error']}")
                    delta = (
                        (body.get("message") or {}).get("content", "")
                        if path == "/api/chat" else body.get("response", "")
                    )
                    if delta:
                        yield delta
                    if body.get("done"):
                        yield {"metadata": _metadata(body)}
                        return
            finally:
                r.close()

        return gen()

    # ----- 卸載：請 daemon 立刻釋放該模型的 GPU 記憶體 -----

    def unload(self, model_name: str) -> bool:
        # 先查出 daemon 上的真名（cache entry 的 model 就是它），再從本地快取移除
        with self._cache_lock:
            entry = self._cache.get(model_name)
            daemon_name = entry.model if entry else None
        removed = super().unload(model_name)
        if daemon_name:
            self._release_on_daemon(daemon_name)
        return removed

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

    def _apply_keep_alive(self, payload: Dict[str, Any], options: Dict[str, Any]) -> None:
        ka = parse_keep_alive(options.get("keep_alive"))
        if ka is None:
            ka = self.default_keep_alive
        if ka is not None:
            # Ollama 接受數字（秒）；負值 = 常駐
            payload["keep_alive"] = int(ka) if float(ka).is_integer() else ka

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise EngineError(f"Ollama daemon unreachable ({self.base_url}): {e}") from e
        if r.status_code != 200:
            raise EngineError(f"Ollama API {path} {r.status_code}: {r.text}")
        return r.json()

    def _is_available(self, name: str) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if r.status_code != 200:
                return False
            names = [m["name"] for m in r.json().get("models", [])]
            return name in names or any(name in n for n in names)
        except requests.RequestException as e:
            logger.warning(f"Ollama /api/tags failed: {e}")
            return False

    def _pull(self, name: str) -> None:
        logger.info(f"pulling Ollama model: {name}")
        try:
            r = requests.post(f"{self.base_url}/api/pull", json={"name": name}, timeout=1200)
        except requests.RequestException as e:
            raise ModelLoadError(f"failed to pull '{name}' from Ollama: {e}") from e
        if r.status_code != 200:
            raise ModelLoadError(f"failed to pull '{name}': {r.status_code} {r.text}")


def _metadata(body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_duration": body.get("total_duration", 0),
        "load_duration": body.get("load_duration", 0),
        "prompt_eval_count": body.get("prompt_eval_count", 0),
        "eval_count": body.get("eval_count", 0),
    }


# ----- options 映射 -----

_OLLAMA_OPT_MAP = {
    "max_length": "num_predict",
    "max_tokens": "num_predict",
    "max_new_tokens": "num_predict",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "repeat_penalty": "repeat_penalty",
    "stop": "stop",
}

_OLLAMA_PASSTHROUGH = {
    "mirostat", "mirostat_eta", "mirostat_tau",
    "num_ctx", "num_batch", "num_gpu", "main_gpu", "low_vram",
    "f16_kv", "use_mmap", "use_mlock", "num_thread",
}


def _to_ollama_options(options: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in options.items():
        if k in _OLLAMA_OPT_MAP:
            out[_OLLAMA_OPT_MAP[k]] = v
        elif k in _OLLAMA_PASSTHROUGH:
            out[k] = v
    return out
