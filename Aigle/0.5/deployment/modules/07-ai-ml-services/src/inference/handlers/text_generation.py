# src/inference/handlers/text_generation.py
"""文本生成 handler — 對應 raw AutoModelForCausalLM + AutoTokenizer 路線。

本檔同時提供 HF text-generation 的共用 helper，raw handler、pipeline 路線
（沿用 pipeline 已載入的 model / tokenizer）與串流路徑都走同一套：

    encode_prompt()        messages / inputs → 實際送入模型的 token IDs
    generation_kwargs()    正規化後的 options → generate() kwargs（max_tokens、greedy ...）
    make_stopping_criteria() 停止字串（只比對生成部分、跨 token）+ 取消旗標 → 真的停止 generate
    generate_text()        非串流生成：只解碼新增 tokens、在停止字串處截斷、以 token IDs 計數 usage
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..options import is_greedy, truncate_at_stop
from .base import BaseHandler

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 512

# 直接轉交 generate() 的 HF 參數（max_tokens / stop 另行處理）
_HF_GEN_KEYS = (
    "max_length", "temperature", "top_p", "top_k", "do_sample", "num_beams",
    "repetition_penalty", "length_penalty", "early_stopping",
)


class TextGenerationHandler(BaseHandler):
    def run(self, loaded: dict[str, Any], spec, data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        if "inputs" not in data and "messages" not in data:
            raise ValueError("text-generation handler requires data['inputs'] or data['messages']")
        return generate_text(
            loaded["model"], loaded["processor"], data, options,
            max_output_tokens=getattr(spec, "max_output_tokens", None),
        )


# ===== 共用 helper =====


def generate_text(model, tokenizer, data: dict[str, Any], options: dict[str, Any],
                  max_output_tokens: int | None = None) -> dict[str, Any]:
    """非串流生成；回傳 {"response", "metadata"}，response 只含新生成且截斷在停止字串前的文字。"""
    import torch

    encoded = encode_prompt(tokenizer, data, options, input_device(model))
    prompt_len = int(encoded["input_ids"].shape[-1])
    gen_kwargs = generation_kwargs(options, max_output_tokens)
    stops = options.get("stop") or []
    criteria = make_stopping_criteria(tokenizer, stops, prompt_len)

    with torch.no_grad():
        outputs = model.generate(**encoded, stopping_criteria=criteria, **gen_kwargs)

    new_ids = outputs[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    text, stopped = truncate_at_stop(text, stops)
    completion_tokens = int(new_ids.shape[-1])
    return {
        "response": text,
        "metadata": {
            "input_length": prompt_len,
            "output_length": int(outputs.shape[-1]),  # 既有欄位：含 prompt 的總序列長度
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "finish_reason": _finish_reason(stopped, completion_tokens, gen_kwargs, prompt_len),
        },
    }


def encode_prompt(tokenizer, data: dict[str, Any], options: dict[str, Any], device=None) -> dict[str, Any]:
    """messages 走 chat template（模板已含特殊 token，不再重複加）；inputs 直接 tokenize。"""
    if "messages" in data:
        prompt, templated = render_messages(tokenizer, data["messages"], options)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=not templated)
    else:
        enc = tokenizer(data["inputs"], return_tensors="pt")
    enc = {k: v for k, v in dict(enc).items() if k in ("input_ids", "attention_mask")}
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def render_messages(tokenizer, messages, options: dict[str, Any] | None = None):
    """OpenAI 式 messages → (prompt 字串, 是否由 chat template 產生)。

    優先用 tokenizer 內建的 chat template（正確處理各家模型的特殊 token）；
    模型沒有 template 時退回簡單的 role-prefix 拼接。think 對齊 Ollama adapter
    的預設（False），以 enable_thinking 傳給支援的模板（如 gemma4），其他模板忽略。
    """
    think = bool((options or {}).get("think", False))
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
        return prompt, True
    except Exception as e:  # noqa: BLE001 -- tokenizer/jinja2 template rendering, varied failure modes; already logged, falls back
        logger.warning(f"apply_chat_template failed ({e}); falling back to plain join")
        lines = [f"{m.get('role', 'user')}: {_text_of(m.get('content', ''))}" for m in messages]
        lines.append("assistant:")
        return "\n".join(lines), False


# 舊名保留給既有 import
def _render_messages(tokenizer, messages) -> str:
    return render_messages(tokenizer, messages)[0]


def generation_kwargs(options: dict[str, Any], max_output_tokens: int | None = None) -> dict[str, Any]:
    """正規化後的 options → generate() kwargs。

    - max_tokens（或未正規化呼叫端的 max_new_tokens）→ max_new_tokens
    - max_length 保留 HF 語意（總序列長度）
    - greedy（temperature=0 或 do_sample=false）→ do_sample=False，且不傳 sampling 參數
    - 都沒指定長度時預設 512，模型有已知上限時取較小者
    """
    out = {k: options[k] for k in _HF_GEN_KEYS if options.get(k) is not None}
    if options.get("max_tokens") is not None:
        out["max_new_tokens"] = int(options["max_tokens"])
    elif options.get("max_new_tokens") is not None:
        out["max_new_tokens"] = int(options["max_new_tokens"])
    if is_greedy(options):
        out["do_sample"] = False
        for k in ("temperature", "top_p", "top_k"):
            out.pop(k, None)
    else:
        out.setdefault("do_sample", True)
    if "max_new_tokens" not in out and "max_length" not in out:
        out["max_new_tokens"] = (
            min(DEFAULT_MAX_NEW_TOKENS, max_output_tokens) if max_output_tokens else DEFAULT_MAX_NEW_TOKENS
        )
    return out


def make_stopping_criteria(tokenizer, stops: list[str], prompt_len: int,
                           cancel: threading.Event | None = None):
    """回傳 StoppingCriteriaList；沒有停止字串也沒有取消旗標時回 None。"""
    stops = [s for s in (stops or []) if s]
    if not stops and cancel is None:
        return None
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    # 一個字元最多 4 個 byte-level token；多留一點餘裕給被略過的特殊 token
    window = 4 * max((len(s) for s in stops), default=0) + 8

    class _GenerationControl(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            hit = cancel is not None and cancel.is_set()
            if not hit and stops:
                generated = input_ids[0, prompt_len:]
                if generated.shape[-1] > 0:
                    tail = tokenizer.decode(generated[-window:], skip_special_tokens=True)
                    hit = any(s in tail for s in stops)
            return torch.full((input_ids.shape[0],), hit, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_GenerationControl()])


def input_device(model):
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _finish_reason(stopped: bool, completion_tokens: int, gen_kwargs: dict[str, Any], prompt_len: int) -> str:
    if stopped:
        return "stop"
    limit = gen_kwargs.get("max_new_tokens")
    if limit is None and gen_kwargs.get("max_length") is not None:
        limit = gen_kwargs["max_length"] - prompt_len
    if limit is not None and completion_tokens >= limit:
        return "length"
    return "stop"


def _text_of(content) -> str:
    """content 可能是字串或 OpenAI 的 parts 列表 — 只取文字部份。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)
