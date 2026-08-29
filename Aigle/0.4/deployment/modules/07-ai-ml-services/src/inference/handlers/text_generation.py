# src/inference/handlers/text_generation.py
"""文本生成 handler — 對應 raw AutoModelForCausalLM + AutoTokenizer 路線。"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .base import BaseHandler

logger = logging.getLogger(__name__)


class TextGenerationHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if "inputs" not in data and "messages" not in data:
            raise ValueError("text-generation handler requires data['inputs'] or data['messages']")

        model = loaded["model"]
        tokenizer = loaded["processor"]
        device = loaded.get("device", "cpu")

        prompt = data["inputs"] if "inputs" in data else _render_messages(tokenizer, data["messages"])
        encoded = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            encoded = {k: v.to(device) for k, v in encoded.items()}

        gen_kwargs = _gen_kwargs(options)
        with torch.no_grad():
            outputs = model.generate(**encoded, **gen_kwargs)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "response": text,
            "metadata": {
                "input_length": int(encoded["input_ids"].shape[-1]),
                "output_length": int(outputs.shape[-1]),
            },
        }


def _render_messages(tokenizer, messages) -> str:
    """OpenAI 式 messages → 模型的 prompt 字串。

    優先用 tokenizer 內建的 chat template（正確處理各家模型的特殊 token）；
    模型沒有 template 時退回簡單的 role-prefix 拼接。
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.warning(f"apply_chat_template failed ({e}); falling back to plain join")
        lines = [f"{m.get('role', 'user')}: {_text_of(m.get('content', ''))}" for m in messages]
        lines.append("assistant:")
        return "\n".join(lines)


def _text_of(content) -> str:
    """content 可能是字串或 OpenAI 的 parts 列表 — 只取文字部份。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def _gen_kwargs(options: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "max_length", "max_new_tokens", "temperature", "top_p", "top_k",
        "do_sample", "num_beams", "repetition_penalty", "length_penalty",
    )
    out = {k: options[k] for k in keys if k in options}
    out.setdefault("do_sample", True)
    if "max_new_tokens" not in out and "max_length" not in out:
        out["max_new_tokens"] = 512
    return out
