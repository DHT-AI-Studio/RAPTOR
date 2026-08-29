# src/inference/handlers/embedding.py
"""
Embedding handler — 文本向量化（bge-m3 / e5 / gte 等 encoder 模型）。

註冊方式（model+processor 路線）：
    model_class     = "AutoModel"
    processor_class = "AutoTokenizer"
    task            = "embedding"

data:
    inputs: str | list[str]        （別名 input 亦可 — 對齊 OpenAI /v1/embeddings）
options:
    pooling:   "cls"（預設，BGE 系列用 CLS） | "mean"
    normalize: bool（預設 True，L2 normalize）
    max_length: int（預設 512）
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseHandler


class EmbeddingHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        texts = data.get("inputs", data.get("input"))
        if texts is None:
            raise ValueError("embedding handler requires data['inputs'] (str or list[str])")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("data['inputs'] must be str or list[str]")

        model = loaded["model"]
        tokenizer = loaded["processor"]
        device = loaded.get("device", "cpu")

        pooling = options.get("pooling", "cls")
        normalize = bool(options.get("normalize", True))
        max_length = int(options.get("max_length", 512))
        batch_size = int(options.get("batch_size", 32))

        vectors: List[List[float]] = []
        total_tokens = 0
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                total_tokens += int(enc["attention_mask"].sum())
                if device == "cuda":
                    enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                hidden = out.last_hidden_state  # (B, T, H)
                if pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                else:  # cls
                    emb = hidden[:, 0]
                if normalize:
                    emb = F.normalize(emb, p=2, dim=1)
                vectors.extend(emb.float().cpu().tolist())

        return {
            "embeddings": vectors,
            "metadata": {
                "dim": len(vectors[0]) if vectors else 0,
                "count": len(vectors),
                "pooling": pooling,
                "normalized": normalize,
                "prompt_eval_count": total_tokens,
            },
        }
