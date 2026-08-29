# src/inference/handlers/rerank.py
"""
Rerank handler — cross-encoder 重排（bge-reranker-v2-m3 / ms-marco MiniLM 等）。

註冊方式（model+processor 路線）：
    model_class     = "AutoModelForSequenceClassification"
    processor_class = "AutoTokenizer"
    task            = "rerank"

data:
    query:     str
    documents: list[str]
options:
    top_n:      int（預設回傳全部）
    normalize:  bool（預設 False；True 時對分數做 sigmoid → 0~1）
    max_length: int（預設 512）
    return_documents: bool（預設 True）
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseHandler


class RerankHandler(BaseHandler):
    def run(self, loaded: Dict[str, Any], spec, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        query = data.get("query")
        documents = data.get("documents")
        if not isinstance(query, str) or not query:
            raise ValueError("rerank handler requires data['query'] (str)")
        if not isinstance(documents, list) or not documents:
            raise ValueError("rerank handler requires data['documents'] (non-empty list[str])")
        documents = [str(d) for d in documents]

        model = loaded["model"]
        tokenizer = loaded["processor"]
        device = loaded.get("device", "cpu")

        max_length = int(options.get("max_length", 512))
        batch_size = int(options.get("batch_size", 32))
        normalize = bool(options.get("normalize", False))
        return_documents = bool(options.get("return_documents", True))
        top_n = options.get("top_n")

        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                enc = tokenizer(
                    [query] * len(batch), batch,
                    padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                if device == "cuda":
                    enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.squeeze(-1)  # (B,) 單分數 cross-encoder
                if normalize:
                    logits = torch.sigmoid(logits)
                scores.extend(logits.float().cpu().tolist())

        ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        if top_n is not None:
            ranked = ranked[: int(top_n)]

        results = []
        for i in ranked:
            item: Dict[str, Any] = {"index": i, "relevance_score": scores[i]}
            if return_documents:
                item["document"] = documents[i]
            results.append(item)

        return {
            "results": results,
            "metadata": {"total_documents": len(documents), "returned": len(results), "normalized": normalize},
        }
