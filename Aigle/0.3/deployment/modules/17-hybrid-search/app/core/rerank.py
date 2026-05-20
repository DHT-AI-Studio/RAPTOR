import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from fastapi import HTTPException, status
import asyncio

from app.core.config import settings
from app.schemas import SearchResult

_rerank_executor = ThreadPoolExecutor(max_workers=2)


class RerankerManager:
    def __init__(self):
        self.model = None
        self.is_ready = False
        self._sem = asyncio.Semaphore(2)  # 同時最多 2 個 rerank task

    def load_model(self):
        if self.model is None:
            logging.info(f"Loading Reranker Model: {settings.RERANKER_MODEL}")
            if torch.cuda.is_available():
                logging.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            self.model = CrossEncoder(settings.RERANKER_MODEL)
            self.is_ready = True
            if torch.cuda.is_available():
                logging.info(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return self.model

    def rerank(
        self,
        query: str,
        documents: List[SearchResult],
        content_extractor: callable,
        top_k: int = 10,
        batch_size: int = 32,
    ) -> List[Dict]:
        if not self.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reranker Model is not loaded yet. Please try again later."
            )

        try:
            pairs = [[query, content_extractor(doc.payload)] for doc in documents]

            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )

            for i, doc in enumerate(documents):
                # doc.payload["original_score"] = doc.score 
                doc.score = float(scores[i])

            documents.sort(key=lambda x: x.score, reverse=True)

            return documents[:top_k]

        except Exception as e:
            logging.error(f"Reranking failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reranking failed: {str(e)}"
            )

    async def rerank_async(
        self,
        query: str,
        documents: List[SearchResult],
        content_extractor: callable,
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[Dict]:
        async with self._sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _rerank_executor,
                self.rerank,
                query,
                documents,
                content_extractor,
                top_k,
                batch_size
            )

    def unload_model(self):
        if self.model is not None:
            logging.info("Unloading Reranker Model")
            del self.model
            self.model = None
            self.is_ready = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info(f"GPU memory after unloading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")