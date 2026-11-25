import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from app.core.config import settings
from app.schemas import SearchResult
from fastapi import HTTPException, status
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
_rerank_executor = ThreadPoolExecutor(max_workers=2)


class RerankerManager:
    def __init__(self):
        self.model = None
        self.is_ready = False
        self._sem = asyncio.Semaphore(2)  # 同時最多 2 個 rerank task

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading Reranker Model: {settings.RERANKER_MODEL}")
            if torch.cuda.is_available():
                logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            self.model = CrossEncoder(settings.RERANKER_MODEL)
            self.is_ready = True
            if torch.cuda.is_available():
                logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return self.model

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        content_extractor: callable,
        top_k: int = 10,
        batch_size: int = 32,
    ) -> list[dict]:
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

        except Exception as e:  # noqa: BLE001 -- torch/sentence-transformers model inference, varied failure modes; already logged
            logger.error(f"Reranking failed: {e!s}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reranking failed: {e!s}"
            )

    async def rerank_async(
        self,
        query: str,
        documents: list[SearchResult],
        content_extractor: callable,
        top_k: int = 10,
        batch_size: int = 32
    ) -> list[dict]:
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
            logger.info("Unloading Reranker Model")
            del self.model
            self.model = None
            self.is_ready = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU memory after unloading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
