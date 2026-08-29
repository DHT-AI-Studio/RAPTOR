import logging
import torch
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException, status
import asyncio

from app.core.config import settings
from app.core.exceptions import EmbeddingError

class EmbeddingManager:
    def __init__(self):
        self.model = None
        self.is_ready = False  # 標記模型是否載入完成
        self._sem = asyncio.Semaphore(2)  # 同時最多 2 個 embedding task

    def load_model(self):
        if self.model is None:
            logging.info(f"Loading Embedding Model: {settings.EMBEDDING_MODEL}")
            if torch.cuda.is_available():
                logging.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.is_ready = True
            if torch.cuda.is_available():
                logging.info(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return self.model

    def encode(self, text: str | list[str], batch_size: int = 32):
        if not self.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding Model is not loaded yet. Please try again later."
            )
        try:
            embeddings = self.model.encode(
                text,
                batch_size=batch_size,
                show_progress_bar=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # 確保 embeddings 回到 CPU
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu()
            # 清空暫存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to encode text: {str(e)}")

    async def encode_async(self, text: str | list[str], batch_size: int = 32):
        async with self._sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self.encode,
                text,
                batch_size
            )

    def unload_model(self):
        if self.model is not None:
            logging.info("Unloading Embedding Model")
            del self.model
            self.model = None
            self.is_ready = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info(f"GPU memory after unloading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

