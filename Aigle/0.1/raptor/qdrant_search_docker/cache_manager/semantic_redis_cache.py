import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, Any
from .base_cache import BaseCache
import logging
import ollama
import asyncio

logger = logging.getLogger(__name__)


class SemanticRedisCache:
    def __init__(
        self,
        base_cache: BaseCache,
        model_name: str = "BAAI/bge-m3",
        similarity_threshold: float = 0.85,
        index_name: str = "semantic_cache_index",
        ollama_url: Optional[str] = None, 
    ):
        self.base_cache = base_cache
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.index_name = index_name
        self._model = None
        self._async_ollama_client = None
        self._sync_ollama_client = None

        if ollama_url:
            logger.info(f"Using Ollama SDK at {ollama_url} for embeddings with model: {model_name}.")
            self._async_ollama_client = ollama.AsyncClient(host=ollama_url)
            self._sync_ollama_client = ollama.Client(host=ollama_url)
            self._is_ollama = True
        else:
            logger.info(f"Using SentenceTransformer for embeddings with model: {model_name}.")
            self._model = SentenceTransformer(model_name)
            self._is_ollama = False

        self._initialize()

    def _get_model_dimension(self) -> int:
        """Get the dimension of the embedding model"""        
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        
        try:
            response = self._sync_ollama_client.embeddings(
                model=self.model_name,
                prompt="test"
            )
            return len(response['embedding'])
        except Exception as e:
            logger.error(f"Failed to get model dimension from Ollama SDK: {e}")
            raise

    def _initialize(self):
        """Initialize the vector index in Redis if it doesn't exist"""
        if self._is_ollama:
            model_list = self._sync_ollama_client.list()
            if self.model_name not in [m.model for m in model_list.models]:
                logger.info(f"Pulling Ollama model: {self.model_name}")
                self._sync_ollama_client.pull(self.model_name)

        dim = self._get_model_dimension()
        try:
            # try to get index info to check if it exists
            self.base_cache.sync_client.execute_command("FT.INFO", self.index_name)
        except Exception:
            # if it doesn't exist, create it
            self.base_cache.sync_client.execute_command(
                'FT.CREATE', self.index_name, 'ON', 'HASH', 'PREFIX', '1', 'sem_cache:',
                'SCHEMA',
                'key', 'TEXT',
                'embedding', 'VECTOR', 'FLAT', '6', 'TYPE', 'FLOAT32', 'DIM', dim, 'DISTANCE_METRIC', 'COSINE'
            )

    def _get_vector(self, text: str) -> bytes:
        """Get embedding vector synchronously"""
        if self._is_ollama:
            response = self._sync_ollama_client.embeddings(
                model=self.model_name,
                prompt=text
            )
            embedding_list = response['embedding']
            return np.array(embedding_list, dtype=np.float32).tobytes()
        else:
            # Fallback path: Local SentenceTransformer
            return self._model.encode(text).astype(np.float32).tobytes()

    async def _aget_vector(self, text: str) -> bytes:
        """Get embedding vector asynchronously"""
        if self._is_ollama:
            response = await self._async_ollama_client.embeddings(
                model=self.model_name,
                prompt=text
            )
            embedding_list = response['embedding']
            return np.array(embedding_list, dtype=np.float32).tobytes()
        else:
            func = self._model.encode
            embedding = await asyncio.to_thread(func, text) 
            return embedding.astype(np.float32).tobytes()

    def _store_vector(self, key: str, text: str, ttl: int = 3600):
        """Store embedding vector in Redis"""
        embedding = self._get_vector(text)
        self.base_cache.sync_client.hset(f"sem_cache:{key}", mapping={
            "key": key,
            "embedding": embedding
        })
        self.base_cache.sync_client.expire(f"sem_cache:{key}", ttl)

    async def _astore_vector(self, key: str, text: str, ttl: int = 3600):
        """Store embedding vector in Redis"""
        embedding = await self._aget_vector(text)
        await self.base_cache.async_client.hset(f"sem_cache:{key}", mapping={
            "key": key,
            "embedding": embedding
        })
        await self.base_cache.async_client.expire(f"sem_cache:{key}", ttl)

    def _search_similar_key(self, query: str) -> Optional[str]:
        """Search for the most similar key based on the query embedding"""
        vec = self._get_vector(query)
        results = self.base_cache.sync_client.execute_command(
            'FT.SEARCH', self.index_name,
            f"*=>[KNN 1 @embedding $vec]",
            'PARAMS', '2', 'vec', vec, 
            'RETURN', '2', '__embedding_score', 'key',
            'DIALECT', '2'
        )
        if not results or len(results) < 2:
            return None
        try:
            fields = results[2]
            field_dict = dict(zip(fields[::2], fields[1::2]))
            score = float(field_dict.get(b'__embedding_score', b'1.0'))
            matched_key = field_dict.get(b'key', None)
            logger.debug(f"score: {1 - score}, matched_key: {matched_key}")
            if matched_key and score <= (1 - self.similarity_threshold):
                return matched_key.decode("utf-8")
            return None
        except Exception as e:
            print(e)

    def get(self, key: str, query: str) -> Optional[Any]:
        """Get value from cache by key and query"""
        semantic_key = self._search_similar_key(query)
        logger.debug(f'semantic_key: {semantic_key}')
        if semantic_key:
            value = self.base_cache.get(semantic_key)
            if value is not None:
                return value
        return self.base_cache.get(key)

    async def aget(self, key: str, query: str) -> Optional[Any]:
        """Get value from cache by key and query"""
        semantic_key = self._search_similar_key(query)
        logger.debug(f'semantic_key: {semantic_key}')
        if semantic_key:
            value = await self.base_cache.aget(semantic_key)
            if value is not None:
                return value
        return await self.base_cache.aget(key)

    def set(self, key: str, value: Any, ttl: int, query: str):
        """Set value in cache by key and query"""
        self.base_cache.set(key, value, ttl)
        self._store_vector(key, query, ttl)

    async def aset(self, key: str, value: Any, ttl: int, query: str):
        """Set value in cache by key and query"""
        await self.base_cache.aset(key, value, ttl)
        await self._astore_vector(key, query, ttl)

    def clear_by_pattern(self, pattern: str):
        """Delete keys matching the given pattern using SCAN"""
        pipeline = self.base_cache.sync_client.pipeline()
        count = 0
        for key in self.base_cache.sync_client.scan_iter(match=pattern):
            key_str = key.decode('utf-8')
            hash_key = f"sem_cache:{key_str}"
            pipeline.delete(key_str)
            pipeline.delete(hash_key)
            count += 1
            if count % 100 == 0:
                pipeline.execute()
        if count % 100 != 0:
            pipeline.execute()
        print(f"Deleted {count} semantic cache entries matching '{pattern}'")

    async def aclear_by_pattern(self, pattern: str):
        """Asynchronous delete keys matching the given pattern using SCAN"""
        pipeline = self.base_cache.async_client.pipeline()
        count = 0
        async for key in self.base_cache.async_client.scan_iter(match=pattern):
            key_str = key.decode('utf-8')
            hash_key = f"sem_cache:{key_str}"
            pipeline.delete(key_str)
            pipeline.delete(hash_key)
            count += 1
            if count % 100 == 0:
                await pipeline.execute()
        if count % 100 != 0:
            await pipeline.execute()
        print(f"Async deleted {count} semantic cache entries matching '{pattern}'")

    def close(self):
        self.base_cache.close()

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis (sync)"""
        return self.base_cache.sync_client.exists(key)

    async def aexists(self, key: str) -> bool:
        """Check if a key exists in Redis (async)"""
        return await self.base_cache.async_client.exists(key)