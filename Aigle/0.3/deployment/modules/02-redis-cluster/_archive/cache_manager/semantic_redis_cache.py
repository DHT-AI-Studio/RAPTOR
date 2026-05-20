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
        model_instance: Optional[SentenceTransformer] = None,
        model_name: str = "BAAI/bge-m3",
        similarity_threshold: float = 0.85,
        index_name: str = "semantic_cache_index",
        ollama_url: Optional[str] = None,
        query_prefix: Optional[str] = None,
        passage_prefix: Optional[str] = None 
    ):
        self.base_cache = base_cache
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.index_name = index_name
        self._model = model_instance
        self._async_ollama_client = None
        self._sync_ollama_client = None

        self.query_prefix = query_prefix or ""
        self.passage_prefix = passage_prefix or ""

        if not model_instance:
            if ollama_url:
                logger.info(f"Using Ollama SDK at {ollama_url} for embeddings with model: {model_name}.")
                self._async_ollama_client = ollama.AsyncClient(host=ollama_url)
                self._sync_ollama_client = ollama.Client(host=ollama_url)
                self._is_ollama = True
            else:
                logger.info(f"Using SentenceTransformer for embeddings with model: {model_name}.")
                self._model = SentenceTransformer(model_name)
                self._is_ollama = False
        else:
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
        """Initialize the vector index in Redis, dropping and recreating if needed."""
        if not self._model and self._is_ollama:
                model_list = self._sync_ollama_client.list()
                if self.model_name not in [m.model for m in model_list.models]:
                    logger.info(f"Pulling Ollama model: {self.model_name}")
                    self._sync_ollama_client.pull(self.model_name)

        new_dim = self._get_model_dimension()

        try:
            index_info = self.base_cache.sync_client.execute_command("FT.INFO", self.index_name)
            current_dim = self._parse_current_dimension(index_info)
            
            if current_dim != new_dim:
                logger.warning(f"Index mismatch detected (dim {current_dim} vs {new_dim}). Rebuilding index.")
                self.base_cache.sync_client.execute_command("FT.DROPINDEX", self.index_name)
            else:
                logger.info("Vector index already exists. Skipping creation.")
                return

        except Exception as e:
            logger.info(f"Vector index {self.index_name} not found or FT.INFO failed: {e}. Proceeding to create.")
            pass

        self.base_cache.sync_client.execute_command(
            'FT.CREATE', self.index_name, 'ON', 'HASH', 'PREFIX', '1', 'sem_cache:',
            'SCHEMA',
            'namespace', 'TAG',
            'key', 'TEXT',
            'embedding', 'VECTOR', 'FLAT', '6', 'TYPE', 'FLOAT32', 'DIM', new_dim, 'DISTANCE_METRIC', 'COSINE'
        )
        logger.info(f"Successfully created/recreated index {self.index_name} with dimension {new_dim}.")

    def _parse_current_dimension(self, index_info):
        """Parses the dimension (DIM) of the 'embedding' field from the FT.INFO list output."""
        try:
            attributes_index = index_info.index(b'attributes')
            attributes_list = index_info[attributes_index + 1]

            for field_def in attributes_list:
                if field_def[1] == b'embedding' and field_def[5] == b'VECTOR':
                    dim_index = field_def.index(b'dim') + 1
                    return field_def[dim_index]
                
        except Exception as e:
            logger.error(f"Failed to parse index dimension: {e}")
            return -1

        return -1

    def _prepare_text(self, text: str, is_query: bool) -> str:
        """Prepare text with appropriate prefix"""
        prefix = self.query_prefix if is_query else self.passage_prefix
        return f"{prefix}{text}"

    def _get_vector(self, text: str, is_query: bool) -> bytes:
        """Get embedding vector synchronously"""
        prepared_text = self._prepare_text(text, is_query)
        if self._is_ollama:
            response = self._sync_ollama_client.embeddings(
                model=self.model_name,
                prompt=prepared_text
            )
            return np.array(response['embedding'], dtype=np.float32).tobytes()
        
        # Fallback path: Local SentenceTransformer
        return self._model.encode(prepared_text).astype(np.float32).tobytes()

    async def _aget_vector(self, text: str, is_query: bool) -> bytes:
        """Get embedding vector asynchronously"""
        prepared_text = self._prepare_text(text, is_query)
        if self._is_ollama:
            response = await self._async_ollama_client.embeddings(
                model=self.model_name,
                prompt=prepared_text
            )
            return np.array(response['embedding'], dtype=np.float32).tobytes()
        
        # Fallback path: Local SentenceTransformer
        embedding = await asyncio.to_thread(self._model.encode, prepared_text)
        return embedding.astype(np.float32).tobytes()

    def _store_vector(self, key: str, text: str, namespace: str, ttl: int = 3600):
        """Store embedding vector in Redis"""
        embedding = self._get_vector(text, is_query=False)
        self.base_cache.sync_client.hset(f"sem_cache:{key}", mapping={
            "key": key,
            "embedding": embedding,
            "namespace": namespace
        })
        self.base_cache.sync_client.expire(f"sem_cache:{key}", ttl)

    async def _astore_vector(self, key: str, text: str, namespace: str, ttl: int = 3600):
        """Store embedding vector in Redis"""
        embedding = await self._aget_vector(text, is_query=False)
        await self.base_cache.async_client.hset(f"sem_cache:{key}", mapping={
            "key": key,
            "embedding": embedding,
            "namespace": namespace
        })
        await self.base_cache.async_client.expire(f"sem_cache:{key}", ttl)

    def _search_similar_key(self, query: str, namespace: str) -> Optional[str]:
        """Search for the most similar key based on the query embedding under the given namespace"""
        vec = self._get_vector(query, is_query=True)
        search_query = f"(@namespace:{{{namespace}}})=>[KNN 1 @embedding $vec]"
        
        try:    
            results = self.base_cache.sync_client.execute_command(
                'FT.SEARCH', self.index_name, search_query,
                'PARAMS', '2', 'vec', vec,
                'RETURN', '2', '__embedding_score', 'key',
                'DIALECT', '2'
            )
            if not results or len(results) < 2: return None
        
            fields = results[2]
            field_dict = dict(zip(fields[::2], fields[1::2]))
            score = float(field_dict.get(b'__embedding_score', b'1.0'))
            matched_key = field_dict.get(b'key', None)

            logger.debug(f"score: {1 - score}, matched_key: {matched_key}")
            if matched_key and score <= (1 - self.similarity_threshold):
                return matched_key.decode("utf-8")
        except Exception as e:
            logger.error(f"Search error: {e}")
        return None

    def get(self, key: str, query: str, namespace: str) -> Optional[Any]:
        """Get value from cache by key and query"""
        semantic_key = self._search_similar_key(query, namespace)
        logger.debug(f'semantic_key: {semantic_key}')
        if semantic_key:
            value = self.base_cache.get(semantic_key)
            if value is not None:
                return value
        return self.base_cache.get(key)

    async def aget(self, key: str, query: str, namespace: str) -> Optional[Any]:
        """Get value from cache by key and query"""
        semantic_key = self._search_similar_key(query, namespace)
        logger.debug(f'semantic_key: {semantic_key}')
        if semantic_key:
            value = await self.base_cache.aget(semantic_key)
            if value is not None:
                return value
        return await self.base_cache.aget(key)

    def set(self, key: str, value: Any, ttl: int, query: str, namespace: str):
        """Set value in cache by key and query"""
        self.base_cache.set(key, value, ttl)
        self._store_vector(key, query, namespace, ttl)

    async def aset(self, key: str, value: Any, ttl: int, query: str, namespace: str):
        """Set value in cache by key and query"""
        await self.base_cache.aset(key, value, ttl)
        await self._astore_vector(key, query, namespace, ttl)

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