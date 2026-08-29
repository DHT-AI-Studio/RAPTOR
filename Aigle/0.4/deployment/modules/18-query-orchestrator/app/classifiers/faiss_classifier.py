"""
FAISS Vector Classifier (CPU)

Stage 2 of HybridClassifier: semantic k-NN retrieval using bge-m3 embeddings.
Index is built from intent taxonomy descriptions + keywords, stored as IndexFlatIP
(inner product = cosine similarity when vectors are L2-normalised).
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseClassifier
from ..models.classification import ClassificationResult, Candidate
from ..models.intent import IntentTaxonomy
from ..core.config import get_settings
from ..data.intent_taxonomy import load_intent_taxonomy

logger = logging.getLogger(__name__)

_TASK_DESC = "Determine the user's intent by understanding the full semantic meaning of the query."


class FAISSClassifier(BaseClassifier):

    def __init__(self, model_name: str = None, taxonomy: IntentTaxonomy = None, build_if_missing: bool = True):
        s = get_settings()
        self.model_name = model_name or s.MODEL_NAME
        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.intent_map: Dict[int, str] = {}
        self.taxonomy = taxonomy or load_intent_taxonomy()
        self.build_if_missing = build_if_missing
        self._load_or_build_index()
        logger.info("FAISSClassifier initialized successfully")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_or_build_index(self):
        s = get_settings()
        index_path = s.FAISS_INDEX_PATH
        if os.path.exists(index_path):
            self._load_index(index_path)
        elif self.build_if_missing:
            logger.info("FAISS index not found at %s — building...", index_path)
            self._build_index()
            self._load_index(index_path)
        else:
            raise RuntimeError(f"FAISS index not found at {index_path}")

    def _load_index(self, index_path: str):
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            self.index = data["index"]
            self.intent_map = data["intent_map"]
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {index_path}: {e}")

    def _build_index(self):
        s = get_settings()
        texts = []
        intent_map = {}
        for intent in self.taxonomy.intents:
            sources = intent.examples if intent.examples else [
                " ".join([intent.description] + (intent.keywords or []))
            ]
            for text in sources:
                intent_map[len(texts)] = intent.id
                texts.append(text)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(embeddings.astype("float32"))

        output_path = s.FAISS_INDEX_PATH
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({"index": cpu_index, "intent_map": intent_map}, f)

        self.index = cpu_index
        self.intent_map = intent_map
        logger.info("FAISS index built → %s (%d vectors, dim=%d)", output_path, cpu_index.ntotal, dim)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        instructed = f"Instruct: {_TASK_DESC}\nQuery: {query}"
        return self.model.encode(
            [instructed], convert_to_numpy=True, normalize_embeddings=True
        )[0]

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------

    def classify(self, query: str, k: int = 1) -> ClassificationResult:
        vec = self._embed_query(query).reshape(1, -1)
        k_safe = min(k, self.index.ntotal)
        distances, indices = self.index.search(vec, k_safe)

        candidates = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            confidence = round(float(distances[0][i]), 4)
            intent_id = self.intent_map[idx]
            intent = self.taxonomy.get_intent(intent_id)
            candidates.append(Candidate(
                intent_id=intent_id,
                description=intent.description if intent else None,
                confidence=confidence,
                target_pipeline=intent.target_pipeline if intent else None,
                rank=i + 1,
            ))

        if not candidates:
            return ClassificationResult(confidence=0.0, intent_type=["unknown"], intents=[])

        intent_type_list, intents = self._determine_intent_type(candidates)
        return ClassificationResult(
            confidence=candidates[0].confidence,
            intent_type=intent_type_list,
            intents=intents,
        )


    def get_all_faiss_scores(self, query: str) -> Dict[str, List[float]]:
        """
         計算 query 對所有 800 個範例的 FAISS 分數，返回按 pipeline 分組的完整分數列表。
        
         與 BM25 的 _group_scores_by_pipeline() 格式一致，供 LightGBM 分佈特徵提取使用。
        
        Returns:
             Dict[pipeline, [score1, score2, ...]] 例如 {"VideoRAG": [0.92, 0.88, ...], "GraphRAG": [0.75, 0.72, ...]}
        """
        vec = self._embed_query(query).reshape(1, -1)
        distances, indices = self.index.search(vec, self.index.ntotal)      # 全量搜尋
        
        scores: Dict[str, List[float]] = {}
        for i in range(len(distances[0])):
            idx = int(indices[0][i])
            confidence = float(distances[0][i])
            intent_id = self.intent_map[idx]
            intent = self.taxonomy.get_intent(intent_id)
            pipeline = intent.target_pipeline if intent else None
            
            if pipeline:
                if pipeline not in scores:
                    scores[pipeline] = []
                scores[pipeline].append(confidence)
        
        return scores

    def batch_classify(self, queries: List[str], k: int = 1) -> List[ClassificationResult]:
        """Batch classify using FAISS vector search."""
        instructed = [f"Instruct: {_TASK_DESC}\nQuery: {q}" for q in queries]
        vecs = self.model.encode(instructed, convert_to_numpy=True, normalize_embeddings=True)
        k_safe = min(k, self.index.ntotal)
        distances, indices = self.index.search(vecs, k_safe)

        results = []
        for i, query in enumerate(queries):
            candidates = []
            for j in range(len(indices[i])):
                idx = int(indices[i][j])
                intent_id = self.intent_map[idx]
                intent = self.taxonomy.get_intent(intent_id)
                candidates.append(Candidate(
                    intent_id=intent_id,
                    description=intent.description if intent else None,
                    confidence=round(float(distances[i][j]), 4),
                    target_pipeline=intent.target_pipeline if intent else None,
                    rank=j + 1,
                 ))
            if not candidates:
                results.append(ClassificationResult(confidence=0.0, intent_type=["unknown"], intents=[]))
                continue
            intent_type_list, intents = self._determine_intent_type(candidates)
            results.append(ClassificationResult(
                confidence=candidates[0].confidence,
                intent_type=intent_type_list,
                intents=intents,
             ))
        return results
