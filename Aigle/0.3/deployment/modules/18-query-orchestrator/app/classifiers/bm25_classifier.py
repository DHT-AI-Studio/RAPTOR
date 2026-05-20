"""
BM25 Okapi Classifier

Stage 1 of HybridClassifier: keyword-based routing gate.
Computes BM25 scores against all taxonomy examples, then Max-Pools per pipeline.
If the best pipeline score < BM25_THRESHOLD the query is considered unknown.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from .base import BaseClassifier
from ..models.classification import ClassificationResult, Candidate
from ..models.intent import IntentTaxonomy
from ..core.config import get_settings
from ..data.intent_taxonomy import load_intent_taxonomy

logger = logging.getLogger(__name__)


class BM25Classifier(BaseClassifier):

    def __init__(self, taxonomy: IntentTaxonomy = None, build_if_missing: bool = True):
        self.bm25 = None
        self.intent_map: Dict[int, str] = {}
        self.intent_pipeline_map: Dict[int, str] = {}
        self.tokenized_corpus: List[List[str]] = []
        self.build_if_missing = build_if_missing
        self.taxonomy = taxonomy or load_intent_taxonomy()
        logger.info("Initializing BM25Classifier...")
        self._load_or_build_index()
        logger.info("BM25Classifier initialized successfully")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_or_build_index(self):
        index_path = get_settings().BM25_INDEX_PATH
        if os.path.exists(index_path):
            self._load_index(index_path)
        elif self.build_if_missing:
            logger.info("BM25 index not found at %s — building...", index_path)
            self._build_index()
            self._load_index(index_path)
        else:
            raise RuntimeError(f"BM25 index not found at {index_path}")

    def _load_index(self, index_path: str):
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.intent_map = data["intent_map"]
            self.intent_pipeline_map = data["intent_pipeline_map"]
            self.tokenized_corpus = data["tokenized_corpus"]
        except Exception as e:
            raise RuntimeError(f"Failed to load BM25 index from {index_path}: {e}")

    def _build_index(self):
        corpus: List[str] = []
        intent_map: Dict[int, str] = {}
        intent_pipeline_map: Dict[int, str] = {}

        for i, intent in enumerate(self.taxonomy.intents):
            parts = [intent.description] + (intent.keywords or [])
            corpus.append(" ".join(parts))
            intent_map[i] = intent.id
            intent_pipeline_map[i] = intent.target_pipeline

        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        output_path = get_settings().BM25_INDEX_PATH
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "intent_map": intent_map,
                "intent_pipeline_map": intent_pipeline_map,
                "tokenized_corpus": tokenized_corpus,
            }, f)

        self.intent_map = intent_map
        self.intent_pipeline_map = intent_pipeline_map
        logger.info("BM25 index built → %s (%d docs)", output_path, len(corpus))

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = jieba.lcut(text.lower())
        return [t.strip() for t in tokens if t.strip()]

    # ------------------------------------------------------------------
    # Scoring helpers (used by HybridClassifier)
    # ------------------------------------------------------------------

    def _compute_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(self._tokenize(query)))

    def _max_pooling(self, scores: np.ndarray) -> Dict[str, float]:
        """Per-pipeline max score over all taxonomy examples."""
        result: Dict[str, float] = {}
        for idx, score in enumerate(scores):
            pipeline = self.intent_pipeline_map.get(idx)
            if pipeline and (pipeline not in result or score > result[pipeline]):
                result[pipeline] = score
        return result

    def _group_scores_by_pipeline(self, scores: np.ndarray) -> Dict[str, List[float]]:
        """
        將所有範例的 BM25 分數按 pipeline 分組。
        
        Returns:
            Dict[pipeline, [score1, score2, ...]]
            例如: {"VideoRAG": [0.9, 0.8, ...], "GraphRAG": [0.7, 0.65, ...]}
        """
        result: Dict[str, List[float]] = {}
        for idx, score in enumerate(scores):
            pipeline = self.intent_pipeline_map.get(idx)
            if pipeline:
                if pipeline not in result:
                    result[pipeline] = []
                result[pipeline].append(float(score))
        return result


    # ------------------------------------------------------------------
    # Standalone classify (Tier-2 or evaluation)
    # ------------------------------------------------------------------

    def classify(self, query: str, k: int = 1) -> ClassificationResult:
        scores = self._compute_scores(query)
        pipeline_scores = self._max_pooling(scores)

        if not pipeline_scores:
            return ClassificationResult(confidence=0.0, intent_type=["unknown"], intents=[])

        best_name, best_score = max(pipeline_scores.items(), key=lambda x: x[1])

        if best_score < get_settings().BM25_THRESHOLD:
            logger.debug("BM25 gate failed (score=%.2f < %.1f)", best_score, get_settings().BM25_THRESHOLD)
            return ClassificationResult(confidence=0.0, intent_type=["unknown"], intents=[])

        candidates = []
        for idx, score in enumerate(scores):
            if self.intent_pipeline_map.get(idx) == best_name:
                intent_id = self.intent_map[idx]
                intent = self.taxonomy.get_intent(intent_id)
                candidates.append(Candidate(
                    intent_id=intent_id,
                    description=intent.description if intent else None,
                    confidence=round(float(score), 4),
                    target_pipeline=best_name,
                    rank=len(candidates) + 1,
                ))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        top = candidates[:k]
        if not top:
            return ClassificationResult(confidence=0.0, intent_type=["unknown"], intents=[])

        intent_type_list, intents = self._determine_intent_type(top)
        return ClassificationResult(
            confidence=top[0].confidence,
            intent_type=intent_type_list,
            intents=intents,
        )

