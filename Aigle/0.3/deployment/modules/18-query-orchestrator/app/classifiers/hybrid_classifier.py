"""
Hybrid Classifier: BM25 gate → FAISS k-NN → weighted score fusion.

Flow:
  Stage 1  BM25 Router   — max-pool per pipeline; reject if best < BM25_THRESHOLD
  Stage 2  FAISS k-NN    — semantic retrieval against taxonomy embeddings
  Stage 3  Score Fusion  — sigmoid(BM25) × BM25_WEIGHT + FAISS × VECTOR_WEIGHT
  Stage 4  Decision      — single / multi / unknown via confidence + gap thresholds
"""
from __future__ import annotations

import logging
import math
from typing import List

from .base import BaseClassifier
from .bm25_classifier import BM25Classifier
from .faiss_classifier import FAISSClassifier
from ..models.classification import ClassificationResult, Candidate
from ..models.intent import IntentTaxonomy
from ..core.config import get_settings
from ..data.intent_taxonomy import load_intent_taxonomy

logger = logging.getLogger(__name__)


class HybridClassifier(BaseClassifier):

    def __init__(self, taxonomy: IntentTaxonomy = None, build_if_missing: bool = True):
        self.taxonomy = taxonomy or load_intent_taxonomy()
        logger.info("Initializing HybridClassifier (BM25 + FAISS)...")
        self.bm25_classifier = BM25Classifier(taxonomy=self.taxonomy, build_if_missing=build_if_missing)
        self.faiss_classifier = FAISSClassifier(taxonomy=self.taxonomy, build_if_missing=build_if_missing)
        logger.info("HybridClassifier ready")

    # ------------------------------------------------------------------
    # Stage 1: BM25 gate
    # ------------------------------------------------------------------

    def _bm25_routing(self, query: str) -> tuple:
        """
        Returns (is_unknown, max_bm25_score, pipeline_scores, bm25_candidates).
        bm25_candidates contains only examples from the winning pipeline.
        """
        scores = self.bm25_classifier._compute_scores(query)
        pipeline_scores = self.bm25_classifier._max_pooling(scores)

        if not pipeline_scores:
            return True, 0.0, {}, []

        best_name, best_score = max(pipeline_scores.items(), key=lambda x: x[1])

        if best_score < get_settings().BM25_THRESHOLD:
            logger.debug("BM25 gate failed: max=%.2f < %.1f", best_score, get_settings().BM25_THRESHOLD)
            return True, best_score, pipeline_scores, []

        bm25_candidates = []
        for idx, score in enumerate(scores):
            if self.bm25_classifier.intent_pipeline_map.get(idx) == best_name:
                intent_id = self.bm25_classifier.intent_map[idx]
                intent = self.taxonomy.get_intent(intent_id)
                bm25_candidates.append(Candidate(
                    intent_id=intent_id,
                    description=intent.description if intent else None,
                    confidence=round(float(score), 4),
                    target_pipeline=best_name,
                    rank=len(bm25_candidates) + 1,
                ))

        bm25_candidates.sort(key=lambda c: c.confidence, reverse=True)
        return False, best_score, pipeline_scores, bm25_candidates

    # ------------------------------------------------------------------
    # Stage 3: Score fusion
    # ------------------------------------------------------------------

    def _sigmoid(self, score: float, k: float = 0.5, x0: float = 3.0) -> float:
        x = max(min(score, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))

    def _fuse_scores(
        self,
        bm25_candidates: List[Candidate],
        faiss_candidates: List[Candidate],
    ) -> List[Candidate]:
        if not bm25_candidates:
            return faiss_candidates
        if not faiss_candidates:
            return bm25_candidates

        bm25_map = {c.intent_id: c.confidence for c in bm25_candidates}
        faiss_map = {c.intent_id: c.confidence for c in faiss_candidates}
        all_ids = set(bm25_map) | set(faiss_map)

        fused = []
        for intent_id in all_ids:
            intent = self.taxonomy.get_intent(intent_id)
            bm25_norm = self._sigmoid(bm25_map.get(intent_id, 0.0))
            faiss_score = faiss_map.get(intent_id, 0.0)
            fused_score = bm25_norm * get_settings().BM25_WEIGHT + faiss_score * get_settings().VECTOR_WEIGHT
            fused.append(Candidate(
                intent_id=intent_id,
                description=intent.description if intent else None,
                confidence=round(fused_score, 4),
                target_pipeline=intent.target_pipeline if intent else None,
            ))

        fused.sort(key=lambda c: c.confidence, reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Main classify
    # ------------------------------------------------------------------

    def classify(self, query: str, k: int = 1) -> ClassificationResult:
        _unknown = ClassificationResult(confidence=0.0, intent_type=["unknown"], intents=[])

        # Stage 1: BM25 gate (pass/fail)
        scores = self.bm25_classifier._compute_scores(query)
        pipeline_scores = self.bm25_classifier._max_pooling(scores)
        if not pipeline_scores:
            return _unknown
        bm25_top_pipeline, bm25_top_score = max(pipeline_scores.items(), key=lambda x: x[1])
        if bm25_top_score < get_settings().BM25_THRESHOLD:
            logger.debug("BM25 gate failed: max=%.2f < %.1f", bm25_top_score, get_settings().BM25_THRESHOLD)
            return _unknown

        # Stage 2: FAISS k-NN (fetch top-5 for consistency check)
        faiss_candidates = self.faiss_classifier.classify(query, k=5).intents
        if not faiss_candidates:
            return _unknown

        # Stage 3: Consistency check (relaxed)
        # Allow top-3 to have at most 1 different pipeline (tolerate noise).
        # Mixed signals mean the query is ambiguous — fall through to rule-based / LLM.
        top3_pipelines = [c.target_pipeline for c in faiss_candidates[:3]]
        unique_pipelines = set(top3_pipelines)
        faiss_top_pipeline = top3_pipelines[0]
        # If top-3 has <= 2 unique pipelines, accept it (at least 2 agree)
        all_agree = len(unique_pipelines) <= 2
        bm25_agrees = bm25_top_pipeline == faiss_top_pipeline

        if not (all_agree and bm25_agrees):
            logger.debug(
                 "Consistency check failed: FAISS_top3=%s BM25_top=%s",
                top3_pipelines, bm25_top_pipeline,
            )
            return _unknown

        # Stage 4: Confidence — boost when both sources fully agree
        base_confidence = faiss_candidates[0].confidence
        confidence = min(round(base_confidence + 0.15, 4), 1.0)

        if confidence < get_settings().DEFAULT_CONFIDENCE_THRESHOLD:
            return _unknown

        top_candidates = faiss_candidates[:k]
        return ClassificationResult(
            confidence=confidence,
            intent_type=[faiss_top_pipeline],
            intents=top_candidates,
        )

