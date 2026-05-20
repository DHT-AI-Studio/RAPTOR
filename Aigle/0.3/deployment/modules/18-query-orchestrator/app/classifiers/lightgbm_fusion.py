"""
LightGBM-based pipeline ranker.

This module uses LightGBM Learning-to-Rank to directly predict the ranking
of pipelines (VideoRAG / GraphRAG / TKG / RDBMS) based on BM25, FAISS, and
rule-based scores.

Usage:
    ranker = LightGBMRanker()
    ranker.train(query_pipeline_features, groups, labels)
    ranker.save("ranker.pkl")

    # At inference:
    ranked = ranker.predict(features)
    # Returns: [("VideoRAG", 0.85), ("GraphRAG", 0.45), ...]

Key differences from classifier:
    - objective: "lambdarank" instead of "multiclass"
    - Input: list of (features, relevance) per query, grouped by query_id
    - Output: ranked list of (pipeline, score) instead of single prediction
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jieba
import numpy as np

logger = logging.getLogger(__name__)

# Pipeline classes
PIPELINE_CLASSES = ["VideoRAG", "GraphRAG", "TKG", "RDBMS"]
PIPELINE_TO_IDX = {p: i for i, p in enumerate(PIPELINE_CLASSES)}

# Keyword patterns for query-level features
_QUESTION_WORDS = re.compile(
    r"\b(什麼|怎麼|如何|哪個|哪些|誰|哪段|哪家的|多少)\b", re.IGNORECASE
)
_TIME_WORDS = re.compile(
    r"\b(時間|之前|之後|順序|演進|歷史|timeline|evolution|before|after|when|during|sequence|chronolog)\b",
    re.IGNORECASE,
)
_RELATION_WORDS = re.compile(
    r"\b(關係|關聯|合作|競爭|連接|relationship|connection|collaborat|competitor|rival|partner|linked|network)\b",
    re.IGNORECASE,
)
_LIST_WORDS = re.compile(
    r"\b(清單|所有|列出|統計|list|count|how many|all|total|catalog|filter|sort)\b",
    re.IGNORECASE,
)
_VISUAL_WORDS = re.compile(
    r"\b(影片|畫面|影像|片段|視覺|video|clip|footage|visual|scene|frame|segment|broadcast)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryFeatures:
    """Query features for LightGBM ranker input.

    Features are divided into three categories:
    1. Query-level: same for all pipelines in a query
    2. Pipeline-level: specific to each pipeline
    3. Cross-pipeline: computed across all pipelines
    """

    # ========================================================================
    # Query-level features (12 features)
    # ========================================================================
    query_length: float = 0.0              # Character count
    query_word_count: float = 0.0          # Token count (jieba)
    query_has_question: float = 0.0        # Has question word (0/1)
    query_has_time_expr: float = 0.0       # Has time expression (0/1)
    query_has_relation_word: float = 0.0   # Has relation word (0/1)
    query_has_list_word: float = 0.0       # Has list word (0/1)
    query_has_visual_word: float = 0.0     # Has visual word (0/1)
    query_complexity: float = 0.0          # Complexity score (0-1)
    bm25_max: float = 0.0                  # Max BM25 score across pipelines
    faiss_max: float = 0.0                 # Max FAISS score across pipelines
    rule_max: float = 0.0                  # Max Rule score across pipelines
    num_matched_rules: float = 0.0         # Number of matched rules

    # ========================================================================
    # Pipeline-level features (18 features x 4 pipelines = 72 features)
    # Stored as dicts: {pipeline_name: value}
    # ========================================================================
    bm25_score: Dict[str, float] = field(default_factory=dict)
    faiss_score: Dict[str, float] = field(default_factory=dict)
    rule_score: Dict[str, float] = field(default_factory=dict)

    bm25_rank: Dict[str, float] = field(default_factory=dict)
    faiss_rank: Dict[str, float] = field(default_factory=dict)
    rule_rank: Dict[str, float] = field(default_factory=dict)

    bm25_top1: Dict[str, float] = field(default_factory=dict)
    faiss_top1: Dict[str, float] = field(default_factory=dict)
    rule_top1: Dict[str, float] = field(default_factory=dict)

    bm25_top3_ratio: Dict[str, float] = field(default_factory=dict)
    faiss_top3_ratio: Dict[str, float] = field(default_factory=dict)
    rule_top3_ratio: Dict[str, float] = field(default_factory=dict)

    bm25_gap: Dict[str, float] = field(default_factory=dict)
    faiss_gap: Dict[str, float] = field(default_factory=dict)
    rule_gap: Dict[str, float] = field(default_factory=dict)

    bm25_rel: Dict[str, float] = field(default_factory=dict)
    faiss_rel: Dict[str, float] = field(default_factory=dict)
    rule_rel: Dict[str, float] = field(default_factory=dict)

    # ========================================================================
    # Pipeline-level distribution features (6 features x 4 pipelines = 24 features)
    # ========================================================================
    bm25_mean: Dict[str, float] = field(default_factory=dict)
    bm25_std: Dict[str, float] = field(default_factory=dict)
    bm25_max_mean_ratio: Dict[str, float] = field(default_factory=dict)

    faiss_mean: Dict[str, float] = field(default_factory=dict)
    faiss_std: Dict[str, float] = field(default_factory=dict)
    faiss_score_density: Dict[str, float] = field(default_factory=dict)

    # ========================================================================
    # Cross-pipeline features (16 features)
    # ========================================================================
    agreement_top1: float = 0.0            # Count of methods agreeing on top-1 (0-3)
    agreement_top3: float = 0.0            # Count of pipelines in all top-3 (0-4)
    score_consistency: float = 0.0         # Inverse of score std (higher = more consistent)
    vote_count: float = 0.0                # Number of top-1 votes for target pipeline (0-3)
    max_min_score_gap: float = 0.0         # Max - min of max scores across methods
    score_entropy: float = 0.0             # Entropy of max scores
    is_consensus_winner: float = 0.0       # 1 if target is consensus top-1
    is_disagreement_candidate: float = 0.0 # 1 if target is in disagreement
    rank_agreement: float = 0.0            # Inverse of rank std (higher = more consistent)
    top1_confidence: float = 0.0           # Max gap across methods
    top3_diversity: float = 0.0            # Unique pipelines in top-3
    cross_method_agreement: float = 0.0    # Combined agreement score (0-1)
    bm25_faiss_corr: float = 0.0           # Correlation between BM25 and FAISS means
    score_dispersion: float = 0.0          # Std of max scores across methods
    top_k_consensus: float = 0.0           # Count of pipelines in all top-3 (by score)
    distribution_overlap: float = 0.0      # Overlap ratio of high-score examples

    # ========================================================================
    # Internal fields (not serialized)
    # ========================================================================
    _target_pipeline: str = ""             # Target pipeline for this sample
    _relevance: float = 0.0                # Relevance label (for training)

    def to_array(self, feature_names: List[str] = None) -> np.ndarray:
        """Convert features to numpy array for LightGBM input."""
        values = []

        # Query-level features (12)
        values.extend([
            self.query_length,
            self.query_word_count,
            self.query_has_question,
            self.query_has_time_expr,
            self.query_has_relation_word,
            self.query_has_list_word,
            self.query_has_visual_word,
            self.query_complexity,
            self.bm25_max,
            self.faiss_max,
            self.rule_max,
            self.num_matched_rules,
        ])

        # Pipeline-level features (18 x 4 = 72)
        for pipeline in PIPELINE_CLASSES:
            values.extend([
                self.bm25_score.get(pipeline, 0.0),
                self.faiss_score.get(pipeline, 0.0),
                self.rule_score.get(pipeline, 0.0),
                self.bm25_rank.get(pipeline, 4.0),
                self.faiss_rank.get(pipeline, 4.0),
                self.rule_rank.get(pipeline, 4.0),
                self.bm25_top1.get(pipeline, 0.0),
                self.faiss_top1.get(pipeline, 0.0),
                self.rule_top1.get(pipeline, 0.0),
                self.bm25_top3_ratio.get(pipeline, 0.0),
                self.faiss_top3_ratio.get(pipeline, 0.0),
                self.rule_top3_ratio.get(pipeline, 0.0),
                self.bm25_gap.get(pipeline, 0.0),
                self.faiss_gap.get(pipeline, 0.0),
                self.rule_gap.get(pipeline, 0.0),
                self.bm25_rel.get(pipeline, 0.0),
                self.faiss_rel.get(pipeline, 0.0),
                self.rule_rel.get(pipeline, 0.0),
            ])

        # Pipeline-level distribution features (6 x 4 = 24)
        for pipeline in PIPELINE_CLASSES:
            values.extend([
                self.bm25_mean.get(pipeline, 0.0),
                self.bm25_std.get(pipeline, 0.0),
                self.bm25_max_mean_ratio.get(pipeline, 0.0),
                self.faiss_mean.get(pipeline, 0.0),
                self.faiss_std.get(pipeline, 0.0),
                self.faiss_score_density.get(pipeline, 0.0),
            ])

        # Cross-pipeline features (16)
        values.extend([
            self.agreement_top1,
            self.agreement_top3,
            self.score_consistency,
            self.vote_count,
            self.max_min_score_gap,
            self.score_entropy,
            self.is_consensus_winner,
            self.is_disagreement_candidate,
            self.rank_agreement,
            self.top1_confidence,
            self.top3_diversity,
            self.cross_method_agreement,
            self.bm25_faiss_corr,
            self.score_dispersion,
            self.top_k_consensus,
            self.distribution_overlap,
        ])

        if feature_names is None:
            feature_names = self._get_feature_names()

        return np.array(values, dtype=np.float32)

    def _get_feature_names(self) -> List[str]:
        """Get full list of feature names."""
        names = []

        # Query-level (12)
        names.extend([
            "query_length",
            "query_word_count",
            "query_has_question",
            "query_has_time_expr",
            "query_has_relation_word",
            "query_has_list_word",
            "query_has_visual_word",
            "query_complexity",
            "bm25_max",
            "faiss_max",
            "rule_max",
            "num_matched_rules",
        ])

        # Pipeline-level (18 x 4 = 72)
        for pipeline in PIPELINE_CLASSES:
            names.extend([
                f"bm25_score_{pipeline}",
                f"faiss_score_{pipeline}",
                f"rule_score_{pipeline}",
                f"bm25_rank_{pipeline}",
                f"faiss_rank_{pipeline}",
                f"rule_rank_{pipeline}",
                f"bm25_top1_{pipeline}",
                f"faiss_top1_{pipeline}",
                f"rule_top1_{pipeline}",
                f"bm25_top3_ratio_{pipeline}",
                f"faiss_top3_ratio_{pipeline}",
                f"rule_top3_ratio_{pipeline}",
                f"bm25_gap_{pipeline}",
                f"faiss_gap_{pipeline}",
                f"rule_gap_{pipeline}",
                f"bm25_rel_{pipeline}",
                f"faiss_rel_{pipeline}",
                f"rule_rel_{pipeline}",
            ])

        # Pipeline-level distribution features (6 x 4 = 24)
        for pipeline in PIPELINE_CLASSES:
            names.extend([
                f"bm25_mean_{pipeline}",
                f"bm25_std_{pipeline}",
                f"bm25_max_mean_ratio_{pipeline}",
                f"faiss_mean_{pipeline}",
                f"faiss_std_{pipeline}",
                f"faiss_score_density_{pipeline}",
            ])

        # Cross-pipeline features (16)
        names.extend([
            "agreement_top1",
            "agreement_top3",
            "score_consistency",
            "vote_count",
            "max_min_score_gap",
            "score_entropy",
            "is_consensus_winner",
            "is_disagreement_candidate",
            "rank_agreement",
            "top1_confidence",
            "top3_diversity",
            "cross_method_agreement",
            "bm25_faiss_corr",
            "score_dispersion",
            "top_k_consensus",
            "distribution_overlap",
        ])

        return names


# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------

def _extract_query_features(
    query: str,
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
) -> Dict[str, float]:
    """Extract query-level features."""
    # Basic features
    query_length = float(len(query))
    tokens = jieba.lcut(query)
    query_word_count = float(len(tokens))

    # Keyword detection
    query_has_question = 1.0 if _QUESTION_WORDS.search(query) else 0.0
    query_has_time_expr = 1.0 if _TIME_WORDS.search(query) else 0.0
    query_has_relation_word = 1.0 if _RELATION_WORDS.search(query) else 0.0
    query_has_list_word = 1.0 if _LIST_WORDS.search(query) else 0.0
    query_has_visual_word = 1.0 if _VISUAL_WORDS.search(query) else 0.0

    # Complexity score
    query_complexity = min(
        1.0,
        (query_word_count / 20.0) * 0.5
        + (query_has_question + query_has_time_expr + query_has_relation_word
           + query_has_list_word + query_has_visual_word) / 5.0,
    )

    # Max scores across pipelines
    bm25_max = max(bm25_scores.values()) if bm25_scores else 0.0
    faiss_max = max(faiss_scores.values()) if faiss_scores else 0.0
    rule_max = max(rule_scores.values()) if rule_scores else 0.0

    return {
        "query_length": query_length,
        "query_word_count": query_word_count,
        "query_has_question": query_has_question,
        "query_has_time_expr": query_has_time_expr,
        "query_has_relation_word": query_has_relation_word,
        "query_has_list_word": query_has_list_word,
        "query_has_visual_word": query_has_visual_word,
        "query_complexity": query_complexity,
        "bm25_max": bm25_max,
        "faiss_max": faiss_max,
        "rule_max": rule_max,
    }


def _extract_pipeline_features(
    pipeline: str,
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
) -> Dict[str, float]:
    """Extract pipeline-level features for a single pipeline."""
    pipelines = PIPELINE_CLASSES

    # Score features
    bm25_score = bm25_scores.get(pipeline, 0.0)
    faiss_score = faiss_scores.get(pipeline, 0.0)
    rule_score = rule_scores.get(pipeline, 0.0)

    # Rank features
    bm25_sorted = sorted(bm25_scores.values(), reverse=True) if bm25_scores else [0.0]
    faiss_sorted = sorted(faiss_scores.values(), reverse=True) if faiss_scores else [0.0]
    rule_sorted = sorted(rule_scores.values(), reverse=True) if rule_scores else [0.0]

    bm25_rank = bm25_sorted.index(bm25_score) + 1 if bm25_score > 0 else 5.0
    faiss_rank = faiss_sorted.index(faiss_score) + 1 if faiss_score > 0 else 5.0
    rule_rank = rule_sorted.index(rule_score) + 1 if rule_score > 0 else 5.0

    # Top-1 features
    bm25_top1_pipeline = max(bm25_scores, key=bm25_scores.get) if bm25_scores else ""
    faiss_top1_pipeline = max(faiss_scores, key=faiss_scores.get) if faiss_scores else ""
    rule_top1_pipeline = max(rule_scores, key=rule_scores.get) if rule_scores else ""

    bm25_top1 = 1.0 if bm25_top1_pipeline == pipeline else 0.0
    faiss_top1 = 1.0 if faiss_top1_pipeline == pipeline else 0.0
    rule_top1 = 1.0 if rule_top1_pipeline == pipeline else 0.0

    # Top-3 ratio features
    bm25_top3 = set(sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:3])
    faiss_top3 = set(sorted(faiss_scores.keys(), key=lambda k: faiss_scores[k], reverse=True)[:3])
    rule_top3 = set(sorted(rule_scores.keys(), key=lambda k: rule_scores[k], reverse=True)[:3])

    bm25_top3_ratio = 1.0 if pipeline in bm25_top3 else 0.0
    faiss_top3_ratio = 1.0 if pipeline in faiss_top3 else 0.0
    rule_top3_ratio = 1.0 if pipeline in rule_top3 else 0.0

    # Gap features (top-1 - top-2)
    bm25_gap = (bm25_sorted[0] - bm25_sorted[1]) if len(bm25_sorted) > 1 else bm25_score
    faiss_gap = (faiss_sorted[0] - faiss_sorted[1]) if len(faiss_sorted) > 1 else faiss_score
    rule_gap = (rule_sorted[0] - rule_sorted[1]) if len(rule_sorted) > 1 else rule_score

    # Relative score features (self / sum)
    bm25_total = sum(bm25_scores.values()) if bm25_scores else 0.0
    faiss_total = sum(faiss_scores.values()) if faiss_scores else 0.0
    rule_total = sum(rule_scores.values()) if rule_scores else 0.0

    bm25_rel = bm25_score / bm25_total if bm25_total > 0 else 0.0
    faiss_rel = faiss_score / faiss_total if faiss_total > 0 else 0.0
    rule_rel = rule_score / rule_total if rule_total > 0 else 0.0

    return {
        "bm25_score": bm25_score,
        "faiss_score": faiss_score,
        "rule_score": rule_score,
        "bm25_rank": bm25_rank,
        "faiss_rank": faiss_rank,
        "rule_rank": rule_rank,
        "bm25_top1": bm25_top1,
        "faiss_top1": faiss_top1,
        "rule_top1": rule_top1,
        "bm25_top3_ratio": bm25_top3_ratio,
        "faiss_top3_ratio": faiss_top3_ratio,
        "rule_top3_ratio": rule_top3_ratio,
        "bm25_gap": bm25_gap,
        "faiss_gap": faiss_gap,
        "rule_gap": rule_gap,
        "bm25_rel": bm25_rel,
        "faiss_rel": faiss_rel,
        "rule_rel": rule_rel,
    }


def _extract_cross_pipeline_features(
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
    target_pipeline: str,
) -> Dict[str, float]:
    """Extract cross-pipeline interaction features (12 features)."""
    bm25_top1 = max(bm25_scores, key=bm25_scores.get) if bm25_scores else None
    faiss_top1 = max(faiss_scores, key=faiss_scores.get) if faiss_scores else None
    rule_top1 = max(rule_scores, key=rule_scores.get) if rule_scores else None
    top1_votes = [bm25_top1, faiss_top1, rule_top1]

    # agreement_top1: how many methods rank target_pipeline as top-1
    agreement_top1 = float(sum(1 for v in top1_votes if v == target_pipeline))

    # top-3 overlap
    bm25_top3 = set(sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:3]) if bm25_scores else set()
    faiss_top3 = set(sorted(faiss_scores, key=faiss_scores.get, reverse=True)[:3]) if faiss_scores else set()
    rule_top3  = set(sorted(rule_scores,  key=rule_scores.get,  reverse=True)[:3]) if rule_scores  else set()
    agreement_top3 = float(len(bm25_top3 & faiss_top3 & rule_top3))

    # score_consistency: 1 / (1 + std of target scores across methods)
    target_scores = [
        bm25_scores.get(target_pipeline, 0.0),
        faiss_scores.get(target_pipeline, 0.0),
        rule_scores.get(target_pipeline, 0.0),
    ]
    score_consistency = 1.0 / (1.0 + float(np.std(target_scores)))

    # vote_count: how many methods rank target_pipeline as top-1
    vote_count = float(sum(1 for v in top1_votes if v == target_pipeline))

    # max_min_score_gap across methods' max scores
    all_max = [
        max(bm25_scores.values()) if bm25_scores else 0.0,
        max(faiss_scores.values()) if faiss_scores else 0.0,
        max(rule_scores.values())  if rule_scores  else 0.0,
    ]
    max_min_score_gap = float(max(all_max) - min(all_max))

    # score_entropy of method max scores
    total = sum(all_max)
    if total > 0:
        probs = [s / total for s in all_max]
        score_entropy = float(-sum(p * np.log2(p + 1e-8) for p in probs))
    else:
        score_entropy = 0.0

    # consensus / disagreement
    is_consensus_winner      = 1.0 if (bm25_top1 == faiss_top1 == rule_top1 == target_pipeline) else 0.0
    unique_tops              = set(v for v in top1_votes if v)
    is_disagreement_candidate = 1.0 if (len(unique_tops) > 2 and target_pipeline in unique_tops) else 0.0

    # rank_agreement: 1 / (1 + std of target's rank across methods)
    def _rank(scores: Dict[str, float], pipeline: str) -> float:
        if not scores:
            return float(len(PIPELINE_CLASSES))
        order = sorted(scores, key=scores.get, reverse=True)
        return float(order.index(pipeline) + 1) if pipeline in order else float(len(PIPELINE_CLASSES))

    ranks = [_rank(bm25_scores, target_pipeline),
             _rank(faiss_scores, target_pipeline),
             _rank(rule_scores, target_pipeline)]
    rank_agreement = 1.0 / (1.0 + float(np.std(ranks)))

    # top1_confidence: max gap between top-1 and top-2 across methods
    def _gap(scores: Dict[str, float]) -> float:
        if not scores or len(scores) < 2:
            return 0.0
        sv = sorted(scores.values(), reverse=True)
        return sv[0] - sv[1]

    top1_confidence = float(max(_gap(bm25_scores), _gap(faiss_scores), _gap(rule_scores)))

    # top3_diversity: unique pipelines across all top-3 sets
    top3_diversity = float(len(bm25_top3 | faiss_top3 | rule_top3))

    # cross_method_agreement: combined agreement score
    cross_method_agreement = (vote_count / 3.0) * 0.5 + (agreement_top3 / 4.0) * 0.5

    return {
        "agreement_top1":           agreement_top1,
        "agreement_top3":           agreement_top3,
        "score_consistency":        score_consistency,
        "vote_count":               vote_count,
        "max_min_score_gap":        max_min_score_gap,
        "score_entropy":            score_entropy,
        "is_consensus_winner":      is_consensus_winner,
        "is_disagreement_candidate": is_disagreement_candidate,
        "rank_agreement":           rank_agreement,
        "top1_confidence":          top1_confidence,
        "top3_diversity":           top3_diversity,
        "cross_method_agreement":   cross_method_agreement,
    }


def _extract_distribution_features(
    bm25_all_scores: Dict[str, List[float]],
    faiss_all_scores: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Extract pipeline-level distribution features.
    
    Args:
        bm25_all_scores: Each pipeline's all example scores
                         {"VideoRAG": [0.9, 0.8, ...], "GraphRAG": [0.7, 0.65, ...]}
        faiss_all_scores: Each pipeline's all FAISS scores
    
    Returns:
        Dict with 6 distribution features per pipeline
    """
    features = {}
    
    # BM25 distribution features (3 per pipeline x 4 pipelines = 12)
    for pipeline, scores in bm25_all_scores.items():
        if scores:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            max_score = max(scores)
            max_mean_ratio = max_score / (mean_score + 1e-8)
            
            features[f"bm25_mean_{pipeline}"] = mean_score
            features[f"bm25_std_{pipeline}"] = std_score
            features[f"bm25_max_mean_ratio_{pipeline}"] = max_mean_ratio
    
    # FAISS distribution features (3 per pipeline x 4 pipelines = 12)
    for pipeline, scores in faiss_all_scores.items():
        if scores:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            
            # Score density: top-5 mean / all mean
            top_k = min(5, len(scores))
            top_k_mean = float(np.mean(sorted(scores, reverse=True)[:top_k]))
            score_density = top_k_mean / (mean_score + 1e-8)
            
            features[f"faiss_mean_{pipeline}"] = mean_score
            features[f"faiss_std_{pipeline}"] = std_score
            features[f"faiss_score_density_{pipeline}"] = score_density
    
    return features


def _extract_cross_pipeline_distribution_features(
    bm25_all_scores: Dict[str, List[float]],
    faiss_all_scores: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Extract cross-pipeline distribution interaction features.
    """
    features = {}
    
    # 1. BM25 vs FAISS correlation (based on each pipeline's mean score)
    bm25_means = []
    faiss_means = []
    for pipeline in PIPELINE_CLASSES:
        bm25_scores_list = bm25_all_scores.get(pipeline, [])
        faiss_scores_list = faiss_all_scores.get(pipeline, [])
        bm25_means.append(float(np.mean(bm25_scores_list)) if bm25_scores_list else 0.0)
        faiss_means.append(float(np.mean(faiss_scores_list)) if faiss_scores_list else 0.0)
    
    if len(bm25_means) > 1 and all(v != 0 for v in bm25_means) and all(v != 0 for v in faiss_means):
        corr = np.corrcoef(bm25_means, faiss_means)[0, 1]
        features["bm25_faiss_corr"] = float(corr) if not np.isnan(corr) else 0.0
    else:
        features["bm25_faiss_corr"] = 0.0
    
    # 2. Score dispersion (std of max scores across methods)
    bm25_maxes = [
        float(max(bm25_all_scores.get(p, [0.0])))
        for p in PIPELINE_CLASSES
    ]
    faiss_maxes = [
        float(max(faiss_all_scores.get(p, [0.0])))
        for p in PIPELINE_CLASSES
    ]
    all_maxes = list(bm25_maxes) + list(faiss_maxes)
    features["score_dispersion"] = float(np.std(all_maxes))
    
    # Per-pipeline means (normalized within each method's own scale)
    bm25_pipeline_means = {
        p: float(np.mean(bm25_all_scores[p])) if bm25_all_scores.get(p) else 0.0
        for p in PIPELINE_CLASSES
    }
    faiss_pipeline_means = {
        p: float(np.mean(faiss_all_scores[p])) if faiss_all_scores.get(p) else 0.0
        for p in PIPELINE_CLASSES
    }
    bm25_global_mean = float(np.mean(list(bm25_pipeline_means.values())))
    faiss_global_mean = float(np.mean(list(faiss_pipeline_means.values())))

    # 3. Top-K consensus: pipelines scoring above their method's global mean in BOTH BM25 and FAISS
    features["top_k_consensus"] = float(sum(
        1 for p in PIPELINE_CLASSES
        if bm25_pipeline_means[p] > bm25_global_mean and faiss_pipeline_means[p] > faiss_global_mean
    ))

    # 4. Distribution overlap: fraction of pipelines where BM25 and FAISS agree on high/low
    # Uses within-method normalisation to avoid cross-scale comparison (BM25 0-20, FAISS 0-1)
    features["distribution_overlap"] = float(sum(
        1 for p in PIPELINE_CLASSES
        if (bm25_pipeline_means[p] > bm25_global_mean) == (faiss_pipeline_means[p] > faiss_global_mean)
    )) / len(PIPELINE_CLASSES)
    
    return features


def extract_features(
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
    query: str,
    matched_rules: List[dict] = None,
    target_pipeline: str = "",
    relevance: float = 0.0,
    # New parameters for distribution features
    bm25_all_scores: Dict[str, List[float]] = None,
    faiss_all_scores: Dict[str, List[float]] = None,
) -> QueryFeatures:
    """
    Extract all features for a single (query, pipeline) pair.

    This is the main entry point for feature extraction. It combines
    query-level, pipeline-level, and cross-pipeline features.

    Args:
        bm25_scores: Raw BM25 pipeline scores
        faiss_scores: Raw FAISS pipeline scores
        rule_scores: Raw rule-based pipeline scores
        query: Original query string
        matched_rules: List of matched rule details
        target_pipeline: The pipeline this sample is for (for training)
        relevance: Relevance label (0-3, for training)
        bm25_all_scores: All BM25 scores per pipeline (for distribution features)
        faiss_all_scores: All FAISS scores per pipeline (for distribution features)

    Returns:
        QueryFeatures: Complete feature vector
    """
    if matched_rules is None:
        matched_rules = []

    # Extract each category of features
    query_feats = _extract_query_features(query, bm25_scores, faiss_scores, rule_scores)
    pipeline_feats = _extract_pipeline_features(target_pipeline, bm25_scores, faiss_scores, rule_scores)
    cross_feats = _extract_cross_pipeline_features(bm25_scores, faiss_scores, rule_scores, target_pipeline)
    
    # Extract distribution features (if full scores are available)
    dist_feats = {}
    cross_dist_feats = {}
    if bm25_all_scores and faiss_all_scores:
        dist_feats = _extract_distribution_features(bm25_all_scores, faiss_all_scores)
        cross_dist_feats = _extract_cross_pipeline_distribution_features(bm25_all_scores, faiss_all_scores)

    # Build QueryFeatures
    features = QueryFeatures(
        # Query-level
        query_length=query_feats["query_length"],
        query_word_count=query_feats["query_word_count"],
        query_has_question=query_feats["query_has_question"],
        query_has_time_expr=query_feats["query_has_time_expr"],
        query_has_relation_word=query_feats["query_has_relation_word"],
        query_has_list_word=query_feats["query_has_list_word"],
        query_has_visual_word=query_feats["query_has_visual_word"],
        query_complexity=query_feats["query_complexity"],
        bm25_max=query_feats["bm25_max"],
        faiss_max=query_feats["faiss_max"],
        rule_max=query_feats["rule_max"],
        num_matched_rules=float(len(matched_rules)),

        # Pipeline-level
        bm25_score={target_pipeline: pipeline_feats["bm25_score"]},
        faiss_score={target_pipeline: pipeline_feats["faiss_score"]},
        rule_score={target_pipeline: pipeline_feats["rule_score"]},
        bm25_rank={target_pipeline: pipeline_feats["bm25_rank"]},
        faiss_rank={target_pipeline: pipeline_feats["faiss_rank"]},
        rule_rank={target_pipeline: pipeline_feats["rule_rank"]},
        bm25_top1={target_pipeline: pipeline_feats["bm25_top1"]},
        faiss_top1={target_pipeline: pipeline_feats["faiss_top1"]},
        rule_top1={target_pipeline: pipeline_feats["rule_top1"]},
        bm25_top3_ratio={target_pipeline: pipeline_feats["bm25_top3_ratio"]},
        faiss_top3_ratio={target_pipeline: pipeline_feats["faiss_top3_ratio"]},
        rule_top3_ratio={target_pipeline: pipeline_feats["rule_top3_ratio"]},
        bm25_gap={target_pipeline: pipeline_feats["bm25_gap"]},
        faiss_gap={target_pipeline: pipeline_feats["faiss_gap"]},
        rule_gap={target_pipeline: pipeline_feats["rule_gap"]},
        bm25_rel={target_pipeline: pipeline_feats["bm25_rel"]},
        faiss_rel={target_pipeline: pipeline_feats["faiss_rel"]},
        rule_rel={target_pipeline: pipeline_feats["rule_rel"]},

        # Pipeline-level distribution features
        bm25_mean={k: v for k, v in dist_feats.items() if k.startswith("bm25_mean_")},
        bm25_std={k: v for k, v in dist_feats.items() if k.startswith("bm25_std_")},
        bm25_max_mean_ratio={k: v for k, v in dist_feats.items() if k.startswith("bm25_max_mean_ratio_")},
        faiss_mean={k: v for k, v in dist_feats.items() if k.startswith("faiss_mean_")},
        faiss_std={k: v for k, v in dist_feats.items() if k.startswith("faiss_std_")},
        faiss_score_density={k: v for k, v in dist_feats.items() if k.startswith("faiss_score_density_")},

        # Cross-pipeline
        agreement_top1=cross_feats["agreement_top1"],
        agreement_top3=cross_feats["agreement_top3"],
        score_consistency=cross_feats["score_consistency"],
        vote_count=cross_feats["vote_count"],
        max_min_score_gap=cross_feats["max_min_score_gap"],
        score_entropy=cross_feats["score_entropy"],
        is_consensus_winner=cross_feats["is_consensus_winner"],
        is_disagreement_candidate=cross_feats["is_disagreement_candidate"],
        rank_agreement=cross_feats["rank_agreement"],
        top1_confidence=cross_feats["top1_confidence"],
        top3_diversity=cross_feats["top3_diversity"],
        cross_method_agreement=cross_feats["cross_method_agreement"],
        bm25_faiss_corr=cross_dist_feats.get("bm25_faiss_corr", 0.0),
        score_dispersion=cross_dist_feats.get("score_dispersion", 0.0),
        top_k_consensus=cross_dist_feats.get("top_k_consensus", 0.0),
        distribution_overlap=cross_dist_feats.get("distribution_overlap", 0.0),

        # Internal
        _target_pipeline=target_pipeline,
        _relevance=relevance,
    )

    return features


def extract_features_for_all_pipelines(
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
    query: str,
    matched_rules: List[dict] = None,
    # New parameters for distribution features
    bm25_all_scores: Dict[str, List[float]] = None,
    faiss_all_scores: Dict[str, List[float]] = None,
) -> List[QueryFeatures]:
    """
    Extract features for all 4 pipelines (used during training).

    Args:
        bm25_scores: Raw BM25 pipeline scores
        faiss_scores: Raw FAISS pipeline scores
        rule_scores: Raw rule-based pipeline scores
        query: Original query string
        matched_rules: List of matched rule details
        bm25_all_scores: All BM25 scores per pipeline (for distribution features)
        faiss_all_scores: All FAISS scores per pipeline (for distribution features)

    Returns:
        List of QueryFeatures, one for each pipeline
    """
    if matched_rules is None:
        matched_rules = []

    return [
        extract_features(
            bm25_scores, faiss_scores, rule_scores, query,
            matched_rules, target_pipeline=pipeline, relevance=0.0,
            bm25_all_scores=bm25_all_scores,
            faiss_all_scores=faiss_all_scores,
        )
        for pipeline in PIPELINE_CLASSES
    ]


# ---------------------------------------------------------------------------
# LightGBM Ranker
# ---------------------------------------------------------------------------

class LightGBMRanker:
    """
    LightGBM-based pipeline ranker.

    Uses Learning-to-Rank (lambdarank / NDCG) to predict the ranking of pipelines.

    Usage:
        ranker = LightGBMRanker()

        # Prepare training data
        # Each query is expanded into 4 samples (one per pipeline)
        train_data = [
            {"features": features_vr, "relevance": 3},   # VideoRAG is perfect match
            {"features": features_gr, "relevance": 0},
            {"features": features_tk, "relevance": 0},
            {"features": features_rb, "relevance": 0},
            # ... more queries
        ]

        # Group sizes: each query has 4 samples
        group_sizes = [4] * num_queries

        # Train
        ranker.train(train_data, group_sizes)
        ranker.save("ranker.pkl")

        # Predict
        ranked = ranker.predict(features)
        # Returns: [("VideoRAG", 0.85), ("GraphRAG", 0.45), ...]
    """

    def __init__(
        self,
        model_path: str = None,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_child_samples: int = 10,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ):
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model = None
        self._loaded = False

        if model_path:
            self.load(model_path)

    def train(
        self,
        train_data: List[dict],
        group_sizes: List[int],
        relevance_gain: List[int] = None,
    ) -> None:
        """
        Train the LightGBM ranker.

        Args:
            train_data: List of {"features": QueryFeatures, "relevance": float}
            group_sizes: Number of samples per query (e.g., [4, 4, 4, ...])
            relevance_gain: Gain values for relevance levels (default: [0, 1, 2, 3])
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required for ranking. "
                "Install with: pip install lightgbm"
            )

        if len(train_data) == 0:
            raise ValueError("No training data provided")

        if len(train_data) != sum(group_sizes):
            raise ValueError(
                f"Train data ({len(train_data)}) != sum of group sizes ({sum(group_sizes)})"
            )

        # Extract features and labels
        X = np.array([item["features"].to_array() for item in train_data])
        y = np.array([item["relevance"] for item in train_data])

        feature_names = train_data[0]["features"]._get_feature_names()

        logger.info(
            f"Training LightGBM ranker with {len(X)} samples, "
            f"{X.shape[1]} features, {len(set(group_sizes))} unique group sizes"
        )

        train_data_obj = lgb.Dataset(X, label=y, group=group_sizes, feature_name=feature_names)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": relevance_gain if relevance_gain else [0, 1, 3, 7],
            "boosting_type": "gbdt",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "num_threads": 4,
            "verbose": -1,
            "seed": 42,
            "force_row_wise": True,
        }

        self.model = lgb.train(params, train_data_obj, num_boost_round=self.n_estimators)

        # Training evaluation
        pred_scores = self.model.predict(X)
        pred_labels = np.zeros_like(pred_scores)
        for i, group_size in enumerate(group_sizes):
            start = sum(group_sizes[:i])
            end = start + group_size
            group_scores = pred_scores[start:end]
            group_labels = y[start:end]
            top_idx = np.argmax(group_scores)
            pred_labels[start + top_idx] = group_labels[top_idx]

        accuracy = np.mean(pred_labels == y)
        logger.info(f"LightGBM ranker trained. Training accuracy: {accuracy:.4f}")

        self._loaded = True

    def predict(self, features_list: List[QueryFeatures]) -> List[Tuple[str, float]]:
        """
        Predict ranked list of pipelines.

        Args:
            features_list: One QueryFeatures per pipeline, in PIPELINE_CLASSES order.
                           lambdarank outputs one score per sample, so we stack
                           4 feature vectors -> (4, n_features) -> get (4,) scores.

        Returns:
            Ranked list of (pipeline, score) tuples, highest score first.
            e.g., [("VideoRAG", 0.85), ("GraphRAG", 0.45), ...]
        """
        if not self._loaded or self.model is None:
            logger.warning("Model not loaded, returning default ranking")
            return [(p, 0.25) for p in PIPELINE_CLASSES]

        X = np.array([f.to_array() for f in features_list])   # (4, n_features)
        raw = self.model.predict(X)                            # (4,) raw lambdarank scores

        # Softmax: convert raw ranking scores to probability-like values (0-1, sum=1)
        exp_s = np.exp(raw - raw.max())
        scores = exp_s / exp_s.sum()

        ranked = sorted(
            zip(PIPELINE_CLASSES, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def predict_top_k(
        self, features_list: List[QueryFeatures], k: int = 3
    ) -> List[Tuple[str, float]]:
        return self.predict(features_list)[:k]

    def predict_with_strategy(
        self,
        features_list: List[QueryFeatures],
        threshold: float = 0.15,
    ) -> Tuple[str, bool]:
        """Returns (top1_pipeline, need_llm_rerank)."""
        ranked = self.predict(features_list)
        top1_pipeline, top1_score = ranked[0]
        _, top2_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
        return (top1_pipeline, (top1_score - top2_score) < threshold)

    def save(self, path: str = None) -> None:
        """Save the ranker to disk."""
        path = path or self.model_path
        if not path:
            raise ValueError("Model path not specified")

        data = {
            "model": self.model,
            "feature_names": self._get_feature_names(),
            "params": {
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_child_samples": self.min_child_samples,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Ranker saved to {path}")

    def load(self, path: str = None) -> None:
        """Load the ranker from disk."""
        path = path or self.model_path
        if not path:
            raise ValueError("Model path not specified")

        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data.get("model")
        params = data.get("params", {})
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.num_leaves = params.get("num_leaves", self.num_leaves)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.max_depth = params.get("max_depth", self.max_depth)
        self.min_child_samples = params.get("min_child_samples", self.min_child_samples)
        self.reg_alpha = params.get("reg_alpha", self.reg_alpha)
        self.reg_lambda = params.get("reg_lambda", self.reg_lambda)

        self._loaded = True
        logger.info(f"Ranker loaded from {path}")

    def _get_feature_names(self) -> List[str]:
        """Get feature names (delegated to QueryFeatures)."""
        dummy = QueryFeatures()
        return dummy._get_feature_names()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def classify_with_lightgbm_ranker(
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
    ranker: LightGBMRanker,
    query: str = "",
    matched_rules: List[dict] = None,
    # New parameters for distribution features
    bm25_all_scores: Dict[str, List[float]] = None,
    faiss_all_scores: Dict[str, List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Rank pipelines using LightGBM ranker.

    Args:
        bm25_scores: Raw BM25 pipeline scores
        faiss_scores: Raw FAISS pipeline scores
        rule_scores: Raw rule-based pipeline scores
        ranker: Trained LightGBMRanker
        query: Original query string
        matched_rules: Matched rule details
        bm25_all_scores: All BM25 scores per pipeline (for distribution features)
        faiss_all_scores: All FAISS scores per pipeline (for distribution features)

    Returns:
        Ranked list of (pipeline, score) tuples
        e.g., [("VideoRAG", 0.85), ("GraphRAG", 0.45), ...]
    """
    if matched_rules is None:
        matched_rules = []

    # One QueryFeatures per pipeline -> (4, n_features) matrix inside predict()
    features_list = extract_features_for_all_pipelines(
        bm25_scores, faiss_scores, rule_scores, query, matched_rules,
        bm25_all_scores=bm25_all_scores,
        faiss_all_scores=faiss_all_scores,
    )

    ranked = ranker.predict(features_list)

    logger.debug(
        "LightGBM ranker: ranked=%s for query: %s...",
        [(p, round(s, 4)) for p, s in ranked[:3]],
        query[:50],
    )

    return ranked


def classify_with_fixed_rules(
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
) -> Tuple[str, float]:
    """
    Classify pipeline using fixed rules (fallback when LightGBM is not available).

    Strategy: Vote from BM25, FAISS, Rule top1 pipelines.
    If tie, use score magnitudes to decide.

    Returns:
        (pipeline_name, confidence)
    """
    # Get top1 from each source
    bm25_top1 = max(bm25_scores, key=bm25_scores.get) if bm25_scores else None
    faiss_top1 = max(faiss_scores, key=faiss_scores.get) if faiss_scores else None
    rule_top1 = max(rule_scores, key=rule_scores.get) if rule_scores else None

    # Count votes
    votes = {}
    scores = {}
    for p, s in [
        (bm25_top1, bm25_scores.get(bm25_top1, 0)) if bm25_top1 else None,
        (faiss_top1, faiss_scores.get(faiss_top1, 0)) if faiss_top1 else None,
        (rule_top1, rule_scores.get(rule_top1, 0)) if rule_top1 else None,
    ]:
        if p:
            votes[p] = votes.get(p, 0) + 1
            scores[p] = max(scores.get(p, 0), s)

    if not votes:
        return ("VideoRAG", 0.1)

    # Find winner
    winner = max(votes, key=votes.get)
    confidence = votes[winner] / 3.0   # Base confidence from vote ratio
    confidence = max(confidence, scores.get(winner, 0.0))   # At least as high as max score

    return (winner, confidence)
