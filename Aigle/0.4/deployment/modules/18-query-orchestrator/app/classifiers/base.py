from abc import ABC, abstractmethod
from typing import List

from ..models.classification import ClassificationResult, Candidate
from ..core.config import get_settings


class BaseClassifier(ABC):
    """Base class for all pipeline classifiers.
    
    Subclasses must implement:
        - classify(): single query classification
        - batch_classify(): optional — a default implementation is provided below
    """

    @abstractmethod
    def classify(self, query: str, k: int = 1) -> ClassificationResult:
        pass

    def batch_classify(self, queries: List[str], k: int = 1) -> List[ClassificationResult]:
        """Default batch implementation — delegate to single classify."""
        return [self.classify(q, k) for q in queries]

    def _determine_intent_type(self, candidates: List[Candidate]) -> tuple:
        """Determine whether the query is single/multi/unknown intent.
        
        Returns (intent_type_list, intents).
        Fixed: uses c1 - gap_threshold for the high-confidence filter (was threshold in FAISS).
        """
        s = get_settings()
        threshold = s.DEFAULT_CONFIDENCE_THRESHOLD
        gap_threshold = s.GAP_THRESHOLD

        if len(candidates) == 1:
            if candidates[0].confidence < threshold:
                return (["unknown"], candidates)
            return ([candidates[0].target_pipeline or "unknown"], [candidates[0]])

        c1, c2 = candidates[0].confidence, candidates[1].confidence
        if c1 < threshold:
            return (["unknown"], candidates)
        if (c1 - c2) > gap_threshold:
            return ([candidates[0].target_pipeline or "unknown"], [candidates[0]])

        high = [c for c in candidates if c.confidence >= c1 - gap_threshold]
        pipelines = list(dict.fromkeys(c.target_pipeline or "unknown" for c in high))
        return (pipelines, high)
