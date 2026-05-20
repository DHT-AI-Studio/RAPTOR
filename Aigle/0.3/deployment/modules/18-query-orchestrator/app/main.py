from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.services.intent_classifier import IntentClassifierService
from app.routers.query import router as classify_router

_logger = logging.getLogger(__name__)

# Secondary relevance (1 = related) for non-target pipelines.
# Reflects semantic overlap between pipeline types; all other pairs default to 0.
_SECONDARY_RELEVANCE: dict[tuple[str, str], int] = {
    ("RDBMS",    "TKG"):      1,
    ("GraphRAG", "TKG"):      1,
    ("TKG",      "GraphRAG"): 1,
}


def _configure_logging(log_level: str) -> None:
    tz_tw = timezone(timedelta(hours=8))

    class _TZFmt(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=tz_tw)
            return dt.strftime(datefmt) if datefmt else dt.isoformat()

    fmt = _TZFmt("%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logging.root.handlers = [handler]
    logging.root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def _train_lightgbm_if_needed() -> None:
    """
    API 啟動時自動訓練 LightGBM Ranker（僅在模型檔案不存在時執行）。

    流程：
      1. 讀取 app/data/training_data.json（{"intent": ..., "query": ...} 格式）
      2. 初始化 BM25Classifier + FAISSClassifier，對每個 query 計算真實分數
      3. 每個 query 展開為 4 筆樣本（每個 pipeline 一筆），relevance=3 給正確
         pipeline，其他依語義相關性給 0 或 1
      4. 以 group_sizes=[4,...] 格式訓練 LightGBMRanker（rank:xcgain / NDCG）
      5. 儲存模型
    """
    settings = get_settings()
    if not settings.use_lightgbm_fusion:
        _logger.info("LightGBM fusion is disabled, skipping auto-training")
        return

    model_path = Path(settings.lightgbm_model_path)
    if model_path.exists():
        _logger.info("LightGBM model already exists at %s, skipping training", model_path)
        return

    data_dir = Path(__file__).parent / "data"
    training_data_path = data_dir / "training_data.json"

    if not training_data_path.exists():
        _logger.warning(
            "training_data.json not found at %s, skipping LightGBM training", training_data_path
        )
        return

    _logger.info(
        "LightGBM model not found — auto-training from %s using real BM25/FAISS scores...",
        training_data_path,
    )

    try:
        from app.classifiers.lightgbm_fusion import LightGBMRanker, extract_features, PIPELINE_CLASSES
        from app.classifiers.bm25_classifier import BM25Classifier
        from app.classifiers.faiss_classifier import FAISSClassifier
        from app.data.intent_taxonomy import load_intent_taxonomy
        from app.services.intent_classifier import _rule_based_classify

        with open(training_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        _logger.info("Loaded %d training samples. Initializing BM25 + FAISS classifiers...", len(raw_data))

        taxonomy = load_intent_taxonomy()
        bm25 = BM25Classifier(taxonomy=taxonomy)
        faiss_clf = FAISSClassifier(taxonomy=taxonomy)

        _logger.info("Classifiers ready. Computing scores for all queries...")

        train_data: list[dict] = []
        group_sizes: list[int] = []
        skipped = 0
        log_interval = max(1, len(raw_data) // 10)

        for i, sample in enumerate(raw_data):
            query: str = sample["query"]
            target_pipeline: str = sample["intent"]

            # Compute real BM25 + FAISS + Rule scores
            try:
                bm25_scores: dict = bm25._max_pooling(bm25._compute_scores(query)) or {}
                # 新增：獲取所有 BM25 分數（按 pipeline 分組）
                bm25_all_scores: dict = bm25._group_scores_by_pipeline(bm25._compute_scores(query))


                faiss_result = faiss_clf.classify(query, k=3)
                faiss_scores: dict = {
                    c.target_pipeline: c.confidence
                    for c in faiss_result.intents if c.target_pipeline
                }

                faiss_all_scores: dict = faiss_clf.get_all_faiss_scores(query)
                rule_scores_sample, _ = _rule_based_classify(query)
            except Exception as e:
                _logger.warning("Score computation failed for sample %d (%r): %s", i, query[:40], e)
                skipped += 1
                continue

            # Expand to 4 samples with relevance labels
            query_samples = []
            for pipeline in PIPELINE_CLASSES:
                relevance = (
                    3 if pipeline == target_pipeline
                    else _SECONDARY_RELEVANCE.get((target_pipeline, pipeline), 0)
                )
                try:
                    features = extract_features(
                        bm25_scores, faiss_scores, rule_scores_sample,
                        query, [],
                        target_pipeline=pipeline,
                        relevance=float(relevance),
                        bm25_all_scores=bm25_all_scores,
                        faiss_all_scores=faiss_all_scores,
                    )
                    query_samples.append({"features": features, "relevance": relevance})
                except Exception as e:
                    _logger.warning(
                        "Feature extraction failed for sample %d pipeline=%s: %s", i, pipeline, e
                    )
                    break

            if len(query_samples) == 4:
                train_data.extend(query_samples)
                group_sizes.append(4)
            else:
                skipped += 1

            if (i + 1) % log_interval == 0:
                _logger.info(
                    "Progress: %d/%d queries processed (%d skipped so far)",
                    i + 1, len(raw_data), skipped,
                )

        if not train_data:
            _logger.warning("No valid training data extracted, skipping LightGBM training")
            return

        _logger.info(
            "Feature extraction complete: %d queries (%d samples), %d skipped. Training Ranker...",
            len(group_sizes), len(train_data), skipped,
        )

        model = LightGBMRanker(
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=200,
            max_depth=6,
        )
        model.train(train_data, group_sizes)
        model.save(str(model_path))
        _logger.info("LightGBM ranker trained and saved to %s", model_path)

    except ImportError as e:
        _logger.warning(
            "Skipping LightGBM auto-training: %s. Install lightgbm to enable.", e
        )
    except Exception as e:
        _logger.error("LightGBM auto-training failed: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configure_logging(settings.log_level)

    # Auto-train LightGBM model if needed
    _train_lightgbm_if_needed()

    _logger.info("Initializing Intent Classifier (QIC Hybrid: BM25 + FAISS bge-m3)…")
    app.state.intent_classifier = IntentClassifierService()
    _logger.info("Intent Classifier ready (Tier1=QIC, Tier2=rule-based, Tier3=LightGBM-Ranker, Tier4=LLM)")
    yield
    _logger.info("Query Orchestrator shutdown complete")


app = FastAPI(title="Raptor Query Orchestrator", version="0.3", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)
app.include_router(classify_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "query-orchestrator"}


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception):
    _logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(exc)})
