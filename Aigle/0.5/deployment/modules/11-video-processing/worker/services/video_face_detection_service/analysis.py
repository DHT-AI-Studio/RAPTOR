# services/video_face_detection_service/analysis.py
# 單支影片的完整人臉處理（v1.4 §4.3 處理流程 1–7）：
#   1–4. 一次解碼：probe、鏡頭切點、8fps 取樣、偵測（pipeline.run_detection）
#   4.   偵測結果邊解碼邊餵給追蹤器，串成軌跡（tracker）
#   5.   每條軌跡用品質最好的幾幀算 embedding 再平均
#   6.   影片內分群：軌跡 → 群（clustering）
#   7.   代表幀與縮圖在追蹤時就已挑好；群的代表臉取成員中品質最好的
# 第 8 步（代表圖上傳 Module 04）由 kafka_handler 呼叫 asset_client 完成；
# 第 9 步（寫入 Module 25）由 kafka_handler 呼叫 personal_db_client 完成。
#
# trace_id / group_id 由影片版本與排序後的位置決定：同一支影片重跑得到相同 ID，
# 之後寫入 Module 25 時可以冪等重試。

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from clustering import TraceGroup, cluster_traces
from pipeline import DetectionResult, run_detection
from tracker import FaceTracker, Trace

logger = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.UUID("6f1c2d3e-8a4b-4c5d-9e0f-a1b2c3d4e5f6")


@dataclass
class FaceAnalysis:
    detection: DetectionResult
    traces: list[Trace] = field(default_factory=list)
    groups: list[TraceGroup] = field(default_factory=list)

    def summary(self) -> dict:
        """給 orchestrator 記錄用的摘要；不含逐幀資料與向量，避免訊息過大。"""
        return {
            **self.detection.summary(),
            "traces": len(self.traces),
            "groups": len(self.groups),
            "traces_per_group": [len(g.traces) for g in self.groups],
        }


def _stable_id(id_seed: str, kind: str, index: int) -> str:
    return uuid.uuid5(_ID_NAMESPACE, f"{id_seed}/{kind}/{index}").hex


def analyze_video(video_path: str, detector: Any, id_seed: str) -> FaceAnalysis:
    """detector 需提供 detect()、align()、embed_aligned()。id_seed 通常是 f"{asset_path}/{version_id}"。"""
    tracker = FaceTracker(aligner=detector.align)
    detection = run_detection(video_path, detector, on_frame=tracker.update)
    traces = tracker.finish()

    for i, trace in enumerate(traces):
        feats = detector.embed_aligned(trace.embed_crops)
        mean = feats.mean(axis=0)
        trace.embedding = mean / max(float(np.linalg.norm(mean)), 1e-12)
        trace.embed_crops = []          # 對齊後的臉只為了算 embedding，算完即釋放
        trace.trace_id = _stable_id(id_seed, "trace", i)

    groups = cluster_traces(traces)
    for i, group in enumerate(groups):
        group.group_id = _stable_id(id_seed, "group", i)

    analysis = FaceAnalysis(detection=detection, traces=traces, groups=groups)
    logger.info(f"Face analysis finished for {video_path}: {analysis.summary()}")
    return analysis
