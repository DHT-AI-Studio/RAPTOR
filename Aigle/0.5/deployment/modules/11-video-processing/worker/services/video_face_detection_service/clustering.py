# services/video_face_detection_service/clustering.py
# 影片內分群（v1.4 §4.3）：有條件的階層式合併，把同一支影片裡同一個人的多條軌跡歸成一群。
#
# 每條軌跡起初自成一群；每輪找出「相似度最高、且符合合併條件」的一對群合併，
# 重算群 embedding（成員軌跡依幀數加權平均），直到沒有任何一對符合條件。
# 合併條件依兩群的軌跡在時間上是否重疊（同框）而不同——這是軟限制，不是絕對禁止
# （子母畫面等情況下，同一人確實可能同時出現在畫面兩處）：
#   - 不重疊：相似度 ≥ FACE_CLUSTER_SIM 即可，不限時間間隔
#   - 重疊：  相似度 ≥ FACE_CLUSTER_SIM_OVERLAP，且每一對重疊軌跡的代表臉都要夠大、夠清楚
# 重疊檢查在「群」的層級做：兩群中任一對軌跡同框就算重疊，
# 所以 A、C 同框時，不會因為 A–B、B–C 各自不重疊而被串成同一群。

from dataclasses import dataclass, field

import numpy as np
from config import (
    FACE_CLUSTER_OVERLAP_MIN_CONF,
    FACE_CLUSTER_OVERLAP_MIN_PX,
    FACE_CLUSTER_SIM,
    FACE_CLUSTER_SIM_OVERLAP,
)


@dataclass
class TraceGroup:
    traces: list                                  # tracker.Trace
    embedding: np.ndarray = field(repr=False)
    frame_count: int = 0
    group_id: str | None = None

    @property
    def representative(self):
        """群的代表臉：成員軌跡中品質最好的代表幀。"""
        return max((t.representative for t in self.traces),
                   key=lambda r: (r.quality.score, r.quality.confidence))


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _group_of(traces: list) -> TraceGroup:
    counts = np.array([len(t.frames) for t in traces], dtype=np.float32)
    embedding = _normalize((np.stack([t.embedding for t in traces]) * counts[:, None]).sum(axis=0))
    return TraceGroup(traces=list(traces), embedding=embedding, frame_count=int(counts.sum()))


def _overlaps(a, b) -> bool:
    return a.start_sec <= b.end_sec and b.start_sec <= a.end_sec


def _clear_enough(trace) -> bool:
    rep = trace.representative
    return rep.min_side >= FACE_CLUSTER_OVERLAP_MIN_PX and rep.quality.confidence >= FACE_CLUSTER_OVERLAP_MIN_CONF


def _may_merge(a: TraceGroup, b: TraceGroup, similarity: float) -> bool:
    overlapping = [(ta, tb) for ta in a.traces for tb in b.traces if _overlaps(ta, tb)]
    if not overlapping:
        return similarity >= FACE_CLUSTER_SIM
    return similarity >= FACE_CLUSTER_SIM_OVERLAP and all(
        _clear_enough(ta) and _clear_enough(tb) for ta, tb in overlapping)


def cluster_traces(traces: list) -> list[TraceGroup]:
    """traces 需已有 embedding。回傳的群依最早出現時間排序。"""
    groups = [_group_of([t]) for t in traces]
    while len(groups) > 1:
        emb = np.stack([g.embedding for g in groups])
        sim = emb @ emb.T
        np.fill_diagonal(sim, -np.inf)
        merged = False
        # 由相似度高到低，找第一對符合條件的群
        order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
        for i, j in order:
            if i >= j:
                continue
            if sim[i, j] < FACE_CLUSTER_SIM:
                break   # 之後的配對連最寬鬆的門檻都達不到
            if _may_merge(groups[i], groups[j], float(sim[i, j])):
                combined = _group_of(groups[i].traces + groups[j].traces)
                groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [combined]
                merged = True
                break
        if not merged:
            break
    groups.sort(key=lambda g: min(t.start_sec for t in g.traces))
    return groups
