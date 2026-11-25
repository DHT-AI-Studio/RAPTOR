"""Prometheus metrics for session queues and service capacity."""
from __future__ import annotations

import bisect
from collections.abc import Iterator
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
    Metric,
)

if TYPE_CHECKING:
    from app.stream import StreamHandler

COUNTERS = {
    "audio_dropped": "Audio chunks dropped because the audio queue was full (QUEUE_OVERFLOW)",
    "stt_overloaded": "Utterances dropped because the STT queue was full (STT_OVERLOADED)",
    "stt_failed": "Utterances where the STT backend raised (WARNING stt_failed)",
    "llm_timeouts": "Voice turns where Module 15 timed out (LLM_TIMEOUT)",
    "response_failed": "Voice turns where speaking the answer raised (WARNING response_failed)",
    "barge_ins": "Answers interrupted by the user (BARGE_IN)",
    "resumes": "Sessions resumed on a new connection (RESUMED)",
    "resume_rejected": "RESUME requests answered SESSION_EXPIRED",
    "sessions_expired": "Dropped sessions not resumed within RT_SESSION_RESUME_WINDOW_S",
    "pong_timeouts": "Connections dropped for a missing PONG",
    "guardrail_blocks": "Transcripts or answer sentences blocked by Module 23 (GUARDRAIL_BLOCK)",
}

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf"))


class LatencyHistogram:
    """Fixed-bucket histogram, cheap to update per observation (``Counter``-sized state, no raw
    samples kept) -- mirrors the hand-rolled style of this module's other metrics rather than
    pulling in a second ``prometheus_client.Histogram`` bound to its own registry."""

    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS) -> None:
        self.buckets = buckets
        self._counts = [0] * len(buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, seconds: float) -> None:
        self.sum += seconds
        self.count += 1
        self._counts[bisect.bisect_left(self.buckets, seconds)] += 1

    def cumulative_counts(self) -> list[int]:
        out, running = [], 0
        for c in self._counts:
            running += c
            out.append(running)
        return out


class _Collector:
    def __init__(self, handler: StreamHandler) -> None:
        self.h = handler

    def collect(self) -> Iterator[Metric]:
        conns = list(self.h.live.values())
        sessions = GaugeMetricFamily("rt_sessions", "Sessions held by this process", labels=["connection"])
        suspended = sum(1 for c in conns if c.suspended)
        sessions.add_metric(["connected"], len(conns) - suspended)
        sessions.add_metric(["suspended"], suspended)
        yield sessions
        yield GaugeMetricFamily("rt_audio_queue_chunks", "Audio chunks waiting for VAD",
                                value=sum(c.audio_q.qsize() for c in conns if c.audio_q is not None))
        yield GaugeMetricFamily("rt_stt_queue_utterances", "Utterances waiting for a final transcript",
                                value=sum(c.utt_q.qsize() for c in conns if c.utt_q is not None))
        yield GaugeMetricFamily("rt_tts_queue_sentences", "Answer sentences waiting for synthesis",
                                value=sum(c._sentences.depth for c in conns if c._sentences is not None))
        yield GaugeMetricFamily("rt_output_queue_ms", "Synthesized answer audio not yet sent (ms)",
                                value=sum(c.outbox.unsent_ms for c in conns))
        for name, svc in (("stt", self.h.stt), ("tts", self.h.tts)):
            yield GaugeMetricFamily(f"rt_{name}_slots", f"{name.upper()} concurrency limit",
                                    value=svc.slots if svc is not None else 0)
            yield GaugeMetricFamily(f"rt_{name}_slots_busy", f"{name.upper()} slots in use",
                                    value=svc.busy if svc is not None else 0)
            yield GaugeMetricFamily(f"rt_{name}_waiting", f"{name.upper()} requests waiting for a slot",
                                    value=svc.waiting if svc is not None else 0)
        for key, doc in COUNTERS.items():
            yield CounterMetricFamily(f"rt_{key}", doc, value=self.h.stats[key])
        hist = self.h.stt_latency
        buckets = [("+Inf" if b == float("inf") else str(b), c)
                   for b, c in zip(hist.buckets, hist.cumulative_counts())]
        yield HistogramMetricFamily("rt_stt_finalize_latency_seconds",
                                    "STT finalize() latency per utterance, decode start to TRANSCRIPT_FINAL",
                                    buckets=buckets, sum_value=hist.sum)


def registry(handler: StreamHandler) -> CollectorRegistry:
    reg = CollectorRegistry(auto_describe=False)
    reg.register(_Collector(handler))
    return reg


def render(reg: CollectorRegistry) -> bytes:
    return generate_latest(reg)
