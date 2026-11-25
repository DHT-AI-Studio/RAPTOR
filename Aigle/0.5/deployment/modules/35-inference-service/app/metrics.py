"""Module 35 custom Prometheus metrics — INF-4.

Defined at module level (not inside create_app()) to avoid duplicate-timeseries
errors when tests spin up multiple throwaway FastAPI apps in the same process.
Same pattern as the Instrumentator() call at the bottom of main.py.

Usage in routers:
    from app.metrics import (
        inf_request_duration_seconds,
        inf_request_errors_total,
        inf_gpu_memory_used_bytes,
    )
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Histogram: inference latency, labelled by task and model.
# Buckets cover fast LLM completions (0.1 s) through long TTS/clone jobs (120 s).
inf_request_duration_seconds = Histogram(
    "inf_request_duration_seconds",
    "End-to-end inference request duration in seconds",
    labelnames=["task", "model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Counter: total inference errors, labelled by task and error_type.
# error_type mirrors the ServiceError code strings already used in errors.py
# (e.g. "upstream_error", "model_not_found", "guardrail_blocked", "unknown").
inf_request_errors_total = Counter(
    "inf_request_errors_total",
    "Total number of failed inference requests",
    labelnames=["task", "error_type"],
)

# Gauge: GPU memory currently allocated by PyTorch (bytes).
# Updated after every voice inference call via torch.cuda.memory_allocated().
# Stays at 0 on text-only deployments where torch / CUDA is unavailable.
inf_gpu_memory_used_bytes = Gauge(
    "inf_gpu_memory_used_bytes",
    "GPU memory currently allocated by PyTorch (bytes), updated per voice inference",
)

# Gauge: number of voice tasks currently holding the GPU semaphore.
# Incremented on acquire in GpuLock.run(), decremented in the done callback.
inf_voice_gpu_active_tasks = Gauge(
    "inf_voice_gpu_active_tasks",
    "Number of voice inference tasks currently holding the GPU lock",
)
