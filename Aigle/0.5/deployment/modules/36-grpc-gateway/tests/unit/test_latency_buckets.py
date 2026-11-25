"""A long stream must land in a bucket that tells its duration, not in +Inf.

The latency histogram used Prometheus' default buckets (largest 10 s), so for DownloadAsset,
UploadAsset, StreamVideoSegment and the other streaming RPCs, which often run longer, every
percentile in the Grafana panel was capped at 10 s.
"""
from __future__ import annotations

import pytest
from interceptors.metrics import LATENCY_BUCKETS, LATENCY_HISTOGRAM
from prometheus_client import REGISTRY

_METHOD = "/test.Service/LongStream"


def _bucket(le: str) -> float:
    return REGISTRY.get_sample_value("grpc_server_handling_seconds_bucket", {"grpc_method": _METHOD, "le": le}) or 0.0


def test_buckets_are_increasing_and_reach_the_hour_the_proxies_allow():
    assert list(LATENCY_BUCKETS) == sorted(set(LATENCY_BUCKETS))
    assert LATENCY_BUCKETS[0] <= 0.005
    assert LATENCY_BUCKETS[-1] >= 3600


def test_the_fast_buckets_that_unary_rpcs_use_are_kept():
    for fast in (0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0):
        assert fast in LATENCY_BUCKETS


@pytest.mark.parametrize(("seconds", "first_bucket_that_holds_it"), [(45.0, "60.0"), (200.0, "300.0"), (1500.0, "1800.0")])
def test_a_long_call_is_counted_in_a_bucket_that_tells_its_duration(seconds, first_bucket_that_holds_it):
    before = {le: _bucket(le) for le in ("10.0", "30.0", first_bucket_that_holds_it, "+Inf")}
    LATENCY_HISTOGRAM.labels(grpc_method=_METHOD).observe(seconds)

    assert _bucket("10.0") == before["10.0"], "a call over 10 s was counted as finishing within 10 s"
    assert _bucket(first_bucket_that_holds_it) == before[first_bucket_that_holds_it] + 1
    assert _bucket("+Inf") == before["+Inf"] + 1
