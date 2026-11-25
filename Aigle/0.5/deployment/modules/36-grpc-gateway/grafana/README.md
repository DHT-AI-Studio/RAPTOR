# Grafana dashboard for the gRPC gateway

`raptor-grpc-gateway.json` is a ready-to-import dashboard for the metrics the gateway
exposes on `:9825/metrics` (`interceptors/metrics.py`). It covers:

| Panel | Query basis |
| --- | --- |
| Active streams per RPC (and a total) | `grpc_server_active_streams` |
| Chunk throughput, bytes/s, sent and received | `grpc_server_stream_bytes_total` |
| RPC latency p50 / p95 / p99 | `grpc_server_handling_seconds_bucket` |
| Cancellation rate (per second and as a share) | `grpc_server_stream_cancellations_total`, `grpc_server_requests_total` |
| `RESOURCE_EXHAUSTED` rate (per second and as a share) | `grpc_server_requests_total{grpc_code="RESOURCE_EXHAUSTED"}` |
| Calls per second by status code, requests per second | `grpc_server_requests_total` |

An **RPC method** variable at the top filters every panel (default: all).

Two things to know when reading it:

- Latency is the whole call, so a streaming RPC shows seconds to minutes and a unary one milliseconds. The
  histogram's buckets run from 5 ms to 1 h (`LATENCY_BUCKETS` in `interceptors/metrics.py`); a call longer
  than 1 h is reported as 1 h. Before this was added the largest bucket was 10 s, which capped every streaming
  percentile at "10 s"; a gateway still running that older version shows the cap.
- A cancel is normal for a client that seeks in a player; the cancellation panels are for spotting a
  change, not for driving the number to zero.

## Get Prometheus to scrape the gateway

Module 14's `prometheus.yml` does not scrape the gateway yet. Add a job (the gateway container is
`raptor-grpc-gateway`, metrics port 9825):

```yaml
  - job_name: 'raptor-grpc-gateway'         # Module 36
    static_configs:
      - targets: ['raptor-grpc-gateway:9825']
    metrics_path: /metrics
```

That file belongs to module 14, so it is not changed here.

## Load the dashboard

- In Grafana: Dashboards, New, Import, upload `raptor-grpc-gateway.json`, choose the Prometheus data source.
- Or copy the file into module 14's `config/grafana/provisioning/dashboards/` (Grafana picks it up
  within 30 s). The JSON has no hard-coded data source, so it uses the default one, like the
  existing `raptor_03_overview.json`.

## What was verified (2026-10-21)

- All 14 queries were run through a real Prometheus 2.55 that scraped the deployed gateway
  (the internal test host, port 9825) while traffic was running: each parsed and returned data.
- Grafana 11.2 provisioned the file without errors (10 panels, the method variable) and ran a panel query
  through its data source.
- `tests/unit/test_grafana_dashboard.py` checks the structure and that every metric the queries name
  still exists in `interceptors/metrics.py`.

Not verified: how the panels look (no screenshot), and a Grafana other than 11.2.
