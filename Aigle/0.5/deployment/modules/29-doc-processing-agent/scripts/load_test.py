#!/usr/bin/env python3
"""DA-09a load test: N concurrent summarize requests, report latency percentiles."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import httpx

BASE_URL = os.getenv("DA_BASE_URL", "http://localhost:8290")
FIXTURE = pathlib.Path(__file__).parent.parent / "tests/fixtures/scope3_emissions.pdf"
CONCURRENCY = int(os.getenv("DA_CONCURRENCY", "10"))
TIMEOUT = float(os.getenv("DA_TIMEOUT", "400"))
TASK = "summarize"
TARGET_P95_S = 15.0


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(pct / 100 * len(s)) - 1)]


async def single_request(client, idx, payload):
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}/api/v1/docagent/process",
            data={"task": TASK},
            files={"file": (FIXTURE.name, payload, "application/pdf")},
            timeout=TIMEOUT,
        )
        return {"idx": idx, "status": resp.status_code,
                "latency": time.perf_counter() - start, "ok": resp.status_code == 200}
    except Exception as exc:  # noqa: BLE001
        return {"idx": idx, "status": 0, "latency": time.perf_counter() - start,
                "ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def run_load_test() -> dict:
    if not FIXTURE.exists():
        sys.exit(f"ERROR: fixture not found: {FIXTURE}")
    payload = FIXTURE.read_bytes()
    fixture_hash = hashlib.sha256(payload).hexdigest()[:16]
    print(f"Fixture: {FIXTURE.name} sha256={fixture_hash}")
    print(f"Concurrency: {CONCURRENCY}  Task: {TASK}  Target: {BASE_URL}")

    async with httpx.AsyncClient() as client:
        warm = await single_request(client, -1, payload)  # warm-up, not counted
        print(f"Warm-up: status={warm['status']} {warm['latency']:.2f}s")
        t0 = time.perf_counter()
        results = await asyncio.gather(*[single_request(client, i, payload)
                                         for i in range(CONCURRENCY)])
        total = time.perf_counter() - t0

    ok_lat = [r["latency"] for r in results if r["ok"]]
    ok_count = len(ok_lat)
    errors = sorted({r.get("error", f"HTTP {r['status']}") for r in results if not r["ok"]})

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture": FIXTURE.name,
        "fixture_hash_sha256_prefix": fixture_hash,
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "concurrency": CONCURRENCY,
        "task": TASK,
        "client_timeout_s": TIMEOUT,
        "warmup_latency_s": round(warm["latency"], 3),
        "total_requests": CONCURRENCY,
        "ok": ok_count,
        "failed": CONCURRENCY - ok_count,
        "errors": errors,
        "latency_p50_s": round(statistics.median(ok_lat), 3) if ok_lat else None,
        "latency_p95_s": round(percentile(ok_lat, 95), 3) if ok_lat else None,
        "latency_max_s": round(max(ok_lat), 3) if ok_lat else None,
        "total_elapsed_s": round(total, 3),
        "p95_target_s": TARGET_P95_S,
    }
    report["p95_pass"] = bool(ok_lat) and ok_count == CONCURRENCY \
        and report["latency_p95_s"] < TARGET_P95_S

    print(f"\n{'=' * 50}")
    print(f"Results: {ok_count}/{CONCURRENCY} OK  errors={errors}")
    print(f"p50={report['latency_p50_s']}s  p95={report['latency_p95_s']}s  "
          f"max={report['latency_max_s']}s")
    print(f"p95 < {TARGET_P95_S:.0f}s (all OK): {'✔ PASS' if report['p95_pass'] else '✖ FAIL'}")

    out = pathlib.Path(__file__).parent.parent / f"performance_baseline_c{CONCURRENCY}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {out}")
    return report


if __name__ == "__main__":
    asyncio.run(run_load_test())
