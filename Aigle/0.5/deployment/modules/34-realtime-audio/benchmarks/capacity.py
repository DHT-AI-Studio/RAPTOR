"""Real concurrent external-brain load runner with a required protocol adapter."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import time
from pathlib import Path

from benchmarks.model_benchmark import load_manifest
from benchmarks.telephony import capacity_gate


async def run(driver, samples: list[dict], calls: int, utterances: int, config: dict) -> dict:
    """Hold every call open through a shared start barrier; failed sessions fail the run.

    driver(sample, call_id, config) must be an async context manager opening one
    real external-brain call; the yielded session has async turn(sample), which
    returns measured stage latencies under the supplied RTA-X8 metric names (other numeric
    keys it returns are summed into ``counters``).
    """
    barrier = asyncio.Event()
    ready = 0
    metrics = {stage: [] for stage in config["slos_ms"]}
    counters: dict[str, float] = {}
    failures = []

    async def one(index):
        nonlocal ready
        try:
            async with driver(samples[index % len(samples)], index, config) as session:
                ready += 1
                if ready == calls:
                    barrier.set()
                await asyncio.wait_for(barrier.wait(), timeout=config.get("start_timeout_s", 30))
                for iteration in range(utterances):
                    measured = await asyncio.wait_for(
                        session.turn(samples[(index + iteration) % len(samples)]),
                        timeout=config.get("turn_timeout_s", 60),
                    )
                    for stage, observations in metrics.items():
                        observations.append(float(measured[stage]))
                    for key, value in measured.items():
                        if key not in metrics:
                            counters[key] = counters.get(key, 0) + value
        except Exception as exc:  # noqa: BLE001 -- arbitrary external-brain driver/protocol adapter; failure recorded per-call, unblocks barrier
            failures.append({"call_id": index, "error": type(exc).__name__, "detail": str(exc)[:200]})
            barrier.set()

    started = time.perf_counter()
    await asyncio.gather(*(one(index) for index in range(calls)))
    return {"mode": "external-brain", "concurrent_calls": calls,
            "gpu_count": config["gpu_count"], "reference_gpu": config["reference_gpu"],
            "failures": len(failures), "errors": failures, "stage_latency_ms": metrics, "counters": counters,
            "wall_time_s": time.perf_counter() - started, "utterances_per_call": utterances}


def main() -> None:
    """No fake driver is supplied; the actual RTA-X11 protocol must be available."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("config", type=Path, help="actual slos_ms, gpu_count, reference_gpu, endpoint/auth configuration")
    parser.add_argument("--driver", required=True, help="module:async_context_manager for real RTA-X11")
    parser.add_argument("--calls", type=int, default=int(os.getenv("RT_TARGET_CONCURRENT_CALLS", "20")))
    parser.add_argument("--utterances", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calls < 1 or args.utterances < 20:
        raise ValueError("positive calls and >=20 utterances per call are required")
    manifest = load_manifest(args.manifest)
    if any(s.get("sample_rate") != 8000 or s.get("codec") not in {"PCMU", "PCMA"}
           for s in manifest["samples"]):
        raise ValueError("capacity requires the 8 kHz G.711 fixture set")
    config = json.loads(args.config.read_text())
    module, name = args.driver.split(":", 1)
    driver = getattr(importlib.import_module(module), name)
    evidence = asyncio.run(run(driver, manifest["samples"], args.calls, args.utterances, config))
    gate = capacity_gate([evidence], config["slos_ms"], args.calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"run": evidence, "gate": gate}, indent=2) + "\n")
    if gate["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
