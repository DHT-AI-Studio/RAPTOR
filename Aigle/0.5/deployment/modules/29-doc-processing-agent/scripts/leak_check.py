#!/usr/bin/env python3
"""DA-09a temp-file leak check: success + failure requests, verify temp dirs clean."""
from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import sys
from collections import Counter

import httpx

BASE_URL = os.getenv("DA_BASE_URL", "http://localhost:8290")
FIXTURE = pathlib.Path(__file__).parent.parent / "tests/fixtures/scope3_emissions.pdf"
CONTAINERS = os.getenv(
    "DA_LEAK_CONTAINERS",
    "raptor-doc-processing-agent-0.5,raptor-document-analysis-0.5",
).split(",")

TMP_GLOB = os.getenv("DA_TMP_GLOB", "/tmp/media_processing/document_processing/sync_*")  # nosec B108 -- read-only glob count via docker exec, never written/opened here
CONCURRENCY = int(os.getenv("DA_LEAK_CONCURRENCY", "3"))
TIMEOUT = float(os.getenv("DA_TIMEOUT", "120"))
TOTAL = 100


def count_tmp_files() -> dict[str, int]:
    counts = {}
    for c in CONTAINERS:
        out = subprocess.run(
            ["docker", "exec", c, "sh", "-c", f"ls -d {TMP_GLOB} 2>/dev/null | wc -l"],
            check=False, capture_output=True, text=True,
        )
        counts[c] = int(out.stdout.strip() or 0) if out.returncode == 0 else -1
    return counts


async def run_request(client, sem, idx, should_fail, payload):
    async with sem:
        try:
            if should_fail:
                files = {"file": ("bad.xyz", b"not a real file", "application/octet-stream")}
            else:
                files = {"file": (FIXTURE.name, payload, "application/pdf")}
            r = await client.post(
                f"{BASE_URL}/api/v1/docagent/process",
                data={"task": "summarize"}, files=files, timeout=TIMEOUT,
            )
            return ("fail" if should_fail else "ok", r.status_code)
        except Exception as exc:  # noqa: BLE001
            return ("fail" if should_fail else "ok", f"{type(exc).__name__}")


async def wait_until_stable(timeout=60):
    """Wait until temp file counts stop changing (in-flight work finished)."""
    last, stable = None, 0
    for _ in range(timeout):
        cur = count_tmp_files()
        stable = stable + 1 if cur == last else 0
        if stable >= 3:
            return cur
        last = cur
        await asyncio.sleep(1)
    return last


async def main():
    if not FIXTURE.exists():
        sys.exit(f"ERROR: fixture not found: {FIXTURE}")
    payload = FIXTURE.read_bytes()

    before = count_tmp_files()
    if any(v < 0 for v in before.values()):
        sys.exit(f"ERROR: cannot docker exec into {before}")
    print(f"Temp files before: {before}")
    print(f"{TOTAL} requests, concurrency={CONCURRENCY}, target={BASE_URL}")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[
            run_request(client, sem, i, i >= TOTAL // 2, payload) for i in range(TOTAL)
        ])

    summary = Counter(results)
    for (kind, status), n in sorted(summary.items(), key=str):
        print(f"  {kind:<5} -> {status}: {n}")

    after = await wait_until_stable()
    leaked = {c: after[c] - before[c] for c in CONTAINERS}
    print(f"Temp files after:  {after}")
    print(f"Leaked:            {leaked}")

    bad_ok = [s for (k, s), n in summary.items() if k == "ok" and s != 200]
    unexpected = [s for (k, s), n in summary.items()
                  if k == "fail" and not (isinstance(s, int) and 400 <= s < 500)]
    ok = not any(v > 0 for v in leaked.values()) and not bad_ok and not unexpected
    if bad_ok:
        print(f"✖ success requests with non-200: {bad_ok}")
    if unexpected:
        print(f"✖ failure requests without 4xx: {unexpected}")
    print(f"Leak check: {'✔ PASS' if ok else '✖ FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
