#!/usr/bin/env python3
"""Measures the gateway process's resident memory while it moves more than 1 GB.

VIE01-160/161 ask that server memory stays constant however big the file is. This samples
VmRSS from /proc/<pid>/status every 50 ms while a client runs, against a gateway started with
the fake asset service from fake_asset.py:

  download (fast consumer) | 3 concurrent downloads | slow consumer, cancelled
  | 20 cancelled downloads | upload | 10 cancelled uploads

and prints, per scenario, the RSS at start, the maximum, and the maximum in each third of the
run (flat thirds = no growth with the amount transferred). See README.md for how to start the
gateway; it must use the same --size-mib as the fake service.

    python measure.py --pid <gateway pid> --size-mib 1200
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "generated"))
from raptor.v1 import raptor_pb2, raptor_pb2_grpc

_AUTH = (("x-api-key", "memtest-key"),)
_STAGING = os.path.join(tempfile.gettempdir(), "raptor-upload-*")


def _status_mib(pid: int, field: str) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"{field} not found for pid {pid}")


class Sampler:
    def __init__(self, pid: int):
        self.pid = pid
        self.rows: list[float] = []
        self._stop = threading.Event()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.rows.append(_status_mib(self.pid, "VmRSS"))
            time.sleep(0.05)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _report(name: str, sampler: Sampler, moved: int, seconds: float, extra: str = "") -> None:
    rows, n = sampler.rows, len(sampler.rows)
    thirds = "/".join(f"{max(rows[i * n // 3:(i + 1) * n // 3] or rows):.0f}" for i in range(3))
    print(
        f"{name:36} moved {moved / 2**20:6.0f} MiB in {seconds:5.1f}s | RSS start {rows[0]:6.1f} MiB, "
        f"max {max(rows):6.1f}, end {rows[-1]:6.1f} | max per third {thirds} MiB {extra}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", type=int, required=True, help="process id of the gateway under test")
    parser.add_argument("--size-mib", type=int, default=1200, help="size of the fake service's blob")
    parser.add_argument("--target", default="127.0.0.1:18825")
    args = parser.parse_args()
    size = args.size_mib << 20
    stub = raptor_pb2_grpc.RaptorServiceStub(
        grpc.insecure_channel(args.target, options=[("grpc.max_receive_message_length", 16 << 20)])
    )
    request = raptor_pb2.DownloadAssetRequest(asset_path="video/mp4/memtest", version_id="v1", preferred_chunk_size=1 << 20)

    def download(slow: float = 0.0, stop_after: int = 0) -> int:
        received = 0
        call = stub.DownloadAsset(request, metadata=_AUTH, timeout=600)
        for message in call:
            if message.HasField("chunk"):
                received += len(message.chunk.data)
                if slow:
                    time.sleep(slow)
                if stop_after and received >= stop_after:
                    call.cancel()
                    break
        return received

    def upload_messages(total: int):
        block = os.urandom(1 << 20)
        yield raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(
            filename="big.mp4", content_type="video/mp4", total_size=size if total == size else 0))
        sent = 0
        while sent < total:
            yield raptor_pb2.UploadAssetRequest(chunk_data=block)
            sent += len(block)

    print(f"gateway pid {args.pid}: idle RSS {_status_mib(args.pid, 'VmRSS'):.1f} MiB; blob {args.size_mib} MiB")
    failures = 0

    with Sampler(args.pid) as sampler:
        start = time.perf_counter()
        got = download()
        seconds = time.perf_counter() - start
    _report("download, fast consumer", sampler, got, seconds, f"| complete: {got == size}")
    failures += got != size

    with Sampler(args.pid) as sampler:
        start = time.perf_counter()
        results = [0, 0, 0]

        def worker(i: int) -> None:
            results[i] = download()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        seconds = time.perf_counter() - start
    _report("3 concurrent downloads", sampler, sum(results), seconds, f"| all complete: {all(r == size for r in results)}")
    failures += any(r != size for r in results)

    with Sampler(args.pid) as sampler:
        start = time.perf_counter()
        got = download(slow=0.05, stop_after=100 << 20)
        seconds = time.perf_counter() - start
    _report("slow consumer (50 ms), cancelled", sampler, got, seconds)

    with Sampler(args.pid) as sampler:
        start = time.perf_counter()
        for _ in range(20):
            download(stop_after=8 << 20)
        seconds = time.perf_counter() - start
    _report("20 cancelled downloads", sampler, 20 * (8 << 20), seconds)

    staged_before = len(glob.glob(_STAGING))
    peak_staged = [0]
    watching = threading.Event()

    def watch_staging() -> None:
        while not watching.is_set():
            for path in glob.glob(_STAGING):
                try:
                    peak_staged[0] = max(peak_staged[0], os.path.getsize(path))
                except OSError:
                    pass
            time.sleep(0.1)

    with Sampler(args.pid) as sampler:
        watcher = threading.Thread(target=watch_staging)
        watcher.start()
        start = time.perf_counter()
        reply = stub.UploadAsset(upload_messages(size), metadata=_AUTH, timeout=900)
        seconds = time.perf_counter() - start
        watching.set()
        watcher.join()
    _report("upload", sampler, reply.bytes_received, seconds,
            f"| complete: {reply.bytes_received == size}; staging file peaked at {peak_staged[0] >> 20} MiB")
    failures += reply.bytes_received != size

    with Sampler(args.pid) as sampler:
        start = time.perf_counter()
        for _ in range(10):
            future = stub.UploadAsset.future(upload_messages(64 << 20), metadata=_AUTH, timeout=60)
            time.sleep(0.15)
            future.cancel()
            time.sleep(0.2)
        time.sleep(2)
        seconds = time.perf_counter() - start
    _report("10 cancelled uploads", sampler, 0, seconds)

    leftover = len(glob.glob(_STAGING)) - staged_before
    print(f"staging files left behind: {leftover}; final RSS {_status_mib(args.pid, 'VmRSS'):.1f} MiB, "
          f"process peak {_status_mib(args.pid, 'VmHWM'):.1f} MiB")
    failures += leftover != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
