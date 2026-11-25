#!/usr/bin/env python3
"""Live tenant-isolation check against a DEPLOYED gateway, with two ordinary accounts.

Account A uploads a small video and starts a job; account B then tries to reach A's data
over gRPC. Every attempt must be refused (NOT_FOUND or PERMISSION_DENIED) and never return
A's bytes, A's job or A's asset id. Anything else is printed as a FINDING and the exit code
is 1. Everything A uploaded is archived and deleted again at the end.

This is the evidence for the security-review item "asset ownership and branch/tenant
restrictions verified" (VIE01-165). It needs the real stack, so it is not part of CI.

Credentials only come from the environment, never from a file or the command line:

    ISO_USER_A=... ISO_PW_A=... ISO_USER_B=... ISO_PW_B=... \\
    ISO_GRPC=<host:port> ISO_REST=<http://host:port> python scripts/isolation_check.py

ISO_GRPC (the gRPC gateway) and ISO_REST (the REST gateway) are required: no deployment address is
kept in the repo. Optional: ISO_SKIP_JOB=1 to skip the job checks (they run the real video pipeline, about 30 s).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import grpc
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generated"))
from raptor.v1 import raptor_pb2, raptor_pb2_grpc

GRPC_TARGET = os.environ.get("ISO_GRPC", "")
REST = os.environ.get("ISO_REST", "").rstrip("/")
DENIED = {grpc.StatusCode.NOT_FOUND, grpc.StatusCode.PERMISSION_DENIED}

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FINDING'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def login(user: str, password: str) -> str:
    r = httpx.post(f"{REST}/api/0.5/sso/login", data={"username": user, "password": password}, timeout=20)
    r.raise_for_status()
    return r.json()["access_token"]


def make_video() -> bytes:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "iso.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
             "-pix_fmt", "yuv420p", str(out)],
            check=True,
        )
        return out.read_bytes()


def upload(stub, token: str, name: str, data: bytes, key: str = "") -> raptor_pb2.UploadAssetResponse:
    def msgs():
        yield raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(
            filename=name, content_type="video/mp4", total_size=len(data), idempotency_key=key))
        yield raptor_pb2.UploadAssetRequest(chunk_data=data)

    return stub.UploadAsset(msgs(), metadata=(("authorization", f"Bearer {token}"),), timeout=120)


def attempt(call):
    """Run a gRPC call to completion; return (status code, bytes seen)."""
    got = 0
    try:
        result = call()
        if hasattr(result, "__iter__") and not hasattr(result, "job_id"):
            for item in result:
                # DownloadAsset wraps MediaChunk in a oneof; StreamVideoSegment sends MediaChunk itself.
                chunk = item.chunk if type(item).__name__ == "DownloadAssetResponse" else item
                got += len(getattr(chunk, "data", b"") or b"")
        return grpc.StatusCode.OK, got, result
    except grpc.RpcError as exc:
        return exc.code(), got, None


def main() -> int:
    env = {k: os.environ.get(k) for k in ("ISO_USER_A", "ISO_PW_A", "ISO_USER_B", "ISO_PW_B", "ISO_GRPC", "ISO_REST")}
    if not all(env.values()):
        print("Set ISO_USER_A, ISO_PW_A, ISO_USER_B, ISO_PW_B, ISO_GRPC and ISO_REST in the environment.", file=sys.stderr)
        return 2

    tok_a = login(env["ISO_USER_A"], env["ISO_PW_A"])
    tok_b = login(env["ISO_USER_B"], env["ISO_PW_B"])
    auth_a = (("authorization", f"Bearer {tok_a}"),)
    auth_b = (("authorization", f"Bearer {tok_b}"),)
    channel = grpc.insecure_channel(GRPC_TARGET)
    stub = raptor_pb2_grpc.RaptorServiceStub(channel)

    created: list[tuple[str, str, str]] = []  # (asset_path, version_id, token)
    data = make_video()
    key = f"iso-{uuid.uuid4().hex[:10]}"
    try:
        print(f"Gateway {GRPC_TARGET}; A uploads {len(data)} bytes.")
        up_a = upload(stub, tok_a, f"iso_{uuid.uuid4().hex[:6]}.mp4", data, key)
        created.append((up_a.asset_id, up_a.version_id, tok_a))
        ref = raptor_pb2.AssetRef(asset_path=up_a.asset_id, version_id=up_a.version_id)

        # A can use its own asset (control: proves the denials below are not just a broken test).
        code, got, _ = attempt(lambda: stub.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path=ref.asset_path, version_id=ref.version_id),
            metadata=auth_a, timeout=60))
        record("control: A downloads its own asset", code == grpc.StatusCode.OK and got > 0, f"{code.name}")

        code, got, _ = attempt(lambda: stub.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path=ref.asset_path, version_id=ref.version_id),
            metadata=auth_b, timeout=60))
        record("B cannot download A's asset", code in DENIED and got == 0, f"{code.name}, {got} data bytes")

        code, got, _ = attempt(lambda: stub.StreamVideoSegment(
            raptor_pb2.StreamVideoSegmentRequest(asset=ref, start_time=0, end_time=1,
                                                 format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264),
            metadata=auth_b, timeout=60))
        record("B cannot stream a clip of A's video", code in DENIED and got == 0, f"{code.name}, {got} bytes")

        # The same idempotency key under another account must not reveal A's asset or block B.
        try:
            up_b = upload(stub, tok_b, f"iso_b_{uuid.uuid4().hex[:6]}.mp4", data, key)
            created.append((up_b.asset_id, up_b.version_id, tok_b))
            record("B reusing A's idempotency key makes B's own asset, not A's",
                   up_b.asset_id != up_a.asset_id, f"A={up_a.asset_id} B={up_b.asset_id}")
        except grpc.RpcError as exc:
            leaked = any(up_a.asset_id in str(v) for _, v in (exc.trailing_metadata() or ()))
            record("B reusing A's idempotency key makes B's own asset, not A's", False,
                   f"{exc.code().name}; A's id in trailing metadata: {leaked}")

        if os.environ.get("ISO_SKIP_JOB") == "1":
            print("  (job checks skipped)")
        else:
            code, _, _job = attempt(lambda: stub.SubmitJob(
                raptor_pb2.SubmitJobRequest(job_type="", input_asset_ids=[ref], request_id=f"iso-b-{key}"),
                metadata=auth_b, timeout=60))
            record("B cannot start a job on A's asset", code in DENIED, f"{code.name}")

            code, _, job_a = attempt(lambda: stub.SubmitJob(
                raptor_pb2.SubmitJobRequest(job_type="", input_asset_ids=[ref], request_id=f"iso-a-{key}"),
                metadata=auth_a, timeout=60))
            if code != grpc.StatusCode.OK:
                record("control: A starts a job on its own asset", False, code.name)
            else:
                record("control: A starts a job on its own asset", True, f"job {job_a.job_id}")
                code, _, _ = attempt(lambda: stub.GetJob(
                    raptor_pb2.GetJobRequest(job_id=job_a.job_id), metadata=auth_b, timeout=30))
                record("B cannot read A's job (GetJob)", code in DENIED, code.name)
                code, _, _ = attempt(lambda: stub.WatchJob(
                    raptor_pb2.WatchJobRequest(job_id=job_a.job_id), metadata=auth_b, timeout=30))
                record("B cannot watch A's job (WatchJob)", code in DENIED, code.name)
                # Let the job finish so cleanup does not race the pipeline.
                deadline = time.time() + 120
                while time.time() < deadline:
                    state = stub.GetJob(raptor_pb2.GetJobRequest(job_id=job_a.job_id), metadata=auth_a, timeout=30).state
                    if state in (raptor_pb2.JOB_STATE_COMPLETE, raptor_pb2.JOB_STATE_FAILED):
                        break
                    time.sleep(3)
    finally:
        print("Cleaning up what A and B uploaded...")
        for path, version, token in created:
            h = {"Authorization": f"Bearer {token}"}
            for step in ("unregister", "filearchive", "delfile"):
                try:
                    r = httpx.post(f"{REST}/api/0.5/asset/{step}/{path}/{version}", headers=h, timeout=30)
                    if r.status_code >= 400:
                        print(f"  cleanup {step} {path}: HTTP {r.status_code}")
                except httpx.HTTPError as exc:
                    print(f"  cleanup {step} {path}: {exc}")
        channel.close()

    bad = [r for r in _results if not r[1]]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} checks passed" + (f", {len(bad)} FINDING(S)" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
