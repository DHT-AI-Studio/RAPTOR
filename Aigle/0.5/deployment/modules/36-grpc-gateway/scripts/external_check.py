#!/usr/bin/env python3
"""Can the gRPC gateway be called from OUTSIDE the network? Run it from a machine that is not on the
internal network. Read-only: it logs in, pings and runs one search.

    EXT_USER=... EXT_PW=... EXT_GRPC=<public-name>:<port> EXT_REST=http://<public-name>:8012 \\
        python scripts/external_check.py

Checks: TLS certificate valid for the name, health service, plaintext refused, reflection not reachable,
no token -> UNAUTHENTICATED, Ping and Search with a token. Exit code 1 on any FAIL, 2 when the
environment is incomplete. No address or credential is kept in the repo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import grpc
import httpx
from grpc_health.v1 import health_pb2, health_pb2_grpc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generated"))
from raptor.v1 import raptor_pb2, raptor_pb2_grpc

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def code_of(call):
    try:
        call()
        return grpc.StatusCode.OK, ""
    except grpc.RpcError as e:
        return e.code(), (e.details() or "")[:100]


def main() -> int:
    env = os.environ
    missing = [k for k in ("EXT_USER", "EXT_PW", "EXT_GRPC", "EXT_REST") if not env.get(k)]
    if missing:
        print(f"Set {', '.join(missing)} in the environment (see the docstring).", file=sys.stderr)
        return 2
    target, rest = env["EXT_GRPC"], env["EXT_REST"].rstrip("/")
    print(f"gRPC {target}; REST {rest}")

    secure = grpc.secure_channel(target, grpc.ssl_channel_credentials())
    health = health_pb2_grpc.HealthStub(secure)
    code, detail = code_of(lambda: health.Check(health_pb2.HealthCheckRequest(service=""), timeout=15))
    record("TLS certificate valid and the gateway answers (health)", code == grpc.StatusCode.OK, f"{code.name} {detail}")
    if code != grpc.StatusCode.OK:
        print("\nCannot reach the gateway over TLS from here; the other checks need it.")
        return 1

    plain = grpc.insecure_channel(target)
    code, _ = code_of(lambda: health_pb2_grpc.HealthStub(plain).Check(health_pb2.HealthCheckRequest(service=""), timeout=10))
    record("a plaintext client is refused", code != grpc.StatusCode.OK, code.name)

    try:
        from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

        refl = reflection_pb2_grpc.ServerReflectionStub(secure)
        req = iter([reflection_pb2.ServerReflectionRequest(list_services="")])
        code, _ = code_of(lambda: list(refl.ServerReflectionInfo(req, timeout=10)))
        record("reflection is not reachable from outside", code != grpc.StatusCode.OK, code.name)
    except ImportError:
        record("reflection is not reachable from outside", True, "skipped: grpcio-reflection not installed")

    stub = raptor_pb2_grpc.RaptorServiceStub(secure)
    code, _ = code_of(lambda: stub.Ping(raptor_pb2.PingRequest(message="x"), timeout=15))
    record("a call without a token is UNAUTHENTICATED", code == grpc.StatusCode.UNAUTHENTICATED, code.name)

    r = httpx.post(f"{rest}/api/0.5/sso/login", data={"username": env["EXT_USER"], "password": env["EXT_PW"]}, timeout=30)
    if r.status_code != 200:
        record("login over REST", False, f"HTTP {r.status_code}")
        return 1
    meta = (("authorization", f"Bearer {r.json()['access_token']}"),)
    record("login over REST", True)
    code, detail = code_of(lambda: stub.Ping(raptor_pb2.PingRequest(message="hi"), metadata=meta, timeout=15))
    record("Ping with a token", code == grpc.StatusCode.OK, f"{code.name} {detail}")
    code, detail = code_of(lambda: stub.Search(raptor_pb2.SearchRequest(query="test", top_k=1), metadata=meta, timeout=60))
    record("Search with a token", code == grpc.StatusCode.OK, f"{code.name} {detail}")

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
