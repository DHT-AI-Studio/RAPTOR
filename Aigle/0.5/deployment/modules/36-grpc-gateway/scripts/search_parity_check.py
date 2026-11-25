#!/usr/bin/env python3
"""Live REST-vs-gRPC search equivalence check against a DEPLOYED stack, with one real account.

For each query it asks the REST gateway (module 13, /api/0.5/search/*) and the gRPC gateway the
same question and compares what comes back:

  hybrid / bm25 / vector   REST /search/{hybrid,bm25,vector}   vs  Search / SearchBm25 / SearchVector
                           (the same hits, as a set; order and scores are reported)
  streams                  StreamSearch / StreamSearchBm25 / StreamSearchVector  vs  the unary call
  graphrag                 REST /search/graphrag vs GraphRagSearch (moments and graph nodes)
  tkg                      REST /search/tkg      vs TKGSearch      (facts and graph nodes)

An RPC the server does not have yet (UNIMPLEMENTED) is reported as SKIP, not a finding, so the
script can be run before a new build is deployed. Anything that differs is a FINDING and the exit
code is 1. It only reads: nothing is uploaded or changed.

This is the "REST and gRPC return equivalent hit sets" gate of VIE01-377 (an RG-8 sign-off
item). It needs the real stack, so it is not part of CI. Credentials only come from the
environment, never from a file or the command line:

    PARITY_USER=... PARITY_PW=... PARITY_GRPC=<host:port> PARITY_REST=<http://host:port> \\
        python scripts/search_parity_check.py

Also required: PARITY_GRPC (host:port of the gRPC gateway) and PARITY_REST (base URL of the REST
gateway) -- there are no defaults, so no deployment address lives in the repo.
Optional: PARITY_QUERIES (comma separated, default "人,黃仁勳,video"), PARITY_TOP_K (default 5).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import grpc
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generated"))
from raptor.v1 import raptor_pb2, raptor_pb2_grpc

GRPC_TARGET = os.environ.get("PARITY_GRPC", "")
REST = os.environ.get("PARITY_REST", "").rstrip("/")
QUERIES = [q.strip() for q in os.environ.get("PARITY_QUERIES", "人,黃仁勳,video").split(",") if q.strip()]
TOP_K = int(os.environ.get("PARITY_TOP_K", "5"))

_results: list[tuple[str, str, str]] = []  # (name, PASS|FINDING|SKIP, detail)


def record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    print(f"  {status:<7} {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def login(user: str, password: str) -> str:
    r = httpx.post(f"{REST}/api/0.5/sso/login", data={"username": user, "password": password}, timeout=20)
    r.raise_for_status()
    return r.json()["access_token"]


def rest_post(token: str, path: str, body: dict) -> dict:
    r = httpx.post(
        f"{REST}/api/0.5/search/{path}", json=body, headers={"Authorization": f"Bearer {token}"}, timeout=120
    )
    r.raise_for_status()
    return r.json()


def chunk_key(version_id: str, chunk_id: str) -> tuple[str, str]:
    return (version_id or "", chunk_id or "")


def rest_chunk_keys(data: dict) -> list[tuple[str, str]]:
    keys = []
    for hit in data.get("results") or []:
        payload = hit.get("payload") or {}
        keys.append(chunk_key(payload.get("version_id"), payload.get("chunk_id")))
    return keys


def grpc_chunk_keys(results) -> list[tuple[str, str]]:
    return [chunk_key(r.version_id, r.chunk_id) for r in results]


def compare_chunks(name: str, rest_keys: list, grpc_keys: list, rest_scores: list, grpc_scores: list, ask_rest_again=None) -> None:
    """The same hits is required. So are the same order and scores, unless REST itself gives a
    different order or scores when asked twice (a reranker can): then it is only a note."""
    only_rest = set(rest_keys) - set(grpc_keys)
    only_grpc = set(grpc_keys) - set(rest_keys)
    if only_rest or only_grpc:
        record(name, "FINDING", f"REST only {len(only_rest)}, gRPC only {len(only_grpc)} of {len(rest_keys)}/{len(grpc_keys)}")
        return
    problems = []
    if rest_keys != grpc_keys:
        problems.append("same hits, different order")
    if not all(math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-3) for a, b in zip(rest_scores, grpc_scores)):
        problems.append("scores differ")
    if not problems:
        record(name, "PASS", f"{len(rest_keys)} hits, same order and scores")
        return
    if ask_rest_again is not None:
        again_keys, again_scores = ask_rest_again()
        rest_stable = again_keys == rest_keys and all(
            math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-3) for a, b in zip(rest_scores, again_scores)
        )
        if not rest_stable:
            record(name, "PASS", f"{len(rest_keys)} hits; {', '.join(problems)}, but REST itself answers differently each time")
            return
    record(name, "FINDING", f"{len(rest_keys)} hits; {', '.join(problems)}")


def grpc_or_skip(name: str, call):
    """Runs a gRPC call; UNIMPLEMENTED means the server predates the RPC -> None and a SKIP."""
    try:
        return call()
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            record(name, "SKIP", "not deployed on this server yet")
            return None
        record(name, "FINDING", f"gRPC {e.code().name}: {e.details()}")
        return None


def check_chunk_searches(stub, token: str, meta, query: str) -> None:
    modes = [
        ("hybrid", "Search", "StreamSearch"),
        ("bm25", "SearchBm25", "StreamSearchBm25"),
        ("vector", "SearchVector", "StreamSearchVector"),
    ]
    request = raptor_pb2.SearchRequest(query=query, top_k=TOP_K)
    for rest_path, unary, streaming in modes:
        label = f"{query!r} {rest_path}"
        data = rest_post(token, rest_path, {"query": query, "top_k": TOP_K})
        reply = grpc_or_skip(f"{label}: {unary}", lambda unary=unary: getattr(stub, unary)(request, metadata=meta, timeout=120))
        if reply is None:
            continue
        def ask_rest_again(rest_path=rest_path):
            again = rest_post(token, rest_path, {"query": query, "top_k": TOP_K})
            return rest_chunk_keys(again), [h.get("score") or 0.0 for h in again.get("results") or []]

        compare_chunks(
            f"{label}: REST vs {unary}",
            rest_chunk_keys(data), grpc_chunk_keys(reply.results),
            [h.get("score") or 0.0 for h in data.get("results") or []], [r.score for r in reply.results],
            ask_rest_again,
        )
        streamed = grpc_or_skip(f"{label}: {streaming}", lambda streaming=streaming: list(getattr(stub, streaming)(request, metadata=meta, timeout=120)))
        if streamed is not None:
            same = [(r.version_id, r.chunk_id, r.score) for r in streamed] == [(r.version_id, r.chunk_id, r.score) for r in reply.results]
            record(f"{label}: {streaming} vs {unary}", "PASS" if same else "FINDING", f"{len(streamed)} items")


def node_ids(nodes) -> set[str]:
    return {str(n.get("id")) for n in nodes or [] if isinstance(n, dict) and n.get("id") is not None}


def check_graphrag(stub, token: str, meta, query: str) -> None:
    label = f"{query!r} graphrag"
    data = rest_post(token, "graphrag", {"query": query})
    reply = grpc_or_skip(f"{label}: GraphRagSearch", lambda: stub.GraphRagSearch(
        raptor_pb2.GraphRagSearchRequest(query=query), metadata=meta, timeout=120))
    if reply is None:
        return
    rest_moments = {chunk_key(m.get("version_id"), m.get("chunk_id")) for m in data.get("matched_moments") or []}
    grpc_moments = set(grpc_chunk_keys(reply.results))
    sub = json.loads(reply.subgraph_json)
    problems = []
    if rest_moments != grpc_moments:
        problems.append(f"moments REST {len(rest_moments)} vs gRPC {len(grpc_moments)}")
    if node_ids(data.get("nodes")) != node_ids(sub.get("nodes")):
        problems.append("graph nodes differ")
    if len(data.get("relationships") or []) != len(sub.get("edges") or []):
        problems.append("graph edges differ")
    record(f"{label}: REST vs GraphRagSearch", "FINDING" if problems else "PASS",
           "; ".join(problems) or f"{len(rest_moments)} moments, {len(node_ids(data.get('nodes')))} nodes")


def check_tkg(stub, token: str, meta, query: str) -> None:
    label = f"{query!r} tkg"
    data = rest_post(token, "tkg", {"query": query})
    reply = grpc_or_skip(f"{label}: TKGSearch", lambda: stub.TKGSearch(
        raptor_pb2.TKGSearchRequest(query=query), metadata=meta, timeout=120))
    if reply is None:
        return
    sub = json.loads(reply.subgraph_json)
    facts = data.get("temporal_facts") or []
    problems = []
    if len(facts) != len(reply.results):
        problems.append(f"facts REST {len(facts)} vs gRPC {len(reply.results)}")
    if node_ids(data.get("subgraph_nodes")) != node_ids(sub.get("nodes")):
        problems.append("graph nodes differ")
    if len(data.get("moment_ids") or []) != len(sub.get("moments") or []):
        problems.append("moments differ")
    record(f"{label}: REST vs TKGSearch", "FINDING" if problems else "PASS",
           "; ".join(problems) or f"{len(facts)} facts, {len(node_ids(data.get('subgraph_nodes')))} nodes")


def main() -> int:
    env = os.environ
    missing = [k for k in ("PARITY_USER", "PARITY_PW", "PARITY_GRPC", "PARITY_REST") if not env.get(k)]
    if missing:
        print(f"Set {', '.join(missing)} in the environment (see the docstring).", file=sys.stderr)
        return 2
    token = login(env["PARITY_USER"], env["PARITY_PW"])
    meta = (("authorization", f"Bearer {token}"),)
    channel = grpc.insecure_channel(GRPC_TARGET)
    stub = raptor_pb2_grpc.RaptorServiceStub(channel)
    print(f"REST {REST} vs gRPC {GRPC_TARGET}; queries {QUERIES}; top_k {TOP_K}")
    for query in QUERIES:
        check_chunk_searches(stub, token, meta, query)
        check_graphrag(stub, token, meta, query)
        check_tkg(stub, token, meta, query)
    channel.close()

    findings = [r for r in _results if r[1] == "FINDING"]
    skipped = [r for r in _results if r[1] == "SKIP"]
    passed = [r for r in _results if r[1] == "PASS"]
    print(f"\n{len(passed)} passed, {len(skipped)} skipped, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
