# Changelog

## v0.5.0 (unreleased)

### Known Limitations

Accepted limitations for this release. Each links its tracking issue; "fixed in a later release" items will move to that release's own section when resolved.

- **MCP (Module 27) stdio transport uses one shared identity for every caller.** The HTTP transport propagates each caller's own JWT; the stdio transport (`MCP_TRANSPORT=stdio`) never sets it, so every tool call made through a stdio-launched process resolves through the shared service account instead of a per-user token. Two different users each running the stdio entrypoint against the same deployment are indistinguishable to every downstream API call. (A second gap this same issue originally reported -- the agent's `client_secret` stored in plaintext in Redis -- has since been fixed: it's now stored Fernet-encrypted.)
  [GitHub #20](https://github.com/DHT-AI-Studio/Raptor_Dev/issues/20)

- **NVENC hardware video encoding has a confirmed driver-level bug on multi-GPU hosts** (reproduced on host MSN-02, driver 595.x): `h264_nvenc` fails deterministically on GPU 0 while working cleanly on GPU 1, independent of CUDA compute. A workaround (pin `VIDEO_CHUNKING_GPU` to a working GPU index) is shipped and confirmed effective end-to-end. The underlying NVIDIA driver bug itself is not fixed, so the same symptom could recur on a different multi-GPU host, or if GPU assignment changes, until NVIDIA ships a driver fix.
  [GitHub #94](https://github.com/DHT-AI-Studio/Raptor_Dev/issues/94)

- **Media pipeline dead-letter queues (modules 09-12, ~21 topics) have no consumer, no monitoring, and no retry mechanism.** A message that lands on any `*-dlq` topic today is written once, read by nobody, and ages out silently. Module 25's own `personal-index-requests-dlq` is a separate, newer system with real alerting and is not affected.
  [GitHub #35](https://github.com/DHT-AI-Studio/Raptor_Dev/issues/35)

- **gRPC gateway (Module 36): `GetAsset`, `Chat`, `GetJobStatus`, and `StreamVideo` RPCs are permanently `UNIMPLEMENTED`** for this release (each has a current replacement: `DownloadAsset`, no replacement yet, `GetJob`/`WatchJob`, and `StreamVideoSegment` respectively). Additionally: `StreamVideoSegment` only supports its `asset` source (`video_id` returns `UNIMPLEMENTED`); `JobSession` only supports the subscribe direction (cancel/control are recorded but not acted on); `AnalyzeStream` is disabled by default (`GRPC_GATEWAY_ANALYZE_STREAM_ENABLED=false`) and, even enabled, never returns analysis events back to the caller.

- **Guardrail policy enforcement is incomplete across entry points for this release:**
  - Module 27 MCP tool-call input screening — not started.
  - gRPC unary interceptor for the Guardrail service — in progress.
  - A2A host orchestrator Guardrail pre-check — shipped.

- **Tenancy model: `branch_id` is always the caller's JWT `sub` claim** — one Keycloak account maps to exactly one per-user database/branch (`user_<sub>`), with no multi-user workspace or shared-tenant model. This is the confirmed, intended v0.5 design (see the v0.4→v0.5 migration tooling's own verification of this mapping), not an interim gap, but it constrains what access-control models this release can support.

- **Module 16 (training-service) does not build or run on ARM64** (GX10/GB10 hosts) — its Dockerfile is still pinned to `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, an amd64-only base image, from the original GX10 porting pass; no fix has shipped yet. Every other currently-deployable module is confirmed to build and run on both amd64 and arm64. Modules 17/19/20 (hybrid-search, graph-database, graph-service — all deprecated in favor of Module 25) have not been specifically verified on arm64 either way, since nothing currently exercises them on that architecture.

### Under review, not yet confirmed as a limitation

- "Module 34/35 preview status" was flagged in the original v0.5 roadmap gap list without a corresponding note anywhere in either module's docs or code. Not included above pending clarification of what this refers to.
