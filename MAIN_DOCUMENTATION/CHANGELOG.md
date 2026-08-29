# Changelog

All notable changes to the RAPTOR AI Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Community feedback integration
- Performance optimizations
- Extended documentation and tutorials
- Additional examples

---

## [Aigle 0.4] - 2026-08

### 🚀 Community Beta — LLM Interoperability & Persistent Memory

### Added

- **MCP Server** (module 27): Raptor's search/RAG/asset/memory APIs exposed as Model Context Protocol tools (22), resources (3), and prompts (10) over Streamable HTTP and stdio — see `Aigle/0.4/MCP_REFERENCE.md`
- **Memory Service** (module 26): persistent per-user/per-session memory on a two-tier Redis (hot) + MemVID (durable) architecture — semantic (BM25 + HNSW) search, multimodal (video/audio/image) indexing, automatic LLM-driven entity/preference extraction, GDPR erasure endpoint; integrated into both module 15 (chat) and module 21 (agent-protocol)
- **Personal DB Service** (modules 24/25): one physically isolated ArcadeDB database per user, replacing the per-user Qdrant + OpenSearch + Neo4j trio with unified hybrid BM25 + vector + graph + temporal-fact search
- **Guardrail Service** (module 23): pluggable LLM input/output content moderation (Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard) plus a separate regex/detector policy engine and violation audit logging; ships disabled by default
- **Benchmark Service** (module 22): user-defined scoring schemas (keyword match, exact/regex match, cosine similarity, LLM-as-judge) with run history and pairwise comparison, for regression-testing chat/RAG/search/classification pipelines
- **A2A protocol documentation**: `Aigle/0.4/A2A_REFERENCE.md` — full write-up of module 21's JSON-RPC orchestrator surface and its five independent, spec-compliant `a2a-sdk` specialist sub-agents (vector/keyword/GraphRAG/TKG/reranker)
- **Module count grows to 27** (from 21 in Aigle 0.3): modules 22–27 above, plus scaffolding/design docs for a still-in-development module 29 (doc-processing-agent, not yet wired into `build.py`)

### Changed

- **Hybrid search & GraphRAG re-platformed onto module 25**: modules 17 (hybrid-search, OpenSearch/Qdrant) and 19/20 (graph-database/graph-service, Neo4j) are now deprecated, kept only for rollback
- Root README, `API_REFERENCE.md`, and `BUILD.md` updated throughout for the 27-module architecture and new service ports

### Known Limitations

- Module 21's A2A `agent.delegate` JSON-RPC method is a stub (accepts and echoes a task without dispatching it); the peer-agent registry doesn't yet feed automatic delegation — tracked in [#21](https://github.com/DHT-AI-Studio/RAPTOR/issues/21)
- MCP tool coverage doesn't yet include a Model Management or Vector DB tool category — tracked in [#19](https://github.com/DHT-AI-Studio/RAPTOR/issues/19)
- Module 11 (video-processing)'s default VLM was downsized (InternVL3.5-4B → 1B) but not yet re-verified on a constrained GPU, and its documented VRAM guidance wasn't updated to match — tracked in [#17](https://github.com/DHT-AI-Studio/RAPTOR/issues/17)
- Docker Compose deployment is still not intended for production scale (Kubernetes planned for v1.0)

---

## [Aigle 0.3] - 2026-06

### 🚀 Community Beta — Modular Platform Release

### Added

- **Modular deployment system**: 21 independent Docker Compose modules under `Aigle/0.3/deployment/modules/`, managed by a single build entry point (`build.py` / `deploy.sh` wrapper) with dependency-ordered start, per-module lifecycle (`--stop/--delete/--restart/--build`), status and logs
- **Hybrid search**: OpenSearch BM25 + Qdrant vector retrieval with RRF fusion and cross-encoder re-ranking (module 17)
- **Graph database & GraphRAG**: Neo4j entity/temporal-fact graph with LLM-powered graph query and reasoning, including relation date normalization (`date_utils`) (modules 19–20)
- **Agent Protocol (A2A)**: agent discovery, orchestration, and multi-agent RAG pipeline over vector / keyword / GraphRAG / TKG agents (module 21)
- **Video Search 2.0**: video-centric search endpoint — multi-recall (BM25 + Vector + GraphRAG + TKG) → RRF fusion → cross-encoder rerank → per-video aggregation with time-coded segments
- **Branch-aware multi-tenancy**: `branch_id` isolation propagated through upload, media processing, indexing, search, and agents
- **Demo frontend**: React + Vite web UI (file upload, natural-language video search, upload history) with Docker deployment (`Aigle/0.3/raptor-demo-frontend/`)
- **Authentication module rework**: restructured FastAPI app (`app/` package) with group and account management, permission endpoint used by the API Gateway, SMTP e-mail notifications, updated Keycloak realm
- **Custom MLflow image** (`raptor/mlflow:0.3`) in module 07
- **Build & source maintenance guide**: `Aigle/0.3/BUILD.md`

### Changed

- **GPU stack upgraded for Blackwell (sm_120 / RTX 50-series)**: CUDA base image 12.6 → 12.8, PyTorch 2.7.1 cu126 → cu128 wheels, PaddlePaddle GPU 3.0/3.1 → 3.3.0 (cu129); shared `raptor/media-worker:0.3` base image for modules 09–12
- **SeaweedFS** upgraded 3.96 → 4.32; LakeFS SDK calls moved off the event loop (thread pool); index/graph status sync added to asset management (module 04)
- **API Gateway permission model**: JWT verification + permission check via authentication module `/auth/permission` (Keycloak UMA removed)
- **Configuration templates**: per-modality model selection (`VIDEO/AUDIO/DOCUMENT_INFERENCE_MODEL`, `IMAGE/VIDEO_VLM_MODEL_PATH`), SMTP block, memory/timeout tuning keys; removed per-service `*_GPU_COUNT`, `GATEWAY_KEYCLOAK_*`, `TKG_AGENT_URL`, `RERANKER_AGENT_URL`
- `search` API `payload_schema` now supports `"contextual"` only

### Known Limitations

- Module 14 (monitoring) is incomplete
- Module 11 (video-processing) requires high GPU VRAM
- Docker Compose deployment is not intended for production scale (Kubernetes planned for v1.0)

---

## [Aigle 0.2] - 2026-02

### Community Beta

- Kafka-based multi-modal processing pipelines (audio / video / image / document)
- MLflow model lifecycle management and Ollama model registration
- Qdrant vector search APIs (video / audio / document / image)
- Redis cluster caching, evaluation & testing API on DHT infrastructure

---

## [Aigle 0.1.0-beta] - 2025-10-22

### 🎉 First Release

This is the first community beta release of the RAPTOR AI Framework by DHT Taiwan Team.

### Added

#### Core Features
- **Multi-Modal Content Processing**: Video, audio, image, and text analysis
- **Semantic Search Engine**: Vector-based similarity search with context understanding
- **AI-Powered Metadata Generation**: Automatic tagging and classification
- **LLM Orchestration Framework**: Flexible integration with multiple language models
- **Content Intelligence Pipeline**: Extract insights from unstructured media
- **Entity Recognition System**: Identify people, places, objects, and concepts
- **Configuration Management**: Flexible configuration for different deployment scenarios
- **Logging and Monitoring**: Comprehensive observability and metrics

#### Documentation
- Comprehensive README with project overview
- Detailed RELEASE_PROCEDURE for future releases
- CONTRIBUTING guidelines for community contributors
- CODE_OF_CONDUCT for community standards
- Apache 2.0 LICENSE file
- Initial API documentation structure

#### Repository Structure
- `/Aigle/0.1/` - Source code for first release
- `/docs/` - Documentation directory
- `/examples/` - Example code and tutorials
- `.github/` - GitHub issue templates and workflows

#### Community
- GitHub issue templates for bug reports and feature requests
- Pull request template
- Community communication channels setup (GitHub, Telegram, Instagram, X)

### Known Limitations

This is a beta release. The following are known limitations:

- Limited documentation coverage for advanced features
- Some features may not be fully optimized for production use
- API may change in future releases based on community feedback

### Breaking Changes

None (first release)

### Security

- All dependencies audited for known vulnerabilities
- Secure default configurations implemented

### Contributors

Special thanks to the DHT Taiwan Team for developing this first release.

### Links

- **GitHub Repository**: https://github.com/DHT-AI-Studio/RAPTOR
- **Company**: https://dhtsolution.com/
- **License**: Apache 2.0

---

## Version History Summary

| Version | Codename | Release Date | Type | Status |
|---------|----------|--------------|------|--------|
| 0.4 | Aigle | 2026-08 | Community Beta | Current |
| 0.3 | Aigle | 2026-06 | Community Beta | Superseded |
| 0.2 | Aigle | 2026-02 | Community Beta | Superseded |
| 0.1.0-beta | Aigle | 2025-10-22 | Beta | Superseded |

---

## How to Read This Changelog

### Version Format
```
[Codename Major.Minor.Patch-stage] - YYYY-MM-DD
```

### Change Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes

### Symbols

- 🎉 Major release or milestone
- 🚀 New feature
- 🐛 Bug fix
- 📝 Documentation
- ⚡ Performance improvement
- 🔒 Security fix
- ⚠️ Breaking change
- 🗑️ Deprecation

---

## Upgrade Guide

### To Aigle 0.1.0-beta

This is the first release, so no upgrade is necessary.

For future upgrades, we'll provide detailed migration guides here.

---

## Feedback and Contributions

We welcome your feedback on this release! Please:

1. **Report bugs**: [GitHub Issues](https://github.com/DHT-AI-Studio/RAPTOR/issues)
2. **Request features**: [GitHub Issues](https://github.com/DHT-AI-Studio/RAPTOR/issues)
3. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Discuss**: Join our community channels

---

**Maintained by DHT Taiwan Team**

For more information, visit [https://dhtsolution.com/](https://dhtsolution.com/)

