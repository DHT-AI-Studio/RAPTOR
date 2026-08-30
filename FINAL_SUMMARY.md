# 🎉 RAPTOR Project Summary

## Project Overview

**RAPTOR** is an open-source **multimodal AI framework and agent harness** — an enterprise AI runtime for building agentic applications over video, audio, images, documents, and organizational knowledge, providing MCP tools, A2A orchestration, persistent memory, hybrid RAG, GraphRAG, evaluation, guardrails, and model lifecycle services.

**Current release: Aigle 0.4 (Community Beta) — August 2026** — pushed to `main`; not yet cut as a GitHub Release/tag (the [latest tagged release](https://github.com/DHT-AI-Studio/RAPTOR/releases) is still `v0.3`).

---

## 📌 Project Information

**Project**: RAPTOR
**Type**: Open-Source Multimodal AI Framework & Agent Harness
**Current Version**: Aigle 0.4 (Community Beta, August 2026)
**License**: Apache 2.0
**Developer**: DHT Taiwan Team
**Company**: [DHT Solutions](https://dhtsolution.com/)
**Repository**: https://github.com/DHT-AI-Studio/RAPTOR
**Evaluation API**: http://raptor_open_0_4_api.dhtsolution.com:8012/

**Key Value Propositions**:
- Agent-native: every capability is callable via MCP tools, A2A orchestration, or plain REST
- 85% reduction in manual content tagging
- 10x faster content discovery
- 60% improvement in content reuse efficiency
- Multi-modal analysis (video, audio, image, document)

---

## 🚀 Release History

| Release | Date | Highlights |
| --- | --- | --- |
| **Aigle 0.1** | October 2025 | First community beta: core framework, system-design documentation, community/GitHub setup |
| **Aigle 0.2** | February 2026 | Kafka-based media pipelines, MLflow model lifecycle, Qdrant vector search APIs, Redis cluster, evaluation API |
| **Aigle 0.3** | June 2026 | 21-module Docker Compose platform (`build.py` build system), hybrid search (BM25 + vector, RRF + cross-encoder rerank), Neo4j knowledge graph & GraphRAG, temporal knowledge graph, A2A agent orchestration, branch-based multi-tenancy, Blackwell-ready GPU stack (CUDA 12.8), reworked Keycloak authentication, React demo frontend |
| **Aigle 0.4** | August 2026 | Grows to 27 modules; MCP Server (tools/resources/prompts), Memory Service (Redis + MemVID, GDPR erasure), Personal DB Service (per-user ArcadeDB replacing the Qdrant/OpenSearch/Neo4j trio), Guardrail Service (LLM content moderation, disabled by default), Benchmark Service (schema-driven scoring); modules 17/19/20 deprecated in favor of module 25 |

Full release notes: [MAIN_DOCUMENTATION/CHANGELOG.md](MAIN_DOCUMENTATION/CHANGELOG.md)

---

## 📂 Repository Structure

```
RAPTOR/
├── README.md                     Main project page (features, install, roadmap)
├── LICENSE                       Apache 2.0
├── GITHUB_SETUP_INFO.md          GitHub configuration & release process reference
├── FINAL_SUMMARY.md              This file
├── MAIN_DOCUMENTATION/
│   └── CHANGELOG.md              Version history
├── COMMUNITY_GUIDELINES/
│   ├── CONTRIBUTING.md · CODE_OF_CONDUCT.md · SECURITY.md
├── .github/                      Issue/PR templates, CI workflow
└── Aigle/
    ├── 0.1/                      First beta (raptor package + design PDFs)
    ├── 0.2/                      Community Beta (Kafka pipelines, root docker-compose)
    ├── 0.3/                      Prior release (kept in-tree, superseded)
    └── 0.4/                      Current release
        ├── README.md             Module reference & deprecated/in-development modules
        ├── BUILD.md              Build & source-maintenance guide
        ├── API_REFERENCE.md      Complete REST API documentation
        ├── MCP_REFERENCE.md      Model Context Protocol tool/resource/prompt reference
        ├── A2A_REFERENCE.md      Agent-to-Agent protocol reference
        ├── deploy.sh             Deployment entry point (wraps build.py)
        ├── test_all_apis.py      End-to-end API test suite
        ├── raptor_client.py      Python client
        ├── deployment/modules/   27 Docker Compose modules (01–27; 29 in development)
        └── raptor-demo-frontend/ React + Vite demo UI
```

---

## 🏗️ Aigle 0.4 Architecture at a Glance

- **27 independent Docker Compose modules** sharing the `raptor` bridge network, deployed in dependency order by `deployment/modules/build.py` (`deploy.sh` wrapper); modules 17/19/20 deprecated, kept only for rollback
- **Media flow**: upload (API Gateway `:8012`) → asset management (LakeFS + SeaweedFS) → Kafka → GPU workers (WhisperX / InternVL / OCR / summaries) → per-user ArcadeDB index (module 25: hybrid BM25 + vector + graph + temporal facts) → search / RAG / A2A / MCP APIs
- **Infrastructure**: NFS, Redis, PostgreSQL, SeaweedFS + LakeFS, Kafka (KRaft), Keycloak, ArcadeDB
- **Reasoning & interop**: LangGraph RAG chat, query orchestrator (intent routing), A2A agent protocol (orchestrator + 5 spec-compliant `a2a-sdk` sub-agents), MCP server (22 tools / 3 resources / 10 prompts), Guardrail content moderation, Memory Service, Benchmark Service
- Configuration via per-module `.env` (templates: `.env.example`; see `Aigle/0.4/BUILD.md`)

---

## 📊 Key Metrics & Features

### Core Capabilities
- ✅ Multi-modal content analysis (video / audio / image / document)
- ✅ Hybrid search: per-user ArcadeDB (BM25 + vector), RRF fusion, cross-encoder rerank
- ✅ Video-centric search with time-coded segment aggregation
- ✅ Knowledge graph, GraphRAG, and temporal facts (ArcadeDB, module 25)
- ✅ A2A agent discovery, orchestration, and multi-agent RAG — plus 5 independent spec-compliant A2A sub-agents
- ✅ MCP tools/resources/prompts for any MCP-compatible client
- ✅ Persistent multimodal memory (Redis + MemVID) with GDPR erasure
- ✅ LLM content moderation (Guardrail), disabled by default
- ✅ Schema-driven pipeline benchmarking and scoring
- ✅ Branch-based multi-tenant data isolation (`branch_id` end-to-end)
- ✅ Model lifecycle management (MLflow) and GPU training service
- ✅ Keycloak authentication with group/account management
- ✅ Demo web frontend (upload · video search · history)

### Strategic Differentiators
1. **AI-Native Architecture** — built around LLM orchestration and vector search
2. **Multi-Modal Understanding** — unified analysis across all media types
3. **Semantic Intelligence** — context-aware, intent-based retrieval
4. **Open + Enterprise** — open-source core with enterprise deployment options
5. **Modular Deployment** — every subsystem independently deployable and testable
6. **LLM Interoperability** — MCP and A2A protocol support alongside plain REST

---

## 🗺️ Roadmap

| Version | Target | Focus |
| --- | --- | --- |
| **v0.4** | Aug 2026 ✅ | MCP integration across core services, Memory Services (Redis + MemVID), Personal Database Service (ArcadeDB) — **Delivered** |
| **v0.5** | Sep 2026 | gRPC API interface, content moderation & GDPR/CCPA compliance, real-time audio processing, plus carry-over work from v0.4 (A2A delegation, MCP tool-catalog gaps, module 11 VRAM re-verification) |
| **v1.0** | Q4 2026 | Production ready: Kubernetes + Helm, ELK Stack observability, 99.9% SLA |

Details: README.md § Future Development Roadmap; open work items tracked on [GitHub Issues](https://github.com/DHT-AI-Studio/RAPTOR/issues) under the v0.5/v1.0 milestones.

---

## 📚 Documentation Reference

### For Users
- **README.md** — overview, installation, quick start, roadmap
- **Aigle/0.4/API_REFERENCE.md** — REST endpoint documentation with examples
- **Aigle/0.4/MCP_REFERENCE.md** — Model Context Protocol interface guide
- **Aigle/0.4/A2A_REFERENCE.md** — Agent-to-Agent protocol guide
- **Aigle/0.4/raptor-demo-frontend/README.md** — demo UI usage

### For Operators
- **Aigle/0.4/BUILD.md** — prerequisites, .env configuration, single/multi-host setup, source-maintenance rules
- **Aigle/0.4/README.md** — module reference, deprecated/in-development modules, service ports
- **Aigle/0.4/deployment/README.md** — dependency graph and `build.py` reference

### For Contributors & Maintainers
- **COMMUNITY_GUIDELINES/** — CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- **GITHUB_SETUP_INFO.md** — repository configuration, labels, milestones, release process
- **MAIN_DOCUMENTATION/CHANGELOG.md** — version history

---

## 🌐 Community

- **Issues & Discussions**: https://github.com/DHT-AI-Studio/RAPTOR
- **Telegram**: [@dhtsupport](https://t.me/dhtsupport)
- **Instagram**: [@DHT.Ai](https://www.instagram.com/DHT.Ai)
- **X / Twitter**: [@dhtsolution2018](https://x.com/dhtsolution2018)
- **Website**: https://dhtsolution.com/

---

*Last updated: August 2026 (Aigle 0.4 release)*
*For: DHT Taiwan Team*
*Status: PUSHED TO MAIN — not yet cut as a GitHub Release; next milestone v0.5 (Sep 2026)*

---

**Made with ❤️ by the DHT Taiwan Team**
