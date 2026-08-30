# GitHub Repository Setup Information

Use this information when setting up the RAPTOR repository on GitHub.

## Repository Configuration

### Basic Information

**Repository Name**: `RAPTOR`

**Short Description** (max 350 characters):
```
Open-source multimodal AI framework & agent harness — an enterprise AI runtime for building agentic applications over video, audio, images, documents, and organizational knowledge. MCP tools, A2A orchestration, persistent memory, hybrid RAG, GraphRAG, evaluation, guardrails, model lifecycle. Apache 2.0 | DHT Taiwan
```

**Website**: `https://dhtsolution.com/`

### Topics/Tags

Add these topics to help discovery:

```
ai
artificial-intelligence
agentic-ai
ai-agents
agent-orchestration
mcp
model-context-protocol
multimodal-ai
machine-learning
semantic-search
vector-database
llm
large-language-models
computer-vision
nlp
natural-language-processing
multimedia-processing
video-analysis
audio-transcription
knowledge-graph
rag
retrieval-augmented-generation
graphrag
guardrails
kafka
docker
python
kubernetes
apache2
open-source
content-intelligence
media-analysis
```

### About Section

**Website**: https://dhtsolution.com/

**Topics**: (Use the tags above - select most relevant 10-15)
- ai
- artificial-intelligence
- agentic-ai
- ai-agents
- mcp
- multimodal-ai
- semantic-search
- vector-database
- llm
- computer-vision
- nlp
- kubernetes
- python
- graphrag
- guardrails

### Social Preview Image

Create a 1200x630px image with:
- RAPTOR logo
- Tagline: "Open-Source Multimodal AI Framework & Agent Harness"
- Key stats: "85% faster tagging | 10x better discovery"
- DHT Solutions branding
- Tech icons: AI, Agent, Video, Audio, Image, Text

### Repository Description (Long)

```markdown
# RAPTOR — Open-Source Multimodal AI Framework & Agent Harness

An enterprise AI runtime for building agentic applications over video, audio, images, documents, and organizational knowledge.

## What is RAPTOR?

RAPTOR is a multimodal AI framework and agent harness — a platform of composable
services (not a single content-search product) that an agent or a human developer
can call directly: MCP tools and A2A orchestration for agent interoperability,
persistent multimodal memory, hybrid RAG, GraphRAG, pipeline evaluation, LLM
guardrails, and model lifecycle management.

## Key Benefits

- Agent-native: every capability is callable via MCP, A2A, or plain REST
- **85% reduction** in manual content tagging
- **10x faster** content discovery
- **60% improvement** in content reuse efficiency
- Real-time insights from video, audio, images, and documents

## Core Capabilities

✨ Multi-modal content analysis (video, audio, image, text)
🔌 MCP tools and resources for any MCP-compatible AI client
🤝 A2A orchestration — agent cards, JSON-RPC, five spec-compliant sub-agents
🔍 Hybrid search — BM25 + vector with RRF fusion and cross-encoder rerank
🕸️ Per-user knowledge graph, GraphRAG, and temporal reasoning (ArcadeDB)
🧠 Persistent multimodal memory (Redis + MemVID) with GDPR erasure
🛡️ LLM guardrails and pluggable content moderation
📊 Pipeline evaluation and benchmarking
🖥️ Demo web frontend for upload, video search, and history
🔒 Keycloak-backed authentication and authorization

## Quick Start

```bash
git clone https://github.com/DHT-AI-Studio/RAPTOR.git
cd RAPTOR/Aigle/0.4
# configure .env files from .env.example (see BUILD.md), then:
bash deploy.sh -m 08 --build   # shared GPU base image
bash deploy.sh                 # start all 27 modules
```

Developed by DHT Taiwan Team | Apache 2.0 License
```

### GitHub Features to Enable

- ✅ Issues
- ✅ Discussions (with categories: Announcements, Q&A, Ideas, Show and Tell, General)
- ✅ Projects (optional)
- ❌ Wiki (use docs/ folder instead)
- ✅ Sponsorships (optional)

### Branch Protection Rules

**Branch**: `main`

Enable:
- ✅ Require a pull request before merging
  - ✅ Require approvals: 1
  - ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - Status checks: CI/CD Pipeline
- ✅ Require conversation resolution before merging
- ✅ Require signed commits (optional, for higher security)
- ✅ Include administrators
- ✅ Restrict who can push to matching branches (maintainers only)

### Labels to Create

**Type Labels**:
- `type: bug` (red) - Something isn't working
- `type: feature` (green) - New feature or request
- `type: enhancement` (blue) - Improvement to existing feature
- `type: documentation` (light blue) - Documentation improvements
- `type: question` (purple) - Further information requested

**Priority Labels**:
- `priority: critical` (dark red) - Critical priority
- `priority: high` (orange) - High priority
- `priority: medium` (yellow) - Medium priority
- `priority: low` (light gray) - Low priority

**Status Labels**:
- `status: needs-triage` (gray) - Needs review and prioritization
- `status: in-progress` (yellow) - Work in progress
- `status: blocked` (red) - Blocked by external dependency
- `status: ready-for-review` (green) - Ready for review

**Component Labels**:
- `component: core` - Core framework
- `component: api` - API and endpoints
- `component: ui` - User interface
- `component: docs` - Documentation
- `component: ci-cd` - CI/CD pipeline
- `component: video` - Video processing
- `component: audio` - Audio processing
- `component: image` - Image processing
- `component: text` - Text processing
- `component: search` - Search functionality
- `component: llm` - LLM integration

**Good First Issue**:
- `good first issue` (green) - Good for newcomers

**Help Wanted**:
- `help wanted` (purple) - Extra attention needed

### Milestones to Create

1. **Aigle 0.1.0-beta** — ✅ Released (October 2025): first community beta

2. **Aigle 0.2** — ✅ Released (February 2026): Kafka media pipelines, MLflow lifecycle, Qdrant search

3. **Aigle 0.3** — ✅ Released (June 2026, tag `v0.3`): 21-module deployment, hybrid search, GraphRAG, A2A agents, demo frontend

4. **v0.4** (Aug 2026) — MCP integration, Memory Services, Personal Database Service (ArcadeDB)

5. **v0.5** (Sep 2026) — gRPC API interface, content moderation, Guardrail Service, GDPR/CCPA, real-time audio

6. **v1.0** (Q4 2026) — Production ready: Kubernetes, ELK Stack, 99.9% SLA

### GitHub Actions Secrets

Add these secrets for CI/CD (if needed):

- `PYPI_API_TOKEN` - For publishing to PyPI
- `DOCKER_USERNAME` - For Docker Hub
- `DOCKER_PASSWORD` - For Docker Hub
- `CODECOV_TOKEN` - For code coverage reporting

### Release Template

Latest published release: **[v0.3 — RAPTOR Aigle 0.3 (Community Beta)](https://github.com/DHT-AI-Studio/RAPTOR/releases/tag/v0.3)**, generated from the `[Aigle 0.3]` section of `MAIN_DOCUMENTATION/CHANGELOG.md`.

When creating releases, use this template:

```markdown
## RAPTOR Aigle <version> (Community Beta)

**Released: <Month Year>**

### Added / Changed / Known Limitations

<copy the release section from MAIN_DOCUMENTATION/CHANGELOG.md>

### Roadmap

<current roadmap table from README.md "Release Timeline">

---

**Full changelog:** [MAIN_DOCUMENTATION/CHANGELOG.md](https://github.com/DHT-AI-Studio/RAPTOR/blob/main/MAIN_DOCUMENTATION/CHANGELOG.md)
**Build guide:** `Aigle/<version>/BUILD.md` (e.g. [Aigle/0.4/BUILD.md](https://github.com/DHT-AI-Studio/RAPTOR/blob/main/Aigle/0.4/BUILD.md))
**API reference:** `Aigle/<version>/API_REFERENCE.md` (e.g. [Aigle/0.4/API_REFERENCE.md](https://github.com/DHT-AI-Studio/RAPTOR/blob/main/Aigle/0.4/API_REFERENCE.md))
```

Release checklist (see also `Aigle/<version>/BUILD.md` §9 Source Maintenance Process):

1. `.env.example` in sync with `.env` keys for every module; no secrets or internal IPs in the tree
2. `MAIN_DOCUMENTATION/CHANGELOG.md` updated; README version badge, evaluation API, and Release Timeline updated
3. Commit via a `release/*` branch, merge to `main`, push
4. Annotated tag `vX.Y` on the merge commit; `gh release create vX.Y` from the changelog section

### Discussion Categories

Create these categories in GitHub Discussions:

1. **📢 Announcements** (Announcement type)
   - Project updates and releases
   
2. **💡 Ideas & Feature Requests**
   - Suggest new features and improvements
   
3. **🙋 Q&A** (Q&A type)
   - Ask questions about RAPTOR
   
4. **🎉 Show and Tell**
   - Share your projects using RAPTOR
   
5. **💬 General**
   - General discussion about RAPTOR

### Security

1. Enable **Security Advisories**
2. Enable **Dependabot alerts**
3. Enable **Dependabot security updates**
4. Add security policy (already in SECURITY.md)

### Insights Settings

Enable these insights:
- Pulse
- Contributors
- Community
- Traffic
- Commits
- Code frequency
- Dependency graph
- Network

---

## Social Media Setup

### Twitter/X Bio

```
🚀 RAPTOR - Open-Source Multimodal AI Framework & Agent Harness
Build agentic apps over video, audio, images & docs
MCP • A2A • RAG • GraphRAG • Guardrails
Open source | Apache 2.0
By @DHT_Taiwan
🔗 github.com/DHT-AI-Studio/RAPTOR
```

### LinkedIn Description

```
RAPTOR is an open-source multimodal AI framework and agent harness — an
enterprise AI runtime for building agentic applications over video, audio,
images, documents, and organizational knowledge.

RAPTOR exposes every capability — search, RAG, GraphRAG, memory, guardrails,
model lifecycle — as MCP tools, A2A-orchestrated agents, and plain REST, so
LLM agents and human developers can build on the same platform:
• Agent-native: MCP tools + A2A orchestration out of the box
• 85% reduction in manual content tagging
• 10x faster content discovery
• Real-time insights from video, audio, images, and documents

Docker Compose today, with a Kubernetes-native production path planned.

Developed by DHT Taiwan Team | Apache 2.0 License
Visit: github.com/DHT-AI-Studio/RAPTOR
```

### Instagram Bio

```
🤖 Open-Source Multimodal AI Framework & Agent Harness
📹 Video | 🎵 Audio | 🖼️ Image | 📄 Text
🔌 MCP Tools | 🤝 A2A Agents | 🔍 Hybrid RAG
⚡ 10x Faster Discovery
🌐 Open Source | Apache 2.0
👉 github.com/DHT-AI-Studio/RAPTOR
```

---

This information should be used when setting up the RAPTOR repository on GitHub 
and associated social media accounts.

