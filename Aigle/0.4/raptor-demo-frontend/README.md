# Raptor Demo Frontend

A React + Vite frontend providing an interface for the Raptor video search system.

## Feature Pages

| Page | Description |
|---|---|
| File Upload | Upload video/audio/documents; the backend automatically triggers the AI processing pipeline |
| Video Search | Natural-language video search (BM25 + Vector + GraphRAG + TKG, cross-encoder reranked) |
| Upload History | Browse upload history, with support for playback, download, archiving, deletion |
| Search *(hidden)* | Segment-level hybrid search (currently hidden in demo mode) |
| Chat *(hidden)* | RAG question-answering (currently hidden in demo mode) |

> The Search and Chat pages are temporarily hidden with JS comments (`Layout.jsx`) — uncomment them to restore when needed.

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
# edit .env, setting API_TARGET and DEMO_PORT
docker compose up -d --build
```

Once the service starts, open `http://<host>:<DEMO_PORT>` in a browser.

### Local development

```bash
npm install
cp .env.example .env
# set VITE_API_TARGET
npm run dev
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_TARGET` | Raptor Gateway address (used by the Docker nginx proxy) | `http://192.168.157.165:8012` |
| `DEMO_PORT` | Public-facing port for the frontend | `3000` |
| `DOCKER_NETWORK` | Docker network name | `raptor` |
| `VITE_API_TARGET` | Vite dev server proxy target (**local development only**, not needed for a Docker deploy) | same as `API_TARGET` |

---

## Architecture

```
nginx (port 80)
 ├── /            → React SPA (dist/)
 ├── /api/*       → proxy → Raptor Gateway (API_TARGET)
 └── /transcode/* → proxy → transcode sidecar (H.265 → H.264)
```

### Transcode Sidecar

`transcode/` is a standalone Python FastAPI service responsible for transcoding H.265 video the browser doesn't support into an H.264 stream in real time. It's started alongside the frontend by `docker-compose.yml`, with nginx automatically proxying to it.

---

## Key Components

| File | Description |
|---|---|
| `src/components/VideoPlayer.jsx` | General-purpose video player, supporting mp4/H.264, TS (mpegts.js), and automatic H.265 transcoding |
| `src/components/GraphView.jsx` | Knowledge-graph visualization via @antv/g6 v5 (drag, zoom, node detail) |
| `src/components/VideoSearchPanel.jsx` | The Video Search page |
| `src/components/HistoryPanel.jsx` | The Upload History page |
| `src/api/client.js` | Central location for all API calls |

---

## Notes

- Login authentication uses Keycloak; the token is managed by `AuthContext`.
- The presigned URL is re-fetched before every playback, never cached, to avoid it expiring.
- Switching tabs doesn't unmount components (CSS `hidden` is used instead), so state is preserved.
