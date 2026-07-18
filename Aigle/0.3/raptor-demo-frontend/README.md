# Raptor Demo Frontend

React + Vite 前端，提供 Raptor 影片搜尋系統的操作介面。

## 功能頁面

| 頁面 | 說明 |
|---|---|
| File Upload | 上傳影片／音訊／文件，後台自動觸發 AI 處理流程 |
| Video Search | 自然語言影片搜尋（BM25 + Vector + GraphRAG + TKG，Cross-Encoder 精排） |
| Upload History | 瀏覽上傳紀錄，支援播放、下載、封存、刪除 |
| Search *(隱藏)* | 段落級 Hybrid Search（目前 demo 模式下隱藏） |
| Chat *(隱藏)* | RAG 問答（目前 demo 模式下隱藏） |

> Search 與 Chat 頁面已用 JS 註解暫時隱藏（`Layout.jsx`），需要時取消註解即可還原。

---

## 快速啟動

### Docker（推薦）

```bash
cp .env.example .env
# 編輯 .env，設定 API_TARGET 與 DEMO_PORT
docker compose up -d --build
```

服務啟動後在瀏覽器開啟 `http://<host>:<DEMO_PORT>`。

### 本機開發

```bash
npm install
cp .env.example .env
# 設定 VITE_API_TARGET
npm run dev
```

---

## 環境變數

| 變數 | 說明 | 預設值 |
|---|---|---|
| `API_TARGET` | Raptor Gateway 位址（Docker nginx proxy 用） | `http://192.168.157.165:8012` |
| `DEMO_PORT` | 前端對外 port | `3000` |
| `DOCKER_NETWORK` | Docker network 名稱 | `raptor` |
| `VITE_API_TARGET` | Vite dev server proxy 目標（**本機開發專用**，Docker 部署不需要） | 同 `API_TARGET` |

---

## 架構

```
nginx (port 80)
 ├── /            → React SPA (dist/)
 ├── /api/*       → proxy → Raptor Gateway (API_TARGET)
 └── /transcode/* → proxy → transcode sidecar (H.265 → H.264)
```

### Transcode Sidecar

`transcode/` 目錄是一個獨立的 Python FastAPI 服務，負責將瀏覽器不支援的 H.265 影片即時轉碼為 H.264 串流。由 `docker-compose.yml` 一起啟動，nginx 自動 proxy。

---

## 主要元件

| 檔案 | 說明 |
|---|---|
| `src/components/VideoPlayer.jsx` | 通用影片播放器，支援 mp4/H.264、TS（mpegts.js）、H.265 自動轉碼 |
| `src/components/GraphView.jsx` | @antv/g6 v5 知識圖譜視覺化（拖曳、縮放、節點詳情） |
| `src/components/VideoSearchPanel.jsx` | Video Search 頁面 |
| `src/components/HistoryPanel.jsx` | Upload History 頁面 |
| `src/api/client.js` | 所有 API 呼叫集中管理 |

---

## 注意事項

- 登入驗證使用 Keycloak，token 由 `AuthContext` 管理。
- Presigned URL 每次播放前都會重新取得，不快取，避免過期失效。
- 切換 tab 不會 unmount 元件（使用 CSS `hidden`），狀態會保留。
