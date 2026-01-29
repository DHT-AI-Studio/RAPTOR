# FastAPI Gateway

> 一個企業級的 FastAPI 應用網關，具有完整的 Docker 部署支持、PostgreSQL 數據庫、Redis 緩存、Kafka 消息隊列和 Qdrant 向量數據庫集成。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-16-336791.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 目錄

- [功能特性](#功能特性)
- [系統要求](#系統要求)
- [快速開始](#快速開始)
- [Docker 部署](#docker-部署)
- [項目結構](#項目結構)
- [API 文檔](#api-文檔)
- [配置管理](#配置管理)
- [故障排除](#故障排除)
- [貢獻指南](#貢獻指南)
- [許可證](#許可證)

---

## ✨ 功能特性

### 核心功能
- 🔐 **JWT 認證**: 使用 HS256 算法的安全 JWT 令牌驗證
- 📤 **文件上傳**: 支持直接上傳和代理到對象存儲服務
- 🔍 **語義搜索**: 集成 Qdrant 向量數據庫，支持多種數據類型搜索
  - 📝 文檔搜索
  - 🎬 視頻搜索
  - 🎵 音頻搜索
  - 🖼️ 圖像搜索
- 📊 **數據資源管理**: 
  - 文件版本管理
  - 資產存檔和恢復
  - 文件下載
  - 批量刪除

### 技術特性
- ⚡ **非同步框架**: 基於 FastAPI 的高性能異步 API
- 🗄️ **數據庫**: PostgreSQL 用於用戶和元數據存儲
- 💾 **緩存**: Redis 支持快速數據訪問
- 📨 **消息隊列**: Kafka 用於異步處理
- 🐳 **Docker 支持**: 完整的 Docker 和 Docker Compose 配置
- 📝 **結構化日誌**: JSON 格式的應用日誌
- 🏥 **健康檢查**: 內置服務健康監控

---

## 📦 系統要求

### 本地開發
- Python 3.10+
- PostgreSQL 16+
- Redis 7+
- Docker & Docker Compose（用於容器化）

### 運行時依賴
- FastAPI 0.111.0
- Uvicorn 0.30.1
- asyncpg 0.29.0
- PyJWT 2.8.0
- passlib 1.7.4 (with argon2)

### 完整依賴列表
見 [requirements.txt](./requirements.txt)

---

## 🚀 快速開始

### 方式 1: 使用 Docker Compose（推薦）

最快的方式是使用提供的部署腳本：

```bash
# 進入項目目錄
cd /opt/home/nelson/VIE/API/fastapi-gateway

# 構建 Docker 映像
./deploy.sh build

# 啟動所有服務
./deploy.sh up

# 查看服務狀態
./deploy.sh ps

# 檢查健康狀態
./deploy.sh health
```

訪問應用：
- 🌐 API 文檔: http://localhost:8012/docs
- 🗄️ 數據庫管理: http://localhost:5480
- 💚 健康檢查: http://localhost:8012/health

### 方式 2: 本地開發

```bash
# 克隆或進入項目目錄
cd /opt/home/nelson/VIE/API/fastapi-gateway

# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt

# 設置環境變數
cp .env.local .env

# 啟動應用
uvicorn app.main:app --reload --port 8012
```

---

## 🐳 Docker 部署

### 架構概述

```
┌─────────────────────────────────────────────────────┐
│         Docker Network: vie-api-gateway             │
│         Subnet: 172.20.0.0/16                       │
│                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ api-gateway  │  │ postgresql │  │  pgadmin   │ │
│  │  :8012       │  │  :5432     │  │  :80       │ │
│  └──────────────┘  └────────────┘  └────────────┘ │
│                                                     │
│  Ports:                                            │
│  - 8012 (API)    → localhost:8012                 │
│  - 5434 (DB)     → localhost:5434                 │
│  - 5480 (PgAdmin) → localhost:5480               │
└─────────────────────────────────────────────────────┘
           ↓ (連接到外部服務)
┌─────────────────────────────────────────────────────┐
│  外部依賴                                            │
│  - Redis: 192.168.157.165:6381                     │
│  - Kafka: 192.168.157.165:19092,19093             │
│  - Qdrant: 192.168.157.124:8801                   │
│  - MinIO: 192.168.157.165:8000                    │
└─────────────────────────────────────────────────────┘
```

### 常用命令

```bash
# 啟動服務
./deploy.sh up

# 停止服務
./deploy.sh down

# 重新啟動
./deploy.sh restart

# 查看日誌（實時）
./deploy.sh logs -f

# 查看特定服務日誌
./deploy.sh logs-gateway -f    # API Gateway
./deploy.sh logs-postgres -f   # PostgreSQL
./deploy.sh logs-pgadmin -f    # PgAdmin

# 進入容器 shell
./deploy.sh shell

# 進入數據庫 shell
./deploy.sh db-shell

# 備份數據庫
./deploy.sh db-backup

# 恢復數據庫
./deploy.sh db-restore backup_file.sql

# 查看所有命令
./deploy.sh help
```

### Docker Compose 直接命令

```bash
# 構建
docker-compose build

# 啟動
docker-compose up -d

# 停止
docker-compose down

# 查看狀態
docker-compose ps

# 查看日誌
docker-compose logs -f api-gateway
```

詳細 Docker 文檔見: [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md)

---

## 📁 項目結構

```
fastapi-gateway/
├── Dockerfile                      # Docker 構建文件
├── docker-compose.yml              # Docker Compose 配置
├── .env                            # 環境變數（本地）
├── .env.docker                     # Docker 環境變數
├── .env.local                      # 本地開發環境變數
├── .dockerignore                   # Docker 構建排除清單
├── deploy.sh                       # 部署管理腳本
├── requirements.txt                # Python 依賴
├── services.yaml                   # 下遊服務配置
│
├── app/
│   ├── main.py                     # 應用入口點
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── dependencies.py         # 依賴注入
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── config.py               # 應用配置
│   │   ├── security.py             # 安全工具
│   │   └── __init__.py
│   │
│   ├── middlewares/
│   │   ├── logging.py              # 日誌中間件
│   │   └── __init__.py
│   │
│   ├── routers/
│   │   ├── auth.py                 # 認證端點
│   │   ├── asset.py                # 資產管理端點
│   │   ├── search.py               # 搜索端點 (Qdrant)
│   │   ├── processing.py           # 處理端點
│   │   ├── upload.py               # 上傳端點
│   │   └── __init__.py
│   │
│   └── services/
│       ├── auth_service.py         # 認證邏輯
│       ├── storage_service.py      # 存儲邏輯
│       ├── kafka_service.py        # Kafka 集成
│       ├── user_service.py         # 用戶管理
│       └── __init__.py
│
├── PostgreSQL/
│   ├── docker-compose.yml          # PostgreSQL 配置（已合併）
│   └── init.sql                    # 數據庫初始化腳本
│
├── data/
│   └── users.json                  # 用戶數據
│
├── docs/
│   └── UPLOAD.md                   # 上傳功能文檔
│
├── README.md                       # 此文件
├── QUICK_START.md                  # 快速開始指南
├── DOCKER_DEPLOYMENT.md            # Docker 部署文檔
├── FILES_GUIDE.md                  # 文件結構說明
├── DEPLOYMENT_COMPLETE.md          # 部署完成總結
├── CHECKLIST.md                    # 驗證清單
├── SUMMARY.txt                     # 視覺化摘要
└── README_DOCKER.md                # Docker 專用 README
```

---

## 📚 API 文檔

### 認證端點

#### 用戶註冊
```
POST /register
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

#### 用戶登錄
```
POST /token
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=secure_password
```

### 文件管理端點

#### 上傳文件
```
POST /fileupload
Headers: Authorization: Bearer {token}
Form-Data:
  - primary_file: (file)
  - archive_ttl: 30
  - destroy_ttl: 30
```

#### 上傳並分析文件
```
POST /fileupload_analysis
Headers: Authorization: Bearer {token}
Form-Data:
  - primary_file: (file)
  - processing_mode: default
```

#### 下載文件
```
GET /filedownload/{asset_path}/{version_id}
Query: return_file_content=true
Headers: Authorization: Bearer {token}
```

#### 列出版本
```
GET /fileversions/{asset_path}/{filename}
Headers: Authorization: Bearer {token}
```

#### 存檔文件
```
POST /filearchive/{asset_path}/{version_id}
Headers: Authorization: Bearer {token}
```

#### 刪除文件
```
POST /delfile/{asset_path}/{version_id}
Headers: Authorization: Bearer {token}
```

### 搜索端點（Qdrant 集成）

#### 搜索視頻
```
POST /video_search
{
  "query_text": "機器學習",
  "embedding_type": "text",
  "type": "video",
  "filename": ["video.mp4"],
  "speaker": ["SPEAKER_00"],
  "limit": 5
}
```

#### 搜索音頻
```
POST /audio_search
{
  "query_text": "語音識別",
  "embedding_type": "text",
  "type": "audio",
  "filename": ["audio.mp3"],
  "speaker": ["SPEAKER_00"],
  "limit": 5
}
```

#### 搜索文檔
```
POST /document_search
{
  "query_text": "財務報告",
  "embedding_type": "text",
  "type": "document",
  "filename": ["report.pdf"],
  "source": "pdf",
  "limit": 5
}
```

#### 搜索圖像
```
POST /image_search
{
  "query_text": "風景照片",
  "embedding_type": "summary",
  "type": "image",
  "filename": ["photo.jpg"],
  "source": "jpg",
  "limit": 5
}
```

### 帶快取的搜索端點

相同的端點帶 `_with_cache` 後綴，具有更快的性能：
- `/video_search_with_cache`
- `/audio_search_with_cache`
- `/document_search_with_cache`
- `/image_search_with_cache`

### 完整 API 文檔

訪問: http://localhost:8012/docs （Swagger UI）
或: http://localhost:8012/redoc （ReDoc）

---

## ⚙️ 配置管理

### 環境變數優先級

1. **環境變數**（最高優先級）
   ```bash
   export GATEWAY_DB_HOST=custom-host
   ```

2. **.env 文件**（中等優先級）
   ```bash
   GATEWAY_DB_HOST=localhost
   ```

3. **config.py 預設值**（最低優先級）
   ```python
   Field(default="postgres", ...)
   ```

### 主要配置參數

```env
# Logging
GATEWAY_LOG_LEVEL=INFO                              # 日誌級別: DEBUG, INFO, WARNING, ERROR

# JWT Settings
GATEWAY_JWT_SECRET_KEY=your-secret-key-here        # JWT 密鑰 (在生產環境中更改！)
GATEWAY_JWT_ALGORITHM=HS256                        # JWT 算法

# Database
GATEWAY_GATEWAY_DB_HOST=postgres                   # 數據庫主機 (Docker: postgres, 本地: localhost)
GATEWAY_GATEWAY_DB_PORT=5432                       # 數據庫端口 (Docker: 5432, 本地: 5434)
GATEWAY_GATEWAY_DB_USER=admin                      # 數據庫用戶
GATEWAY_GATEWAY_DB_PASSWORD=admin123               # 數據庫密碼 (在生產環境中更改！)
GATEWAY_GATEWAY_DB_NAME=mydb                       # 數據庫名稱
GATEWAY_GATEWAY_DB_POOL_MIN_SIZE=1                 # 最小連接池大小
GATEWAY_GATEWAY_DB_POOL_MAX_SIZE=10                # 最大連接池大小

# Redis Cache
GATEWAY_REDIS_HOST=192.168.157.165                 # Redis 主機
GATEWAY_REDIS_PORT=6381                            # Redis 端口
GATEWAY_REDIS_DB=0                                 # Redis 數據庫索引

# Kafka
GATEWAY_KAFKA_BOOTSTRAP_SERVERS=[...]              # Kafka 服務器列表
GATEWAY_KAFKA_TOPICS={...}                         # Kafka 主題映射

# External Services
GATEWAY_OBJECT_STORAGE_URL=http://192.168.157.165:8000  # MinIO URL
GATEWAY_QDRANT_HOST=192.168.157.124                # Qdrant 主機
GATEWAY_QDRANT_PORT=8801                           # Qdrant 端口

# Request Settings
GATEWAY_REQUEST_TIMEOUT=60.0                       # 請求超時（秒）
```

### 環境檔案

- **`.env`** - 當前活動配置（不要提交到 Git）
- **`.env.docker`** - Docker 環境專用變數
- **`.env.local`** - 本地開發環境變數
- **`.env.example`** - 配置示例（可提交到 Git）

---

## 🔧 開發指南

### 安裝開發依賴

```bash
# 使用 requirements.txt 安裝所有依賴
pip install -r requirements.txt

# 或只安裝基本依賴
pip install fastapi uvicorn sqlalchemy asyncpg
```

### 運行應用

```bash
# 開發模式（自動重載）
uvicorn app.main:app --reload --port 8012

# 生產模式
uvicorn app.main:app --host 0.0.0.0 --port 8012
```

### 運行測試

```bash
# 使用 pytest（如果已安裝）
pytest

# 查看覆蓋率
pytest --cov=app
```

### 代碼格式化

```bash
# 使用 black 格式化代碼
black app/

# 使用 isort 整理導入
isort app/

# 使用 flake8 檢查代碼風格
flake8 app/
```

---

## 🐛 故障排除

### 問題 1: 容器無法啟動

**症狀**: Docker 容器立即退出

**診斷**:
```bash
./deploy.sh logs-gateway
```

**常見原因**:
- 環境變數配置不正確
- 外部服務不可訪問
- 端口被占用

### 問題 2: 無法連接 PostgreSQL

**症狀**: `psycopg2.OperationalError: could not translate host name "postgres"`

**診斷**:
```bash
./deploy.sh health
docker network inspect vie-api-gateway_vie-api-gateway
```

**解決方案**:
```bash
# 重啟服務
./deploy.sh restart

# 或清理並重新啟動
./deploy.sh down
./deploy.sh up
```

### 問題 3: 端口已被占用

**症狀**: `Error starting userland proxy: listen tcp 0.0.0.0:8012`

**診斷**:
```bash
lsof -i :8012
```

**解決方案**:
- 終止占用的進程，或
- 在 `docker-compose.yml` 中修改端口映射

### 問題 4: 外部服務無法訪問

**診斷**:
```bash
docker-compose exec api-gateway ping 192.168.157.165
curl http://192.168.157.165:6381  # 測試 Redis
```

**解決方案**:
- 驗證防火牆設置
- 驗證外部服務正在運行
- 檢查 IP 地址和端口配置

### 獲取更多幫助

```bash
# 查看完整日誌
./deploy.sh logs -f

# 進入容器進行診斷
./deploy.sh shell

# 查看部署文檔
cat DOCKER_DEPLOYMENT.md
```

---

## 🔐 安全建議

### 生產環境檢查清單

- [ ] 更改所有默認密碼
- [ ] 生成強大的 JWT 密鑰
- [ ] 配置 HTTPS/TLS 證書
- [ ] 設置防火牆規則
- [ ] 啟用日誌監控和警報
- [ ] 定期備份數據庫
- [ ] 使用密鑰管理服務（如 Vault）
- [ ] 限制容器資源（CPU、內存）
- [ ] 進行安全審計
- [ ] 禁用調試模式

### 密碼策略

```bash
# 生成強密碼
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 或使用 openssl
openssl rand -base64 32
```

---

## 📊 監控和日誌

### 查看日誌

```bash
# 查看所有服務日誌
./deploy.sh logs -f

# 查看特定服務
./deploy.sh logs-gateway -f

# 限制日誌行數
./deploy.sh logs --tail=100
```

### 日誌格式

應用使用 JSON 格式的結構化日誌：

```json
{
  "timestamp": "2025-10-17T12:34:56.789Z",
  "log_level": "INFO",
  "name": "app.routers.auth",
  "message": "User authenticated",
  "user_id": "user123"
}
```

### 性能監控

```bash
# 查看容器資源使用情況
docker stats

# 查看特定容器
docker stats vie-api-gateway

# 查看數據庫連接
./deploy.sh db-shell
# SELECT datname, count(*) FROM pg_stat_activity GROUP BY datname;
```

---

## 📈 性能優化

### 調整數據庫連接池

編輯 `.env.docker`:
```env
GATEWAY_GATEWAY_DB_POOL_MAX_SIZE=20
```

### 優化 Docker 資源限制

編輯 `docker-compose.yml`:
```yaml
api-gateway:
  resources:
    limits:
      cpus: '2'
      memory: 1G
    reservations:
      cpus: '1'
      memory: 512M
```

### 啟用 Redis 快取搜索

使用帶 `_with_cache` 後綴的搜索端點獲得更快的性能。

---

## 🤝 貢獻指南

### 開發工作流程

1. 創建特性分支
   ```bash
   git checkout -b feature/your-feature
   ```

2. 進行更改並測試
   ```bash
   ./deploy.sh up
   # 測試更改
   ./deploy.sh down
   ```

3. 提交更改
   ```bash
   git add .
   git commit -m "feat: description of changes"
   git push origin feature/your-feature
   ```

4. 創建 Pull Request

### 代碼風格

- 遵循 PEP 8 標準
- 使用類型提示
- 包含文檔字符串
- 編寫單元測試

---

## 📚 文檔

| 文檔 | 描述 |
|------|------|
| [README.md](./README.md) | 此文件 - 項目概述 |
| [QUICK_START.md](./QUICK_START.md) | 快速開始指南（5 分鐘） |
| [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) | 完整 Docker 部署文檔 |
| [FILES_GUIDE.md](./FILES_GUIDE.md) | 項目文件結構說明 |
| [DEPLOYMENT_COMPLETE.md](./DEPLOYMENT_COMPLETE.md) | 部署完成總結 |
| [CHECKLIST.md](./CHECKLIST.md) | 部署驗證清單 |
| [SUMMARY.txt](./SUMMARY.txt) | 視覺化部署摘要 |

---

## 📅 更新日誌

### v1.0 (2025-10-17)

**初始版本**
- ✅ FastAPI 應用網關
- ✅ JWT 認證
- ✅ PostgreSQL 數據庫
- ✅ Redis 緩存
- ✅ Kafka 消息隊列
- ✅ Qdrant 搜索
- ✅ 完整 Docker 支持
- ✅ 詳細文檔

---

## 📞 支持

### 獲取幫助

1. **快速問題**: 查看 [QUICK_START.md](./QUICK_START.md)
2. **配置問題**: 查看 [FILES_GUIDE.md](./FILES_GUIDE.md)
3. **部署問題**: 查看 [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md)
4. **故障排除**: 見本文件的[故障排除](#故障排除)部分

### 報告問題

```bash
# 收集診斷信息
./deploy.sh ps
./deploy.sh logs > diagnostics.log
./deploy.sh health
```

---

## 📄 許可證

此項目根據 MIT 許可證授權。詳見 [LICENSE](LICENSE) 文件。

---

## 👥 相關項目

- [VIE API Gateway](https://github.com/DHT-AI-Studio/VIE01) - 主項目
- [FastAPI](https://fastapi.tiangolo.com/) - 文檔
- [Docker](https://www.docker.com/get-started/) - 容器指南
- [PostgreSQL](https://www.postgresql.org/docs/) - 數據庫文檔

---

## 🎯 下一步

1. 閱讀 [QUICK_START.md](./QUICK_START.md)
2. 運行 `./deploy.sh up`
3. 訪問 http://localhost:8012/docs
4. 探索 API 文檔
5. 自定義配置

---

**最後更新**: 2025-10-17  
**版本**: 1.0  
**狀態**: ✅ 準備就緒  
**維護者**: DHT AI Studio

---

## 🌟 特別感謝

感謝所有貢獻者、測試人員和用戶的支持！

**快樂編碼！** 🚀
