# Qdrant Search System Docker

This project is built on the **Qdrant vector database** and the **BAAI/bge-m3 semantic embedding model**, providing a high-performance semantic similarity search service for multimedia content.
The system supports **Video**, **Audio**, **Image**, and **Document** search, integrated with a **Redis caching mechanism** to enhance query performance and response speed.

---

## 🚀 Quick Start

### 1. Install Dependencies

Please install Python dependencies first:

```bash
pip install -r requirements.txt
```

### 2. Start Services

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View live logs
docker-compose logs -f
```

### 3. Verify Deployment

```bash
# Check Qdrant health status
curl http://localhost:6334/health

# Check Video Search API
curl http://localhost:8821/health

# Other APIs can be tested at ports 8822, 8823, 8824, 8815
```

---

## 🧩 System Services Overview

| Service Name        | Container Port | Host Port | Description             | API Docs URL                                                       |
| ------------------- | -------------- | --------- | ----------------------- | ------------------------------------------------------------------ |
| Qdrant Database     | 6333           | 6334      | Vector database service | [http://localhost:6334/dashboard](http://localhost:6334/dashboard) |
| Insert API          | 8815           | 8815      | Vector data insertion   | [http://localhost:8815/docs](http://localhost:8815/docs)           |
| Video Search API    | 8811           | 8821      | Video content search    | [http://localhost:8821/docs](http://localhost:8821/docs)           |
| Audio Search API    | 8812           | 8822      | Audio content search    | [http://localhost:8822/docs](http://localhost:8822/docs)           |
| Document Search API | 8813           | 8823      | Document content search | [http://localhost:8823/docs](http://localhost:8823/docs)           |
| Image Search API    | 8814           | 8824      | Image content search    | [http://localhost:8824/docs](http://localhost:8824/docs)           |

---

## 📘 Usage Examples

### 1. Insert Data

```bash
# Create a JSON data file
cat > data.json << EOF
{
  "id": "unique-id-123",
  "payload": {
    "type": "videos",
    "embedding_type": "summary",
    "filename": "demo.mp4",
    "summary": "This is a tutorial video about machine learning.",
    "status": "active"
  }
}
EOF

# Upload data
curl -X POST "http://localhost:8815/insert_json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.json"
```

---

### 2. Search Videos

```bash
curl -X POST "http://localhost:8821/video_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "machine learning tutorial",
    "embedding_type": "summary",
    "limit": 5
  }'
```

---

### 3. Conditional Search

```bash
curl -X POST "http://localhost:8821/video_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "deep learning",
    "embedding_type": "text",
    "filename": ["demo.mp4", "tutorial.mp4"],
    "speaker": ["SPEAKER_01"],
    "limit": 10
  }'
```

---

### 4. Search Audio

```bash
curl -X POST "http://localhost:8822/audio_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "product introduction",
    "embedding_type": "text",
    "speaker": ["SPEAKER_01", "SPEAKER_02"],
    "limit": 5
  }'
```

---

### 5. Search Documents

```bash
curl -X POST "http://localhost:8823/document_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "financial report analysis",
    "embedding_type": "text",
    "source": "pdf",
    "limit": 5
  }'
```

---

### 6. Search Images

```bash
curl -X POST "http://localhost:8824/image_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "a scenic photo with blue sky and white clouds",
    "embedding_type": "summary",
    "source": "jpg",
    "limit": 5
  }'
```

---

## 🧱 Data Structure

### 🔍 Search Request Format

```json
{
  "query_text": "search keywords",
  "embedding_type": "summary",    # "summary" or "text"
  "type": "videos",               # optional: audio/video/document/image
  "filename": ["file1.mp4"],      # optional
  "speaker": ["SPEAKER_01"],      # optional (audio/video only)
  "source": "jpg",                # optional (image/document only)
  "limit": 5                      # number of results (1–100)
}
```

---

### 💾 Insertion Data Format

#### Video / Audio

```json
{
  "id": "unique-uuid",
  "payload": {
    "type": "videos",
    "embedding_type": "summary",
    "filename": "example.mp4",
    "summary": "Content summary",
    "text": "Full text content",
    "speaker": "SPEAKER_01",
    "start_time": 0.0,
    "end_time": 120.5,
    "status": "active"
  }
}
```

#### Image / Document

```json
{
  "id": "unique-uuid",
  "payload": {
    "type": "images",
    "embedding_type": "summary",
    "filename": "photo.jpg",
    "summary": "Image description",
    "text": "Extracted text content",
    "source": "jpg",
    "status": "active"
  }
}
```

---

## ⚡ Cache Configuration

You can configure Redis cache parameters in each API module:

```python
cm = CacheManager(
    host='192.168.157.123',    # Redis host
    port=7000,                 # Redis port
    password="dht888888",      # Redis password
    max_connections=1000,      # Maximum connections
    ttl=3600,                  # Cache time-to-live (seconds)
    ttl_multiplier=1e-2,       # TTL multiplier
    is_cluster=False           # Set to True if using Redis Cluster
)
```

---

## 🐳 Docker Management Commands

### Basic Operations

```bash
docker-compose up -d          # Start all services
docker-compose down           # Stop all services
docker-compose restart video_search_api
docker-compose logs -f video_search_api
docker-compose exec video_search_api bash
```

### Cleanup & Rebuild

```bash
docker-compose down           # Stop containers (keep volumes)
docker-compose down -v        # Stop and remove all volumes
docker-compose build --no-cache
docker-compose up -d --build  # Rebuild and start services
```

### Resource Management

```bash
docker stats                  # Check resource usage
docker images | grep search_api
docker image prune -a         # Remove unused images
```

