FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY cache_manager/ ./cache_manager/

COPY api_*.py .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

EXPOSE 8811 8812 8813 8814 8815

CMD ["python", "-m", "uvicorn", "api_video_search_with_cache:app", "--host", "0.0.0.0", "--port", "8811"]
