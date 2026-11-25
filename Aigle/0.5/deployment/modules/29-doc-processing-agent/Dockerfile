FROM python:3.11-slim

WORKDIR /app

# System deps (pymupdf needs libmupdf; python-docx needs libxml2)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmupdf-dev \
        libxml2-dev \
        libxslt1-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV DA_HOST=0.0.0.0 \
    DA_PORT=8290 \
    DA_LOG_LEVEL=INFO

EXPOSE 8290

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8290/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8290"]
