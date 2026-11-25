FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /server

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

EXPOSE 8027

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8027/health || exit 1

# app/__main__.py dispatches on MCP_TRANSPORT: "http" (default, uvicorn on
# app.main:app) or "stdio" (app.stdio_main). Invoking uvicorn directly here
# would bypass that switch and MCP_TRANSPORT=stdio would have no effect.
CMD ["python", "-m", "app"]
