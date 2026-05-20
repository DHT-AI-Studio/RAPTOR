FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "redis[asyncio]" \
    python-dotenv \
    prometheus-fastapi-instrumentator \
    httpx \
    "pydantic>=2.0" \
    loguru \
    asyncpg \
    "a2a-sdk[http-server]==0.3.26" \
    "smolagents[litellm]" \
    duckduckgo-search \
    litellm \
    opencc-python-reimplemented

COPY app/ .

EXPOSE 8030

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8030"]
