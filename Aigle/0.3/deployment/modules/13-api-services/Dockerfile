# # API Gateway Dockerfile
# # Aigle Platform v0.2

# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY app/ /app/app/
# COPY services.yaml /app/

# # Create directories
# RUN mkdir -p /app/data /app/logs

# # Set environment variables
# ENV PYTHONPATH=/app
# ENV PYTHONUNBUFFERED=1

# # Expose port
# EXPOSE 8012

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
#     CMD curl -f http://localhost:8012/health || exit 1

# # Run the application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8012"]


# 使用 slim 版本 Python 3.10
FROM python:3.10-slim

# 設定環境變數
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt requirements.txt

# # 複製 wheels 資料夾（如果存在）
# COPY wheels /wheels

# 安裝依賴，優先使用 wheels，如果沒有則從 PyPI 安裝
RUN if [ -d /wheels ] && [ "$(ls -A /wheels 2>/dev/null | grep -v '^\.gitkeep$')" ]; then \
        echo "Installing from wheels..."; \
        pip install --no-index --find-links=/wheels -r requirements.txt; \
    else \
        echo "No wheels found, installing from PyPI..."; \
        pip install -r requirements.txt; \
    fi

# # 複製自訂 keycloak 套件並安裝
# COPY keycloak /app/keycloak
# RUN pip install --no-deps -e /app/keycloak

# 複製其他專案程式碼
#COPY . /src
COPY ./app /app
COPY .env /app

# 暴露應用端口
EXPOSE 8012

# 啟動 uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]
