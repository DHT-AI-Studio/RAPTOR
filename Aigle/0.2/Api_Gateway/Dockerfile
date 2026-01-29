# # ============================================================================
# # VIE API Gateway - Multi-stage Docker Build
# # ============================================================================

# # Stage 1: Base image with Python
# # FROM python:3.11-slim as base

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     g++ \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Stage 2: Dependencies
# FROM base as builder

# WORKDIR /tmp

# # Copy local editable packages
# COPY keycloak ./keycloak

# # Copy requirements file
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --user --no-cache-dir -r requirements.txt

# # Stage 3: Runtime
# FROM base as runtime

# # Create app user for security
# RUN useradd -m -u 1000 appuser && \
#     mkdir -p /app /app/logs /app/data && \
#     chown -R appuser:appuser /app

# # Set working directory
# WORKDIR /app

# # Copy Python dependencies from builder
# COPY --from=builder /root/.local /home/appuser/.local

# # Copy application code
# COPY --chown=appuser:appuser ./app ./app
# COPY --chown=appuser:appuser ./data ./data

# # Copy services configuration (optional, can be mounted as volume)
# # COPY --chown=appuser:appuser services.yaml .

# # Update PATH to include user-installed packages
# ENV PATH=/home/appuser/.local/bin:$PATH

# # Switch to non-root user
# USER appuser

# # Expose the application port
# EXPOSE 8012

# # Health check
# HEALTHCHECK --interval=1h --timeout=10s --start-period=40s --retries=3 \
#     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8012/health').read()" || exit 1

# # Run the application with uvicorn
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8012", "--workers", "1"]


# # -------------------- dockerfile(minnie) --------------------
# # FROM python:3.11-slim

# # 設定工作目錄
# WORKDIR /app

# # 安裝系統依賴 + 安裝 keycloak + 安裝其他依賴
# # 使用一條 RUN 減少層數
# COPY ./keycloak ./keycloak
# COPY requirements.txt .

# RUN apt-get update && \
#     apt-get install -y build-essential && \
#     pip install --editable ./keycloak && \
#     pip install -r requirements.txt && \
#     apt-get remove -y build-essential && \
#     apt-get autoremove -y && \
#     rm -rf /var/lib/apt/lists/*

# # 複製 app 代碼
# COPY ./app ./app

# # uvicorn 啟動
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8012"]

# # -------------------- minnie --------------------
# FROM python:3.11-slim

# # 設定工作目錄
# WORKDIR /app

# # 安裝系統依賴（build C 擴展需要）
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# # 複製 keycloak 並安裝
# COPY ./keycloak ./keycloak
# RUN pip install --no-cache-dir --editable ./keycloak

# # 複製 app 代碼
# COPY ./app ./app

# # 安裝其他依賴
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # uvicorn 啟動指令
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8012"]


# # =========================
# # 基礎映像
# # =========================
# FROM python:3.11-slim

# # =========================
# # 環境變數
# # =========================
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     PATH=/home/appuser/.local/bin:$PATH

# # =========================
# # 系統依賴（編譯 C 擴展用）
# # =========================
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         build-essential \
#         wget \
#         curl \
#         git \
#     && rm -rf /var/lib/apt/lists/*

# # =========================
# # 建立非 root 用戶
# # =========================
# RUN useradd -m -u 1000 appuser
# USER appuser
# WORKDIR /app

# # =========================
# # ML / Vector wheels 安裝
# # =========================
# COPY wheels/ml /wheels/ml
# COPY requirements-ml.txt .

# RUN pip install --no-cache-dir --no-index \
#     --find-links=/wheels/ml \
#     -r requirements-ml.txt

# # =========================
# # 安裝 local package（keycloak）
# # =========================
# # 切換回 root
# USER root

# COPY ./keycloak ./keycloak
# RUN pip install --no-cache-dir --editable ./keycloak

# # 切回 appuser
# USER appuser
# WORKDIR /app

# # =========================
# # 安裝其他小套件
# # =========================
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # =========================
# # 複製應用程式
# # =========================
# COPY ./app ./app

# # =========================
# # 開放端口 & CMD
# # =========================
# EXPOSE 8012
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8012"]



# 使用 slim 版本 Python 3.11
FROM python:3.11-slim

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

# 複製 requirements.txt、wheels 資料夾
COPY requirements.txt ./requirements.txt
COPY wheels /wheels

# 安裝依賴，優先使用 wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# # 複製自訂 keycloak 套件並安裝
# COPY keycloak /app/keycloak
# RUN pip install --no-deps -e /app/keycloak

# 複製其他專案程式碼
COPY . /app

# 暴露應用端口
EXPOSE 8012

# 啟動 uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8012"]
