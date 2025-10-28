# Dockerfile - AI Model Lifecycle Management Platform
# Multi-stage build for optimized production deployment

# --- Stage 1: Base Image ---
# 使用一個官方的、輕量級的 Python 映像檔作為基礎
FROM python:3.10-slim

# Upgrade system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- 環境變數 ---
# 設定工作目錄
WORKDIR /app

# 防止 Python 寫入 .pyc 檔案
ENV PYTHONDONTWRITEBYTECODE=1
# 確保 Python 輸出是即時的，方便在 Docker logs 中查看
ENV PYTHONUNBUFFERED=1

# --- 安裝 Python 依賴 ---
# 複製需求文件
COPY requirements.txt .

# 使用 --no-cache-dir 來減少映像檔大小
RUN pip install --no-cache-dir -r requirements.txt

# --- 複製應用程式程式碼 ---
# 將 src 目錄下的所有內容複製到工作目錄的 src 子目錄中
COPY ./src ./src

# --- 端口暴露 ---
# 我們的 FastAPI 應用預計在 8000 端口運行
EXPOSE ${FASTAPI_PORT}

# --- 啟動命令 ---
# 當容器啟動時，執行此命令來啟動 Uvicorn 伺服器
# 綁定到 0.0.0.0 以便從容器外部訪問
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "${FASTAPI_PORT}", "--reload"]