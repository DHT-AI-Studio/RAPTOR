# RAPTOR Training Service Dockerfile
# Multi-stage build for optimized production deployment

# --- Stage 1: Base Image ---
# 使用 NVIDIA PyTorch/CUDA 映像檔以支持 GPU 和 DeepSpeed
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Upgrade system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libaio-dev \
    time \
    && rm -rf /var/lib/apt/lists/*

# --- 環境變數 ---
# 設定工作目錄
WORKDIR /app

# 防止 Python 寫入 .pyc 檔案
ENV PYTHONDONTWRITEBYTECODE=1
# 確保 Python 輸出是即時的，方便在 Docker logs 中查看
ENV PYTHONUNBUFFERED=1

# --- 安裝 Python 依賴 ---
# 使用 requirements.lock.txt 確保版本一致（從 uv.lock 導出）
COPY requirements.lock.txt ./

COPY pyproject.toml ./

# 安裝 uv 並使用鎖定文件安裝依賴
# --system: 安裝到系統 Python（Docker 容器中不需要虛擬環境）
# --no-cache: 不使用快取，確保全新安裝
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.lock.txt

# --- DeepSpeed 專門編譯步驟 ---
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.18.2 --no-cache-dir \
    && rm -rf /root/.cache/pip

# --- 複製應用程式程式碼 ---
COPY src/ ./src/
COPY config.yaml ./

# --- 端口暴露 ---
# EXPOSE ${PORT_FASTAPI}

# --- 啟動命令 ---
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "${PORT_FASTAPI}", "--reload"]

CMD sh -c "uvicorn src.main:app --host 0.0.0.0 --port ${PORT_FASTAPI:-8009} --reload"