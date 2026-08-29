# Vision Processing Worker
# Aigle Platform v0.2
# GPU required for:
#   image_processing       (InternVL 3.5-4B — VLM description + OCR)
#   video_frame_description (InternVL 3.5-4B — frame VLM)
#   video_ocr_frame        (PaddleOCR)

# CUDA 12.8 is the minimum runtime that ships sm_120 (Blackwell, RTX 50-series) kernels.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# PaddlePaddle GPU (cu129) — required for sm_120 (Blackwell / RTX 50-series).
# paddlepaddle-gpu==3.3.0 is the first official release shipping sm_120 kernels.
RUN pip install --no-cache-dir --timeout 600 paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
    --extra-index-url https://pypi.org/simple

# PyTorch built against CUDA 12.8 — required for RTX 50-series (sm_120).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1

# Remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dispatcher entrypoint + all service code
COPY main.py /app/
COPY services/ /app/services/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
