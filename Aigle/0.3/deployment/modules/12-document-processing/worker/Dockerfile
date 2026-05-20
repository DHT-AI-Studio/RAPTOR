# Document Processing Worker
# Aigle Platform v0.2
# GPU required for document_analysis_service (InternVL OCR/VLM)

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
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

# PaddlePaddle GPU (document_analysis_service OCR) — requires custom index
RUN pip install --no-cache-dir paddlepaddle-gpu==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Dispatcher entrypoint + all service code
COPY main.py /app/
COPY services/ /app/services/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
